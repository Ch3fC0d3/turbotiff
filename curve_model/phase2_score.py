"""Soft fusion and confidence calculations for Phase 2 neural outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .integration import normalize_probability


@dataclass(frozen=True)
class Phase2ScoreConfig:
    centerline_weight: float = 0.45
    distance_weight: float = 0.30
    stroke_weight: float = 0.20
    grid_weight: float = 0.15
    grid_overlap_relief: float = 0.75
    skeleton_bonus: float = 0.08
    phase2_weight: float = 0.70
    classic_weight: float = 0.30


def normalize_direction_field(direction: np.ndarray) -> np.ndarray:
    field = np.asarray(direction, dtype=np.float32)
    if field.ndim != 3 or field.shape[0] != 2:
        raise ValueError("Direction field must have shape [2, H, W]")
    field = np.where(np.isfinite(field), field, 0.0)
    norm = np.linalg.norm(field, axis=0, keepdims=True)
    return np.divide(field, np.maximum(norm, 1e-6), out=np.zeros_like(field), where=norm > 1e-6)


def build_phase2_trace_score(
    stroke_probability: np.ndarray,
    centerline_probability: np.ndarray,
    distance_field: np.ndarray,
    direction_field: np.ndarray,
    grid_probability: np.ndarray,
    classic_probability: Optional[np.ndarray] = None,
    skeleton_probability: Optional[np.ndarray] = None,
    config: Optional[Phase2ScoreConfig] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    config = config or Phase2ScoreConfig()
    config = Phase2ScoreConfig(
        centerline_weight=float(np.clip(float(config.centerline_weight), 0.0, 2.0)),
        distance_weight=float(np.clip(float(config.distance_weight), 0.0, 2.0)),
        stroke_weight=float(np.clip(float(config.stroke_weight), 0.0, 2.0)),
        grid_weight=float(np.clip(float(config.grid_weight), 0.0, 2.0)),
        grid_overlap_relief=float(np.clip(float(config.grid_overlap_relief), 0.0, 1.0)),
        skeleton_bonus=float(np.clip(float(config.skeleton_bonus), 0.0, 2.0)),
        phase2_weight=float(np.clip(float(config.phase2_weight), 0.0, 2.0)),
        classic_weight=float(np.clip(float(config.classic_weight), 0.0, 2.0)),
    )
    stroke = normalize_probability(stroke_probability)
    centerline = normalize_probability(centerline_probability)
    distance = np.clip(np.nan_to_num(distance_field, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    grid = normalize_probability(grid_probability)
    if not (stroke.shape == centerline.shape == distance.shape == grid.shape):
        raise ValueError("All Phase 2 scalar fields must have the same shape")
    direction = normalize_direction_field(direction_field)
    if direction.shape[1:] != stroke.shape:
        raise ValueError("Direction field dimensions do not match scalar fields")

    positive_weight = max(1e-6, config.centerline_weight + config.distance_weight + config.stroke_weight)
    positive = (
        config.centerline_weight * centerline
        + config.distance_weight * distance
        + config.stroke_weight * stroke
    ) / positive_weight
    overlap_evidence = np.maximum(centerline, distance)
    grid_penalty = config.grid_weight * grid * (1.0 - config.grid_overlap_relief * overlap_evidence)
    score = positive - grid_penalty
    if skeleton_probability is not None:
        skeleton = normalize_probability(skeleton_probability)
        if skeleton.shape != score.shape:
            raise ValueError("Skeleton evidence dimensions do not match Phase 2 fields")
        score = score * (1.0 + config.skeleton_bonus * skeleton)
    score = np.clip(score, 0.0, 1.0)

    if classic_probability is not None:
        classic = normalize_probability(classic_probability)
        total = max(1e-6, config.phase2_weight + config.classic_weight)
        score = (config.phase2_weight * score + config.classic_weight * classic) / total
    return np.clip(score, 0.0, 1.0).astype(np.float32), direction, asdict(config)


def phase2_confidence(
    path_x: np.ndarray,
    score: np.ndarray,
    centerline_probability: np.ndarray,
    distance_field: np.ndarray,
    direction_field: np.ndarray,
    grid_probability: np.ndarray,
    classic_probability: Optional[np.ndarray] = None,
    low_threshold: float = 0.35,
) -> tuple[np.ndarray, dict]:
    path = np.asarray(path_x, dtype=np.float32).reshape(-1)
    height, width = score.shape
    confidence = np.zeros(height, dtype=np.float32)
    direction = normalize_direction_field(direction_field)
    classic = normalize_probability(classic_probability) if classic_probability is not None else None
    previous = None
    for row in range(min(height, path.size)):
        if not np.isfinite(path[row]):
            previous = None
            continue
        x = int(np.clip(round(float(path[row])), 0, width - 1))
        selected = float(score[row, x])
        candidates = score[row].copy()
        candidates[max(0, x - 2):min(width, x + 3)] = -1.0
        separation = max(0.0, selected - float(np.max(candidates)))
        direction_agreement = 0.5
        if previous is not None:
            transition = np.array([float(x - previous), 1.0], dtype=np.float32)
            transition /= max(float(np.linalg.norm(transition)), 1e-6)
            direction_agreement = 0.5 * (1.0 + float(np.dot(transition, direction[:, row, x])))
        classic_agreement = 0.5
        if classic is not None:
            classic_row = classic[row]
            classic_x = int(np.argmax(classic_row))
            classic_agreement = max(0.0, 1.0 - abs(classic_x - x) / max(1.0, width * 0.2))
        entropy = -(selected * np.log(max(selected, 1e-6)) + (1.0 - selected) * np.log(max(1.0 - selected, 1e-6))) / np.log(2.0)
        confidence[row] = np.clip(
            0.24 * float(centerline_probability[row, x])
            + 0.20 * float(distance_field[row, x])
            + 0.18 * separation
            + 0.16 * direction_agreement
            + 0.10 * (1.0 - float(grid_probability[row, x]))
            + 0.07 * classic_agreement
            + 0.05 * (1.0 - entropy),
            0.0,
            1.0,
        )
        previous = x
    low = confidence < float(low_threshold)
    longest = current = 0
    for value in low:
        current = current + 1 if value else 0
        longest = max(longest, current)
    finite_rows = np.isfinite(path[:height])
    selected_confidence = confidence[finite_rows] if finite_rows.any() else confidence
    summary = {
        "mean_confidence": float(np.mean(selected_confidence)) if selected_confidence.size else 0.0,
        "minimum_confidence": float(np.min(selected_confidence)) if selected_confidence.size else 0.0,
        "low_confidence_fraction": float(np.mean(low[finite_rows])) if finite_rows.any() else 1.0,
        "longest_low_confidence_run": int(longest),
        "low_confidence_threshold": float(low_threshold),
    }
    return confidence, summary
