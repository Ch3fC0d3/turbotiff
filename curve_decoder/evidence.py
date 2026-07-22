"""Structured decoder evidence and calibrated observation scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import DecoderConfig


@dataclass
class CurveEvidence:
    centerline_probability: np.ndarray
    stroke_probability: Optional[np.ndarray] = None
    distance_field: Optional[np.ndarray] = None
    direction_field: Optional[np.ndarray] = None
    grid_probability: Optional[np.ndarray] = None
    classic_probability: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    wrap_probability_right_to_left: Optional[np.ndarray] = None
    wrap_probability_left_to_right: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(np.asarray(self.centerline_probability).shape)

    def validate(self) -> tuple[int, int]:
        center = np.asarray(self.centerline_probability)
        if center.ndim != 2 or not center.size:
            raise ValueError("centerline_probability must be a non-empty [H, W] array")
        height, width = center.shape
        for name in ("stroke_probability", "distance_field", "grid_probability", "classic_probability", "valid_mask"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != (height, width):
                raise ValueError(f"{name} must match centerline dimensions")
        if self.direction_field is not None and np.asarray(self.direction_field).shape != (2, height, width):
            raise ValueError("direction_field must have shape [2, H, W]")
        for name in ("wrap_probability_right_to_left", "wrap_probability_left_to_right"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).reshape(-1).size != height:
                raise ValueError(f"{name} must contain one value per row")
        return height, width


def _soft_field(value, shape, default=0.0):
    if value is None:
        return np.full(shape, float(default), dtype=np.float32)
    field = np.asarray(value, dtype=np.float32)
    field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
    if field.size and float(field.max()) > 1.0:
        field = field / 255.0
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def normalized_direction(evidence: CurveEvidence) -> np.ndarray:
    height, width = evidence.validate()
    if evidence.direction_field is None:
        direction = np.zeros((2, height, width), dtype=np.float32)
        direction[1] = 1.0
        return direction
    direction = np.nan_to_num(np.asarray(evidence.direction_field, dtype=np.float32), nan=0.0)
    norm = np.linalg.norm(direction, axis=0, keepdims=True)
    return np.divide(direction, np.maximum(norm, 1e-6), out=np.zeros_like(direction), where=norm > 1e-6)


def calculate_observation_score(evidence: CurveEvidence, config: DecoderConfig) -> np.ndarray:
    shape = evidence.validate()
    center = _soft_field(evidence.centerline_probability, shape)
    stroke = _soft_field(evidence.stroke_probability, shape)
    distance = _soft_field(evidence.distance_field, shape)
    classic = _soft_field(evidence.classic_probability, shape)
    grid = _soft_field(evidence.grid_probability, shape)
    weighted_fields = [(center, float(config.centerline_weight))]
    if evidence.stroke_probability is not None:
        weighted_fields.append((stroke, float(config.stroke_weight)))
    if evidence.distance_field is not None:
        weighted_fields.append((distance, float(config.distance_weight)))
    if evidence.classic_probability is not None:
        weighted_fields.append((classic, float(config.classic_weight)))
    positive_total = max(1e-6, sum(weight for _, weight in weighted_fields))
    score = sum(field * weight for field, weight in weighted_fields) / positive_total
    overlap = np.maximum(center, distance)
    score -= float(config.grid_weight) * grid * (1.0 - float(config.grid_overlap_relief) * overlap)
    if evidence.valid_mask is not None:
        valid = np.asarray(evidence.valid_mask, dtype=bool)
        score = np.where(valid, score, 0.0)
    return np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0).astype(np.float32)
