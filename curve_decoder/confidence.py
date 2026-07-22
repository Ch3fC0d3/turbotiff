"""Row and wrap-event confidence for topology-aware paths."""

from __future__ import annotations

import numpy as np

from .config import DecoderConfig
from .evidence import CurveEvidence, normalized_direction, _soft_field


def calculate_confidence(
    x_by_row: np.ndarray,
    unwrapped_x_by_row: np.ndarray,
    observation_score: np.ndarray,
    evidence: CurveEvidence,
    alternative_margin: np.ndarray,
    config: DecoderConfig,
) -> tuple[np.ndarray, dict]:
    x = np.asarray(x_by_row, dtype=np.float32)
    unwrapped = np.asarray(unwrapped_x_by_row, dtype=np.float32)
    height, width = observation_score.shape
    direction = normalized_direction(evidence)
    grid = _soft_field(evidence.grid_probability, (height, width))
    confidence = np.zeros(height, dtype=np.float32)
    for row in range(height):
        if not np.isfinite(x[row]):
            continue
        column = int(np.clip(round(float(x[row])), 0, width - 1))
        observation = float(observation_score[row, column])
        margin = float(np.tanh(max(0.0, alternative_margin[row])))
        agreement = 0.5
        slope_margin = 1.0
        if row > 0 and np.isfinite(unwrapped[row - 1]):
            delta = float(unwrapped[row] - unwrapped[row - 1])
            vector = np.array([delta, 1.0], dtype=np.float32)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            agreement = 0.5 * (1.0 + float(np.dot(vector, direction[:, row, column])))
            slope_margin = max(0.0, 1.0 - abs(delta) / max(1.0, float(config.max_slope)))
        bridged = 1.0 if observation < 0.08 else 0.0
        confidence[row] = np.clip(
            0.40 * observation
            + 0.22 * margin
            + 0.18 * agreement
            + 0.10 * (1.0 - float(grid[row, column]))
            + 0.10 * slope_margin
            - 0.08 * bridged,
            0.0,
            1.0,
        )
    low = confidence < float(config.low_confidence_threshold)
    longest = current = 0
    for value in low:
        current = current + 1 if value else 0
        longest = max(longest, current)
    summary = {
        "mean_confidence": float(np.mean(confidence)) if confidence.size else 0.0,
        "minimum_confidence": float(np.min(confidence)) if confidence.size else 0.0,
        "longest_low_confidence_run": int(longest),
        "percentage_low_confidence_rows": float(100.0 * np.mean(low)) if low.size else 100.0,
        "low_confidence_threshold": float(config.low_confidence_threshold),
    }
    return confidence, summary

