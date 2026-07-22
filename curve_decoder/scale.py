"""Scale conversion using decoder-provided wrap indexes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleConfig:
    track_width: int
    left_value: float
    right_value: float
    scale_type: str = "linear"


def path_to_values(x_by_row: np.ndarray, wrap_index_by_row: np.ndarray, config: ScaleConfig) -> np.ndarray:
    x = np.asarray(x_by_row, dtype=np.float64).reshape(-1)
    wrap = np.asarray(wrap_index_by_row, dtype=np.int32).reshape(-1)
    if x.shape != wrap.shape:
        raise ValueError("Visible X and wrap-index arrays must have equal length")
    if int(config.track_width) < 2:
        raise ValueError("Scale conversion requires a track width of at least two pixels")
    values = np.full(x.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    fraction = np.clip(x[valid], 0.0, float(config.track_width - 1)) / float(config.track_width - 1)
    left = float(config.left_value)
    right = float(config.right_value)
    scale_type = str(config.scale_type or "linear").lower().strip()
    if scale_type == "log" and left > 0 and right > 0:
        base = np.power(10.0, np.log10(left) + fraction * (np.log10(right) - np.log10(left)))
        values[valid] = base * np.power(right / left, wrap[valid])
    else:
        cycle = right - left
        values[valid] = left + fraction * cycle + wrap[valid] * cycle
    return values.astype(np.float32)

