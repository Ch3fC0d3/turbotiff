"""Serializable path, wrap event, and rendering-segment structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CurveSegment:
    start_row: int
    end_row: int
    wrap_index: int
    points: list[tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "start_row": int(self.start_row),
            "end_row": int(self.end_row),
            "wrap_index": int(self.wrap_index),
            "points": [[float(x), float(y)] for x, y in self.points],
        }


@dataclass
class CurvePathResult:
    x_by_row: np.ndarray
    unwrapped_x_by_row: np.ndarray
    wrap_index_by_row: np.ndarray
    slope_by_row: np.ndarray
    confidence_by_row: np.ndarray
    wrap_events: list[dict] = field(default_factory=list)
    visible_segments: list[CurveSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    observation_score_by_row: np.ndarray | None = None
    transition_score_by_row: np.ndarray | None = None
    break_rows: set[int] = field(default_factory=set)

    def to_dict(self, include_arrays: bool = True) -> dict:
        payload = {
            "wrap_events": list(self.wrap_events),
            "visible_segments": [segment.to_dict() for segment in self.visible_segments],
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            payload.update({
                "x_by_row": self.x_by_row.tolist(),
                "unwrapped_x_by_row": self.unwrapped_x_by_row.tolist(),
                "wrap_index_by_row": self.wrap_index_by_row.tolist(),
                "slope_by_row": self.slope_by_row.tolist(),
                "confidence_by_row": self.confidence_by_row.tolist(),
            })
        return payload

