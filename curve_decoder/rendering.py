"""Convert topology paths into visible segments without cross-track connectors."""

from __future__ import annotations

import numpy as np

from .path_result import CurveSegment


def build_visible_segments(
    x_by_row: np.ndarray,
    wrap_index_by_row: np.ndarray,
    break_rows=None,
    discontinuity_threshold: float = 24.0,
) -> list[CurveSegment]:
    x = np.asarray(x_by_row, dtype=np.float32).reshape(-1)
    wrap = np.asarray(wrap_index_by_row, dtype=np.int32).reshape(-1)
    if x.shape != wrap.shape:
        raise ValueError("Visible X and wrap-index arrays must have equal length")
    breaks = {int(row) for row in (break_rows or set())}
    segments = []
    points = []
    start = None
    active_wrap = 0

    def finish(end_row):
        nonlocal points, start, active_wrap
        if points:
            segments.append(CurveSegment(int(start), int(end_row), int(active_wrap), points))
        points = []
        start = None

    for row in range(x.size):
        valid = np.isfinite(x[row])
        split = row in breaks
        if points and valid:
            split = split or wrap[row] != active_wrap or abs(float(x[row]) - points[-1][0]) > float(discontinuity_threshold)
        if not valid:
            finish(row - 1)
            continue
        if split:
            finish(row - 1)
        if not points:
            start = row
            active_wrap = int(wrap[row])
        points.append((float(x[row]), float(row)))
    finish(x.size - 1)
    return segments


def has_cross_track_connector(segments: list[CurveSegment], track_width: int, threshold_fraction: float = 0.5) -> bool:
    threshold = max(1.0, float(track_width) * float(threshold_fraction))
    for segment in segments:
        for first, second in zip(segment.points, segment.points[1:]):
            if abs(float(second[0]) - float(first[0])) > threshold:
                return True
    return False

