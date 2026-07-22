"""Pure path editing helpers for points, breaks, and explicit wrap transitions."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .path_result import CurvePathResult
from .rendering import build_visible_segments


def _refresh(result: CurvePathResult, track_width: int) -> CurvePathResult:
    result.unwrapped_x_by_row = result.x_by_row + result.wrap_index_by_row.astype(np.float32) * float(track_width)
    result.slope_by_row = np.concatenate((np.array([0.0], dtype=np.float32), np.diff(result.unwrapped_x_by_row).astype(np.float32)))
    result.visible_segments = build_visible_segments(result.x_by_row, result.wrap_index_by_row, result.break_rows)
    events = []
    for row in np.flatnonzero(np.diff(result.wrap_index_by_row) != 0) + 1:
        before = int(result.wrap_index_by_row[row - 1])
        after = int(result.wrap_index_by_row[row])
        events.append({
            "row_before": int(row - 1), "row_after": int(row),
            "direction": "right_to_left" if after > before else "left_to_right",
            "wrap_index_before": before, "wrap_index_after": after,
            "x_before": float(result.x_by_row[row - 1]), "x_after": float(result.x_by_row[row]),
            "unwrapped_delta": float(result.unwrapped_x_by_row[row] - result.unwrapped_x_by_row[row - 1]),
            "confidence": float(min(result.confidence_by_row[row - 1], result.confidence_by_row[row])),
            "manual": True,
        })
    result.wrap_events = events
    return result


def move_points(result: CurvePathResult, updates: dict[int, float], track_width: int) -> CurvePathResult:
    edited = deepcopy(result)
    for row, x in updates.items():
        if 0 <= int(row) < edited.x_by_row.size:
            edited.x_by_row[int(row)] = float(np.clip(x, 0, track_width - 1))
    return _refresh(edited, track_width)


def add_path_break(result: CurvePathResult, row: int, track_width: int) -> CurvePathResult:
    edited = deepcopy(result)
    edited.break_rows.add(int(row))
    return _refresh(edited, track_width)


def remove_path_break(result: CurvePathResult, row: int, track_width: int) -> CurvePathResult:
    edited = deepcopy(result)
    edited.break_rows.discard(int(row))
    return _refresh(edited, track_width)


def set_wrap_transition(result: CurvePathResult, row: int, direction: str, track_width: int) -> CurvePathResult:
    if direction not in {"right_to_left", "left_to_right"}:
        raise ValueError("Invalid wrap direction")
    edited = deepcopy(result)
    row = int(row)
    if row <= 0 or row >= edited.wrap_index_by_row.size:
        raise ValueError("Wrap transition row must be inside the path")
    change = 1 if direction == "right_to_left" else -1
    edited.wrap_index_by_row[row:] += change
    return _refresh(edited, track_width)


def remove_wrap_transition(result: CurvePathResult, row: int, track_width: int) -> CurvePathResult:
    edited = deepcopy(result)
    row = int(row)
    if row <= 0 or row >= edited.wrap_index_by_row.size:
        raise ValueError("Wrap transition row must be inside the path")
    change = int(edited.wrap_index_by_row[row] - edited.wrap_index_by_row[row - 1])
    edited.wrap_index_by_row[row:] -= change
    return _refresh(edited, track_width)

