"""Quantitative metrics for visible and unwrapped topology paths."""

from __future__ import annotations

import numpy as np

from .rendering import build_visible_segments, has_cross_track_connector


def _event_rows(wrap_index: np.ndarray) -> list[dict]:
    wrap = np.asarray(wrap_index, dtype=np.int32).reshape(-1)
    events = []
    for row in np.flatnonzero(np.diff(wrap) != 0) + 1:
        delta = int(wrap[row] - wrap[row - 1])
        events.append({
            "row": int(row),
            "direction": "right_to_left" if delta > 0 else "left_to_right",
            "delta": delta,
        })
    return events


def _match_wrap_events(predicted: list[dict], truth: list[dict], tolerance_rows: int) -> dict:
    unmatched = set(range(len(truth)))
    matches = []
    for prediction in predicted:
        candidates = [
            index for index in unmatched
            if truth[index]["direction"] == prediction["direction"]
            and abs(truth[index]["row"] - prediction["row"]) <= int(tolerance_rows)
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda index: (abs(truth[index]["row"] - prediction["row"]), index))
        unmatched.remove(chosen)
        matches.append((prediction, truth[chosen]))
    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - true_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "mean_row_error": float(np.mean([
            abs(first["row"] - second["row"]) for first, second in matches
        ])) if matches else None,
    }


def calculate_topology_metrics(
    predicted_x: np.ndarray,
    predicted_unwrapped_x: np.ndarray,
    predicted_wrap_index: np.ndarray,
    truth_x: np.ndarray,
    truth_unwrapped_x: np.ndarray,
    truth_wrap_index: np.ndarray,
    track_width: int,
    tolerance_rows: int = 3,
) -> dict:
    predicted_x = np.asarray(predicted_x, dtype=np.float32).reshape(-1)
    predicted_unwrapped_x = np.asarray(predicted_unwrapped_x, dtype=np.float32).reshape(-1)
    predicted_wrap_index = np.asarray(predicted_wrap_index, dtype=np.int32).reshape(-1)
    truth_x = np.asarray(truth_x, dtype=np.float32).reshape(-1)
    truth_unwrapped_x = np.asarray(truth_unwrapped_x, dtype=np.float32).reshape(-1)
    truth_wrap_index = np.asarray(truth_wrap_index, dtype=np.int32).reshape(-1)
    lengths = {array.size for array in (
        predicted_x, predicted_unwrapped_x, predicted_wrap_index,
        truth_x, truth_unwrapped_x, truth_wrap_index,
    )}
    if len(lengths) != 1:
        raise ValueError("All topology metric vectors must have equal length")
    valid = np.isfinite(predicted_unwrapped_x) & np.isfinite(truth_unwrapped_x)
    errors = np.abs(predicted_unwrapped_x[valid] - truth_unwrapped_x[valid])
    predicted_events = _event_rows(predicted_wrap_index)
    truth_events = _event_rows(truth_wrap_index)
    event_metrics = _match_wrap_events(predicted_events, truth_events, tolerance_rows)
    segments = build_visible_segments(predicted_x, predicted_wrap_index)
    predicted_slope = np.diff(predicted_unwrapped_x)
    truth_slope = np.diff(truth_unwrapped_x)
    slope_valid = np.isfinite(predicted_slope) & np.isfinite(truth_slope)
    curvature = np.diff(predicted_unwrapped_x, n=2)
    return {
        "unwrapped_mean_absolute_error": float(np.mean(errors)) if errors.size else None,
        "unwrapped_p95_absolute_error": float(np.percentile(errors, 95)) if errors.size else None,
        "unwrapped_maximum_absolute_error": float(np.max(errors)) if errors.size else None,
        "wrap_index_accuracy": float(np.mean(predicted_wrap_index == truth_wrap_index)) if predicted_wrap_index.size else 0.0,
        "wrap_events": event_metrics,
        "false_wraps": event_metrics["false_positive"],
        "missed_wraps": event_metrics["false_negative"],
        "cross_track_connector": has_cross_track_connector(segments, int(track_width)),
        "slope_mean_absolute_error": float(np.mean(np.abs(predicted_slope[slope_valid] - truth_slope[slope_valid]))) if slope_valid.any() else None,
        "curvature_p95": float(np.percentile(np.abs(curvature[np.isfinite(curvature)]), 95)) if np.isfinite(curvature).any() else None,
    }
