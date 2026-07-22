"""Quantitative centerline metrics for classic/neural/hybrid comparison."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover
    find_peaks = None


def _constant_runs(values: np.ndarray, valid: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    rounded = np.rint(values)
    runs = []
    start = None
    for index in range(values.size):
        same = index > 0 and valid[index] and valid[index - 1] and rounded[index] == rounded[index - 1]
        if same and start is None:
            start = index - 1
        if start is not None and (not same or index == values.size - 1):
            end = index if same and index == values.size - 1 else index - 1
            if end - start + 1 >= minimum_length:
                runs.append((start, end))
            start = None
    return runs


def _extrema(values: np.ndarray, valid: np.ndarray, prominence: float) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 3 or valid.sum() < 3:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    filled = values.copy().astype(np.float64)
    rows = np.arange(values.size)
    filled[~valid] = np.interp(rows[~valid], rows[valid], filled[valid])
    if find_peaks is not None:
        peaks = find_peaks(filled, prominence=prominence)[0]
        valleys = find_peaks(-filled, prominence=prominence)[0]
    else:
        peaks = np.where((filled[1:-1] > filled[:-2]) & (filled[1:-1] >= filled[2:]))[0] + 1
        valleys = np.where((filled[1:-1] < filled[:-2]) & (filled[1:-1] <= filled[2:]))[0] + 1
    return peaks.astype(np.int32), valleys.astype(np.int32)


def _match_extrema(predicted: np.ndarray, truth: np.ndarray, tolerance_rows: int = 8) -> dict:
    used = set()
    errors = []
    missed = 0
    for target in truth.tolist():
        choices = [(abs(int(candidate) - int(target)), index, candidate) for index, candidate in enumerate(predicted) if index not in used]
        if not choices:
            missed += 1
            continue
        distance, index, _ = min(choices)
        if distance <= tolerance_rows:
            used.add(index)
            errors.append(float(distance))
        else:
            missed += 1
    return {
        "missed": int(missed),
        "extra": int(max(0, predicted.size - len(used))),
        "average_position_error_rows": float(np.mean(errors)) if errors else None,
    }


def calculate_trace_metrics(
    predicted_x: np.ndarray,
    correct_x: np.ndarray,
    valid_row_mask: Optional[np.ndarray] = None,
    constant_run_min: int = 20,
    extrema_prominence: float = 2.0,
) -> dict:
    predicted = np.asarray(predicted_x, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct_x, dtype=np.float64).reshape(-1)
    if predicted.shape != correct.shape:
        raise ValueError("Predicted and correct traces must have the same number of rows")
    valid_truth = np.isfinite(correct)
    if valid_row_mask is not None:
        mask = np.asarray(valid_row_mask, dtype=bool).reshape(-1)
        if mask.shape != correct.shape:
            raise ValueError("Valid-row mask must match the trace length")
        valid_truth &= mask
    usable = valid_truth & np.isfinite(predicted)
    errors = np.abs(predicted[usable] - correct[usable])
    missing_rows = int(np.sum(valid_truth & ~np.isfinite(predicted)))

    if errors.size:
        error_metrics = {
            "mean_absolute_error": float(np.mean(errors)),
            "median_absolute_error": float(np.median(errors)),
            "p90_absolute_error": float(np.percentile(errors, 90)),
            "p95_absolute_error": float(np.percentile(errors, 95)),
            "maximum_absolute_error": float(np.max(errors)),
        }
    else:
        error_metrics = {key: None for key in (
            "mean_absolute_error", "median_absolute_error", "p90_absolute_error",
            "p95_absolute_error", "maximum_absolute_error",
        )}

    tolerance_accuracy = {
        f"within_{tolerance}px": float(np.mean(errors <= tolerance)) if errors.size else 0.0
        for tolerance in (1, 2, 3, 5, 10)
    }
    constant_runs = _constant_runs(predicted, np.isfinite(predicted) & valid_truth, int(constant_run_min))
    false_grid_runs = [
        (start, end)
        for start, end in constant_runs
        if np.ptp(correct[start:end + 1][np.isfinite(correct[start:end + 1])]) > 1.0
    ]
    predicted_peaks, predicted_valleys = _extrema(predicted, np.isfinite(predicted) & valid_truth, extrema_prominence)
    truth_peaks, truth_valleys = _extrema(correct, valid_truth, extrema_prominence)

    return {
        **error_metrics,
        "accuracy": tolerance_accuracy,
        "valid_rows": int(valid_truth.sum()),
        "usable_rows": int(usable.sum()),
        "missing_rows": missing_rows,
        "missing_fraction": missing_rows / max(1, int(valid_truth.sum())),
        "grid_lock": {
            "false_constant_runs": len(false_grid_runs),
            "maximum_false_run_length": max((end - start + 1 for start, end in false_grid_runs), default=0),
        },
        "peaks": _match_extrema(predicted_peaks, truth_peaks),
        "valleys": _match_extrema(predicted_valleys, truth_valleys),
    }


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def calculate_phase2_metrics(
    predicted_x: np.ndarray,
    correct_x: np.ndarray,
    valid_row_mask: Optional[np.ndarray] = None,
    grid_mask: Optional[np.ndarray] = None,
    stroke_mask: Optional[np.ndarray] = None,
    major_error_threshold: float = 8.0,
) -> dict:
    predicted = np.asarray(predicted_x, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct_x, dtype=np.float64).reshape(-1)
    if predicted.shape != correct.shape:
        raise ValueError("Predicted and correct traces must have the same number of rows")
    valid = np.isfinite(correct)
    if valid_row_mask is not None:
        valid &= np.asarray(valid_row_mask, dtype=bool).reshape(-1)
    usable = valid & np.isfinite(predicted)
    errors = np.full(correct.shape, np.nan, dtype=np.float64)
    errors[usable] = np.abs(predicted[usable] - correct[usable])
    major = usable & (errors > float(major_error_threshold))
    gap_runs = _true_runs(major)
    recovery = []
    for _, end in gap_runs:
        recovery_rows = np.arange(end + 1, min(correct.size, end + 6))
        recovery_rows = recovery_rows[usable[recovery_rows]]
        if recovery_rows.size:
            recovery.append(float(np.mean(errors[recovery_rows] <= 3.0)))

    truth_dx = np.gradient(correct)
    predicted_dx = np.gradient(predicted)
    direction_valid = usable & np.isfinite(truth_dx) & np.isfinite(predicted_dx)
    if direction_valid.any():
        truth_angle = np.arctan2(truth_dx[direction_valid], np.ones(direction_valid.sum()))
        predicted_angle = np.arctan2(predicted_dx[direction_valid], np.ones(direction_valid.sum()))
        angular_error = np.degrees(np.abs(np.arctan2(np.sin(predicted_angle - truth_angle), np.cos(predicted_angle - truth_angle))))
        direction_metrics = {
            "mean_error_degrees": float(np.mean(angular_error)),
            "median_error_degrees": float(np.median(angular_error)),
            "p95_error_degrees": float(np.percentile(angular_error, 95)),
        }
    else:
        direction_metrics = {"mean_error_degrees": None, "median_error_degrees": None, "p95_error_degrees": None}

    crossing_rows = np.array([], dtype=np.int32)
    if grid_mask is not None:
        grid = np.asarray(grid_mask)
        if grid.shape[0] != correct.size:
            raise ValueError("Grid mask height must match the trace")
        crossing = np.zeros(correct.size, dtype=bool)
        for row in np.flatnonzero(valid):
            x = int(np.clip(round(float(correct[row])), 0, grid.shape[1] - 1))
            crossing[row] = bool(grid[row, x] > 0)
        crossing_rows = np.flatnonzero(crossing)
    crossing_errors = errors[crossing_rows]
    crossing_errors = crossing_errors[np.isfinite(crossing_errors)]

    thick_errors = np.array([], dtype=np.float64)
    if stroke_mask is not None:
        stroke = np.asarray(stroke_mask) > 0
        widths = stroke.sum(axis=1)
        thick_rows = usable & (widths >= 3)
        thick_errors = errors[thick_rows]

    return {
        "connectivity": {
            "major_path_gaps": len(gap_runs),
            "average_gap_length": float(np.mean([end - start + 1 for start, end in gap_runs])) if gap_runs else 0.0,
            "maximum_gap_length": max((end - start + 1 for start, end in gap_runs), default=0),
            "recovery_accuracy": float(np.mean(recovery)) if recovery else None,
        },
        "grid_crossings": {
            "rows": int(crossing_rows.size),
            "mean_error": float(np.mean(crossing_errors)) if crossing_errors.size else None,
            "p95_error": float(np.percentile(crossing_errors, 95)) if crossing_errors.size else None,
            "rows_over_threshold": int(np.sum(crossing_errors > major_error_threshold)),
        },
        "direction_consistency": direction_metrics,
        "center_of_stroke": {
            "thick_rows": int(thick_errors.size),
            "mean_error": float(np.mean(thick_errors)) if thick_errors.size else None,
            "p95_error": float(np.percentile(thick_errors, 95)) if thick_errors.size else None,
        },
    }
