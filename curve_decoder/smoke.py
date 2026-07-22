"""Deterministic decoder-only Phase 3 smoke comparison."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .config import DecoderConfig
from .evidence import CurveEvidence
from .metrics import calculate_topology_metrics
from .cylindrical_dp import decode_curve_path


def _case_unwrapped(name: str, width: int, height: int) -> tuple[np.ndarray, str]:
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)
    if name == "bounded":
        return width * (0.48 + 0.22 * np.sin(2.0 * np.pi * 1.7 * t)), "bounded"
    if name == "right_to_left":
        return width * (0.68 + 0.70 * t + 0.025 * np.sin(4 * np.pi * t)), "cylindrical"
    if name == "left_to_right":
        return width * (0.32 - 0.70 * t + 0.025 * np.sin(4 * np.pi * t)), "cylindrical"
    if name == "multiple":
        return width * (0.18 + 2.35 * t + 0.025 * np.sin(5 * np.pi * t)), "cylindrical"
    if name == "turn_away":
        return width * (0.28 + 0.68 * np.sin(np.pi * t)), "cylindrical"
    raise ValueError(name)


def _evidence(unwrapped: np.ndarray, width: int, missing: bool) -> tuple[CurveEvidence, np.ndarray, np.ndarray]:
    visible = np.mod(unwrapped, float(width)).astype(np.float32)
    wrap = np.floor_divide(unwrapped, float(width)).astype(np.int32)
    columns = np.arange(width, dtype=np.float32)[None, :]
    distance = np.abs(columns - visible[:, None])
    distance = np.minimum(distance, float(width) - distance)
    center = np.exp(-0.5 * np.square(distance / 0.85)).astype(np.float32)
    smooth_distance = np.clip(1.0 - distance / 9.0, 0.0, 1.0).astype(np.float32)
    if missing:
        center[int(0.42 * unwrapped.size):int(0.55 * unwrapped.size)] = 0.0
    slope = np.gradient(unwrapped)
    magnitude = np.sqrt(slope * slope + 1.0)
    direction = np.zeros((2, unwrapped.size, width), dtype=np.float32)
    direction[0] = (slope / magnitude)[:, None]
    direction[1] = (1.0 / magnitude)[:, None]
    rtl = np.zeros(unwrapped.size, dtype=np.float32)
    ltr = np.zeros(unwrapped.size, dtype=np.float32)
    for row in np.flatnonzero(np.diff(wrap) != 0) + 1:
        (rtl if wrap[row] > wrap[row - 1] else ltr)[row] = 1.0
    return CurveEvidence(
        centerline_probability=center,
        distance_field=smooth_distance,
        direction_field=direction,
        wrap_probability_right_to_left=rtl,
        wrap_probability_left_to_right=ltr,
    ), visible, wrap


def run_smoke(width: int = 128, height: int = 256, beam_width: int = 96) -> dict:
    rows = []
    for name in ("bounded", "right_to_left", "left_to_right", "multiple", "turn_away"):
        truth_unwrapped, topology = _case_unwrapped(name, width, height)
        evidence, truth_x, truth_wrap = _evidence(truth_unwrapped, width, missing=name == "right_to_left")
        started = time.perf_counter()
        result = decode_curve_path(evidence, DecoderConfig(
            topology=topology,
            max_step=8,
            max_slope=8,
            slope_bins=17,
            beam_width=beam_width,
            edge_transition_width=max(4, width // 16),
            maximum_wrap_count=4,
            minimum_rows_between_wraps=8,
            rendering_discontinuity=max(20.0, width * 0.3),
        ))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = calculate_topology_metrics(
            result.x_by_row, result.unwrapped_x_by_row, result.wrap_index_by_row,
            truth_x, truth_unwrapped, truth_wrap, width,
        )
        rows.append({
            "case": name,
            "topology": topology,
            "unwrapped_mae": metrics["unwrapped_mean_absolute_error"],
            "unwrapped_p95": metrics["unwrapped_p95_absolute_error"],
            "wrap_precision": metrics["wrap_events"]["precision"],
            "wrap_recall": metrics["wrap_events"]["recall"],
            "false_wraps": metrics["false_wraps"],
            "missed_wraps": metrics["missed_wraps"],
            "cross_track_connector": metrics["cross_track_connector"],
            "decode_ms": elapsed_ms,
            "states_evaluated": result.metadata["states_evaluated"],
        })
    return {
        "width": width,
        "height": height,
        "beam_width": beam_width,
        "mean_unwrapped_mae": float(np.mean([row["unwrapped_mae"] for row in rows])),
        "mean_decode_ms": float(np.mean([row["decode_ms"] for row in rows])),
        "false_wraps": int(sum(row["false_wraps"] for row in rows)),
        "missed_wraps": int(sum(row["missed_wraps"] for row in rows)),
        "cross_track_connectors": int(sum(row["cross_track_connector"] for row in rows)),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Phase 3 decoder smoke cases")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=96)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.width, args.height, args.beam_width), indent=2))


if __name__ == "__main__":
    main()
