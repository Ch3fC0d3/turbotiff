"""Compare classic, neural, and hybrid probability maps on golden tracks."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .integration import build_phase1_probability
from .metrics import calculate_phase2_metrics, calculate_trace_metrics
from .phase2_decode import decode_phase2_path
from .phase2_integration import PHASE2_MODES, build_phase2_probability


MODES = ("classic", "neural_phase1", "hybrid_phase1", "neural_phase2", "hybrid_phase2")


def _load_cases(golden_dir: Path) -> list[dict]:
    manifest = golden_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Golden manifest not found: {manifest}")
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decode_with_existing_dp(probability: np.ndarray, curve_type: str = "GR") -> tuple[np.ndarray, float]:
    # Import lazily so metrics and dataset tools do not initialize the web app.
    import web_app

    started = time.perf_counter()
    xs, _ = web_app.trace_curve_with_dp(
        probability,
        scale_min=0.0,
        scale_max=100.0,
        curve_type=curve_type or "GR",
        max_step=max(3, min(80, probability.shape[1] // 3)),
        smooth_lambda=0.005,
        curv_lambda=0.0,
    )
    return xs.astype(np.float32), (time.perf_counter() - started) * 1000.0


def _classic_probability(image: np.ndarray, color_mode: str) -> tuple[np.ndarray, float]:
    import web_app

    started = time.perf_counter()
    probability = web_app.compute_prob_map(image, mode=color_mode or "black")
    return probability, (time.perf_counter() - started) * 1000.0


def _save_overlay(path: Path, image: np.ndarray, truth: np.ndarray, predicted: np.ndarray) -> None:
    overlay = image.copy()
    truth_valid = np.isfinite(truth)
    predicted_valid = np.isfinite(predicted)
    truth_points = np.column_stack((np.rint(truth[truth_valid]), np.flatnonzero(truth_valid))).astype(np.int32)
    prediction_points = np.column_stack((np.rint(predicted[predicted_valid]), np.flatnonzero(predicted_valid))).astype(np.int32)
    if truth_points.size:
        cv2.polylines(overlay, [truth_points.reshape(-1, 1, 2)], False, (0, 190, 0), 1, cv2.LINE_AA)
    if prediction_points.size:
        cv2.polylines(overlay, [prediction_points.reshape(-1, 1, 2)], False, (0, 0, 230), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def _aggregate(rows: list[dict], mode: str) -> dict:
    selected = [row for row in rows if row["mode"] == mode]
    numeric_keys = [
        "mean_absolute_error", "median_absolute_error", "p90_absolute_error",
        "p95_absolute_error", "maximum_absolute_error", "missing_fraction",
        "probability_ms", "decode_ms", "total_ms", "total_ms_per_megapixel",
        "mean_direction_error_degrees", "grid_crossing_mean_error", "center_of_stroke_mean_error",
    ]
    aggregate = {"cases": len(selected)}
    for key in numeric_keys:
        values = [float(row[key]) for row in selected if row.get(key) is not None]
        aggregate[key] = float(np.mean(values)) if values else None
    for tolerance in (1, 2, 3, 5, 10):
        key = f"within_{tolerance}px"
        values = [float(row[key]) for row in selected]
        aggregate[key] = float(np.mean(values)) if values else 0.0
    aggregate["false_grid_runs"] = int(sum(row["false_grid_runs"] for row in selected))
    aggregate["missed_peaks"] = int(sum(row["missed_peaks"] for row in selected))
    aggregate["missed_valleys"] = int(sum(row["missed_valleys"] for row in selected))
    aggregate["major_path_gaps"] = int(sum(row["major_path_gaps"] for row in selected))
    aggregate["maximum_gap_length"] = max((row["maximum_gap_length"] for row in selected), default=0)
    return aggregate


def evaluate_golden(
    golden_dir: Path | str,
    model_path: Path | str,
    output_dir: Optional[Path | str] = None,
    device: Optional[str] = None,
    phase2_model_path: Optional[Path | str] = None,
) -> dict:
    golden_dir = Path(golden_dir)
    output_dir = Path(output_dir) if output_dir else golden_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(golden_dir)
    rows = []
    overlay_candidates = []

    for case in cases:
        image = cv2.imread(str(golden_dir / case["image"]), cv2.IMREAD_COLOR)
        trace_data = np.load(golden_dir / case["trace"])
        truth = trace_data["centerline_x_by_row"].astype(np.float32)
        valid = trace_data["valid_row_mask"].astype(bool) if "valid_row_mask" in trace_data else np.isfinite(truth)
        metadata_path = golden_dir / case["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if image is None or image.shape[0] != truth.size:
            raise ValueError(f"Golden case {case['id']} has inconsistent image/trace dimensions")
        classic, classic_ms = _classic_probability(image, metadata.get("curve_color") or "black")

        for mode in MODES:
            probability_started = time.perf_counter()
            auxiliary = {}
            if mode == "classic":
                probability = classic.copy()
                phase_meta = {"fallback_occurred": False, "tracing_mode": "classic"}
                probability_ms = classic_ms
            elif mode not in PHASE2_MODES:
                probability, phase_meta = build_phase1_probability(
                    image,
                    classic,
                    mode=mode,
                    model_path=str(model_path),
                    device=device,
                )
                probability_ms = (time.perf_counter() - probability_started) * 1000.0
            else:
                probability, phase_meta, auxiliary = build_phase2_probability(
                    image,
                    classic,
                    mode=mode,
                    phase2_model_path=str(phase2_model_path) if phase2_model_path else None,
                    phase1_model_path=str(model_path),
                    device=device,
                )
                probability_ms = (time.perf_counter() - probability_started) * 1000.0
            if phase_meta.get("tracing_mode") in PHASE2_MODES and auxiliary.get("direction_field") is not None:
                decode_started = time.perf_counter()
                predicted = decode_phase2_path(
                    probability.astype(np.float32) / 255.0,
                    auxiliary["direction_field"],
                    max_step=max(3, min(80, probability.shape[1] // 3)),
                )
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
            else:
                predicted, decode_ms = _decode_with_existing_dp(probability, metadata.get("curve_type") or "GR")
            metrics = calculate_trace_metrics(predicted, truth, valid)
            phase2_metrics = calculate_phase2_metrics(
                predicted,
                truth,
                valid,
                grid_mask=trace_data["grid_mask"] if "grid_mask" in trace_data else None,
                stroke_mask=trace_data["stroke_mask"] if "stroke_mask" in trace_data else None,
            )
            megapixels = max(1e-6, image.shape[0] * image.shape[1] / 1_000_000.0)
            total_ms = probability_ms + decode_ms
            row = {
                "case_id": case["id"],
                "mode": mode,
                "fallback_occurred": bool(phase_meta.get("fallback_occurred")),
                "actual_mode": phase_meta.get("actual_mode") or phase_meta.get("tracing_mode"),
                "mean_absolute_error": metrics["mean_absolute_error"],
                "median_absolute_error": metrics["median_absolute_error"],
                "p90_absolute_error": metrics["p90_absolute_error"],
                "p95_absolute_error": metrics["p95_absolute_error"],
                "maximum_absolute_error": metrics["maximum_absolute_error"],
                "missing_fraction": metrics["missing_fraction"],
                "false_grid_runs": metrics["grid_lock"]["false_constant_runs"],
                "missed_peaks": metrics["peaks"]["missed"],
                "missed_valleys": metrics["valleys"]["missed"],
                "probability_ms": probability_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "total_ms_per_megapixel": total_ms / megapixels,
                "major_path_gaps": phase2_metrics["connectivity"]["major_path_gaps"],
                "maximum_gap_length": phase2_metrics["connectivity"]["maximum_gap_length"],
                "mean_direction_error_degrees": phase2_metrics["direction_consistency"]["mean_error_degrees"],
                "grid_crossing_mean_error": phase2_metrics["grid_crossings"]["mean_error"],
                "center_of_stroke_mean_error": phase2_metrics["center_of_stroke"]["mean_error"],
            }
            row.update(metrics["accuracy"])
            rows.append(row)
            ranking_error = row["mean_absolute_error"]
            if ranking_error is None:
                ranking_error = float("inf")
            overlay_candidates.append((ranking_error, case["id"], mode, image, truth, predicted))

    aggregate = {mode: _aggregate(rows, mode) for mode in MODES}
    report = {
        "schema_version": 1,
        "golden_dir": str(golden_dir.resolve()),
        "model": str(Path(model_path).expanduser().resolve()),
        "phase2_model": str(Path(phase2_model_path).expanduser().resolve()) if phase2_model_path else None,
        "case_count": len(cases),
        "aggregate": aggregate,
        "cases": rows,
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    markdown = [
        "# Neural Curve Phase 1 and Phase 2 Evaluation",
        "",
        f"Cases: {len(cases)}",
        "",
        "| Mode | Mean error | P95 error | Within 3 px | Gaps | Missing | Total ms/MP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        item = aggregate[mode]
        markdown.append(
            f"| {mode} | {item['mean_absolute_error'] or 0:.3f} | {item['p95_absolute_error'] or 0:.3f} | "
            f"{100.0 * item['within_3px']:.1f}% | {item['major_path_gaps']} | {100.0 * (item['missing_fraction'] or 0):.1f}% | "
            f"{item['total_ms_per_megapixel'] or 0:.1f} |"
        )
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    worst = sorted(overlay_candidates, key=lambda item: item[0], reverse=True)[:12]
    for rank, (_, case_id, mode, image, truth, predicted) in enumerate(worst, start=1):
        _save_overlay(output_dir / "worst" / f"{rank:02d}_{case_id}_{mode}.png", image, truth, predicted)

    candidate_map = {(case_id, mode): (image, truth, predicted) for _, case_id, mode, image, truth, predicted in overlay_candidates}
    categories = {
        "grid_lock": sorted(rows, key=lambda row: row["false_grid_runs"], reverse=True),
        "connectivity": sorted(rows, key=lambda row: row["maximum_gap_length"], reverse=True),
        "missed_extrema": sorted(rows, key=lambda row: row["missed_peaks"] + row["missed_valleys"], reverse=True),
    }
    for category, ranked_rows in categories.items():
        for rank, row in enumerate(ranked_rows[:6], start=1):
            candidate = candidate_map.get((row["case_id"], row["mode"]))
            if candidate:
                _save_overlay(
                    output_dir / "worst" / category / f"{rank:02d}_{row['case_id']}_{row['mode']}.png",
                    *candidate,
                )

    by_case_mode = {(row["case_id"], row["mode"]): row for row in rows}
    comparisons = []
    for case in cases:
        case_id = case["id"]
        phase1 = by_case_mode.get((case_id, "neural_phase1"))
        phase2 = by_case_mode.get((case_id, "neural_phase2"))
        hybrid2 = by_case_mode.get((case_id, "hybrid_phase2"))
        if phase1 and phase2 and phase1["mean_absolute_error"] is not None and phase2["mean_absolute_error"] is not None:
            comparisons.append((phase2["mean_absolute_error"] - phase1["mean_absolute_error"], "phase2_worse_than_phase1", phase2))
        if phase2 and hybrid2 and phase2["mean_absolute_error"] is not None and hybrid2["mean_absolute_error"] is not None:
            comparisons.append((hybrid2["mean_absolute_error"] - phase2["mean_absolute_error"], "hybrid_worse_than_neural", hybrid2))
    for category in ("phase2_worse_than_phase1", "hybrid_worse_than_neural"):
        selected = sorted(
            (item for item in comparisons if item[1] == category and item[0] > 0),
            key=lambda item: item[0],
            reverse=True,
        )
        for rank, (_, _, row) in enumerate(selected[:6], start=1):
            candidate = candidate_map.get((row["case_id"], row["mode"]))
            if candidate:
                _save_overlay(output_dir / "worst" / category / f"{rank:02d}_{row['case_id']}_{row['mode']}.png", *candidate)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classic, neural, and hybrid curve detection")
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--phase2-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = evaluate_golden(
        args.golden_dir,
        args.model,
        output_dir=args.output_dir,
        device=args.device,
        phase2_model_path=args.phase2_model,
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
