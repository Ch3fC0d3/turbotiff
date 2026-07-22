"""Evaluate legacy and topology decoders on the same golden evidence."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from curve_model.phase2_integration import build_phase2_probability

from .config import DecoderConfig
from .diagnostics import write_diagnostic_csv
from .evidence import CurveEvidence
from .metrics import calculate_topology_metrics
from .rendering import build_visible_segments
from .cylindrical_dp import decode_curve_path


COMBINATIONS = (
    ("classic", "legacy_dp"),
    ("neural_phase2", "legacy_dp"),
    ("hybrid_phase2", "legacy_dp"),
    ("neural_phase2", "topology_dp"),
    ("hybrid_phase2", "topology_dp"),
)


def _records(golden_dir: Path) -> list[dict]:
    manifest = golden_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Golden manifest not found: {manifest}")
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def _classic_probability(image: np.ndarray, color_mode: str) -> np.ndarray:
    import web_app

    return web_app.compute_prob_map(image, mode=color_mode or "black")


def _legacy_decode(probability: np.ndarray, topology: str, curve_type: str) -> tuple[np.ndarray, float]:
    import web_app

    started = time.perf_counter()
    predicted, _ = web_app.trace_curve_with_dp(
        probability,
        scale_min=0.0,
        scale_max=100.0,
        curve_type=curve_type or "GR",
        max_step=max(3, min(80, probability.shape[1] // 3)),
        smooth_lambda=0.005,
        curv_lambda=0.0,
        wrap_enabled=topology == "cylindrical",
    )
    return predicted.astype(np.float32), (time.perf_counter() - started) * 1000.0


def _derive_legacy_wrap(x: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    wrap = np.zeros(x.shape, dtype=np.int32)
    for row in range(1, x.size):
        wrap[row] = wrap[row - 1]
        if not np.isfinite(x[row - 1]) or not np.isfinite(x[row]):
            continue
        delta = float(x[row] - x[row - 1])
        if delta < -0.5 * width:
            wrap[row:] += 1
        elif delta > 0.5 * width:
            wrap[row:] -= 1
    return x + wrap.astype(np.float32) * float(width), wrap


def _topology_evidence(probability: np.ndarray, auxiliary: dict, classic: np.ndarray) -> CurveEvidence:
    score = probability.astype(np.float32)
    if score.size and float(score.max()) > 1.0:
        score /= 255.0
    return CurveEvidence(
        centerline_probability=auxiliary.get("centerline_probability", score),
        stroke_probability=auxiliary.get("stroke_probability"),
        distance_field=auxiliary.get("distance_field"),
        direction_field=auxiliary.get("direction_field"),
        grid_probability=auxiliary.get("grid_probability"),
        classic_probability=classic.astype(np.float32) / 255.0,
    )


def _draw_segments(image: np.ndarray, truth_x: np.ndarray, truth_wrap: np.ndarray, predicted_x: np.ndarray, predicted_wrap: np.ndarray) -> np.ndarray:
    output = image.copy()
    for x, wrap, color in ((truth_x, truth_wrap, (0, 180, 0)), (predicted_x, predicted_wrap, (0, 0, 230))):
        for segment in build_visible_segments(x, wrap):
            points = np.asarray(segment.points, dtype=np.float32)
            if points.shape[0] >= 2:
                cv2.polylines(output, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    return output


def _aggregate(rows: list[dict], combination: str) -> dict:
    selected = [row for row in rows if row["combination"] == combination]
    numeric = (
        "unwrapped_mean_absolute_error", "unwrapped_p95_absolute_error",
        "unwrapped_maximum_absolute_error", "wrap_index_accuracy",
        "wrap_precision", "wrap_recall", "slope_mean_absolute_error",
        "decode_ms", "total_ms_per_megapixel",
    )
    result = {"cases": len(selected)}
    for key in numeric:
        values = [float(row[key]) for row in selected if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    result["false_wraps"] = int(sum(row["false_wraps"] for row in selected))
    result["missed_wraps"] = int(sum(row["missed_wraps"] for row in selected))
    result["cross_track_connectors"] = int(sum(bool(row["cross_track_connector"]) for row in selected))
    result["fallback_cases"] = int(sum(bool(row["fallback_occurred"]) for row in selected))
    return result


def evaluate_phase3(
    golden_dir: Path | str,
    output_dir: Path | str,
    phase2_model_path: Path | str | None = None,
    phase1_model_path: Path | str | None = None,
    device: str | None = None,
) -> dict:
    golden_dir = Path(golden_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    overlays = []
    diagnostics_dir = output_dir / "diagnostics"

    for record in _records(golden_dir):
        case_id = str(record["id"])
        image = cv2.imread(str(golden_dir / record["image"]), cv2.IMREAD_COLOR)
        trace = np.load(golden_dir / record["trace"], allow_pickle=False)
        metadata_path = golden_dir / record["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        truth_x = trace["correct_x_by_row"] if "correct_x_by_row" in trace else trace["centerline_x_by_row"]
        truth_unwrapped = trace["correct_unwrapped_x_by_row"] if "correct_unwrapped_x_by_row" in trace else truth_x
        truth_wrap = trace["correct_wrap_index_by_row"] if "correct_wrap_index_by_row" in trace else np.zeros(truth_x.shape, dtype=np.int32)
        if image is None or image.shape[0] != truth_x.size:
            raise ValueError(f"Golden case {case_id} has inconsistent image and trace dimensions")
        topology = str(metadata.get("topology") or "bounded")
        classic_started = time.perf_counter()
        classic = _classic_probability(image, metadata.get("curve_color") or "black")
        classic_ms = (time.perf_counter() - classic_started) * 1000.0
        detector_cache = {"classic": (classic, {"fallback_occurred": False, "actual_mode": "classic"}, {}, classic_ms)}

        for detector in ("neural_phase2", "hybrid_phase2"):
            started = time.perf_counter()
            probability, phase_metadata, auxiliary = build_phase2_probability(
                image,
                classic,
                mode=detector,
                phase2_model_path=str(phase2_model_path) if phase2_model_path else None,
                phase1_model_path=str(phase1_model_path) if phase1_model_path else None,
                device=device,
            )
            detector_cache[detector] = (
                probability, phase_metadata, auxiliary,
                (time.perf_counter() - started) * 1000.0,
            )

        for detector, decoder in COMBINATIONS:
            probability, phase_metadata, auxiliary, probability_ms = detector_cache[detector]
            if decoder == "legacy_dp":
                predicted_x, decode_ms = _legacy_decode(probability, topology, metadata.get("curve_type") or "GR")
                predicted_unwrapped, predicted_wrap = _derive_legacy_wrap(predicted_x, image.shape[1])
                decoder_metadata = {"decoder": "legacy_dp", "topology": topology}
                segments = build_visible_segments(predicted_x, predicted_wrap)
            else:
                started = time.perf_counter()
                result = decode_curve_path(
                    _topology_evidence(probability, auxiliary, classic),
                    DecoderConfig(
                        topology=topology,
                        max_step=max(3, min(16, image.shape[1] // 4)),
                        max_slope=max(3, min(16, image.shape[1] // 4)),
                        slope_bins=min(33, max(7, 2 * min(16, image.shape[1] // 4) + 1)),
                        beam_width=96,
                        edge_transition_width=max(2, min(10, image.shape[1] // 8)),
                    ),
                )
                decode_ms = (time.perf_counter() - started) * 1000.0
                predicted_x = result.x_by_row
                predicted_unwrapped = result.unwrapped_x_by_row
                predicted_wrap = result.wrap_index_by_row
                decoder_metadata = result.metadata
                segments = result.visible_segments
                write_diagnostic_csv(
                    diagnostics_dir / f"{case_id}_{detector}_{decoder}.csv",
                    result,
                )
            metrics = calculate_topology_metrics(
                predicted_x, predicted_unwrapped, predicted_wrap,
                truth_x, truth_unwrapped, truth_wrap, image.shape[1],
            )
            combination = f"{detector}+{decoder}"
            total_ms = probability_ms + decode_ms
            megapixels = max(1e-6, image.shape[0] * image.shape[1] / 1_000_000.0)
            row = {
                "case_id": case_id,
                "combination": combination,
                "detector": detector,
                "decoder": decoder,
                "topology": topology,
                "actual_detector": phase_metadata.get("actual_mode") or phase_metadata.get("tracing_mode") or detector,
                "fallback_occurred": bool(phase_metadata.get("fallback_occurred")),
                "fallback_reason": phase_metadata.get("fallback_reason") or phase_metadata.get("fallback_chain"),
                "unwrapped_mean_absolute_error": metrics["unwrapped_mean_absolute_error"],
                "unwrapped_p95_absolute_error": metrics["unwrapped_p95_absolute_error"],
                "unwrapped_maximum_absolute_error": metrics["unwrapped_maximum_absolute_error"],
                "wrap_index_accuracy": metrics["wrap_index_accuracy"],
                "wrap_precision": metrics["wrap_events"]["precision"],
                "wrap_recall": metrics["wrap_events"]["recall"],
                "false_wraps": metrics["false_wraps"],
                "missed_wraps": metrics["missed_wraps"],
                "cross_track_connector": metrics["cross_track_connector"],
                "slope_mean_absolute_error": metrics["slope_mean_absolute_error"],
                "curvature_p95": metrics["curvature_p95"],
                "probability_ms": probability_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "total_ms_per_megapixel": total_ms / megapixels,
                "segment_count": len(segments),
                "decoder_metadata": json.dumps(decoder_metadata, sort_keys=True),
            }
            rows.append(row)
            overlays.append((float(metrics["unwrapped_mean_absolute_error"] or 0.0), case_id, combination, image, truth_x, truth_wrap, predicted_x, predicted_wrap))

    combinations = [f"{detector}+{decoder}" for detector, decoder in COMBINATIONS]
    aggregate = {combination: _aggregate(rows, combination) for combination in combinations}
    report = {
        "schema_version": 3,
        "golden_dir": str(golden_dir.resolve()),
        "phase1_model": str(Path(phase1_model_path).resolve()) if phase1_model_path else None,
        "phase2_model": str(Path(phase2_model_path).resolve()) if phase2_model_path else None,
        "case_count": len(_records(golden_dir)),
        "aggregate": aggregate,
        "cases": rows,
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    markdown = [
        "# Topology Decoder Phase 3 Evaluation", "", f"Cases: {report['case_count']}", "",
        "| Combination | Unwrapped MAE | P95 | Wrap precision | Wrap recall | False | Missed | Connectors | ms/MP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for combination in combinations:
        item = aggregate[combination]
        markdown.append(
            f"| {combination} | {item['unwrapped_mean_absolute_error'] or 0:.3f} | "
            f"{item['unwrapped_p95_absolute_error'] or 0:.3f} | {item['wrap_precision'] or 0:.3f} | "
            f"{item['wrap_recall'] or 0:.3f} | {item['false_wraps']} | {item['missed_wraps']} | "
            f"{item['cross_track_connectors']} | {item['total_ms_per_megapixel'] or 0:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    for rank, candidate in enumerate(sorted(overlays, reverse=True)[:15], start=1):
        _, case_id, combination, image, truth_x, truth_wrap, predicted_x, predicted_wrap = candidate
        overlay = _draw_segments(image, truth_x, truth_wrap, predicted_x, predicted_wrap)
        path = output_dir / "worst" / f"{rank:02d}_{case_id}_{combination}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), overlay)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 detector and decoder combinations")
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase2-model", type=Path, default=None)
    parser.add_argument("--phase1-model", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = evaluate_phase3(
        args.golden_dir, args.output_dir,
        phase2_model_path=args.phase2_model,
        phase1_model_path=args.phase1_model,
        device=args.device,
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
