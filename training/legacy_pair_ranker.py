"""Rank legacy TIFF/LAS configurations for review without trusting old predictions.

Legacy TurboTIFF configurations are useful association and setup *seeds*.  They
are not labels: their depth mapping covered the full image, their tracks were
evenly divided, and their scales came from observed LAS extrema.  This module
therefore keeps every result in ``automatic_draft`` until a reviewer approves
the raster/LAS overlay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .real_log_dataset import (
    PairCandidate,
    _payload_hash,
    _write_json,
    assign_split,
    audit_pair,
    project_las_curves,
    validate_alignment,
)


@dataclass(frozen=True)
class ResolvedLegacyPair:
    legacy_index: int
    source: str
    well_id: str
    image_path: str
    las_path: str
    config: dict
    pair_id: str
    historical_exposure: str = "legacy_training_or_evaluation"


def load_legacy_configs(path: str | Path) -> list[dict]:
    """Load either the old JSON array or the later JSONL format."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("Legacy JSON must contain a list")
            return payload
        return [json.loads(line) for line in handle if line.strip()]


def resolve_legacy_configs(configs: Iterable[dict], dataset_root: str | Path) -> tuple[list[ResolvedLegacyPair], dict]:
    """Resolve obsolete drive paths by well and filename under the current corpus."""
    root = Path(dataset_root)
    source_dirs = sorted(path for path in root.iterdir() if (path / "pairs").is_dir())
    resolved: list[ResolvedLegacyPair] = []
    counts = Counter()
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(configs):
        legacy_path = PureWindowsPath(str(item.get("image_path", "")))
        well_id, filename = legacy_path.parent.name, legacy_path.name
        if not well_id or not filename:
            counts["invalid_legacy_path"] += 1
            continue
        matches = []
        for source_dir in source_dirs:
            image_path = source_dir / "pairs" / well_id / filename
            if image_path.is_file():
                las_files = sorted(image_path.parent.glob("*.las"))
                matches.append((source_dir.name.lower(), image_path, las_files))
        if len(matches) != 1:
            counts["missing_or_ambiguous_image"] += 1
            continue
        source, image_path, las_files = matches[0]
        if len(las_files) != 1:
            counts["missing_or_ambiguous_las"] += 1
            continue
        key = (str(image_path.resolve()).lower(), str(las_files[0].resolve()).lower())
        if key in seen:
            counts["duplicate_config"] += 1
            continue
        seen.add(key)
        pair_id = hashlib.sha256(f"legacy|{source}|{well_id}|{filename}|{las_files[0].name}".encode()).hexdigest()[:20]
        resolved.append(
            ResolvedLegacyPair(
                index,
                source,
                well_id,
                str(image_path.resolve()),
                str(las_files[0].resolve()),
                item.get("config", {}),
                pair_id,
            )
        )
        counts["resolved"] += 1
    counts["input_configs"] = index + 1 if "index" in locals() else 0
    return resolved, dict(counts)


def legacy_role(well_id: str, seed: str = "turbotiff-legacy-review-v1") -> str:
    """Legacy-exposed wells can train or diagnose, but never be a final test."""
    split = assign_split(well_id, seed)
    return "train" if split == "train" else "validation_diagnostic"


def legacy_config_to_alignment(item: ResolvedLegacyPair, image_size: tuple[int, int]) -> dict:
    width, height = image_size
    depth = item.config.get("depth", {})
    top_row = min(max(float(depth.get("top_px", 0)), 0.0), max(0.0, height - 1.0))
    bottom_row = min(max(float(depth.get("bottom_px", height - 1)), 0.0), max(0.0, height - 1.0))
    tracks = []
    identifiers = Counter()
    for curve in item.config.get("curves", []):
        mnemonic = str(curve.get("las_mnemonic") or curve.get("name") or "").strip().upper()
        if not mnemonic:
            continue
        identifiers[mnemonic] += 1
        track_id = mnemonic if identifiers[mnemonic] == 1 else f"{mnemonic}_{identifiers[mnemonic]}"
        tracks.append(
            {
                "track_id": track_id,
                "mnemonic": mnemonic,
                "unit": str(curve.get("las_unit") or ""),
                "x_left": float(curve.get("left_px", 0)),
                "x_right": float(curve.get("right_px", width - 1)),
                "value_left": float(curve.get("left_value", 0)),
                "value_right": float(curve.get("right_value", 1)),
                "scale_type": "linear",
                "color": str(curve.get("mode") or "black").lower(),
            }
        )
    return {
        "schema_version": 1,
        "pair_id": item.pair_id,
        "review_status": "automatic_draft",
        "review_required": True,
        "depth_unit": str(depth.get("unit") or "FT"),
        "depth_control_points": [
            {"depth": float(depth.get("top_depth")), "row": top_row},
            {"depth": float(depth.get("bottom_depth")), "row": bottom_row},
        ],
        "curve_tracks": tracks,
        "proposal_origin": "legacy_heuristic_seed",
        "proposal_warnings": [
            "Full-image depth mapping was not read from printed depth labels.",
            "Track bounds were evenly distributed rather than detected from the raster.",
            "Scale endpoints came from observed LAS extrema rather than printed scales.",
            "Old digitizer predictions are excluded from labels and scoring.",
        ],
        "historical_exposure": item.historical_exposure,
        "allowed_dataset_roles": ["train", "validation_diagnostic"],
        "prohibited_dataset_roles": ["final_unbiased_test"],
    }


def _preview_image(path: str | Path, maximum_width: int = 1600, maximum_height: int = 5000) -> tuple[np.ndarray, float, float]:
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            target_width = min(rgb.width, maximum_width)
            target_height = min(rgb.height, maximum_height)
            if (target_width, target_height) != rgb.size:
                rgb = rgb.resize((target_width, target_height), Image.Resampling.BILINEAR)
            array = np.asarray(rgb)
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR), target_width / source.width, target_height / source.height
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def score_raster_alignment(
    image_path: str | Path,
    records: list[dict],
    alignment: dict,
    radius_pixels: float = 2.0,
) -> tuple[dict, np.ndarray, tuple[float, float]]:
    """Score projected points against de-gridded dark raster evidence and controls."""
    preview, scale_x, scale_y = _preview_image(image_path)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    dark = gray < 115
    # Suppress long grid/border lines before distance scoring. Text remains, so
    # the same-row uniform-X controls are still essential to the evidence lift.
    vertical = dark.mean(axis=0) > 0.42
    horizontal = dark.mean(axis=1) > 0.42
    evidence = dark.copy()
    evidence[:, vertical] = False
    evidence[horizontal, :] = False
    distance = cv2.distanceTransform((~evidence).astype(np.uint8), cv2.DIST_L2, 3)
    tracks = {str(track.get("track_id") or track["mnemonic"]): track for track in alignment["curve_tracks"]}
    metrics = {}
    for track_id in sorted(tracks):
        subset = [record for record in records if str(record.get("track_id") or record["mnemonic"]) == track_id]
        if not subset:
            continue
        if len(subset) > 2500:
            indexes = np.linspace(0, len(subset) - 1, 2500).astype(int)
            subset = [subset[index] for index in indexes]
        xs = np.asarray([item["x"] for item in subset], float) * scale_x
        ys = np.asarray([item["y"] for item in subset], float) * scale_y
        inside = (xs >= 0) & (xs < preview.shape[1]) & (ys >= 0) & (ys < preview.shape[0])
        xs, ys = xs[inside], ys[inside]
        if not len(xs):
            continue
        xi = np.clip(np.rint(xs).astype(int), 0, preview.shape[1] - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, preview.shape[0] - 1)
        projected = distance[yi, xi]
        track = tracks[track_id]
        left = float(track["x_left"]) * scale_x
        right = float(track["x_right"]) * scale_x
        sequence = np.arange(len(yi), dtype=float)
        control_hits, control_distances = [], []
        for offset in (0.07, 0.23, 0.41, 0.67, 0.83):
            fractions = np.mod(sequence * 0.61803398875 + offset, 1.0)
            control_x = np.clip(np.rint(left + fractions * (right - left)).astype(int), 0, preview.shape[1] - 1)
            values = distance[yi, control_x]
            control_hits.append(float(np.mean(values <= radius_pixels)))
            control_distances.append(float(np.median(values)))
        hit = float(np.mean(projected <= radius_pixels))
        control_hit = float(np.median(control_hits))
        median_distance = float(np.median(projected))
        control_median = float(np.median(control_distances))
        lift = hit - control_hit
        metrics[track_id] = {
            "mnemonic": subset[0]["mnemonic"],
            "point_count": int(len(projected)),
            "hit_fraction_within_radius": hit,
            "control_hit_fraction": control_hit,
            "hit_lift_over_control": lift,
            "median_distance_pixels": median_distance,
            "control_median_distance_pixels": control_median,
            "radius_pixels": float(radius_pixels),
            "preview_scale_x": scale_x,
            "preview_scale_y": scale_y,
        }
    return metrics, preview, (scale_x, scale_y)


def _candidate_prefilter_key(item: ResolvedLegacyPair) -> tuple:
    curves = item.config.get("curves", [])
    names = {str(curve.get("las_mnemonic") or curve.get("name") or "").upper() for curve in curves}
    size = Path(item.image_path).stat().st_size
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(item.image_path) as image:
            landscape = image.width >= image.height
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    return ("GR" not in names, landscape, size, item.pair_id)


def assess_required_evidence(metrics: dict, required_mnemonic: str) -> str:
    """Apply a deliberately conservative gate to heuristic raster evidence."""
    matching = [value for value in metrics.values() if value["mnemonic"].upper() == required_mnemonic.upper()]
    if not matching:
        return "insufficient"
    best = max(matching, key=lambda value: value["hit_lift_over_control"])
    if (
        best["hit_fraction_within_radius"] >= 0.50
        and best["hit_lift_over_control"] >= 0.15
        and best["median_distance_pixels"] <= best["radius_pixels"]
    ):
        return "strong_review_candidate"
    return "weak_legacy_seed"


def rank_legacy_pairs(
    resolved: list[ResolvedLegacyPair],
    required_mnemonic: str = "GR",
    score_limit_train: int = 36,
    score_limit_validation: int = 14,
) -> list[dict]:
    """Audit and independently score a bounded deterministic candidate pool."""
    required = required_mnemonic.upper()
    pools = defaultdict(list)
    for item in resolved:
        names = {str(curve.get("las_mnemonic") or curve.get("name") or "").upper() for curve in item.config.get("curves", [])}
        if required and required not in names:
            continue
        pools[legacy_role(item.well_id)].append(item)
    chosen = []
    for role, limit in (("train", score_limit_train), ("validation_diagnostic", score_limit_validation)):
        chosen.extend(sorted(pools[role], key=_candidate_prefilter_key)[:limit])

    rankings = []
    for item in chosen:
        candidate = PairCandidate(
            item.pair_id,
            item.source,
            item.well_id,
            (item.image_path,),
            item.las_path,
            "legacy_config_exact_tiff_same_well",
        )
        audit = audit_pair(candidate, content_hashes=False, split_seed="turbotiff-legacy-review-v1")
        image_summary = audit.tiff_summaries[0] if audit.tiff_summaries else {}
        alignment = legacy_config_to_alignment(item, (image_summary.get("width", 0), image_summary.get("height", 0)))
        errors = validate_alignment(alignment, audit.las_summary)
        if errors or audit.status == "rejected":
            rankings.append({
                "pair_id": item.pair_id,
                "well_id": item.well_id,
                "source": item.source,
                "dataset_role": legacy_role(item.well_id),
                "score": 0.0,
                "status": "rejected",
                "errors": errors,
                "image_path": item.image_path,
                "las_path": item.las_path,
            })
            continue
        records = project_las_curves(item.las_path, alignment, sample_depth_interval=1.0)
        metrics, _, _ = score_raster_alignment(item.image_path, records, alignment)
        required_metrics = [value for value in metrics.values() if value["mnemonic"].upper() == required]
        if not required_metrics:
            evidence_score = 0.0
        else:
            best = max(required_metrics, key=lambda value: value["hit_lift_over_control"])
            distance_advantage = max(0.0, best["control_median_distance_pixels"] - best["median_distance_pixels"])
            evidence_score = max(0.0, best["hit_lift_over_control"]) * 0.8 + min(1.0, distance_advantage / 8.0) * 0.2
        curve_names = {curve["mnemonic"].upper() for curve in audit.las_summary.get("curves", [])}
        config_names = {track["mnemonic"].upper() for track in alignment["curve_tracks"]}
        curve_coverage = len(curve_names & config_names) / max(1, len(config_names))
        score = 0.8 * evidence_score + 0.15 * curve_coverage + 0.05 * audit.association_confidence
        rankings.append({
            "pair_id": item.pair_id,
            "legacy_index": item.legacy_index,
            "well_id": item.well_id,
            "source": item.source,
            "dataset_role": legacy_role(item.well_id),
            "historical_exposure": item.historical_exposure,
            "allowed_dataset_roles": ["train", "validation_diagnostic"],
            "prohibited_dataset_roles": ["final_unbiased_test"],
            "score": float(score),
            "evidence_score": float(evidence_score),
            "curve_coverage": float(curve_coverage),
            "status": "alignment_review_required",
            "image_path": item.image_path,
            "las_path": item.las_path,
            "image_summary": image_summary,
            "las_summary": audit.las_summary,
            "alignment_seed": alignment,
            "raster_metrics": metrics,
            "required_mnemonic": required,
            "evidence_status": assess_required_evidence(metrics, required),
        })
    return sorted(rankings, key=lambda row: (-row["score"], row["pair_id"]))


def select_review_batch(rankings: list[dict], count: int = 12, minimum_validation: int = 2) -> list[dict]:
    valid = [row for row in rankings if row.get("status") == "alignment_review_required"]
    validation = [row for row in valid if row["dataset_role"] == "validation_diagnostic"][:minimum_validation]
    selected_ids = {row["pair_id"] for row in validation}
    selected = validation + [row for row in valid if row["pair_id"] not in selected_ids][: max(0, count - len(validation))]
    return sorted(selected[:count], key=lambda row: (-row["score"], row["pair_id"]))


def _attach_source_hashes(alignment: dict, ranking: dict) -> dict:
    candidate = PairCandidate(
        ranking["pair_id"], ranking["source"], ranking["well_id"],
        (ranking["image_path"],), ranking["las_path"], "legacy_config_exact_tiff_same_well",
    )
    audit = audit_pair(candidate, content_hashes=True, split_seed="turbotiff-legacy-review-v1")
    output = dict(alignment)
    output["source_files"] = {
        "tiffs": [
            {key: summary.get(key) for key in ("path", "content_sha256", "stat_fingerprint")}
            for summary in audit.tiff_summaries
        ],
        "las": {key: audit.las_summary.get(key) for key in ("path", "content_sha256", "stat_fingerprint")},
    }
    output["alignment_hash"] = _payload_hash(output)
    return output


def _write_overlay(preview: np.ndarray, records: list[dict], scale: tuple[float, float], output_path: Path, mnemonic: str | None = None) -> None:
    scale_x, scale_y = scale
    output = preview.copy()
    if mnemonic:
        records = [record for record in records if record["mnemonic"].upper() == mnemonic.upper()]
    for track_id in sorted({str(record.get("track_id") or record["mnemonic"]) for record in records}):
        subset = [record for record in records if str(record.get("track_id") or record["mnemonic"]) == track_id]
        points = np.asarray([(round(row["x"] * scale_x), round(row["y"] * scale_y)) for row in subset], np.int32)
        if len(points) > 1:
            cv2.polylines(output, [points.reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), output):
        raise ValueError(f"Cannot write overlay: {output_path}")


def write_review_queue(
    output_dir: str | Path,
    resolution_summary: dict,
    rankings: list[dict],
    selected: list[dict],
) -> dict:
    output = Path(output_dir)
    proposals = output / "proposals"
    proposals.mkdir(parents=True, exist_ok=True)
    ranking_path = output / "rankings.jsonl"
    with ranking_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rankings:
            compact = {key: value for key, value in row.items() if key != "alignment_seed"}
            handle.write(json.dumps(compact, sort_keys=True, default=str) + "\n")
    manifest = []
    contact_panels = []
    for order, ranking in enumerate(selected, 1):
        directory = proposals / f"{order:02d}_{ranking['well_id']}_{ranking['pair_id']}"
        directory.mkdir(parents=True, exist_ok=True)
        alignment = _attach_source_hashes(ranking["alignment_seed"], ranking)
        records = project_las_curves(ranking["las_path"], alignment, sample_depth_interval=1.0)
        metrics, preview, scale = score_raster_alignment(ranking["image_path"], records, alignment)
        _write_json(directory / "alignment.json", alignment)
        _write_json(directory / "raster_metrics.json", {"pair_id": ranking["pair_id"], "metrics": metrics})
        with (directory / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        overlay_path = directory / "overlay_preview.png"
        _write_overlay(preview, records, scale, overlay_path, ranking.get("required_mnemonic"))
        panel = cv2.resize(cv2.imread(str(overlay_path)), (320, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (319, 42), (255, 255, 255), -1)
        cv2.putText(panel, f"{order:02d} {ranking['well_id']}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, f"score {ranking['score']:.3f} {ranking['dataset_role']}", (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        contact_panels.append(panel)
        manifest.append({
            "review_order": order,
            "pair_id": ranking["pair_id"],
            "well_id": ranking["well_id"],
            "source": ranking["source"],
            "dataset_role": ranking["dataset_role"],
            "score": ranking["score"],
            "review_status": "automatic_draft",
            "evidence_status": ranking.get("evidence_status", "insufficient"),
            "training_eligible": False,
            "alignment_hash": alignment["alignment_hash"],
            "proposal_directory": str(directory.resolve()),
            "image_path": ranking["image_path"],
            "las_path": ranking["las_path"],
        })
    manifest_path = output / "review_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if contact_panels:
        rows = []
        for start in range(0, len(contact_panels), 4):
            panels = contact_panels[start:start + 4]
            panels.extend([np.full_like(contact_panels[0], 255)] * (4 - len(panels)))
            rows.append(np.concatenate(panels, axis=1))
        cv2.imwrite(str(output / "review_contact_sheet.png"), np.concatenate(rows, axis=0))
    summary = {
        "resolution": resolution_summary,
        "scored_candidates": len(rankings),
        "selected_for_review": len(manifest),
        "selected_by_role": dict(Counter(row["dataset_role"] for row in manifest)),
        "selected_by_evidence": dict(Counter(row["evidence_status"] for row in manifest)),
        "training_eligible": 0,
        "policy": "Legacy-exposed pairs require alignment review and cannot serve as final unbiased test wells.",
        "old_predictions_used_as_labels": False,
    }
    _write_json(output / "summary.json", summary)
    return summary


def write_resolved_manifest(output_dir: str | Path, resolved: list[ResolvedLegacyPair], required_mnemonic: str) -> Path:
    """Persist all reusable source associations without copying source files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "resolved_legacy_pairs.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(resolved, key=lambda value: (value.source, value.well_id, value.pair_id)):
            mnemonics = sorted({str(curve.get("las_mnemonic") or curve.get("name") or "").upper() for curve in item.config.get("curves", [])})
            row = {
                "pair_id": item.pair_id,
                "legacy_index": item.legacy_index,
                "source": item.source,
                "well_id": item.well_id,
                "image_path": item.image_path,
                "las_path": item.las_path,
                "mnemonics": mnemonics,
                "contains_required_mnemonic": required_mnemonic.upper() in mnemonics,
                "dataset_role": legacy_role(item.well_id),
                "historical_exposure": item.historical_exposure,
                "allowed_dataset_roles": ["train", "validation_diagnostic"],
                "prohibited_dataset_roles": ["final_unbiased_test"],
                "alignment_status": "not_reviewed",
                "training_eligible": False,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rank old TIFF/LAS configs for safe reuse")
    parser.add_argument("--legacy-configs", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--required-mnemonic", default="GR")
    parser.add_argument("--review-count", type=int, default=12)
    parser.add_argument("--score-limit-train", type=int, default=36)
    parser.add_argument("--score-limit-validation", type=int, default=14)
    args = parser.parse_args(argv)
    configs = load_legacy_configs(args.legacy_configs)
    resolved, resolution = resolve_legacy_configs(configs, args.dataset_root)
    write_resolved_manifest(args.output_dir, resolved, args.required_mnemonic)
    rankings = rank_legacy_pairs(
        resolved,
        required_mnemonic=args.required_mnemonic,
        score_limit_train=args.score_limit_train,
        score_limit_validation=args.score_limit_validation,
    )
    selected = select_review_batch(rankings, args.review_count)
    summary = write_review_queue(args.output_dir, resolution, rankings, selected)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
