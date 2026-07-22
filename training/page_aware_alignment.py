"""Page-aware alignment of a colored LAS curve to a raster well log."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

from .legacy_pair_ranker import _preview_image, legacy_role
from .real_log_dataset import PairCandidate, _stable_id, _write_json, alignment_training_eligible, audit_pair, review_alignment, write_alignment_bundle


def color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    blue, green, red = cv2.split(image)
    return {
        "red": (red > 115) & (red > green * 1.28) & (red > blue * 1.28),
        "green": (green > 95) & (green > red * 1.22) & (green > blue * 1.12),
        "blue": (blue > 105) & (blue > green * 1.18) & (blue > red * 1.18),
    }


def detect_colored_log_body(masks: dict[str, np.ndarray], image: np.ndarray | None = None) -> tuple[int, int]:
    combined = np.logical_or.reduce(list(masks.values()))
    minimum = max(3, int(round(combined.shape[1] * 0.002)))
    active = (combined.sum(axis=1) >= minimum).astype(np.uint8)
    kernel = np.ones(81, np.uint8)
    closed = cv2.morphologyEx(active.reshape(-1, 1), cv2.MORPH_CLOSE, kernel.reshape(-1, 1)).ravel() > 0
    runs = []
    start = None
    for row, enabled in enumerate(np.r_[closed, False]):
        if enabled and start is None:
            start = row
        elif not enabled and start is not None:
            runs.append((start, row))
            start = None
    if not runs:
        raise ValueError("No sustained colored log-body region was detected")
    top, bottom = max(runs, key=lambda pair: pair[1] - pair[0])
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        long_rules = np.flatnonzero((gray < 180).sum(axis=1) >= gray.shape[1] * 0.70)
        header_rules = long_rules[long_rules < gray.shape[0] * 0.45]
        if len(header_rules):
            # The final near-full-width rule in the upper page separates the
            # header/scale legend from the continuous log body.
            proposed_top = int(header_rules[-1] + 2)
            if proposed_top < bottom - gray.shape[0] * 0.25:
                top = max(top, proposed_top)
    if bottom - top < combined.shape[0] * 0.25:
        raise ValueError("Colored region is too short to support page-aware alignment")
    return int(top), int(bottom - 1)


def _las_curve(las_path: str | Path, mnemonic: str) -> tuple[np.ndarray, np.ndarray, str]:
    import lasio

    las = lasio.read(str(las_path), engine="normal", ignore_header_errors=True)
    depth = np.asarray(las.index, float)
    curve = next((curve for curve in las.curves if curve.mnemonic.upper() == mnemonic.upper()), None)
    if curve is None:
        raise ValueError(f"LAS curve is absent: {mnemonic}")
    values = np.asarray(curve.data, float)
    valid = np.isfinite(depth) & np.isfinite(values)
    depth, values = depth[valid], values[valid]
    if len(depth) < 20:
        raise ValueError(f"LAS curve has insufficient valid samples: {mnemonic}")
    if len(depth) > 1400:
        indexes = np.linspace(0, len(depth) - 1, 1400).astype(int)
        depth, values = depth[indexes], values[indexes]
    return depth, values, str(curve.unit or "")


def _scale_candidates(mnemonic: str, values: np.ndarray) -> list[tuple[float, float]]:
    low, high = float(np.nanpercentile(values, 0.5)), float(np.nanpercentile(values, 99.5))
    if mnemonic.upper() == "GR":
        candidates = [(0.0, 100.0), (0.0, 150.0), (0.0, 200.0), (0.0, 250.0), (0.0, 300.0)]
    else:
        candidates = [(low, high)]
    if high > low:
        candidates.append((low, high))
    return list(dict.fromkeys(candidates))


def optimize_colored_curve_alignment(
    image_path: str | Path,
    las_path: str | Path,
    mnemonic: str = "GR",
    pair_id: str | None = None,
    radius_pixels: float = 3.0,
    fixed_scale: tuple[float, float] | None = None,
    allowed_colors: tuple[str, ...] | None = None,
) -> tuple[dict, dict]:
    """Fit one colored LAS curve using a coarse-to-fine affine raster search."""
    preview, scale_x, scale_y = _preview_image(image_path)
    masks = color_masks(preview)
    body_top, body_bottom = detect_colored_log_body(masks, preview)
    depth, values, unit = _las_curve(las_path, mnemonic)
    depth_min, depth_max = float(np.min(depth)), float(np.max(depth))
    depth_span = max(1e-6, depth_max - depth_min)
    body_height = body_bottom - body_top
    maximum_distance = float(np.hypot(*preview.shape[:2]))
    distance_maps = {
        color: np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3), maximum_distance)
        for color, mask in masks.items()
        if allowed_colors is None or color in allowed_colors
    }
    if not distance_maps:
        raise ValueError("No supported curve colors were selected")
    scale_options = [fixed_scale] if fixed_scale is not None else _scale_candidates(mnemonic, values)
    width = preview.shape[1]
    x_left_fractions = np.linspace(0.01, 0.76, 16)
    x_width_fractions = (0.12, 0.16, 0.20, 0.24, 0.28)
    slope_base = body_height / depth_span
    slopes = slope_base * np.linspace(0.82, 1.18, 13)
    rows_at_min = body_top + body_height * np.linspace(-0.18, 0.16, 18)
    best = None
    for value_left, value_right in scale_options:
        value_fraction = (values - value_left) / (value_right - value_left)
        value_valid = (value_fraction >= 0) & (value_fraction <= 1)
        if value_valid.sum() < 20:
            continue
        for slope in slopes:
            for row_at_min in rows_at_min:
                ys = row_at_min + (depth - depth_min) * slope
                y_valid = (ys >= body_top) & (ys <= body_bottom)
                valid = value_valid & y_valid
                if valid.sum() < 20:
                    continue
                yi = np.rint(ys[valid]).astype(int)
                fractions = value_fraction[valid]
                for left_fraction in x_left_fractions:
                    left = left_fraction * width
                    for width_fraction in x_width_fractions:
                        right = left + width_fraction * width
                        if right >= width:
                            continue
                        xi = np.rint(left + fractions * (right - left)).astype(int)
                        for color, distance in distance_maps.items():
                            sampled = distance[yi, xi]
                            hit = float(np.mean(sampled <= radius_pixels))
                            median = float(np.median(sampled))
                            soft_hit = float(np.mean(np.exp(-sampled / max(1.0, radius_pixels))))
                            objective = 0.7 * hit + 0.3 * soft_hit
                            if best is None or objective > best["objective"]:
                                best = {
                                    "objective": objective,
                                    "hit_fraction_within_radius": hit,
                                    "median_distance_pixels": median,
                                    "color": color,
                                    "value_left": value_left,
                                    "value_right": value_right,
                                    "x_left_preview": left,
                                    "x_right_preview": right,
                                    "row_at_depth_min_preview": row_at_min,
                                    "slope_preview_pixels_per_depth_unit": slope,
                                    "point_count": int(valid.sum()),
                                }
    if best is None:
        raise ValueError("No valid colored-curve alignment candidate was found")

    # Coordinate-descent refinement removes the several-pixel quantization of
    # the broad search without exploding the four-dimensional search space.
    distance = distance_maps[best["color"]]
    for _ in range(3):
        fixed_y = best["row_at_depth_min_preview"] + (depth - depth_min) * best["slope_preview_pixels_per_depth_unit"]
        value_rights = np.asarray([best["value_right"]]) if fixed_scale is not None else np.linspace(max(25.0, best["value_right"] - 40.0), best["value_right"] + 40.0, 17)
        lefts = np.linspace(max(0.0, best["x_left_preview"] - width * 0.04), min(width - 2.0, best["x_left_preview"] + width * 0.04), 17)
        rights = np.linspace(max(2.0, best["x_right_preview"] - width * 0.04), min(width - 1.0, best["x_right_preview"] + width * 0.04), 17)
        for value_right in value_rights:
            fractions = (values - best["value_left"]) / (value_right - best["value_left"])
            valid = (fractions >= 0) & (fractions <= 1) & (fixed_y >= body_top) & (fixed_y <= body_bottom)
            if valid.sum() < 20:
                continue
            yi = np.rint(fixed_y[valid]).astype(int)
            for left in lefts:
                for right in rights:
                    if right <= left + 2:
                        continue
                    xi = np.rint(left + fractions[valid] * (right - left)).astype(int)
                    sampled = distance[yi, xi]
                    hit = float(np.mean(sampled <= radius_pixels))
                    soft_hit = float(np.mean(np.exp(-sampled / max(1.0, radius_pixels))))
                    objective = 0.7 * hit + 0.3 * soft_hit
                    if objective > best["objective"]:
                        best.update({
                            "objective": objective,
                            "hit_fraction_within_radius": hit,
                            "median_distance_pixels": float(np.median(sampled)),
                            "value_right": float(value_right),
                            "x_left_preview": float(left),
                            "x_right_preview": float(right),
                            "point_count": int(valid.sum()),
                        })
        fractions = (values - best["value_left"]) / (best["value_right"] - best["value_left"])
        rows_at_min_fine = np.linspace(best["row_at_depth_min_preview"] - 100.0, best["row_at_depth_min_preview"] + 100.0, 25)
        slopes_fine = np.linspace(max(1e-6, best["slope_preview_pixels_per_depth_unit"] - 0.45), best["slope_preview_pixels_per_depth_unit"] + 0.45, 25)
        for slope in slopes_fine:
            for row_at_min in rows_at_min_fine:
                ys = row_at_min + (depth - depth_min) * slope
                valid = (fractions >= 0) & (fractions <= 1) & (ys >= body_top) & (ys <= body_bottom)
                if valid.sum() < 20:
                    continue
                xi = np.rint(best["x_left_preview"] + fractions[valid] * (best["x_right_preview"] - best["x_left_preview"])).astype(int)
                yi = np.rint(ys[valid]).astype(int)
                sampled = distance[yi, xi]
                hit = float(np.mean(sampled <= radius_pixels))
                soft_hit = float(np.mean(np.exp(-sampled / max(1.0, radius_pixels))))
                objective = 0.7 * hit + 0.3 * soft_hit
                if objective > best["objective"]:
                    best.update({
                        "objective": objective,
                        "hit_fraction_within_radius": hit,
                        "median_distance_pixels": float(np.median(sampled)),
                        "row_at_depth_min_preview": float(row_at_min),
                        "slope_preview_pixels_per_depth_unit": float(slope),
                        "point_count": int(valid.sum()),
                    })

    row_at_min = best["row_at_depth_min_preview"]
    slope = best["slope_preview_pixels_per_depth_unit"]
    chosen_fraction = (values - best["value_left"]) / (best["value_right"] - best["value_left"])
    chosen_y = row_at_min + (depth - depth_min) * slope
    chosen_valid = (chosen_fraction >= 0) & (chosen_fraction <= 1) & (chosen_y >= body_top) & (chosen_y <= body_bottom)
    chosen_yi = np.rint(chosen_y[chosen_valid]).astype(int)
    sequence = np.arange(chosen_valid.sum(), dtype=float)
    control_hits = []
    distance = distance_maps[best["color"]]
    for offset in (0.07, 0.23, 0.41, 0.67, 0.83):
        random_fraction = np.mod(sequence * 0.61803398875 + offset, 1.0)
        control_x = np.rint(best["x_left_preview"] + random_fraction * (best["x_right_preview"] - best["x_left_preview"])).astype(int)
        control_hits.append(float(np.mean(distance[chosen_yi, control_x] <= radius_pixels)))
    control_hit = float(np.median(control_hits))
    hit_lift = best["hit_fraction_within_radius"] - control_hit
    chosen_x = np.rint(best["x_left_preview"] + chosen_fraction[chosen_valid] * (best["x_right_preview"] - best["x_left_preview"])).astype(int)
    color_comparison = {}
    for color, comparison_distance in distance_maps.items():
        sampled = comparison_distance[chosen_yi, chosen_x]
        color_comparison[color] = {
            "hit_fraction_within_radius": float(np.mean(sampled <= radius_pixels)),
            "median_distance_pixels": float(np.median(sampled)),
        }
    original_row_min = row_at_min / scale_y
    original_row_max = (row_at_min + depth_span * slope) / scale_y
    alignment = {
        "schema_version": 1,
        "pair_id": pair_id or _stable_id(str(Path(image_path).resolve()), str(Path(las_path).resolve())),
        "review_status": "automatic_draft",
        "review_required": True,
        "depth_unit": "FT",
        "depth_control_points": [
            {"depth": depth_min, "row": float(original_row_min)},
            {"depth": depth_max, "row": float(original_row_max)},
        ],
        "curve_tracks": [
            {
                "track_id": f"{mnemonic.upper()}_page_aware",
                "mnemonic": mnemonic.upper(),
                "unit": unit,
                "x_left": float(best["x_left_preview"] / scale_x),
                "x_right": float(best["x_right_preview"] / scale_x),
                "value_left": float(best["value_left"]),
                "value_right": float(best["value_right"]),
                "scale_type": "linear",
                "color": best["color"],
            }
        ],
        "proposal_origin": "page_aware_colored_curve_optimizer",
        "proposal_warnings": [
            "Raster correlation is an alignment proposal, not human approval.",
            "Printed depth labels and scale endpoints require visual confirmation.",
        ],
    }
    evidence_status = "strong_review_candidate" if (
        best["hit_fraction_within_radius"] >= 0.50 and hit_lift >= 0.15 and best["median_distance_pixels"] <= radius_pixels
    ) else "weak_alignment_candidate"
    metrics = {
        **best,
        "control_hit_fraction": control_hit,
        "hit_lift_over_control": hit_lift,
        "evidence_status": evidence_status,
        "body_bounds_preview": {"top": body_top, "bottom": body_bottom},
        "preview_scale_x": scale_x,
        "preview_scale_y": scale_y,
        "color_comparison_at_selected_geometry": color_comparison,
    }
    return alignment, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a page-aware colored-curve alignment proposal")
    parser.add_argument("--image", required=True)
    parser.add_argument("--las", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mnemonic", default="GR")
    parser.add_argument("--source", default="legacy")
    parser.add_argument("--well-id", required=True)
    parser.add_argument("--color", choices=("red", "green", "blue"))
    parser.add_argument("--value-left", type=float)
    parser.add_argument("--value-right", type=float)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--review-notes", default="")
    args = parser.parse_args(argv)
    if (args.value_left is None) != (args.value_right is None):
        parser.error("--value-left and --value-right must be provided together")
    pair_id = _stable_id("page-aware", args.source, args.well_id, Path(args.image).name, Path(args.las).name)
    candidate = PairCandidate(pair_id, args.source, args.well_id, (str(Path(args.image).resolve()),), str(Path(args.las).resolve()), "legacy_config_exact_tiff_same_well")
    audit = audit_pair(candidate, content_hashes=True, split_seed="turbotiff-legacy-review-v1")
    fixed_scale = (args.value_left, args.value_right) if args.value_left is not None else None
    allowed_colors = (args.color,) if args.color else None
    alignment, metrics = optimize_colored_curve_alignment(args.image, args.las, args.mnemonic, pair_id, fixed_scale=fixed_scale, allowed_colors=allowed_colors)
    alignment.update({
        "historical_exposure": "legacy_training_or_evaluation",
        "dataset_role": legacy_role(args.well_id),
        "allowed_dataset_roles": ["train", "validation_diagnostic"],
        "prohibited_dataset_roles": ["final_unbiased_test"],
    })
    outputs = write_alignment_bundle(args.output_dir, audit, alignment)
    _write_json(Path(args.output_dir) / "optimization_metrics.json", metrics)
    training_eligible = False
    if args.approve:
        if not args.reviewer:
            parser.error("--reviewer is required with --approve")
        if metrics["evidence_status"] != "strong_review_candidate":
            raise PermissionError("Only a strong review candidate can be approved")
        bundled = json.loads((Path(args.output_dir) / "alignment.json").read_text(encoding="utf-8"))
        reviewed = review_alignment(bundled, audit, args.reviewer, "approved", args.review_notes)
        reviewed_dir = Path(args.output_dir) / "reviewed_alignment"
        outputs.extend(write_alignment_bundle(reviewed_dir, audit, reviewed))
        reviewed_payload = json.loads((reviewed_dir / "alignment.json").read_text(encoding="utf-8"))
        training_eligible = alignment_training_eligible(reviewed_payload, audit)
        if not training_eligible:
            raise PermissionError("Reviewed alignment failed the source/hash eligibility gate")
    print(json.dumps({"metrics": metrics, "training_eligible": training_eligible, "outputs": [str(path) for path in outputs]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
