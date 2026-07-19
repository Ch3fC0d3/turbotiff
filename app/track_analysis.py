"""Vision-model guidance for the deterministic well-log tracing pipeline."""

from __future__ import annotations

import base64
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_TRACK_ANALYST_MODEL = "gpt-5.6-luna"
MAX_TRACKS = 6
MAX_CURVES_PER_TRACK = 6
MAX_SEEDS_PER_CURVE = 12
MAX_REGIONS = 12


class TrackAnalysisError(RuntimeError):
    """A user-safe failure from the advisory vision service."""


def _nullable(kind: str) -> Dict[str, Any]:
    return {"type": [kind, "null"]}


REGION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x1", "y1", "x2", "y2", "reason", "confidence"],
    "properties": {
        "x1": {"type": "number"},
        "y1": {"type": "number"},
        "x2": {"type": "number"},
        "y2": {"type": "number"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
}


TRACK_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis_confidence", "tracks", "global_ignore_regions", "notes"],
    "properties": {
        "analysis_confidence": {"type": "number"},
        "tracks": {
            "type": "array",
            "maxItems": MAX_TRACKS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id", "left_x", "right_x", "top_y", "bottom_y",
                    "scale_type", "scale_min", "scale_max", "unit",
                    "horizontal_grid_spacing_px", "vertical_grid_spacing_px",
                    "wraparound", "confidence", "curves", "ignore_regions",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "left_x": {"type": "number"},
                    "right_x": {"type": "number"},
                    "top_y": {"type": "number"},
                    "bottom_y": {"type": "number"},
                    "scale_type": {
                        "type": "string",
                        "enum": ["linear", "log", "centered", "unknown"],
                    },
                    "scale_min": _nullable("number"),
                    "scale_max": _nullable("number"),
                    "unit": _nullable("string"),
                    "horizontal_grid_spacing_px": _nullable("number"),
                    "vertical_grid_spacing_px": _nullable("number"),
                    "wraparound": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "curves": {
                        "type": "array",
                        "maxItems": MAX_CURVES_PER_TRACK,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "id", "mnemonic", "color", "estimated_start_x",
                                "wrap_enabled", "confidence", "max_jump_px",
                                "seed_points", "low_confidence_sections",
                            ],
                            "properties": {
                                "id": {"type": "string"},
                                "mnemonic": {"type": "string"},
                                "color": {
                                    "type": "string",
                                    "enum": [
                                        "black", "red", "green", "blue", "cyan",
                                        "magenta", "yellow", "orange", "purple", "unknown",
                                    ],
                                },
                                "estimated_start_x": _nullable("number"),
                                "wrap_enabled": {"type": "boolean"},
                                "confidence": {"type": "number"},
                                "max_jump_px": _nullable("number"),
                                "seed_points": {
                                    "type": "array",
                                    "maxItems": MAX_SEEDS_PER_CURVE,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["x", "y", "confidence"],
                                        "properties": {
                                            "x": {"type": "number"},
                                            "y": {"type": "number"},
                                            "confidence": {"type": "number"},
                                        },
                                    },
                                },
                                "low_confidence_sections": {
                                    "type": "array",
                                    "maxItems": MAX_REGIONS,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["y1", "y2", "reason"],
                                        "properties": {
                                            "y1": {"type": "number"},
                                            "y2": {"type": "number"},
                                            "reason": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "ignore_regions": {
                        "type": "array",
                        "maxItems": MAX_REGIONS,
                        "items": REGION_SCHEMA,
                    },
                },
            },
        },
        "global_ignore_regions": {
            "type": "array",
            "maxItems": MAX_REGIONS,
            "items": REGION_SCHEMA,
        },
        "notes": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
    },
}


TRACK_ANALYST_PROMPT = """
You are the visual track analyst for a paper well-log digitization system.
Analyze the supplied full-color preview and return layout guidance only. A local
OpenCV/Viterbi pipeline will calculate every final curve coordinate and LAS value.

Use preview pixel coordinates, with (0, 0) at the top-left. The preview may be
scaled independently in X and Y, so judge horizontal and vertical distances in
the displayed preview coordinate system. Identify up to six logging tracks and
up to six visible curves in each track. For each curve, provide sparse seed
points on the visible ink, including points before and after an apparent wrap.
Do not invent a curve when it is not visible. Use "unknown" when a color or
scale cannot be determined.

Ignore regions must cover text, legends, handwritten marks, stamps, or other
annotations that should not attract a pixel tracer. Do not mark ordinary grid
lines as ignore regions. Estimate grid spacing separately. A low-confidence
section should identify a vertical interval where the curve is obscured,
overlapping, faint, or ambiguous. max_jump_px is the largest ordinary horizontal
movement between adjacent preview rows; wrap transitions are excluded.
""".strip()


def _finite(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    number = _finite(value, default)
    return max(low, min(high, float(number)))


def _clamp_confidence(value: Any) -> float:
    return _clamp(value, 0.0, 1.0, 0.0)


def _bounded_list(value: Any, limit: int) -> List[Any]:
    return value[:limit] if isinstance(value, list) else []


def build_preview(
    image_bgr: np.ndarray,
    region: Optional[Dict[str, Any]] = None,
    max_width: int = 1600,
    max_height: int = 2200,
) -> Tuple[str, Dict[str, int]]:
    """Create a bounded full-color preview and retain its source mapping."""
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        raise TrackAnalysisError("The uploaded image could not be analyzed")

    source_h, source_w = image_bgr.shape[:2]
    region = region if isinstance(region, dict) else {}
    left = int(_clamp(region.get("left_px", 0), 0, max(0, source_w - 1)))
    right = int(_clamp(region.get("right_px", source_w), left + 1, source_w, source_w))
    top = int(_clamp(region.get("top_px", 0), 0, max(0, source_h - 1)))
    bottom = int(_clamp(region.get("bottom_px", source_h), top + 1, source_h, source_h))
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        raise TrackAnalysisError("The selected analysis region is empty")

    crop_h, crop_w = crop.shape[:2]
    preview_w = max(2, min(int(max_width), crop_w))
    preview_h = max(2, min(int(max_height), crop_h))
    interpolation = cv2.INTER_AREA if preview_w < crop_w or preview_h < crop_h else cv2.INTER_LINEAR
    preview = cv2.resize(crop, (preview_w, preview_h), interpolation=interpolation)
    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 84])
    if not ok:
        raise TrackAnalysisError("The analysis preview could not be encoded")

    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    meta = {
        "source_width": int(source_w),
        "source_height": int(source_h),
        "region_left": int(left),
        "region_right": int(right),
        "region_top": int(top),
        "region_bottom": int(bottom),
        "preview_width": int(preview_w),
        "preview_height": int(preview_h),
    }
    return data_url, meta


def _extract_output_text(response_payload: Dict[str, Any]) -> str:
    direct = response_payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for item in response_payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    raise TrackAnalysisError("The track analyst returned no structured result")


def _map_x(value: Any, meta: Dict[str, int]) -> float:
    preview_span = max(1, meta["preview_width"] - 1)
    source_span = max(1, meta["region_right"] - meta["region_left"] - 1)
    x = _clamp(value, 0, preview_span)
    return float(meta["region_left"]) + (x / preview_span) * source_span


def _map_y(value: Any, meta: Dict[str, int]) -> float:
    preview_span = max(1, meta["preview_height"] - 1)
    source_span = max(1, meta["region_bottom"] - meta["region_top"] - 1)
    y = _clamp(value, 0, preview_span)
    return float(meta["region_top"]) + (y / preview_span) * source_span


def _normalize_region(region: Any, meta: Dict[str, int]) -> Optional[Dict[str, Any]]:
    if not isinstance(region, dict):
        return None
    x1, x2 = sorted((_map_x(region.get("x1"), meta), _map_x(region.get("x2"), meta)))
    y1, y2 = sorted((_map_y(region.get("y1"), meta), _map_y(region.get("y2"), meta)))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "reason": str(region.get("reason") or "annotation")[:160],
        "confidence": _clamp_confidence(region.get("confidence")),
    }


def normalize_analysis(raw: Dict[str, Any], meta: Dict[str, int]) -> Dict[str, Any]:
    """Clamp untrusted model output and remap preview coordinates to source pixels."""
    if not isinstance(raw, dict):
        raise TrackAnalysisError("The track analyst returned an invalid result")

    preview_w = meta["preview_width"]
    preview_h = meta["preview_height"]
    x_scale = max(1, meta["region_right"] - meta["region_left"] - 1) / max(1, preview_w - 1)
    tracks: List[Dict[str, Any]] = []
    for track_index, candidate in enumerate(_bounded_list(raw.get("tracks"), MAX_TRACKS)):
        if not isinstance(candidate, dict):
            continue
        left_x, right_x = sorted((
            _map_x(candidate.get("left_x"), meta),
            _map_x(candidate.get("right_x"), meta),
        ))
        top_y, bottom_y = sorted((
            _map_y(candidate.get("top_y"), meta),
            _map_y(candidate.get("bottom_y"), meta),
        ))
        if right_x - left_x < 3.0 or bottom_y - top_y < 3.0:
            continue

        scale_type = str(candidate.get("scale_type") or "unknown").lower()
        if scale_type not in {"linear", "log", "centered", "unknown"}:
            scale_type = "unknown"
        track_id = str(candidate.get("id") or f"track_{track_index + 1}")[:80]
        ignore_regions = [
            normalized for normalized in (
                _normalize_region(item, meta)
                for item in _bounded_list(candidate.get("ignore_regions"), MAX_REGIONS)
            ) if normalized
        ]

        curves: List[Dict[str, Any]] = []
        for curve_index, curve in enumerate(_bounded_list(candidate.get("curves"), MAX_CURVES_PER_TRACK)):
            if not isinstance(curve, dict):
                continue
            color = str(curve.get("color") or "unknown").lower()
            if color not in {
                "black", "red", "green", "blue", "cyan", "magenta",
                "yellow", "orange", "purple", "unknown",
            }:
                color = "unknown"
            estimated_start_x = _finite(curve.get("estimated_start_x"))
            if estimated_start_x is not None:
                estimated_start_x = _map_x(estimated_start_x, meta)

            seed_points = []
            for seed in _bounded_list(curve.get("seed_points"), MAX_SEEDS_PER_CURVE):
                if not isinstance(seed, dict):
                    continue
                seed_points.append({
                    "x": _map_x(seed.get("x"), meta),
                    "y": _map_y(seed.get("y"), meta),
                    "confidence": _clamp_confidence(seed.get("confidence")),
                })
            seed_points.sort(key=lambda item: item["y"])

            sections = []
            for section in _bounded_list(curve.get("low_confidence_sections"), MAX_REGIONS):
                if not isinstance(section, dict):
                    continue
                y1, y2 = sorted((_map_y(section.get("y1"), meta), _map_y(section.get("y2"), meta)))
                if y2 - y1 < 1.0:
                    continue
                sections.append({
                    "y1": y1,
                    "y2": y2,
                    "reason": str(section.get("reason") or "ambiguous trace")[:160],
                })

            jump_preview = _finite(curve.get("max_jump_px"))
            max_jump_px = None if jump_preview is None else max(1.0, jump_preview * x_scale)
            curves.append({
                "id": str(curve.get("id") or f"curve_{curve_index + 1}")[:80],
                "mnemonic": str(curve.get("mnemonic") or "OTHER").upper()[:32],
                "color": color,
                "estimated_start_x": estimated_start_x,
                "wrap_enabled": bool(curve.get("wrap_enabled") or candidate.get("wraparound")),
                "confidence": _clamp_confidence(curve.get("confidence")),
                "max_jump_px": max_jump_px,
                "seed_points": seed_points,
                "low_confidence_sections": sections,
            })

        tracks.append({
            "id": track_id,
            "left_x": left_x,
            "right_x": right_x,
            "top_y": top_y,
            "bottom_y": bottom_y,
            "scale_type": scale_type,
            "scale_min": _finite(candidate.get("scale_min")),
            "scale_max": _finite(candidate.get("scale_max")),
            "unit": str(candidate.get("unit"))[:32] if candidate.get("unit") else None,
            "horizontal_grid_spacing_px": (
                None if _finite(candidate.get("horizontal_grid_spacing_px")) is None
                else max(1.0, float(candidate["horizontal_grid_spacing_px"]) * x_scale)
            ),
            "vertical_grid_spacing_px": (
                None if _finite(candidate.get("vertical_grid_spacing_px")) is None
                else max(
                    1.0,
                    float(candidate["vertical_grid_spacing_px"])
                    * max(1, meta["region_bottom"] - meta["region_top"] - 1)
                    / max(1, preview_h - 1),
                )
            ),
            "wraparound": bool(candidate.get("wraparound")),
            "confidence": _clamp_confidence(candidate.get("confidence")),
            "curves": curves,
            "ignore_regions": ignore_regions,
        })

    tracks.sort(key=lambda item: item["left_x"])
    global_regions = [
        normalized for normalized in (
            _normalize_region(item, meta)
            for item in _bounded_list(raw.get("global_ignore_regions"), MAX_REGIONS)
        ) if normalized
    ]
    return {
        "schema_version": 1,
        "analysis_confidence": _clamp_confidence(raw.get("analysis_confidence")),
        "source": {
            "width": meta["source_width"],
            "height": meta["source_height"],
            "region": {
                "left_px": meta["region_left"],
                "right_px": meta["region_right"],
                "top_px": meta["region_top"],
                "bottom_px": meta["region_bottom"],
            },
        },
        "preview": {"width": preview_w, "height": preview_h},
        "tracks": tracks,
        "global_ignore_regions": global_regions,
        "notes": [str(note)[:240] for note in _bounded_list(raw.get("notes"), 12)],
    }


def analyze_with_openai(
    image_bgr: np.ndarray,
    api_key: str,
    model: str = DEFAULT_TRACK_ANALYST_MODEL,
    region: Optional[Dict[str, Any]] = None,
    post: Callable[..., Any] = requests.post,
) -> Dict[str, Any]:
    """Request strict visual guidance from the OpenAI Responses API."""
    if not api_key:
        raise TrackAnalysisError("Track analysis is not configured")
    preview_url, meta = build_preview(image_bgr, region=region)
    prompt = (
        TRACK_ANALYST_PROMPT
        + f"\n\nPreview dimensions: {meta['preview_width']} x {meta['preview_height']} pixels."
    )
    payload = {
        "model": model,
        "store": False,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": preview_url, "detail": "high"},
            ],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "well_log_track_analysis",
                "strict": True,
                "schema": TRACK_ANALYSIS_SCHEMA,
            },
        },
        "max_output_tokens": 6000,
    }
    try:
        response = post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 90),
        )
    except requests.RequestException as exc:
        raise TrackAnalysisError("The track analyst could not be reached") from exc

    if response.status_code != 200:
        raise TrackAnalysisError(f"Track analysis failed with status {response.status_code}")
    try:
        response_payload = response.json()
        raw_analysis = json.loads(_extract_output_text(response_payload))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TrackAnalysisError("The track analyst returned invalid structured data") from exc

    normalized = normalize_analysis(raw_analysis, meta)
    normalized["provider"] = "openai"
    normalized["model"] = model
    return normalized


def guided_max_step(default_step: int, guidance: Any, track_width: int) -> int:
    """Apply an advisory jump estimate without allowing unsafe path expansion."""
    default_step = max(1, int(default_step))
    if not isinstance(guidance, dict) or guidance.get("wrap_enabled"):
        return default_step
    estimate = _finite(guidance.get("max_jump_px"))
    if estimate is None:
        return default_step
    suggested = max(3, int(round(estimate * 1.5)))
    return max(1, min(default_step, suggested, max(1, int(track_width) - 1)))


def apply_curve_guidance(
    probability_mask: np.ndarray,
    roi_bgr: np.ndarray,
    guidance: Any,
    roi_left: int,
    roi_top: int,
) -> np.ndarray:
    """Gently bias real image evidence near seeds and suppress annotations."""
    if not isinstance(guidance, dict) or probability_mask is None or probability_mask.size == 0:
        return probability_mask
    h, w = probability_mask.shape[:2]
    if roi_bgr is None or roi_bgr.shape[:2] != (h, w):
        return probability_mask

    probability = probability_mask.astype(np.float32) / 255.0
    seed_points = []
    seed_items = guidance.get("seed_points")
    if not isinstance(seed_items, list):
        seed_items = []
    for seed in seed_items[:MAX_SEEDS_PER_CURVE]:
        if not isinstance(seed, dict):
            continue
        x = _finite(seed.get("x"))
        y = _finite(seed.get("y"))
        if x is None or y is None:
            continue
        x_local = x - float(roi_left)
        y_local = y - float(roi_top)
        if -4 <= x_local < w + 4 and -4 <= y_local < h + 4:
            seed_points.append((
                float(np.clip(x_local, 0, w - 1)),
                float(np.clip(y_local, 0, h - 1)),
                _clamp_confidence(seed.get("confidence")),
            ))
    seed_points.sort(key=lambda item: item[1])

    if seed_points:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        sample_colors = []
        for x, y, _ in seed_points:
            ix, iy = int(round(x)), int(round(y))
            x1, x2 = max(0, ix - 2), min(w, ix + 3)
            y1, y2 = max(0, iy - 2), min(h, iy + 3)
            patch = lab[y1:y2, x1:x2]
            if patch.size:
                sample_colors.append(np.median(patch.reshape(-1, 3), axis=0))

        if sample_colors:
            seed_y = np.array([item[1] for item in seed_points], dtype=np.float32)
            seed_x = np.array([item[0] for item in seed_points], dtype=np.float32)
            x_axis = np.arange(w, dtype=np.float32)[None, :]
            y_margin = max(24.0, h * 0.04)
            for row_start in range(0, h, 256):
                row_end = min(h, row_start + 256)
                lab_chunk = lab[row_start:row_end].astype(np.float32)
                color_distance = np.full((row_end - row_start, w), np.inf, dtype=np.float32)
                for sample in sample_colors:
                    distance = np.linalg.norm(lab_chunk - sample.reshape(1, 1, 3), axis=2)
                    color_distance = np.minimum(color_distance, distance)
                color_likelihood = np.exp(-np.square(color_distance / 34.0))
                y_axis = np.arange(row_start, row_end, dtype=np.float32)[:, None]

                if len(seed_points) >= 2:
                    guide_x = np.interp(y_axis[:, 0], seed_y, seed_x)[:, None]
                    corridor = np.exp(-np.square((x_axis - guide_x) / max(8.0, w * 0.08)))
                    y_gate = np.clip(
                        np.minimum(
                            (y_axis - (seed_y[0] - y_margin)) / y_margin,
                            ((seed_y[-1] + y_margin) - y_axis) / y_margin,
                        ),
                        0.0,
                        1.0,
                    )
                    corridor *= y_gate
                else:
                    sx, sy, confidence = seed_points[0]
                    corridor = np.exp(
                        -np.square((x_axis - sx) / max(10.0, w * 0.1))
                        -np.square((y_axis - sy) / max(30.0, h * 0.08))
                    ) * max(0.25, confidence)

                evidence = np.clip(color_likelihood * corridor, 0.0, 1.0)
                probability[row_start:row_end] = np.maximum(
                    probability[row_start:row_end],
                    evidence * 0.62,
                )

    start_x = _finite(guidance.get("estimated_start_x"))
    if start_x is not None:
        start_local = float(np.clip(start_x - roi_left, 0, w - 1))
        x_bias = np.exp(-np.square((np.arange(w, dtype=np.float32) - start_local) / max(8.0, w * 0.1)))
        top_rows = max(1, min(h, int(round(h * 0.08))))
        probability[:top_rows] *= 1.0 + 0.35 * x_bias[None, :]

    ignore_regions = guidance.get("ignore_regions")
    if not isinstance(ignore_regions, list):
        ignore_regions = []
    for region in ignore_regions[:MAX_REGIONS]:
        if not isinstance(region, dict):
            continue
        x1 = int(math.floor((_finite(region.get("x1"), roi_left) or roi_left) - roi_left))
        x2 = int(math.ceil((_finite(region.get("x2"), roi_left) or roi_left) - roi_left))
        y1 = int(math.floor((_finite(region.get("y1"), roi_top) or roi_top) - roi_top))
        y2 = int(math.ceil((_finite(region.get("y2"), roi_top) or roi_top) - roi_top))
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 > x1 and y2 > y1:
            probability[y1:y2, x1:x2] *= 0.08

    return np.clip(probability * 255.0, 0, 255).astype(np.uint8)
