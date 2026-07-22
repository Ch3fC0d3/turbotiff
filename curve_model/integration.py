"""Feature-flagged adapter between Phase 1 probability maps and TurboTIFF DP."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .infer import predict_curve_probability


PHASE1_MODES = {"classic", "neural_phase1", "hybrid_phase1"}


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not np.isfinite(number):
        number = default
    return min(maximum, max(minimum, number))


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return min(maximum, max(minimum, number))


def normalize_probability(probability: np.ndarray) -> np.ndarray:
    array = np.asarray(probability, dtype=np.float32)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    array = np.where(finite, array, 0.0)
    if float(array.max()) > 1.0:
        array = array / 255.0
    upper = float(np.percentile(array, 99.5))
    if upper > 1e-8:
        array = array / upper
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def build_phase1_probability(
    image: np.ndarray,
    classic_probability: np.ndarray,
    mode: str = "classic",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    neural_weight: float = 0.65,
    classic_weight: float = 0.35,
    centerline_weight: float = 0.70,
    stroke_weight: float = 0.30,
    tile_height: int = 512,
    overlap: int = 96,
    predictor: Callable[..., dict] = predict_curve_probability,
) -> tuple[np.ndarray, dict]:
    selected_mode = str(mode or "classic").lower().strip()
    if selected_mode not in PHASE1_MODES:
        selected_mode = "classic"
    neural_weight = _bounded_float(neural_weight, 0.65, 0.0, 1.0)
    classic_weight = _bounded_float(classic_weight, 0.35, 0.0, 1.0)
    centerline_weight = _bounded_float(centerline_weight, 0.70, 0.0, 1.0)
    stroke_weight = _bounded_float(stroke_weight, 0.30, 0.0, 1.0)
    tile_height = _bounded_int(tile_height, 512, 64, 4096)
    overlap = _bounded_int(overlap, 96, 0, tile_height - 1)
    classic_u8 = np.asarray(classic_probability, dtype=np.uint8)
    metadata = {
        "requested_mode": selected_mode,
        "tracing_mode": "classic",
        "model_checkpoint": Path(model_path).name if model_path else None,
        "model_version": None,
        "neural_weight": float(neural_weight),
        "classic_weight": float(classic_weight),
        "centerline_weight": float(centerline_weight),
        "stroke_weight": float(stroke_weight),
        "inference_resolution": [int(image.shape[0]), int(image.shape[1])],
        "inference_duration_ms": 0.0,
        "fallback_occurred": False,
        "fallback_reason": None,
    }
    if selected_mode == "classic":
        return classic_u8.copy(), metadata

    try:
        if not model_path:
            raise FileNotFoundError("TURBOTIFF_PHASE1_MODEL_PATH is not configured")
        prediction = predictor(
            image,
            model_path=model_path,
            device=device,
            tile_height=tile_height,
            overlap=overlap,
        )
        stroke = normalize_probability(prediction["stroke_probability"])
        centerline = normalize_probability(prediction["centerline_probability"])
        if stroke.shape != classic_u8.shape or centerline.shape != classic_u8.shape:
            raise ValueError("Phase 1 inference output does not match the track crop")
        neural_total = max(1e-8, float(centerline_weight) + float(stroke_weight))
        neural = (
            float(centerline_weight) * centerline
            + float(stroke_weight) * stroke
        ) / neural_total
        if selected_mode == "neural_phase1":
            combined = neural
        else:
            classic = normalize_probability(classic_u8)
            total = max(1e-8, float(neural_weight) + float(classic_weight))
            combined = (float(neural_weight) * neural + float(classic_weight) * classic) / total
        prediction_meta = prediction.get("metadata") or {}
        metadata.update({
            "tracing_mode": selected_mode,
            "model_version": prediction_meta.get("model_version"),
            "inference_resolution": prediction_meta.get("inference_resolution", metadata["inference_resolution"]),
            "inference_duration_ms": float(prediction_meta.get("inference_duration_ms") or 0.0),
        })
        return np.clip(combined * 255.0, 0, 255).astype(np.uint8), metadata
    except Exception as exc:
        metadata.update({
            "tracing_mode": "classic",
            "fallback_occurred": True,
            "fallback_reason": str(exc),
        })
        return classic_u8.copy(), metadata
