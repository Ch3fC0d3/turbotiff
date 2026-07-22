"""Explicit Phase 2 modes and observable fallback chaining."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .integration import build_phase1_probability
from .phase2_infer import predict_phase2_geometry
from .phase2_score import Phase2ScoreConfig, build_phase2_trace_score


PHASE2_MODES = {"neural_phase2", "hybrid_phase2"}


def build_phase2_probability(
    image: np.ndarray,
    classic_probability: np.ndarray,
    mode: str,
    phase2_model_path: Optional[str],
    phase1_model_path: Optional[str] = None,
    device: Optional[str] = None,
    tile_height: int = 512,
    overlap: int = 96,
    score_config: Optional[Phase2ScoreConfig] = None,
    predictor: Callable[..., dict] = predict_phase2_geometry,
) -> tuple[np.ndarray, dict, dict]:
    requested = str(mode or "classic").lower().strip()
    if requested not in PHASE2_MODES:
        raise ValueError(f"Unsupported Phase 2 mode: {requested}")
    metadata = {
        "requested_mode": requested,
        "actual_mode": "classic",
        "tracing_mode": "classic",
        "fallback": False,
        "fallback_occurred": False,
        "fallback_reason": None,
        "fallback_chain": [],
        "model_checkpoint": Path(phase2_model_path).name if phase2_model_path else None,
        "model_version": None,
        "model_format_version": None,
        "direction_adjustment_available": False,
    }
    try:
        if not phase2_model_path:
            raise FileNotFoundError("TURBOTIFF_PHASE2_MODEL_PATH is not configured")
        prediction = predictor(
            image,
            model_path=phase2_model_path,
            device=device,
            tile_height=tile_height,
            overlap=overlap,
        )
        score, direction, fusion_config = build_phase2_trace_score(
            prediction["stroke_probability"],
            prediction["centerline_probability"],
            prediction["distance_field"],
            prediction["direction_field"],
            prediction["grid_probability"],
            classic_probability=classic_probability if requested == "hybrid_phase2" else None,
            config=score_config,
        )
        prediction_metadata = prediction.get("metadata") or {}
        metadata.update({
            "actual_mode": requested,
            "tracing_mode": requested,
            "model_version": prediction_metadata.get("model_version"),
            "model_format_version": prediction_metadata.get("model_format_version"),
            "inference_resolution": prediction_metadata.get("inference_resolution"),
            "inference_duration_ms": float(prediction_metadata.get("inference_duration_ms") or 0.0),
            "direction_adjustment_available": True,
            "fusion_config": fusion_config,
        })
        auxiliary = {
            "direction_field": direction,
            "stroke_probability": prediction["stroke_probability"],
            "centerline_probability": prediction["centerline_probability"],
            "distance_field": prediction["distance_field"],
            "grid_probability": prediction["grid_probability"],
        }
        return np.clip(score * 255.0, 0, 255).astype(np.uint8), metadata, auxiliary
    except Exception as phase2_error:
        metadata["fallback"] = True
        metadata["fallback_occurred"] = True
        metadata["fallback_reason"] = str(phase2_error)
        metadata["fallback_chain"].append({"mode": requested, "reason": str(phase2_error)})

    phase1_mode = "hybrid_phase1" if requested == "hybrid_phase2" else "neural_phase1"
    phase1_probability, phase1_metadata = build_phase1_probability(
        image,
        classic_probability,
        mode=phase1_mode,
        model_path=phase1_model_path,
        device=device,
        tile_height=tile_height,
        overlap=overlap,
    )
    if not phase1_metadata.get("fallback_occurred"):
        metadata.update({
            "actual_mode": phase1_mode,
            "tracing_mode": phase1_mode,
            "model_version": phase1_metadata.get("model_version"),
        })
        metadata["fallback_chain"].append({"mode": phase1_mode, "reason": None})
        return phase1_probability, metadata, {}

    metadata["fallback_chain"].append({
        "mode": phase1_mode,
        "reason": phase1_metadata.get("fallback_reason"),
    })
    metadata.update({"actual_mode": "classic", "tracing_mode": "classic"})
    return np.asarray(classic_probability, dtype=np.uint8).copy(), metadata, {}

