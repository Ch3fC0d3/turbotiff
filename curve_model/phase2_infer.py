"""Version-validated, overlap-tiled inference for Phase 2 checkpoints."""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .infer import _blend_window, _pad_to_multiple, _resolve_device, _tile_starts
from .model import require_torch, torch
from .phase2_model import CurvePhase2UNet


PHASE2_OUTPUTS = ("stroke_probability", "centerline_probability", "distance_field", "direction_field", "grid_probability")


def validate_phase2_checkpoint(checkpoint: dict) -> None:
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Phase 2 checkpoint must contain a state_dict")
    model_config = checkpoint.get("model_config") or {}
    version = checkpoint.get("model_format_version", model_config.get("model_format_version"))
    phase = checkpoint.get("phase", model_config.get("phase"))
    outputs = checkpoint.get("outputs", model_config.get("outputs")) or []
    if int(version or 0) != 2 or int(phase or 0) != 2:
        raise ValueError("Checkpoint is not a Phase 2 model format")
    missing = set(CurvePhase2UNet.outputs) - set(outputs)
    if missing:
        raise ValueError(f"Phase 2 checkpoint is missing output declarations: {sorted(missing)}")
    state_keys = set(checkpoint["state_dict"])
    required_heads = {"distance_head.weight", "direction_head.weight", "grid_head.weight"}
    if not required_heads.issubset(state_keys):
        raise ValueError("Checkpoint is missing one or more Phase 2 output heads")


@functools.lru_cache(maxsize=4)
def _load_phase2_cached(resolved_path: str, device: str, modified_ns: int):
    del modified_ns
    require_torch()
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=True)
    validate_phase2_checkpoint(checkpoint)
    config = checkpoint.get("model_config") or {}
    model = CurvePhase2UNet(
        in_channels=int(config.get("in_channels", 3)),
        base_channels=int(config.get("base_channels", 16)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    metadata = {
        "model_version": checkpoint.get("model_version") or model.model_version,
        "model_format_version": 2,
        "training_config": checkpoint.get("training_config") or {},
        "loss_configuration": checkpoint.get("loss_configuration") or {},
        "dataset_version": checkpoint.get("dataset_version"),
    }
    return model, metadata


def load_phase2_model(model_path: Path | str, device: Optional[str] = None):
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Phase 2 curve model not found: {path}")
    selected_device = _resolve_device(device)
    model, metadata = _load_phase2_cached(str(path), selected_device, path.stat().st_mtime_ns)
    return model, selected_device, dict(metadata)


def _predict_tile(model, device: str, tile_bgr: np.ndarray) -> dict:
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    padded, original_shape = _pad_to_multiple(rgb)
    tensor = torch.from_numpy(np.transpose(padded, (2, 0, 1))).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = model(tensor)
        result = {
            "stroke_probability": torch.sigmoid(outputs["stroke_logits"])[0, 0].cpu().numpy(),
            "centerline_probability": torch.sigmoid(outputs["centerline_logits"])[0, 0].cpu().numpy(),
            "distance_field": outputs["distance_field"][0, 0].cpu().numpy(),
            "direction_field": outputs["direction"][0].cpu().numpy(),
            "grid_probability": torch.sigmoid(outputs["grid_logits"])[0, 0].cpu().numpy(),
        }
    height, width = original_shape
    for key in ("stroke_probability", "centerline_probability", "distance_field", "grid_probability"):
        result[key] = result[key][:height, :width].astype(np.float32)
    result["direction_field"] = result["direction_field"][:, :height, :width].astype(np.float32)
    return result


def blend_phase2_tiles(image: np.ndarray, tile_height: int, overlap: int, predict_tile) -> dict:
    height, width = image.shape[:2]
    tile_height = max(16, min(int(tile_height), max(16, height)))
    starts = _tile_starts(height, tile_height, overlap)
    scalar_keys = ("stroke_probability", "centerline_probability", "distance_field", "grid_probability")
    sums = {key: np.zeros((height, width), dtype=np.float32) for key in scalar_keys}
    direction_sum = np.zeros((2, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, 1), dtype=np.float32)
    for start in starts:
        end = min(height, start + tile_height)
        tile = predict_tile(image[start:end])
        weights = _blend_window(end - start, overlap)
        if start == 0:
            weights[:min(overlap, weights.shape[0])] = 1.0
        if end == height:
            weights[max(0, weights.shape[0] - overlap):] = 1.0
        for key in scalar_keys:
            if tile[key].shape != (end - start, width):
                raise ValueError(f"Phase 2 tile output {key} has the wrong shape")
            sums[key][start:end] += tile[key] * weights
        if tile["direction_field"].shape != (2, end - start, width):
            raise ValueError("Phase 2 direction tile has the wrong shape")
        direction_sum[:, start:end] += tile["direction_field"] * weights[None, ...]
        weight_sum[start:end] += weights
    denominator = np.maximum(weight_sum, 1e-6)
    result = {key: sums[key] / denominator for key in scalar_keys}
    direction = direction_sum / denominator[None, ...]
    norm = np.linalg.norm(direction, axis=0, keepdims=True)
    result["direction_field"] = np.divide(direction, np.maximum(norm, 1e-6), out=np.zeros_like(direction), where=norm > 1e-6)
    return result


def predict_phase2_geometry(
    image: np.ndarray,
    model_path: str,
    device: str | None = None,
    tile_height: int = 512,
    overlap: int = 96,
) -> dict:
    if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Phase 2 inference expects a BGR image shaped [H, W, 3]")
    started = time.perf_counter()
    model, selected_device, checkpoint_meta = load_phase2_model(model_path, device)
    outputs = blend_phase2_tiles(
        image,
        tile_height,
        overlap,
        lambda tile: _predict_tile(model, selected_device, tile),
    )
    for key in ("stroke_probability", "centerline_probability", "distance_field", "grid_probability"):
        outputs[key] = np.clip(outputs[key], 0.0, 1.0).astype(np.float32)
    outputs["direction_field"] = outputs["direction_field"].astype(np.float32)
    outputs["metadata"] = {
        **checkpoint_meta,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "device": selected_device,
        "inference_resolution": [int(image.shape[0]), int(image.shape[1])],
        "tile_height": int(tile_height),
        "tile_overlap": int(overlap),
        "inference_duration_ms": (time.perf_counter() - started) * 1000.0,
    }
    return outputs

