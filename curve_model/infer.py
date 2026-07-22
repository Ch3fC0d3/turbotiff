"""Cached, overlap-tiled inference for the Phase 1 curve detector."""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .model import CurvePhase1UNet, require_torch, torch


def _resolve_device(device: Optional[str]) -> str:
    require_torch()
    if device:
        requested = str(device).lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=4)
def _load_model_cached(resolved_path: str, device: str, modified_ns: int):
    del modified_ns  # Included in the cache key so updated checkpoints reload.
    require_torch()
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Phase 1 checkpoint must contain a state_dict")
    model_config = checkpoint.get("model_config") or {}
    model = CurvePhase1UNet(
        in_channels=int(model_config.get("in_channels", 3)),
        base_channels=int(model_config.get("base_channels", 16)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    metadata = {
        "model_version": checkpoint.get("model_version") or model.model_version,
        "model_config": model.configuration(),
        "training_config": checkpoint.get("training_config") or {},
    }
    return model, metadata


def load_model(model_path: Path | str, device: Optional[str] = None):
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 curve model not found: {path}")
    selected_device = _resolve_device(device)
    model, metadata = _load_model_cached(str(path), selected_device, path.stat().st_mtime_ns)
    return model, selected_device, dict(metadata)


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    length = int(length)
    tile_size = max(1, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _blend_window(height: int, overlap: int) -> np.ndarray:
    weights = np.ones(int(height), dtype=np.float32)
    fade = min(max(0, int(overlap)), max(0, int(height) // 2))
    if fade:
        ramp = np.linspace(0.05, 1.0, fade, dtype=np.float32)
        weights[:fade] = ramp
        weights[-fade:] = ramp[::-1]
    return weights[:, None]


def blend_vertical_tiles(
    image: np.ndarray,
    tile_height: int,
    overlap: int,
    predict_tile: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Run and blend a tile predictor without seams at overlap boundaries."""
    height, width = image.shape[:2]
    tile_height = max(16, min(int(tile_height), max(16, height)))
    starts = _tile_starts(height, tile_height, overlap)
    stroke_sum = np.zeros((height, width), dtype=np.float32)
    center_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, 1), dtype=np.float32)

    for start in starts:
        end = min(height, start + tile_height)
        tile = image[start:end]
        stroke, centerline = predict_tile(tile)
        if stroke.shape != tile.shape[:2] or centerline.shape != tile.shape[:2]:
            raise ValueError("Tile predictor returned a probability map with the wrong shape")
        weights = _blend_window(end - start, overlap)
        if start == 0:
            weights[:min(overlap, weights.shape[0])] = 1.0
        if end == height:
            weights[max(0, weights.shape[0] - overlap):] = 1.0
        stroke_sum[start:end] += stroke.astype(np.float32) * weights
        center_sum[start:end] += centerline.astype(np.float32) * weights
        weight_sum[start:end] += weights

    weight_sum = np.maximum(weight_sum, 1e-6)
    return stroke_sum / weight_sum, center_sum / weight_sum


def _pad_to_multiple(image_rgb: np.ndarray, multiple: int = 4) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_rgb.shape[:2]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if not pad_h and not pad_w:
        return image_rgb, (height, width)
    padded = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return padded, (height, width)


def _model_predict_tile(model, device: str, tile_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    padded, original_shape = _pad_to_multiple(rgb)
    tensor = torch.from_numpy(np.transpose(padded, (2, 0, 1))).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = model(tensor)
        stroke = torch.sigmoid(outputs["stroke_logits"])[0, 0].cpu().numpy()
        centerline = torch.sigmoid(outputs["centerline_logits"])[0, 0].cpu().numpy()
    height, width = original_shape
    return stroke[:height, :width].astype(np.float32), centerline[:height, :width].astype(np.float32)


def predict_curve_probability(
    image: np.ndarray,
    model_path: str,
    device: str | None = None,
    tile_height: int = 512,
    overlap: int = 96,
) -> dict:
    """Return full-resolution stroke and centerline probabilities for a BGR track crop."""
    if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Phase 1 inference expects a BGR image shaped [H, W, 3]")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("Phase 1 inference received an empty image")

    started = time.perf_counter()
    model, selected_device, checkpoint_meta = load_model(model_path, device)

    def predictor(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _model_predict_tile(model, selected_device, tile)

    stroke, centerline = blend_vertical_tiles(image, tile_height, overlap, predictor)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    height, width = image.shape[:2]
    return {
        "stroke_probability": np.clip(stroke, 0.0, 1.0).astype(np.float32),
        "centerline_probability": np.clip(centerline, 0.0, 1.0).astype(np.float32),
        "metadata": {
            "model_path": str(Path(model_path).expanduser().resolve()),
            "model_version": checkpoint_meta["model_version"],
            "device": selected_device,
            "inference_resolution": [int(height), int(width)],
            "tile_height": int(tile_height),
            "tile_overlap": int(overlap),
            "inference_duration_ms": elapsed_ms,
        },
    }

