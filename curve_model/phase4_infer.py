"""Common CPU/GPU inference interface for Phase 4 lightweight and adapter models."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

from .model import require_torch, torch
from .phase4_model import CurvePhase4UNet
from .advanced import CurveAdapterModel


@lru_cache(maxsize=4)
def _load(path: str, modified_ns: int, device_name: str):
    require_torch(); checkpoint=torch.load(path,map_location=device_name,weights_only=True)
    config=checkpoint.get("model_config") or {}; architecture=config.get("architecture","")
    cls=CurveAdapterModel if architecture=="curve_adapter_v1" else CurvePhase4UNet
    model=cls(base_channels=int(config.get("base_channels",24 if cls is CurveAdapterModel else 16)),prompt_conditioned=bool(config.get("prompt_conditioned",True)))
    model.load_state_dict(checkpoint["state_dict"]); model.to(device_name).eval()
    return model,checkpoint


def predict_phase4_geometry(image: np.ndarray, model_path: str, prompt_map: np.ndarray | None = None, device: str | None = None) -> dict:
    require_torch(); path=Path(model_path).resolve()
    if not path.exists(): raise FileNotFoundError(path)
    selected=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model,checkpoint=_load(str(path),path.stat().st_mtime_ns,selected)
    bgr=np.asarray(image)
    if bgr.ndim!=3 or bgr.shape[2]!=3: raise ValueError("Phase 4 image must have shape [H, W, 3]")
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    tensor=torch.from_numpy(np.transpose(rgb,(2,0,1))[None]).to(selected)
    prompt=None
    if prompt_map is not None:
        prompt_array=np.asarray(prompt_map,dtype=np.float32)
        if prompt_array.shape!=bgr.shape[:2]: raise ValueError("prompt_map must match image dimensions")
        prompt=torch.from_numpy(np.clip(prompt_array,0,1)[None,None]).to(selected)
    with torch.inference_mode(): output=model(tensor,prompt)
    wrap=torch.softmax(output["wrap_logits"],dim=1)[0].cpu().numpy()
    return {
        "stroke_probability":torch.sigmoid(output["stroke_logits"])[0,0].cpu().numpy(),
        "centerline_probability":torch.sigmoid(output["centerline_logits"])[0,0].cpu().numpy(),
        "distance_field":output["distance_field"][0,0].cpu().numpy(),
        "direction_field":output["direction"][0].cpu().numpy(),
        "grid_probability":torch.sigmoid(output["grid_logits"])[0,0].cpu().numpy(),
        "wrap_probability_right_to_left":wrap[1],
        "wrap_probability_left_to_right":wrap[2],
        "metadata":{"model_version":checkpoint.get("model_version"),"architecture":checkpoint.get("model_config",{}).get("architecture"),"device":selected},
    }
