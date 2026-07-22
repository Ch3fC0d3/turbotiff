try:
    import torch
    import torch.nn as nn
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
import numpy as np
import cv2
from pathlib import Path

class CurveTraceNet(nn.Module if nn is not None else object):
    def __init__(self, in_ch: int = 1, base: int = 16):
        if nn is None:
            raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x):
        h = self.enc(x)
        logits = self.dec(h).squeeze(1)
        return torch.softmax(logits, dim=-1)


def _prediction_to_normalized_x(prediction: np.ndarray) -> np.ndarray:
    """Normalize coordinate-vector or probability-map model output to one x per row."""
    pred = np.asarray(prediction, dtype=np.float32).squeeze()
    if pred.ndim == 1:
        return np.clip(pred, 0.0, 1.0)
    if pred.ndim != 2 or pred.shape[1] < 1:
        raise ValueError(f"Unexpected AI trace output shape: {pred.shape}")

    # Preserve discrete candidate peaks. An expected coordinate can land in
    # blank space when two curves have similar probability in the same row.
    peak_indices = np.argmax(pred, axis=1).astype(np.float32)
    return peak_indices / max(1, pred.shape[1] - 1)


class AITracer:
    def __init__(self, model_path="curve_trace_model.pt"):
        self.model = None
        self.input_h = 256
        self.input_w = 128

        if torch is None:
            self.device = None
            print(f"[WARN] AI model unavailable because PyTorch could not load: {TORCH_IMPORT_ERROR}")
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if available
        p = Path(model_path)
        if p.exists():
            try:
                ckpt = torch.load(str(p), map_location=self.device, weights_only=True)
                self.input_h = ckpt.get('input_h', 256)
                self.input_w = ckpt.get('input_w', 128)
                
                self.model = CurveTraceNet().to(self.device)
                self.model.load_state_dict(ckpt['state_dict'])
                self.model.eval()
                print(f"[OK] AI Model loaded from {model_path} (Device: {self.device})")
            except Exception as e:
                print(f"[WARN] Failed to load AI model: {e}")
        else:
            print(f"[WARN] AI model not found at {model_path}")

    def is_available(self):
        return self.model is not None

    def predict_probability_map(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return the model's curve heatmap at the original ROI resolution."""
        if self.model is None:
            raise RuntimeError("AI model not loaded.")

        orig_h, orig_w = roi_bgr.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return np.zeros((orig_h, orig_w), dtype=np.float32)

        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        x_tensor = torch.from_numpy(roi_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            prediction = self.model(x_tensor).cpu().numpy()

        pred = np.asarray(prediction, dtype=np.float32).squeeze()
        if pred.ndim == 1:
            # Backward compatibility for a custom model that still emits one
            # normalized coordinate per row.
            coords = np.clip(pred, 0.0, 1.0)
            heatmap = np.zeros((coords.size, self.input_w), dtype=np.float32)
            indices = np.rint(coords * max(1, self.input_w - 1)).astype(np.int32)
            heatmap[np.arange(coords.size), np.clip(indices, 0, self.input_w - 1)] = 1.0
        elif pred.ndim == 2:
            heatmap = np.clip(pred, 0.0, None)
        else:
            raise ValueError(f"Unexpected AI heatmap output shape: {pred.shape}")

        heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        row_max = heatmap.max(axis=1, keepdims=True)
        row_max[row_max <= 1e-8] = 1.0
        return np.clip(heatmap / row_max, 0.0, 1.0).astype(np.float32)

    def trace(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Runs the AI model on a cropped BGR image of the curve track.
        Returns a 1D numpy array of x-coordinates for each row in the original roi.
        """
        orig_h, orig_w = roi_bgr.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return np.array([])
        heatmap = self.predict_probability_map(roi_bgr)
        return np.argmax(heatmap, axis=1).astype(np.float32)
