"""Dataset loader for generated and golden Phase 1 curve data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover
    torch = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def _letterbox(array: np.ndarray, target_size: tuple[int, int], interpolation: int, fill: int = 0) -> np.ndarray:
    target_h, target_w = map(int, target_size)
    height, width = array.shape[:2]
    scale = min(target_w / max(1, width), target_h / max(1, height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(array, (resized_w, resized_h), interpolation=interpolation)
    if array.ndim == 3:
        output = np.full((target_h, target_w, array.shape[2]), fill, dtype=array.dtype)
    else:
        output = np.full((target_h, target_w), fill, dtype=array.dtype)
    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    output[top:top + resized_h, left:left + resized_w] = resized
    return output


class SyntheticCurveDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        records: Optional[Sequence[dict]] = None,
        target_size: tuple[int, int] = (512, 256),
    ):
        if torch is None:
            raise RuntimeError(f"PyTorch is required for training datasets: {TORCH_IMPORT_ERROR}")
        self.data_dir = Path(data_dir)
        self.target_size = tuple(map(int, target_size))
        if records is None:
            manifest = self.data_dir / "manifest.jsonl"
            if not manifest.exists():
                raise FileNotFoundError(f"Synthetic manifest not found: {manifest}")
            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = cv2.imread(str(self.data_dir / record["image"]), cv2.IMREAD_COLOR)
        stroke = cv2.imread(str(self.data_dir / record["stroke_mask"]), cv2.IMREAD_GRAYSCALE)
        centerline = cv2.imread(str(self.data_dir / record["centerline_mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or stroke is None or centerline is None:
            raise ValueError(f"Could not load synthetic sample {record.get('id', index)}")

        image = _letterbox(image, self.target_size, cv2.INTER_AREA, fill=235)
        stroke = _letterbox(stroke, self.target_size, cv2.INTER_NEAREST, fill=0)
        centerline = _letterbox(centerline, self.target_size, cv2.INTER_NEAREST, fill=0)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(np.transpose(image_rgb, (2, 0, 1))).float()
        stroke_tensor = torch.from_numpy((stroke > 127).astype(np.float32)[None, ...])
        centerline_tensor = torch.from_numpy((centerline > 127).astype(np.float32)[None, ...])
        sample = {
            "image": image_tensor,
            "stroke_mask": stroke_tensor,
            "centerline_mask": centerline_tensor,
            "sample_id": str(record.get("id", index)),
            "source": str(record.get("source", "synthetic")),
            "hard_case": bool(record.get("hard_case", False)),
            "stroke_label_valid": torch.ones((1, 1, 1), dtype=torch.float32),
            "grid_label_valid": torch.ones((1, 1, 1), dtype=torch.float32),
        }
        phase2_keys = ("distance_field", "direction_field", "grid_mask", "valid_direction_mask")
        if all(record.get(key) for key in phase2_keys):
            distance = np.load(self.data_dir / record["distance_field"], allow_pickle=False).astype(np.float32)
            direction = np.load(self.data_dir / record["direction_field"], allow_pickle=False).astype(np.float32)
            grid = cv2.imread(str(self.data_dir / record["grid_mask"]), cv2.IMREAD_GRAYSCALE)
            valid_direction = cv2.imread(str(self.data_dir / record["valid_direction_mask"]), cv2.IMREAD_GRAYSCALE)
            if direction.ndim != 3 or direction.shape[0] != 2 or grid is None or valid_direction is None:
                raise ValueError(f"Invalid Phase 2 labels for sample {record.get('id', index)}")
            distance = _letterbox(distance, self.target_size, cv2.INTER_LINEAR, fill=0)
            direction_hwc = np.transpose(direction, (1, 2, 0))
            direction_hwc = _letterbox(direction_hwc, self.target_size, cv2.INTER_LINEAR, fill=0)
            grid = _letterbox(grid, self.target_size, cv2.INTER_NEAREST, fill=0)
            valid_direction = _letterbox(valid_direction, self.target_size, cv2.INTER_NEAREST, fill=0)
            direction_norm = np.linalg.norm(direction_hwc, axis=2, keepdims=True)
            direction_hwc = np.divide(
                direction_hwc,
                np.maximum(direction_norm, 1e-6),
                out=np.zeros_like(direction_hwc),
                where=direction_norm > 1e-6,
            )
            sample.update({
                "distance_field": torch.from_numpy(distance[None, ...]).float(),
                "direction_field": torch.from_numpy(np.transpose(direction_hwc, (2, 0, 1))).float(),
                "grid_mask": torch.from_numpy((grid > 127).astype(np.float32)[None, ...]),
                "valid_direction_mask": torch.from_numpy((valid_direction > 127).astype(np.float32)[None, ...]),
            })
        return sample


def _derive_centerline_geometry(
    centerline_x: np.ndarray,
    valid_rows: np.ndarray,
    width: int,
    maximum_distance: float = 16.0,
    direction_radius: int = 5,
    tube_radius: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = centerline_x.size
    rows = np.arange(height)
    valid_rows = np.asarray(valid_rows, dtype=bool) & np.isfinite(centerline_x)
    if not valid_rows.any():
        raise ValueError("Real Phase 2 sample has no valid centerline rows")
    filled = centerline_x.astype(np.float32).copy()
    filled[~valid_rows] = np.interp(rows[~valid_rows], rows[valid_rows], filled[valid_rows])
    centerline = np.zeros((height, width), dtype=np.uint8)
    for row in np.flatnonzero(valid_rows):
        x = int(np.clip(round(float(filled[row])), 0, width - 1))
        cv2.circle(centerline, (x, int(row)), 1, 255, -1, cv2.LINE_8)
    columns = np.arange(width, dtype=np.float32)[None, :]
    pixel_distance = np.abs(columns - filled[:, None])
    distance = np.clip(1.0 - pixel_distance / max(1.0, float(maximum_distance)), 0.0, 1.0)
    distance[~valid_rows] = 0.0
    before = np.maximum(0, rows - max(1, int(direction_radius)))
    after = np.minimum(height - 1, rows + max(1, int(direction_radius)))
    dx = filled[after] - filled[before]
    dy = (after - before).astype(np.float32)
    magnitude = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-6)
    valid_direction = (pixel_distance <= max(0, int(tube_radius))) & valid_rows[:, None]
    direction = np.zeros((2, height, width), dtype=np.float32)
    direction[0] = np.where(valid_direction, (dx / magnitude)[:, None], 0.0)
    direction[1] = np.where(valid_direction, (dy / magnitude)[:, None], 0.0)
    return centerline, distance.astype(np.float32), direction, (valid_direction.astype(np.uint8) * 255)


class Phase2RealCurveDataset(Dataset):
    """Golden real-track loader that derives geometry without inventing stroke labels."""

    def __init__(self, data_dir: Path | str, records: Optional[Sequence[dict]] = None, target_size=(512, 256)):
        if torch is None:
            raise RuntimeError(f"PyTorch is required for training datasets: {TORCH_IMPORT_ERROR}")
        self.data_dir = Path(data_dir)
        self.target_size = tuple(map(int, target_size))
        self.records = list(records) if records is not None else load_manifest(self.data_dir)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = cv2.imread(str(self.data_dir / record["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load real Phase 2 image {record.get('id', index)}")
        trace = np.load(self.data_dir / record["trace"], allow_pickle=False)
        centerline_x = trace["centerline_x_by_row"].astype(np.float32)
        valid_rows = trace["valid_row_mask"].astype(bool) if "valid_row_mask" in trace else np.isfinite(centerline_x)
        if image.shape[0] != centerline_x.size:
            raise ValueError(f"Real Phase 2 trace length mismatch for {record.get('id', index)}")
        centerline, distance, direction, valid_direction = _derive_centerline_geometry(
            centerline_x, valid_rows, image.shape[1]
        )
        stroke_path = record.get("stroke_mask")
        grid_path = record.get("grid_mask")
        stroke = cv2.imread(str(self.data_dir / stroke_path), cv2.IMREAD_GRAYSCALE) if stroke_path else None
        grid = cv2.imread(str(self.data_dir / grid_path), cv2.IMREAD_GRAYSCALE) if grid_path else None
        stroke_valid = stroke is not None
        grid_valid = grid is not None
        stroke = stroke if stroke is not None else np.zeros(image.shape[:2], dtype=np.uint8)
        grid = grid if grid is not None else np.zeros(image.shape[:2], dtype=np.uint8)

        image = _letterbox(image, self.target_size, cv2.INTER_AREA, fill=235)
        stroke = _letterbox(stroke, self.target_size, cv2.INTER_NEAREST, fill=0)
        centerline = _letterbox(centerline, self.target_size, cv2.INTER_NEAREST, fill=0)
        distance = _letterbox(distance, self.target_size, cv2.INTER_LINEAR, fill=0)
        direction_hwc = _letterbox(np.transpose(direction, (1, 2, 0)), self.target_size, cv2.INTER_LINEAR, fill=0)
        grid = _letterbox(grid, self.target_size, cv2.INTER_NEAREST, fill=0)
        valid_direction = _letterbox(valid_direction, self.target_size, cv2.INTER_NEAREST, fill=0)
        norm = np.linalg.norm(direction_hwc, axis=2, keepdims=True)
        direction_hwc = np.divide(direction_hwc, np.maximum(norm, 1e-6), out=np.zeros_like(direction_hwc), where=norm > 1e-6)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(np.transpose(image_rgb, (2, 0, 1))).float(),
            "stroke_mask": torch.from_numpy((stroke > 127).astype(np.float32)[None]),
            "centerline_mask": torch.from_numpy((centerline > 127).astype(np.float32)[None]),
            "distance_field": torch.from_numpy(distance[None]).float(),
            "direction_field": torch.from_numpy(np.transpose(direction_hwc, (2, 0, 1))).float(),
            "grid_mask": torch.from_numpy((grid > 127).astype(np.float32)[None]),
            "valid_direction_mask": torch.from_numpy((valid_direction > 127).astype(np.float32)[None]),
            "stroke_label_valid": torch.full((1, 1, 1), float(stroke_valid)),
            "grid_label_valid": torch.full((1, 1, 1), float(grid_valid)),
            "sample_id": str(record.get("id", index)),
            "source": "real",
            "hard_case": False,
        }


def load_manifest(data_dir: Path | str) -> list[dict]:
    manifest = Path(data_dir) / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Synthetic manifest not found: {manifest}")
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
