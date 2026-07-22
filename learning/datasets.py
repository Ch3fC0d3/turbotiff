"""Immutable dataset manifests, correction export, duplicate and leakage checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from curve_model.dataset import _derive_centerline_geometry
from .corrections import CorrectionStore


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return f"{sum((1 << i) for i, value in enumerate((small >= small.mean()).flat) if value):016x}"


def hamming_hash(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


class DatasetRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        if not self.registry_path.exists():
            self.registry_path.write_text('{"datasets": []}\n', encoding="utf-8")

    def entries(self):
        return json.loads(self.registry_path.read_text(encoding="utf-8"))["datasets"]

    def register(self, dataset_id: str, samples: list[dict], parent_dataset=None, notes="") -> dict:
        directory = self.root / dataset_id
        if any(item["dataset_id"] == dataset_id for item in self.entries()) or (directory / "manifest.json").exists():
            raise FileExistsError(f"Dataset version already exists: {dataset_id}")
        canonical = json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
        label_names = ("centerline", "wrap", "stroke", "grid")
        manifest = {
            "dataset_id": dataset_id, "created_at": _now(), "parent_dataset": parent_dataset,
            "sample_count": len(samples),
            "source_types": {name: sum(s.get("source_type") == name for s in samples) for name in sorted({s.get("source_type", "unknown") for s in samples})},
            "label_counts": {name: sum(bool(s.get("labels", {}).get(name)) for s in samples) for name in label_names},
            "source_record_ids": [s.get("record_id") for s in samples if s.get("record_id")],
            "content_hash": hashlib.sha256(canonical).hexdigest(), "notes": notes,
        }
        directory.mkdir(exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        (directory / "samples.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in samples), encoding="utf-8")
        registry = {"datasets": self.entries() + [manifest]}
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.registry_path)
        return manifest


def export_approved_corrections(store_root: Path | str, datasets_root: Path | str, dataset_id: str) -> dict:
    store = CorrectionStore(store_root)
    registry = DatasetRegistry(datasets_root)
    output = Path(datasets_root) / dataset_id
    records = store.approved_for_training()
    samples = []
    for record in records:
        correction = record["correction"]
        x = np.asarray(correction["x_by_row"], dtype=np.float32)
        valid = np.asarray(correction.get("valid_row_mask", np.isfinite(x)), dtype=bool)
        width = int(record.get("track_dimensions", [len(x), 0])[1] or record.get("track_width", 0))
        if width < 1:
            raise ValueError(f"Record {record['record_id']} has no track width")
        center, distance, direction, valid_direction = _derive_centerline_geometry(x, valid, width)
        wrap_index = np.asarray(correction.get("wrap_index_by_row", np.zeros(x.size)), dtype=np.int32)
        wrap_event_class = np.zeros(x.size, dtype=np.uint8)
        if x.size > 1:
            change = np.diff(wrap_index)
            wrap_event_class[1:][change > 0] = 1
            wrap_event_class[1:][change < 0] = 2
        sample_dir = output / "samples" / record["record_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        source_image = Path(store_root) / record["track_image_path"]
        image_target = sample_dir / source_image.name
        shutil.copy2(source_image, image_target)
        cv2.imwrite(str(sample_dir / "centerline.png"), center)
        np.savez_compressed(sample_dir / "geometry.npz", distance_field=distance, direction_field=direction,
                            valid_direction_mask=valid_direction, wrap_index_by_row=wrap_index,
                            wrap_event_class_by_row=wrap_event_class, valid_row_mask=valid)
        samples.append({
            "id": record["record_id"], "record_id": record["record_id"], "source_type": "human_corrected",
            "source_group": record.get("source_image_id"), "image_checksum": record.get("image_checksum"),
            "image": str(image_target.relative_to(output)), "labels": {"centerline": True, "wrap": True, "stroke": False, "grid": False},
            "centerline_mask": str((sample_dir / "centerline.png").relative_to(output)),
            "geometry": str((sample_dir / "geometry.npz").relative_to(output)),
            "label_quality": {"centerline_quality": "human_approved", "wrap_quality": "human_approved", "stroke_quality": "not_available", "grid_quality": "not_available"},
        })
    return registry.register(dataset_id, samples, notes="Approved real curve corrections")


def leakage_report(training_samples: list[dict], golden_samples: list[dict], perceptual_tolerance: int = 4) -> dict:
    exact_golden = {s.get("image_checksum") for s in golden_samples if s.get("image_checksum")}
    groups = {s.get("source_group") for s in golden_samples if s.get("source_group")}
    exact = [s.get("id") for s in training_samples if s.get("image_checksum") in exact_golden]
    grouped = [s.get("id") for s in training_samples if s.get("source_group") in groups]
    near = []
    for train in training_samples:
        if not train.get("perceptual_hash"):
            continue
        if any(g.get("perceptual_hash") and hamming_hash(train["perceptual_hash"], g["perceptual_hash"]) <= perceptual_tolerance for g in golden_samples):
            near.append(train.get("id"))
    return {"exact_duplicates": exact, "source_group_overlap": grouped, "near_duplicates": near,
            "blocked": bool(exact or grouped or near)}


def grouped_split(samples: list[dict], validation_fraction: float = 0.2, seed: int = 0):
    groups = sorted({str(sample.get("source_group") or sample.get("id")) for sample in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation = set(groups[:max(1, round(len(groups) * validation_fraction))]) if groups else set()
    return ([s for s in samples if str(s.get("source_group") or s.get("id")) not in validation],
            [s for s in samples if str(s.get("source_group") or s.get("id")) in validation])
