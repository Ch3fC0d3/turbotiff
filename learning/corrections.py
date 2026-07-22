"""Durable, privacy-aware curve-correction records and content-addressed images."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


STATUSES = {"pending", "approved", "rejected", "needs_more_review"}
DATA_USES = {"training_allowed", "evaluation_only", "do_not_retain", "client_restricted", "internal_only"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class CorrectionStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        for name in ("pending", "reviewed", "rejected", "exported", "assets"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _location(self, record_id: str) -> Path | None:
        for folder in ("pending", "reviewed", "rejected", "exported"):
            path = self.root / folder / f"{record_id}.json"
            if path.exists():
                return path
        return None

    def capture(self, image_bytes: bytes, prediction: dict, correction: dict, **context) -> dict:
        data_use = str(context.pop("data_use", "do_not_retain"))
        if data_use not in DATA_USES:
            raise ValueError(f"Invalid data-use status: {data_use}")
        checksum = hashlib.sha256(image_bytes).hexdigest()
        asset = None
        context_asset = None
        if data_use != "do_not_retain":
            suffix = str(context.pop("image_suffix", ".png"))
            asset = self.root / "assets" / f"{checksum}{suffix}"
            if not asset.exists():
                asset.write_bytes(image_bytes)
            context_bytes = context.pop("context_image_bytes", None)
            if context_bytes:
                context_checksum = hashlib.sha256(context_bytes).hexdigest()
                context_asset = self.root / "assets" / f"{context_checksum}{suffix}"
                if not context_asset.exists():
                    context_asset.write_bytes(context_bytes)
        record_id = str(context.pop("record_id", uuid.uuid4()))
        record = {
            "schema_version": 4,
            "record_id": record_id,
            "source_image_id": str(context.pop("source_image_id", checksum)),
            "track_image_path": str(asset.relative_to(self.root)) if asset else None,
            "context_image_path": str(context_asset.relative_to(self.root)) if context_asset else None,
            "image_checksum": checksum,
            "track_bounds": list(context.pop("track_bounds", [])),
            "track_dimensions": list(context.pop("track_dimensions", [])),
            "curve_name": context.pop("curve_name", None),
            "curve_color": context.pop("curve_color", None),
            "scale_type": str(context.pop("scale_type", "linear")),
            "topology": str(context.pop("topology", "bounded")),
            "preprocessing_config": context.pop("preprocessing_config", {}),
            "rotation_degrees": float(context.pop("rotation_degrees", 0.0)),
            "deskew_config": context.pop("deskew_config", {}),
            "scale_config": context.pop("scale_config", {}),
            "curve_selection_config": context.pop("curve_selection_config", {}),
            "source_file_id": context.pop("source_file_id", None),
            "page_number": context.pop("page_number", None),
            "depth_range": context.pop("depth_range", None),
            "prediction": prediction,
            "correction": correction,
            "edit_history": list(context.pop("edit_history", [])),
            "review_status": "pending",
            "data_use": data_use,
            "created_at": _now(),
            "reviewed_at": None,
            "metadata": context.pop("metadata", {}),
            **context,
        }
        self._validate(record)
        _atomic_json(self.root / "pending" / f"{record_id}.json", record)
        return record

    @staticmethod
    def _validate(record: dict) -> None:
        if record.get("review_status") not in STATUSES:
            raise ValueError("Invalid review status")
        if record.get("data_use") not in DATA_USES:
            raise ValueError("Invalid data-use status")
        for section in ("prediction", "correction"):
            if not isinstance(record.get(section), dict) or "x_by_row" not in record[section]:
                raise ValueError(f"{section}.x_by_row is required")
        predicted = np.asarray(record["prediction"]["x_by_row"]).reshape(-1)
        corrected = np.asarray(record["correction"]["x_by_row"]).reshape(-1)
        if predicted.size == 0 or predicted.size != corrected.size:
            raise ValueError("Prediction and correction must contain equal nonzero row counts")
        for key in ("unwrapped_x_by_row", "wrap_index_by_row", "confidence_by_row", "model_version", "decoder_version"):
            if key not in record["prediction"]:
                raise ValueError(f"prediction.{key} is required")
        for key in ("unwrapped_x_by_row", "wrap_index_by_row", "valid_row_mask", "wrap_events"):
            if key not in record["correction"]:
                raise ValueError(f"correction.{key} is required")

    def load(self, record_id: str) -> dict:
        path = self._location(record_id)
        if path is None:
            raise KeyError(record_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def append_edit(self, record_id: str, operation: str, payload: dict | None = None) -> dict:
        path = self._location(record_id)
        if path is None:
            raise KeyError(record_id)
        record = self.load(record_id)
        record["edit_history"].append({"operation": str(operation), "payload": payload or {}, "at": _now()})
        _atomic_json(path, record)
        return record

    def review(self, record_id: str, status: str, reviewer: str, notes: str | None = None) -> dict:
        if status not in STATUSES - {"pending"}:
            raise ValueError("Review must resolve to approved, rejected, or needs_more_review")
        source = self._location(record_id)
        if source is None:
            raise KeyError(record_id)
        record = self.load(record_id)
        record.update({"review_status": status, "reviewed_at": _now(), "reviewed_by": reviewer, "review_notes": notes})
        folder = "reviewed" if status in {"approved", "needs_more_review"} else "rejected"
        target = self.root / folder / source.name
        _atomic_json(target, record)
        if source != target:
            source.unlink()
        return record

    def approved_for_training(self) -> list[dict]:
        records = []
        for path in sorted((self.root / "reviewed").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("review_status") == "approved" and record.get("data_use") in {"training_allowed", "internal_only"}:
                records.append(record)
        return records
