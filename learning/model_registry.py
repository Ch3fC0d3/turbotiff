"""Auditable candidate, production, promotion, and rollback registry."""

from __future__ import annotations
import json, os, shutil
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"candidate", "approved", "production", "rejected", "archived"}

def _now(): return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class ModelRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        for folder in ("candidates", "production", "archived"): (self.root / folder).mkdir(exist_ok=True)
        self.path = self.root / "registry.json"
        if not self.path.exists(): self.path.write_text('{"models": [], "production_model_id": null, "history": []}\n', encoding="utf-8")

    def _read(self): return json.loads(self.path.read_text(encoding="utf-8"))
    def _write(self, data):
        temp = self.path.with_suffix(".tmp"); temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"); os.replace(temp, self.path)

    def register_candidate(self, model_id: str, checkpoint: Path | str, architecture: str, dataset_ids: list[str], **metadata):
        data = self._read()
        if any(m["model_id"] == model_id for m in data["models"]): raise FileExistsError(model_id)
        source = Path(checkpoint); target = self.root / "candidates" / model_id / source.name
        target.parent.mkdir(); shutil.copy2(source, target)
        entry = {"model_id": model_id, "status": "candidate", "architecture": architecture, "phase": 4,
                 "checkpoint_path": str(target.relative_to(self.root)), "training_dataset_ids": dataset_ids,
                 "created_at": _now(), "approved_by": None, "promotion_notes": None, **metadata}
        data["models"].append(entry); self._write(data); return entry

    def get(self, model_id):
        for item in self._read()["models"]:
            if item["model_id"] == model_id: return item
        raise KeyError(model_id)

    def active(self):
        data = self._read(); return self.get(data["production_model_id"]) if data["production_model_id"] else None

    def promote(self, model_id: str, approved_by: str, reason: str, gates: dict):
        data = self._read(); candidate = next((m for m in data["models"] if m["model_id"] == model_id), None)
        if not candidate or candidate["status"] not in {"candidate", "approved"}: raise ValueError("Only a candidate may be promoted")
        required = ("evaluation_completed", "no_leakage", "thresholds_passed", "regression_report")
        if (not approved_by or not all(bool(gates.get(key)) for key in required)
                or not candidate.get("evaluation_completed") or not candidate.get("metrics")):
            raise PermissionError("Promotion gates, registered evaluation, or human approval are incomplete")
        previous = data.get("production_model_id")
        if previous:
            prior = next(m for m in data["models"] if m["model_id"] == previous); prior["status"] = "archived"
        candidate.update(status="production", approved_by=approved_by, promotion_notes=reason, promoted_at=_now())
        data["production_model_id"] = model_id
        data["history"].append({"action": "promote", "model_id": model_id, "previous": previous, "by": approved_by, "at": _now(), "reason": reason})
        self._write(data)
        (self.root / "production" / "current.json").write_text(json.dumps({"model_id":model_id,"checkpoint_path":candidate["checkpoint_path"]},indent=2),encoding="utf-8")
        return candidate

    def rollback(self, model_id: str, approved_by: str, reason: str):
        data = self._read(); target = next((m for m in data["models"] if m["model_id"] == model_id), None)
        if not target or target["status"] not in {"archived", "production"}: raise ValueError("Rollback target is not a preserved production model")
        current = data.get("production_model_id")
        if current and current != model_id: next(m for m in data["models"] if m["model_id"] == current)["status"] = "archived"
        target["status"] = "production"; data["production_model_id"] = model_id
        data["history"].append({"action": "rollback", "model_id": model_id, "previous": current, "by": approved_by, "at": _now(), "reason": reason})
        self._write(data)
        (self.root / "production" / "current.json").write_text(json.dumps({"model_id":model_id,"checkpoint_path":target["checkpoint_path"]},indent=2),encoding="utf-8")
        return target


def resolve_production_checkpoint(root: Path | str) -> Path | None:
    registry = ModelRegistry(root); active = registry.active()
    return registry.root / active["checkpoint_path"] if active else None
