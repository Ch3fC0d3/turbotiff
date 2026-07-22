"""Offline training command for the TurboTIFF Phase 1 curve model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dataset import SyntheticCurveDataset, load_manifest
from .losses import CurveDetectionLoss, LossWeights
from .model import CurvePhase1UNet, require_torch, torch


def set_deterministic_seed(seed: int) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _split_records(records: list[dict], validation_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if len(records) < 2:
        raise ValueError("Training requires at least two generated samples")
    explicit_train=[record for record in records if record.get("split")=="train"]
    explicit_validation=[record for record in records if record.get("split")=="validation"]
    if explicit_train and explicit_validation:
        return explicit_train,explicit_validation
    order = np.random.default_rng(seed).permutation(len(records))
    validation_count = max(1, min(len(records) - 1, int(round(len(records) * validation_fraction))))
    validation_indices = set(order[:validation_count].tolist())
    train_records = [record for index, record in enumerate(records) if index not in validation_indices]
    validation_records = [record for index, record in enumerate(records) if index in validation_indices]
    return train_records, validation_records


def _batch_metrics(outputs: dict, batch: dict) -> dict:
    stroke_probability = torch.sigmoid(outputs["stroke_logits"])
    center_probability = torch.sigmoid(outputs["centerline_logits"])
    stroke_target = batch["stroke_mask"]
    center_target = batch["centerline_mask"]
    stroke_prediction = stroke_probability >= 0.5
    intersection = (stroke_prediction * (stroke_target > 0.5)).sum().item()
    union = ((stroke_prediction | (stroke_target > 0.5))).sum().item()
    stroke_iou = intersection / max(1.0, union)
    predicted_x = center_probability[:, 0].argmax(dim=2).float()
    target_rows = center_target[:, 0].sum(dim=2) > 0
    target_x = center_target[:, 0].argmax(dim=2).float()
    if target_rows.any():
        center_mae = torch.abs(predicted_x[target_rows] - target_x[target_rows]).mean().item()
    else:
        center_mae = 0.0
    return {"stroke_iou": float(stroke_iou), "centerline_mae_px": float(center_mae)}


def _save_prediction_preview(output_path: Path, batch: dict, outputs: dict) -> None:
    image = batch["image"][0].detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    stroke_target = (batch["stroke_mask"][0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    center_target = (batch["centerline_mask"][0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    stroke_prediction = (torch.sigmoid(outputs["stroke_logits"])[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    center_prediction = (torch.sigmoid(outputs["centerline_logits"])[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    overlay = image.copy()
    predicted_x = np.argmax(center_prediction, axis=1)
    points = np.column_stack((predicted_x, np.arange(predicted_x.size))).astype(np.int32)
    cv2.polylines(overlay, [points.reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)
    panels = [
        image,
        cv2.cvtColor(stroke_target, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(stroke_prediction, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(center_target, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(center_prediction, cv2.COLOR_GRAY2BGR),
        overlay,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(panels, axis=1))


def _checkpoint_payload(model, optimizer, scaler, epoch: int, best_loss: float, training_config: dict) -> dict:
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_validation_loss": float(best_loss),
        "model_version": model.model_version,
        "model_config": model.configuration(),
        "training_config": training_config,
    }


def train_phase1(
    data_dir: Path | str,
    output_dir: Path | str,
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 1234,
    target_size: tuple[int, int] = (512, 256),
    base_channels: int = 16,
    device: Optional[str] = None,
    resume: Optional[Path | str] = None,
    max_batches_per_epoch: Optional[int] = None,
) -> dict:
    require_torch()
    set_deterministic_seed(int(seed))
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(selected_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    records = load_manifest(data_dir)
    train_records, validation_records = _split_records(records, validation_fraction, seed)
    train_dataset = SyntheticCurveDataset(data_dir, train_records, target_size)
    validation_dataset = SyntheticCurveDataset(data_dir, validation_records, target_size)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CurvePhase1UNet(base_channels=base_channels).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    amp_enabled = str(selected_device).startswith("cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover - compatibility with older supported torch releases
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = CurveDetectionLoss(LossWeights())
    start_epoch = 0
    best_loss = float("inf")

    training_config = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "target_size": list(map(int, target_size)),
        "base_channels": int(base_channels),
        "device": str(selected_device),
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
    }
    (output_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    if resume:
        resume_path = Path(resume)
        checkpoint = torch.load(str(resume_path), map_location=selected_device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("best_validation_loss", best_loss))

    history = []
    log_path = output_dir / "losses.jsonl"
    for epoch in range(start_epoch, int(epochs)):
        started = time.perf_counter()
        model.train()
        train_totals = []
        for batch_index, batch in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_index >= max_batches_per_epoch:
                break
            image = batch["image"].to(selected_device)
            targets = {
                "stroke_mask": batch["stroke_mask"].to(selected_device),
                "centerline_mask": batch["centerline_mask"].to(selected_device),
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(image)
                losses = criterion(outputs, targets)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            train_totals.append(float(losses["total"].detach().cpu()))

        model.eval()
        validation_totals = []
        validation_metrics = []
        preview_batch = None
        preview_outputs = None
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if max_batches_per_epoch is not None and batch_index >= max_batches_per_epoch:
                    break
                image = batch["image"].to(selected_device)
                targets = {
                    "stroke_mask": batch["stroke_mask"].to(selected_device),
                    "centerline_mask": batch["centerline_mask"].to(selected_device),
                }
                outputs = model(image)
                losses = criterion(outputs, targets)
                validation_totals.append(float(losses["total"].detach().cpu()))
                validation_metrics.append(_batch_metrics(outputs, targets))
                if preview_batch is None:
                    preview_batch = {key: value.detach().cpu() if hasattr(value, "detach") else value for key, value in batch.items()}
                    preview_outputs = {key: value.detach().cpu() for key, value in outputs.items()}

        train_loss = float(np.mean(train_totals)) if train_totals else float("nan")
        validation_loss = float(np.mean(validation_totals)) if validation_totals else float("nan")
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "stroke_iou": float(np.mean([item["stroke_iou"] for item in validation_metrics])) if validation_metrics else 0.0,
            "centerline_mae_px": float(np.mean([item["centerline_mae_px"] for item in validation_metrics])) if validation_metrics else 0.0,
            "duration_seconds": time.perf_counter() - started,
        }
        history.append(metrics)
        with log_path.open("a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(json.dumps(metrics, sort_keys=True) + "\n")

        payload = _checkpoint_payload(model, optimizer, scaler, epoch, min(best_loss, validation_loss), training_config)
        torch.save(payload, output_dir / "last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            payload["best_validation_loss"] = best_loss
            torch.save(payload, output_dir / "best.pt")
        if preview_batch is not None and preview_outputs is not None:
            _save_prediction_preview(output_dir / "samples" / f"epoch_{epoch:03d}.png", preview_batch, preview_outputs)

    summary = {
        "best_validation_loss": float(best_loss),
        "epochs_completed": len(history),
        "last_metrics": history[-1] if history else None,
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TurboTIFF Phase 1 curve detector")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-height", type=int, default=512)
    parser.add_argument("--target-width", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    result = train_phase1(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        target_size=(args.target_height, args.target_width),
        base_channels=args.base_channels,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
