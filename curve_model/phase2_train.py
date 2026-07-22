"""Staged training command for the TurboTIFF Phase 2 multitask model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dataset import Phase2RealCurveDataset, SyntheticCurveDataset, load_manifest
from .phase2_infer import validate_phase2_checkpoint
from .phase2_losses import CapeConfig, CurvePhase2Loss, Phase2LossWeights
from .phase2_model import CurvePhase2UNet, transfer_phase1_weights
from .train import _split_records, set_deterministic_seed
from .model import require_torch, torch


TARGET_KEYS = (
    "stroke_mask", "centerline_mask", "distance_field", "direction_field",
    "grid_mask", "valid_direction_mask", "stroke_label_valid", "grid_label_valid",
)


def _device_batch(batch: dict, device: str) -> dict:
    return {key: batch[key].to(device) for key in TARGET_KEYS}


def _save_preview(path: Path, batch: dict, outputs: dict) -> None:
    image = batch["image"][0].detach().cpu().numpy().transpose(1, 2, 0)
    image = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    scalar_panels = [
        batch["centerline_mask"][0, 0].detach().cpu().numpy(),
        torch.sigmoid(outputs["centerline_logits"])[0, 0].detach().cpu().numpy(),
        outputs["distance_field"][0, 0].detach().cpu().numpy(),
        batch["grid_mask"][0, 0].detach().cpu().numpy(),
        torch.sigmoid(outputs["grid_logits"])[0, 0].detach().cpu().numpy(),
    ]
    panels = [image] + [cv2.cvtColor(np.clip(panel * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for panel in scalar_panels]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def _checkpoint_payload(model, optimizer, scaler, epoch, best_loss, training_config, criterion, transfer_report):
    configuration = model.configuration()
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_validation_loss": float(best_loss),
        "model_version": model.model_version,
        "model_format_version": 2,
        "phase": 2,
        "architecture": "small_unet_multitask",
        "outputs": list(model.outputs),
        "model_config": configuration,
        "training_resolution": training_config["target_size"],
        "training_config": training_config,
        "loss_configuration": criterion.configuration(),
        "dataset_version": "synthetic-v2",
        "transfer_report": transfer_report,
    }


def train_phase2(
    data_dir: Path | str,
    output_dir: Path | str,
    epochs: int = 30,
    batch_size: int = 4,
    shared_learning_rate: float = 1e-4,
    head_learning_rate: float = 5e-4,
    validation_fraction: float = 0.2,
    seed: int = 1234,
    target_size: tuple[int, int] = (512, 256),
    base_channels: int = 16,
    device: Optional[str] = None,
    phase1_checkpoint: Optional[Path | str] = None,
    resume: Optional[Path | str] = None,
    real_data_dir: Optional[Path | str] = None,
    synthetic_ratio: float = 0.75,
    real_ratio: float = 0.25,
    hard_example_multiplier: float = 2.0,
    cape_config: Optional[CapeConfig] = None,
    loss_weights: Optional[Phase2LossWeights] = None,
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
    synthetic_train = SyntheticCurveDataset(data_dir, train_records, target_size)
    validation_dataset = SyntheticCurveDataset(data_dir, validation_records, target_size)
    datasets = [synthetic_train]
    sample_weights = []
    synthetic_total_weight = max(0.0, float(synthetic_ratio))
    for record in train_records:
        hard_multiplier = float(hard_example_multiplier) if record.get("hard_case") else 1.0
        sample_weights.append(synthetic_total_weight * hard_multiplier / max(1, len(train_records)))

    real_count = 0
    if real_data_dir:
        real_records = load_manifest(real_data_dir)
        real_dataset = Phase2RealCurveDataset(real_data_dir, real_records, target_size)
        datasets.append(real_dataset)
        real_count = len(real_dataset)
        sample_weights.extend([max(0.0, float(real_ratio)) / max(1, real_count)] * real_count)
    train_dataset = torch.utils.data.ConcatDataset(datasets) if len(datasets) > 1 else synthetic_train
    generator = torch.Generator().manual_seed(int(seed))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CurvePhase2UNet(base_channels=base_channels).to(selected_device)
    transfer_report = None
    if phase1_checkpoint and not resume:
        checkpoint = torch.load(str(Path(phase1_checkpoint)), map_location=selected_device, weights_only=True)
        transfer_report = transfer_phase1_weights(model, checkpoint)
    new_head_names = ("distance_head", "direction_head", "grid_head")
    shared_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith(new_head_names)]
    new_head_parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(new_head_names)]
    optimizer = torch.optim.AdamW([
        {"params": shared_parameters, "lr": float(shared_learning_rate)},
        {"params": new_head_parameters, "lr": float(head_learning_rate)},
    ])
    amp_enabled = str(selected_device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = CurvePhase2Loss(loss_weights, cape_config)
    start_epoch = 0
    best_loss = float("inf")

    training_config = {
        "epochs": int(epochs), "batch_size": int(batch_size),
        "shared_learning_rate": float(shared_learning_rate), "head_learning_rate": float(head_learning_rate),
        "validation_fraction": float(validation_fraction), "seed": int(seed),
        "target_size": list(map(int, target_size)), "base_channels": int(base_channels),
        "device": str(selected_device), "synthetic_train_samples": len(train_records),
        "validation_samples": len(validation_records), "real_samples": real_count,
        "data_mix": {"synthetic": float(synthetic_ratio), "real": float(real_ratio)},
        "hard_example_multiplier": float(hard_example_multiplier),
        "phase1_checkpoint": str(phase1_checkpoint) if phase1_checkpoint else None,
    }
    (output_dir / "training_config.json").write_text(json.dumps({**training_config, "loss": criterion.configuration()}, indent=2), encoding="utf-8")

    if resume:
        checkpoint = torch.load(str(Path(resume)), map_location=selected_device, weights_only=True)
        validate_phase2_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("best_validation_loss", best_loss))
        transfer_report = checkpoint.get("transfer_report")

    history = []
    log_path = output_dir / "losses.jsonl"
    for epoch in range(start_epoch, int(epochs)):
        started = time.perf_counter()
        model.train()
        train_parts: dict[str, list[float]] = {}
        for batch_index, batch in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_index >= max_batches_per_epoch:
                break
            image = batch["image"].to(selected_device)
            targets = _device_batch(batch, selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(image)
                losses = criterion(outputs, targets, epoch=epoch)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            for key, value in losses.items():
                if key != "cape_active":
                    train_parts.setdefault(key, []).append(float(value.detach().cpu()))

        model.eval()
        validation_parts: dict[str, list[float]] = {}
        preview = None
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if max_batches_per_epoch is not None and batch_index >= max_batches_per_epoch:
                    break
                image = batch["image"].to(selected_device)
                outputs = model(image)
                losses = criterion(outputs, _device_batch(batch, selected_device), epoch=epoch)
                for key, value in losses.items():
                    if key != "cape_active":
                        validation_parts.setdefault(key, []).append(float(value.detach().cpu()))
                if preview is None:
                    preview = (batch, {key: value.detach().cpu() for key, value in outputs.items()})
        metrics = {
            "epoch": epoch,
            "cape_active": criterion.cape_active(epoch),
            "duration_seconds": time.perf_counter() - started,
            **{f"train_{key}": float(np.mean(values)) for key, values in train_parts.items()},
            **{f"validation_{key}": float(np.mean(values)) for key, values in validation_parts.items()},
        }
        validation_loss = metrics.get("validation_total", float("nan"))
        history.append(metrics)
        with log_path.open("a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(json.dumps(metrics, sort_keys=True) + "\n")
        payload = _checkpoint_payload(model, optimizer, scaler, epoch, min(best_loss, validation_loss), training_config, criterion, transfer_report)
        torch.save(payload, output_dir / "last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            payload["best_validation_loss"] = best_loss
            torch.save(payload, output_dir / "best.pt")
        if preview:
            _save_preview(output_dir / "samples" / f"epoch_{epoch:03d}.png", *preview)

    summary = {
        "best_validation_loss": float(best_loss), "epochs_completed": len(history),
        "last_metrics": history[-1] if history else None,
        "best_checkpoint": str(output_dir / "best.pt"), "last_checkpoint": str(output_dir / "last.pt"),
        "transfer_report": transfer_report,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TurboTIFF Phase 2 geometry-aware detector")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase1-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--real-data-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shared-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--target-height", type=int, default=512)
    parser.add_argument("--target-width", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device")
    parser.add_argument("--cape", action="store_true")
    parser.add_argument("--cape-start-epoch", type=int, default=10)
    args = parser.parse_args()
    result = train_phase2(
        args.data_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        shared_learning_rate=args.shared_learning_rate, head_learning_rate=args.head_learning_rate,
        seed=args.seed, target_size=(args.target_height, args.target_width), base_channels=args.base_channels,
        device=args.device, phase1_checkpoint=args.phase1_checkpoint, resume=args.resume,
        real_data_dir=args.real_data_dir, cape_config=CapeConfig(enabled=args.cape, start_epoch=args.cape_start_epoch),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

