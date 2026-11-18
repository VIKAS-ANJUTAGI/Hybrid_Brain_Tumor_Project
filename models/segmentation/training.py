"""CPU-first training and checkpoint utilities for the segmentation baseline."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .dataset import BraTSSegmentationDataset, ManifestSplitError, validate_patient_assignments
from .inference import NORMALIZATION_CONFIG
from .losses import BCEDiceLoss
from .metrics import segmentation_metrics
from .unet import UNet2D


@dataclass(frozen=True)
class TrainingConfig:
    """Conservative defaults suitable for a CPU-only baseline."""

    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-3
    patience: int = 5
    num_workers: int = 0
    threshold: float = 0.5
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ValueError("this baseline is CPU-only; device must be 'cpu'")
        if min(self.epochs, self.batch_size, self.patience) < 1 or self.num_workers < 0:
            raise ValueError("epochs, batch_size, and patience must be positive; num_workers cannot be negative")
        if self.learning_rate <= 0 or not 0 <= self.threshold <= 1:
            raise ValueError("learning_rate must be positive and threshold must be in [0, 1]")


def build_datasets(manifest_path: str | Path, *, dataset_root: str | Path | None = None, cache_size: int = 1) -> tuple[BraTSSegmentationDataset, BraTSSegmentationDataset]:
    """Construct disjoint train/validation datasets without loading test data."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assignments = validate_patient_assignments(manifest)
    for split in ("train", "validation"):
        if not assignments[split]:
            raise ManifestSplitError(f"manifest must contain non-empty {split!r} IDs")
    return (
        BraTSSegmentationDataset(manifest_path, "train", dataset_root=dataset_root, cache_size=cache_size),
        BraTSSegmentationDataset(manifest_path, "validation", dataset_root=dataset_root, cache_size=cache_size),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(model: UNet2D, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], criterion: BCEDiceLoss, *, optimizer: Adam | None, threshold: float) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metric_totals = {name: 0.0 for name in ("dice", "iou", "precision", "recall")}
    for images, masks in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        metrics = segmentation_metrics(logits.detach(), masks, threshold)
        total_loss += float(loss.detach().item()) * images.shape[0]
        for name in metric_totals:
            metric_totals[name] += metrics[name] * images.shape[0]
    count = len(loader.dataset)
    if count == 0:
        raise ValueError("cannot train or validate with an empty dataset")
    return {"loss": total_loss / count, **{name: value / count for name, value in metric_totals.items()}}


def _architecture_config(model: UNet2D) -> dict[str, int]:
    return {"in_channels": model.in_channels, "out_channels": model.out_channels, "base_width": model.base_width}


def reconstruct_model(checkpoint: Mapping[str, Any]) -> UNet2D:
    """Rebuild a U-Net from checkpoint metadata, defaulting legacy files safely."""
    architecture = checkpoint.get("model_architecture", {})
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint model_architecture must be a mapping")
    defaults = {"in_channels": 4, "out_channels": 1, "base_width": 16}
    return UNet2D(**{name: int(architecture.get(name, value)) for name, value in defaults.items()})


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a CPU checkpoint and validate its minimum model payload."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict")
    return checkpoint


def _checkpoint_payload(*, model: UNet2D, optimizer: Adam, scheduler: ReduceLROnPlateau, epoch: int, best_dice: float, config: TrainingConfig, manifest_path: str | Path, train_ids: tuple[str, ...], validation_ids: tuple[str, ...], history: list[dict[str, float | int]]) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch, "best_validation_dice": best_dice, "config": asdict(config), "model_architecture": _architecture_config(model),
        "input_channels": model.in_channels, "output_channels": model.out_channels, "manifest_path": str(Path(manifest_path).resolve()),
        "train_patient_ids": train_ids, "validation_patient_ids": validation_ids, "random_seed": config.seed,
        "metric_threshold": config.threshold, "normalization_config": NORMALIZATION_CONFIG,
        "metric_convention": "Dice/IoU: both empty=1, one empty=0; precision/recall: zero denominator=1", "history": history,
    }


def _validate_resume(checkpoint: Mapping[str, Any], *, config: TrainingConfig, train_ids: tuple[str, ...], validation_ids: tuple[str, ...]) -> None:
    missing = [name for name in ("optimizer_state_dict", "scheduler_state_dict", "epoch", "best_validation_dice") if name not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint cannot resume because it lacks {missing}")
    expected = {"train_patient_ids": train_ids, "validation_patient_ids": validation_ids, "metric_threshold": config.threshold, "normalization_config": NORMALIZATION_CONFIG}
    for name, value in expected.items():
        if name in checkpoint and checkpoint[name] != value:
            raise ValueError(f"resume checkpoint {name} does not match this training run")
    saved_config = checkpoint.get("config")
    if isinstance(saved_config, Mapping):
        for name in ("batch_size", "learning_rate", "patience", "seed"):
            if saved_config.get(name) != getattr(config, name):
                raise ValueError(f"resume checkpoint config {name} does not match this training run")


def train_segmentation(manifest_path: str | Path, checkpoint_path: str | Path, *, dataset_root: str | Path | None = None, config: TrainingConfig = TrainingConfig(), resume_checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Train, save best/last/history artifacts, and optionally resume exactly."""
    _seed_everything(config.seed)
    train_dataset, validation_dataset = build_datasets(manifest_path, dataset_root=dataset_root)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    model = UNet2D()
    criterion, optimizer = BCEDiceLoss(), Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, config.patience // 2))
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint, history_path = checkpoint.with_name("last_checkpoint.pt"), checkpoint.with_name("training_history.json")
    best_dice, stale_epochs, history, start_epoch = float("-inf"), 0, [], 1

    if resume_checkpoint is not None:
        resumed = load_checkpoint(resume_checkpoint)
        if _architecture_config(reconstruct_model(resumed)) != _architecture_config(model):
            raise ValueError("resume checkpoint architecture does not match the training architecture")
        _validate_resume(resumed, config=config, train_ids=train_dataset.patient_ids, validation_ids=validation_dataset.patient_ids)
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        best_dice, history, start_epoch = float(resumed["best_validation_dice"]), list(resumed.get("history", [])), int(resumed["epoch"]) + 1

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer=optimizer, threshold=config.threshold)
        with torch.no_grad():
            validation_metrics = _run_epoch(model, validation_loader, criterion, optimizer=None, threshold=config.threshold)
        scheduler.step(validation_metrics["dice"])
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}})
        if validation_metrics["dice"] > best_dice:
            best_dice, stale_epochs = validation_metrics["dice"], 0
            torch.save(_checkpoint_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_dice=best_dice, config=config, manifest_path=manifest_path, train_ids=train_dataset.patient_ids, validation_ids=validation_dataset.patient_ids, history=history), checkpoint)
        else:
            stale_epochs += 1
        torch.save(_checkpoint_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_dice=best_dice, config=config, manifest_path=manifest_path, train_ids=train_dataset.patient_ids, validation_ids=validation_dataset.patient_ids, history=history), last_checkpoint)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale_epochs >= config.patience:
            break
    return {"best_validation_dice": best_dice, "epochs_completed": len(history), "checkpoint_path": str(checkpoint), "last_checkpoint_path": str(last_checkpoint), "history_path": str(history_path), "history": history}
