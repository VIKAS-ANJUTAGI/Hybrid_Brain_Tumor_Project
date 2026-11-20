"""CPU-first training and checkpoint utilities for the segmentation baseline."""

from __future__ import annotations

import json
import random
import time
import os
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from fractions import Fraction
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler

from .dataset import BraTSSegmentationDataset, ManifestSplitError, validate_patient_assignments
from .inference import NORMALIZATION_CONFIG
from .losses import BCEDiceLoss
from .metrics import segmentation_metrics
from .unet import UNet2D


EMPTY_SLICE_FP_CONSTRAINT = Fraction(85, 384)
SELECTION_CRITERION = "maximize patient-macro tumor-containing-slice Dice subject to empty-slice false-positive rate <= 85/384"


class PatientBatchSampler(Sampler[list[int]]):
    """Shuffle patients and their slices while keeping each batch patient-local.

    Grouping changes short-range stochastic batch composition and ordering, but
    does not change the slices, loss, or patient-level partition.
    """
    def __init__(self, dataset: BraTSSegmentationDataset, batch_size: int, seed: int = 42) -> None:
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
        self.patient_indices: dict[str, list[int]] = {pid: [] for pid in dataset.patient_ids}
        for index, (patient_id, _) in enumerate(dataset._index):
            self.patient_indices[patient_id].append(index)

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int:
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self.patient_indices.values())
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        patients = list(self.patient_indices)
        rng.shuffle(patients)
        for patient in patients:
            indices = self.patient_indices[patient].copy()
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


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


def _per_slice_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:
    """Return metric vectors, preserving the documented empty-mask convention."""
    predicted, target = torch.sigmoid(logits) >= threshold, targets > 0.5
    dimensions = tuple(range(1, logits.ndim))
    intersection = (predicted & target).sum(dim=dimensions, dtype=torch.float32)
    predicted_size = predicted.sum(dim=dimensions, dtype=torch.float32)
    target_size = target.sum(dim=dimensions, dtype=torch.float32)
    false_positive = (predicted & ~target).sum(dim=dimensions, dtype=torch.float32)
    false_negative = (~predicted & target).sum(dim=dimensions, dtype=torch.float32)
    both_empty = (predicted_size == 0) & (target_size == 0)
    dice = torch.where(both_empty, torch.ones_like(intersection), 2 * intersection / (predicted_size + target_size).clamp_min(1))
    iou = torch.where(both_empty, torch.ones_like(intersection), intersection / (predicted_size + target_size - intersection).clamp_min(1))
    precision = torch.where(intersection + false_positive == 0, torch.ones_like(intersection), intersection / (intersection + false_positive))
    recall = torch.where(intersection + false_negative == 0, torch.ones_like(intersection), intersection / (intersection + false_negative))
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "target_positive": target_size > 0}


def _run_epoch(model: UNet2D, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], criterion: BCEDiceLoss, *, optimizer: Adam | None, threshold: float, progress=None, status=None, epoch: int = 0) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metric_totals = {name: 0.0 for name in ("dice", "iou", "precision", "recall")}
    positive_dice_total, positive_count, empty_count, empty_fp_count = 0.0, 0, 0, 0
    patient_sums: dict[str, float] = {}
    patient_counts: dict[str, int] = {}
    patient_empty_counts: dict[str, int] = {}
    sample_offset = 0
    sample_patient_ids = [] if training else [patient_id for patient_id, _ in getattr(loader.dataset, "_index", [])]
    started = time.perf_counter(); batch_seconds = 0.0
    batch_started = started
    samples_processed = 0
    total_batches = len(loader)
    for batch_number, (images, masks) in enumerate(loader, 1):
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        metrics = _per_slice_metrics(logits.detach(), masks, threshold)
        actual_batch_size = int(images.shape[0])
        total_loss += float(loss.detach().item()) * actual_batch_size
        samples_processed += actual_batch_size
        for name in metric_totals:
            metric_totals[name] += float(metrics[name].sum().item())
        positive = metrics["target_positive"]
        positive_count += int(positive.sum().item())
        empty_count += int((~positive).sum().item())
        positive_dice_total += float(metrics["dice"][positive].sum().item())
        predicted = torch.sigmoid(logits.detach()) >= threshold
        batch_patient_ids = sample_patient_ids[sample_offset:sample_offset + actual_batch_size]
        if batch_patient_ids:
            for item, patient_id in enumerate(batch_patient_ids):
                if bool(positive[item].item()):
                    patient_sums[patient_id] = patient_sums.get(patient_id, 0.0) + float(metrics["dice"][item].item())
                    patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
                else:
                    patient_empty_counts[patient_id] = patient_empty_counts.get(patient_id, 0) + 1
                    empty_fp_count += int(bool(predicted[item].any().item()))
        else:
            empty_fp_count += int(((~positive) & predicted.flatten(1).any(dim=1)).sum().item())
        sample_offset += actual_batch_size
        duration = time.perf_counter() - batch_started
        batch_seconds += duration
        if progress and training and (batch_number % 10 == 0 or batch_number == total_batches):
            avg = total_loss / samples_processed
            elapsed = time.perf_counter() - started
            percentage = 100.0 * batch_number / total_batches if total_batches else 100.0
            remaining = max(total_batches - batch_number, 0)
            line = f"Epoch {epoch} {batch_number}/{total_batches} ({percentage:.1f}%) loss={float(loss.detach()):.6f} avg_loss={avg:.6f} elapsed={elapsed:.1f}s avg_batch={batch_seconds/batch_number:.2f}s eta={((elapsed/batch_number)*remaining):.1f}s"
            print(line, flush=True); progress.write(line + "\n"); progress.flush()
        if status and training:
            status(batch_number, total_batches, float(loss.detach()), total_loss / samples_processed)
        batch_started = time.perf_counter()
    if len(loader.dataset) == 0:
        raise ValueError("cannot train or validate with an empty dataset")
    if samples_processed == 0:
        raise ValueError("data loader produced no samples")
    all_patient_ids = list(getattr(loader.dataset, "patient_ids", tuple(patient_sums)))
    patient_dice = {patient_id: (patient_sums[patient_id] / patient_counts[patient_id] if patient_counts.get(patient_id, 0) else None) for patient_id in all_patient_ids}
    tumor_patient_scores = [value for value in patient_dice.values() if value is not None]
    empty_fp_rate = empty_fp_count / empty_count if empty_count else 0.0
    return {
        "loss": total_loss / samples_processed,
        **{name: value / samples_processed for name, value in metric_totals.items()},
        "positive_slice_dice": positive_dice_total / positive_count if positive_count else 0.0,
        "tumor_positive_slice_count": positive_count,
        "empty_slice_count": empty_count,
        "patient_tumor_dice": patient_dice,
        "patient_tumor_slice_counts": {patient_id: patient_counts.get(patient_id, 0) for patient_id in all_patient_ids},
        "patient_empty_slice_counts": {patient_id: patient_empty_counts.get(patient_id, 0) for patient_id in all_patient_ids},
        "patients_without_tumor_slices": [patient_id for patient_id in all_patient_ids if not patient_counts.get(patient_id, 0)],
        "patient_macro_tumor_dice": float(sum(tumor_patient_scores) / len(tumor_patient_scores)) if tumor_patient_scores else 0.0,
        "empty_slice_false_positive_count": empty_fp_count,
        "empty_slice_false_positive_rate": empty_fp_rate,
        "constraint_value": float(EMPTY_SLICE_FP_CONSTRAINT),
        "constraint_satisfied": empty_fp_rate <= float(EMPTY_SLICE_FP_CONSTRAINT),
    }


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


def _checkpoint_payload(*, model: UNet2D, optimizer: Adam, scheduler: ReduceLROnPlateau, epoch: int, best_dice: float, config: TrainingConfig, manifest_path: str | Path, train_ids: tuple[str, ...], validation_ids: tuple[str, ...], history: list[dict[str, Any]], selection_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch, "best_validation_dice": best_dice, "config": asdict(config), "model_architecture": _architecture_config(model),
        "input_channels": model.in_channels, "output_channels": model.out_channels, "manifest_path": str(Path(manifest_path).resolve()),
        "train_patient_ids": train_ids, "validation_patient_ids": validation_ids, "random_seed": config.seed,
        "metric_threshold": config.threshold, "segmentation_threshold": config.threshold, "normalization_config": NORMALIZATION_CONFIG,
        "metric_convention": "Dice/IoU: both empty=1, one empty=0; precision/recall: zero denominator=1", "history": history,
        "selection_metadata": dict(selection_metadata or {}),
    }


def _select_checkpoint(history: list[Mapping[str, Any]]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Select the best epoch under the fixed empty-slice FP constraint."""
    eligible = [entry for entry in history if entry.get("validation_constraint_satisfied") is True]
    fallback = not eligible
    candidates = eligible or list(history)
    if not candidates:
        raise ValueError("cannot select a checkpoint without validation history")
    if fallback:
        chosen = min(candidates, key=lambda e: (float(e["validation_empty_slice_false_positive_rate"]), -float(e["validation_patient_macro_tumor_dice"]), float(e["validation_loss"])))
    else:
        chosen = min(candidates, key=lambda e: (-float(e["validation_patient_macro_tumor_dice"]), float(e["validation_loss"])))
    metadata = {
        "selection_criterion": SELECTION_CRITERION,
        "patient_macro_tumor_dice": float(chosen["validation_patient_macro_tumor_dice"]),
        "empty_slice_false_positive_rate": float(chosen["validation_empty_slice_false_positive_rate"]),
        "constraint_value": float(EMPTY_SLICE_FP_CONSTRAINT),
        "constraint_satisfied": bool(chosen["validation_constraint_satisfied"]),
        "fallback_used": fallback,
        "all_slice_dice": float(chosen["validation_dice"]),
        "validation_loss": float(chosen["validation_loss"]),
        "epoch": int(chosen["epoch"]),
        "patient_level_dice_values": chosen["validation_patient_tumor_dice"],
        "segmentation_threshold": float(chosen["segmentation_threshold"]),
    }
    return chosen, metadata


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
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint, history_path = checkpoint.with_name("last_checkpoint.pt"), checkpoint.with_name("training_history.json")
    status_path = checkpoint.with_name("training_status.json")
    run_started = datetime.now(timezone.utc).isoformat()
    status_data = {"run_id": str(uuid.uuid4()), "pid": os.getpid(), "process_id": os.getpid(), "start_time": run_started, "status": "INITIALIZING", "current_epoch": 0, "total_epochs": config.epochs, "current_batch": 0, "total_batches": 0, "latest_loss": None, "running_average_loss": None, "elapsed_time": 0.0, "last_updated_time": run_started, "last_update_time": run_started, "last_successful_checkpoint": None}
    def update_status(**updates):
        now = datetime.now(timezone.utc).isoformat()
        status_data.update(updates, last_updated_time=now, last_update_time=now, elapsed_time=time.monotonic() - mono_started)
        _atomic_json(status_path, status_data)
    mono_started = time.monotonic()
    update_status(status="INITIALIZING")
    log_stream = None
    try:
        _seed_everything(config.seed)
        train_dataset, validation_dataset = build_datasets(manifest_path, dataset_root=dataset_root)
        train_sampler = PatientBatchSampler(train_dataset, config.batch_size, config.seed)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config.num_workers)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        model = UNet2D()
        criterion, optimizer = BCEDiceLoss(), Adam(model.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, config.patience // 2))
        update_status(total_batches=len(train_loader))
        log_stream = checkpoint.with_name("training_progress.log").open("a", encoding="utf-8")
        best_dice, stale_epochs, history, start_epoch = float("-inf"), 0, [], 1
        best_selection_metadata: dict[str, Any] | None = None
        best_model_state: dict[str, torch.Tensor] | None = None
        train_dataset.reset_counters()
        validation_dataset.reset_counters()

        if resume_checkpoint is not None:
            resumed = load_checkpoint(resume_checkpoint)
            if _architecture_config(reconstruct_model(resumed)) != _architecture_config(model):
                raise ValueError("resume checkpoint architecture does not match the training architecture")
            _validate_resume(resumed, config=config, train_ids=train_dataset.patient_ids, validation_ids=validation_dataset.patient_ids)
            model.load_state_dict(resumed["model_state_dict"])
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
            best_dice, history, start_epoch = float(resumed["best_validation_dice"]), list(resumed.get("history", [])), int(resumed["epoch"]) + 1
            saved_selection = resumed.get("selection_metadata")
            if isinstance(saved_selection, Mapping) and saved_selection.get("epoch") is not None:
                best_selection_metadata = dict(saved_selection)

        for epoch in range(start_epoch, config.epochs + 1):
            epoch_started = time.perf_counter()
            train_sampler.set_epoch(epoch - 1)
            update_status(status="RUNNING", current_epoch=epoch, current_batch=0, total_batches=len(train_loader))
            def batch_status(batch, total, loss, average):
                update_status(current_batch=batch, total_batches=total, latest_loss=loss, running_average_loss=average)
            train_metrics = _run_epoch(model, train_loader, criterion, optimizer=optimizer, threshold=config.threshold, progress=log_stream, status=batch_status, epoch=epoch)
            update_status(status="VALIDATING", current_batch=0, total_batches=len(validation_loader))
            with torch.no_grad():
                validation_metrics = _run_epoch(model, validation_loader, criterion, optimizer=None, threshold=config.threshold)
            scheduler.step(validation_metrics["patient_macro_tumor_dice"])
            epoch_record = {"epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]), "duration_seconds": time.perf_counter() - epoch_started, "segmentation_threshold": config.threshold, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}}
            history.append(epoch_record)
            selectable_history = [item for item in history if "validation_constraint_satisfied" in item]
            selected_record, new_selection_metadata = _select_checkpoint(selectable_history)
            is_new_best = int(selected_record["epoch"]) == epoch
            if is_new_best:
                best_selection_metadata = new_selection_metadata
                best_dice = float(validation_metrics["dice"])
                stale_epochs = 0
                best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                torch.save(_checkpoint_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_dice=best_dice, config=config, manifest_path=manifest_path, train_ids=train_dataset.patient_ids, validation_ids=validation_dataset.patient_ids, history=history, selection_metadata=best_selection_metadata), checkpoint)
            else:
                stale_epochs += 1
            torch.save(_checkpoint_payload(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_dice=best_dice, config=config, manifest_path=manifest_path, train_ids=train_dataset.patient_ids, validation_ids=validation_dataset.patient_ids, history=history, selection_metadata=best_selection_metadata), last_checkpoint)
            update_status(status="RUNNING", current_batch=0, total_batches=len(train_loader), last_successful_checkpoint=str(last_checkpoint))
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            if stale_epochs >= config.patience:
                break
        if best_model_state is None:
            # A resumed run may not improve on a pre-existing best checkpoint.
            best_model_state = {name: value.detach().cpu().clone() for name, value in load_checkpoint(checkpoint)["model_state_dict"].items()}
        cache_metrics = {}
        for name, dataset in (("train", train_dataset), ("validation", validation_dataset)):
            cache_metrics[name] = {"cache_hits": dataset.cache_hits, "cache_misses": dataset.cache_misses, "nifti_loads": dataset.volume_load_count, "patient_switches": dataset.patient_switches, "data_loading_seconds": dataset.data_loading_seconds}
        cache_line = "Cache metrics " + json.dumps(cache_metrics, sort_keys=True)
        print(cache_line, flush=True)
        log_stream.write(cache_line + "\n")
        log_stream.flush()
        update_status(status="COMPLETED", cache_metrics=cache_metrics)
        return {"best_validation_dice": best_dice, "selection_metadata": best_selection_metadata, "epochs_completed": len(history), "checkpoint_path": str(checkpoint), "last_checkpoint_path": str(last_checkpoint), "history_path": str(history_path), "best_model_state_dict": best_model_state, "cache_metrics": cache_metrics}
    except Exception as exc:
        try:
            update_status(status="FAILED", error_type=type(exc).__name__, error_message=str(exc))
        except Exception:
            # Preserve the original training error if the status write also fails.
            pass
        raise
    finally:
        if log_stream is not None:
            log_stream.close()
