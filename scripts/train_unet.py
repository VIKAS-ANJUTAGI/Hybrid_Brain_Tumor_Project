"""Run the approved five-epoch CPU U-Net baseline and write its artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from models.segmentation.inference import NORMALIZATION_CONFIG, predict_slice
from models.segmentation.metrics import segmentation_metrics
from models.segmentation.training import (
    TrainingConfig, build_datasets, load_checkpoint, reconstruct_model, train_segmentation,
)


METRIC_CONVENTION = (
    "Per-slice macro averaging; Dice/IoU are 1 for empty prediction plus empty target, "
    "0 when only one is empty; zero-denominator precision/recall are 1."
)


def _new_run_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate, suffix = root / f"unet_baseline_{stamp}", 1
    while candidate.exists():
        candidate = root / f"unet_baseline_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _slice_values(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[dict[str, float], int, int]:
    metrics = segmentation_metrics(logits, target, threshold)
    predicted, actual = torch.sigmoid(logits) >= threshold, target > 0.5
    false_positive = int((predicted & ~actual).sum().item())
    false_negative = int((~predicted & actual).sum().item())
    return metrics, false_positive, false_negative


def _save_visualization(path: Path, record: dict[str, Any], category: str, note: str | None) -> None:
    image, target, prediction = record["image"][0].numpy(), record["target"][0].numpy(), record["prediction"][0].numpy()
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("T1C MRI")
    axes[1].imshow(target, cmap="magma"); axes[1].set_title("Ground truth")
    axes[2].imshow(prediction, cmap="magma"); axes[2].set_title("Predicted mask")
    axes[3].imshow(image, cmap="gray"); axes[3].imshow(prediction, cmap="spring", alpha=0.45); axes[3].set_title("Prediction overlay")
    for axis in axes:
        axis.axis("off")
    title = f"{category}: {record['patient_id']} slice {record['slice_index']}"
    if note:
        title += f" ({note})"
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_best_checkpoint(checkpoint_path: Path, manifest_path: Path, dataset_root: Path, threshold: float, output_dir: Path) -> dict[str, Any]:
    """Reload the best model, evaluate every validation slice, and save five PNGs."""
    checkpoint = load_checkpoint(checkpoint_path)
    model = reconstruct_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, validation = build_datasets(manifest_path, dataset_root=dataset_root)
    totals = {name: 0.0 for name in ("dice", "iou", "precision", "recall")}
    positive_dice, positive_count, empty_count = 0.0, 0, 0
    selections: dict[str, dict[str, Any] | None] = {name: None for name in ("tumor_presence", "highest_tumor_dice", "empty_ground_truth", "largest_false_positive", "largest_false_negative")}
    fallbacks: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(validation)):
            image, target = validation[index]
            logits = model(image.unsqueeze(0))
            metrics, false_positive, false_negative = _slice_values(logits, target.unsqueeze(0), threshold)
            patient_id, slice_index = validation._index[index]  # Dataset index is constructed only from validation IDs.
            prediction = (torch.sigmoid(logits).squeeze(0) >= threshold).cpu()
            record = {"patient_id": patient_id, "slice_index": slice_index, "image": image.cpu(), "target": target.cpu(), "prediction": prediction, "dice": metrics["dice"], "false_positive": false_positive, "false_negative": false_negative}
            fallbacks.append(record)
            for name, value in metrics.items():
                totals[name] += value
            tumor_positive = bool((target > 0.5).any().item())
            if tumor_positive:
                positive_count += 1; positive_dice += metrics["dice"]
                if selections["tumor_presence"] is None:
                    selections["tumor_presence"] = record
                if selections["highest_tumor_dice"] is None or metrics["dice"] > selections["highest_tumor_dice"]["dice"]:
                    selections["highest_tumor_dice"] = record
            else:
                empty_count += 1
                if selections["empty_ground_truth"] is None:
                    selections["empty_ground_truth"] = record
            if false_positive and (selections["largest_false_positive"] is None or false_positive > selections["largest_false_positive"]["false_positive"]):
                selections["largest_false_positive"] = record
            if false_negative and (selections["largest_false_negative"] is None or false_negative > selections["largest_false_negative"]["false_negative"]):
                selections["largest_false_negative"] = record

    used: set[tuple[str, int]] = set()
    visualization_notes: dict[str, str | None] = {}
    for category, record in selections.items():
        if record is None or (record["patient_id"], record["slice_index"]) in used:
            record = next(item for item in fallbacks if (item["patient_id"], item["slice_index"]) not in used)
            visualization_notes[category] = "requested category unavailable; deterministic representative used"
        else:
            visualization_notes[category] = None
        used.add((record["patient_id"], record["slice_index"]))
        selections[category] = record
        _save_visualization(output_dir / f"validation_{category}.png", record, category.replace("_", " "), visualization_notes[category])

    count = len(validation)
    return {
        "checkpoint": str(checkpoint_path.resolve()), "validation_slice_count": count,
        "validation_empty_slice_count": empty_count, "validation_tumor_positive_slice_count": positive_count,
        "metric_convention": METRIC_CONVENTION,
        "metrics": {name: value / count for name, value in totals.items()},
        "tumor_positive_slice_dice": positive_dice / positive_count if positive_count else 0.0,
        "visualizations": [{"category": category, "file": f"visualizations/validation_{category}.png", "patient_id": record["patient_id"], "slice_index": record["slice_index"], "note": visualization_notes[category]} for category, record in selections.items()],
    }


def run(manifest_path: Path, dataset_root: Path) -> Path:
    config = TrainingConfig(epochs=5, batch_size=2, learning_rate=1e-3, patience=3, num_workers=0, threshold=0.5, seed=42, device="cpu")
    output_dir = _new_run_directory(settings.RESULTS_DIR / "segmentation_runs")
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir()
    (output_dir / "training_config.json").write_text(json.dumps({"training_config": asdict(config), "model_architecture": {"in_channels": 4, "out_channels": 1, "base_width": 16}, "modality_order": list(settings.MRI_MODALITIES), "manifest_path": str(manifest_path.resolve()), "dataset_root": str(dataset_root.resolve()), "device": "cpu"}, indent=2), encoding="utf-8")
    result = train_segmentation(manifest_path, output_dir / "best_checkpoint.pt", dataset_root=dataset_root, config=config)
    metrics = evaluate_best_checkpoint(output_dir / "best_checkpoint.pt", manifest_path, dataset_root, config.threshold, visualization_dir)

    # Compare an in-memory copy captured when the best checkpoint was saved with the reloaded model.
    _, validation = build_datasets(manifest_path, dataset_root=dataset_root)
    image, _ = validation[0]
    pre_reload = reconstruct_model(load_checkpoint(output_dir / "best_checkpoint.pt")); pre_reload.load_state_dict(result["best_model_state_dict"])
    reloaded = reconstruct_model(load_checkpoint(output_dir / "best_checkpoint.pt")); reloaded.load_state_dict(load_checkpoint(output_dir / "best_checkpoint.pt")["model_state_dict"])
    before_probability, before_mask = predict_slice(pre_reload, image, threshold=config.threshold)
    after_probability, after_mask = predict_slice(reloaded, image, threshold=config.threshold)
    metrics["reload_validation"] = {"input_shape": list(image.shape), "probability_shape": list(after_probability.shape), "binary_mask_shape": list(after_mask.shape), "probability_finite": bool(torch.isfinite(after_probability).all().item()), "probability_min": float(after_probability.min().item()), "probability_max": float(after_probability.max().item()), "binary_mask_dtype": str(after_mask.dtype), "max_absolute_probability_difference_pre_reload_vs_reloaded": float((before_probability - after_probability).abs().max().item()), "binary_masks_equal": bool(torch.equal(before_mask, after_mask))}
    checkpoint = load_checkpoint(output_dir / "best_checkpoint.pt")
    metrics["normalization_config"] = checkpoint["normalization_config"]
    metrics["model_architecture"] = checkpoint["model_architecture"]
    metrics["input_channels"] = checkpoint["input_channels"]
    metrics["output_channels"] = checkpoint["output_channels"]
    metrics["reload_validation"]["preprocessing_restored"] = checkpoint["normalization_config"] == NORMALIZATION_CONFIG
    metrics["metric_aggregation"] = "Per-slice macro average across the complete validation cohort; loss is sample-weighted by slice count."
    (output_dir / "segmentation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    config_path = output_dir / "training_config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_config["patient_splits"] = {"train": list(checkpoint["train_patient_ids"]), "validation": list(checkpoint["validation_patient_ids"])}
    saved_config["normalization_config"] = checkpoint["normalization_config"]
    saved_config["segmentation_threshold"] = config.threshold
    saved_config["cache_statistics"] = result["cache_metrics"]
    config_path.write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    print(json.dumps({"run_directory": str(output_dir), "epochs_completed": result["epochs_completed"], "best_validation_dice": result["best_validation_dice"]}, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results" / "phase2_full_dataset_seed42" / "manifest.json")
    parser.add_argument("--dataset-root", type=Path, default=Path(settings.DATASET_ROOT))
    arguments = parser.parse_args()
    run(arguments.manifest, arguments.dataset_root)


if __name__ == "__main__":
    main()
