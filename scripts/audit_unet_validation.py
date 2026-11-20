"""Validation-only quality audit for a saved U-Net segmentation checkpoint.

This script reloads an existing checkpoint and evaluates only the validation
patients named by the run configuration. It never trains or saves model state.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.segmentation.dataset import BraTSSegmentationDataset
from models.segmentation.training import load_checkpoint, reconstruct_model

RUN = ROOT / "results" / "segmentation_runs" / "unet_baseline_20261120T173712Z"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
BASELINE = 0.50
EMPTY_CONVENTION = (
    "Per-slice Dice/IoU equal 1 when target and prediction are both empty and 0 when exactly one is empty; "
    "per-slice precision/recall equal 1 when their denominator is zero. Global precision/recall are undefined "
    "only if the corresponding global denominator is zero (not the case here)."
)


def div(a: float, b: float, zero: float = 0.0) -> float:
    return float(a / b) if b else float(zero)


def slice_metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = p >= threshold
    gt = y > 0.5
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    ps, gs = tp + fp, tp + fn
    union = tp + fp + fn
    return {
        "dice": 1.0 if ps + gs == 0 else div(2 * tp, ps + gs),
        "iou": 1.0 if union == 0 else div(tp, union),
        "precision": div(tp, ps, 1.0), "recall": div(tp, gs, 1.0),
        "tp": tp, "fp": fp, "fn": fn, "predicted_pixels": ps, "ground_truth_pixels": gs,
        "target_positive": bool(gs), "prediction_positive": bool(ps),
    }


def summarize(rows: list[dict[str, Any]], threshold: float, *, patient_id: str | None = None) -> dict[str, Any]:
    chosen = [r for r in rows if patient_id is None or r["patient_id"] == patient_id]
    metrics = [slice_metrics(r["probability"], r["target"], threshold) for r in chosen]
    positive = [m for m in metrics if m["target_positive"]]
    tp, fp, fn = (sum(m[k] for m in metrics) for k in ("tp", "fp", "fn"))
    return {
        "slice_count": len(chosen),
        "empty_ground_truth_slices": sum(not m["target_positive"] for m in metrics),
        "tumor_containing_slices": len(positive),
        "all_slice_macro_dice": div(sum(m["dice"] for m in metrics), len(metrics)),
        "tumor_containing_slice_macro_dice": div(sum(m["dice"] for m in positive), len(positive)),
        "macro_iou": div(sum(m["iou"] for m in metrics), len(metrics)),
        "macro_slice_precision": div(sum(m["precision"] for m in metrics), len(metrics)),
        "macro_slice_recall": div(sum(m["recall"] for m in metrics), len(metrics)),
        "global_voxel_dice": div(2 * tp, 2 * tp + fp + fn),
        "global_voxel_iou": div(tp, tp + fp + fn),
        "global_pixel_precision": div(tp, tp + fp),
        "global_pixel_recall": div(tp, tp + fn),
        "true_positive_pixels": tp, "false_positive_pixels": fp, "false_negative_pixels": fn,
        "empty_gt_slices_with_empty_prediction": sum((not m["target_positive"]) and (not m["prediction_positive"]) for m in metrics),
        "empty_gt_slices_with_false_positive": sum((not m["target_positive"]) and m["prediction_positive"] for m in metrics),
        "tumor_slices_with_empty_prediction": sum(m["target_positive"] and not m["prediction_positive"] for m in metrics),
    }


def save_case(path: Path, row: dict[str, Any], title: str) -> None:
    image = row["image"]
    gt = row["target"]
    pred = row["probability"] >= BASELINE
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("T1C MRI")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Prediction (0.5)")
    axes[3].imshow(image, cmap="gray"); axes[3].imshow(gt, cmap="Greens", alpha=.45); axes[3].imshow(pred, cmap="Reds", alpha=.4); axes[3].set_title("Overlay: GT green, pred red")
    for ax in axes: ax.axis("off")
    sm = slice_metrics(row["probability"], row["target"], BASELINE)
    fig.suptitle(f"{title} | {row['patient_id']} slice {row['slice_index']} | Dice {sm['dice']:.3f}, FP {sm['fp']}, FN {sm['fn']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> Path:
    config = json.loads((RUN / "training_config.json").read_text(encoding="utf-8"))
    manifest = Path(config["manifest_path"])
    dataset_root = Path(config["dataset_root"])
    validation_ids = config["patient_splits"]["validation"]
    out_root = ROOT / "results"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / f"segmentation_validation_audit_{stamp}"
    suffix = 1
    while out.exists():
        out = out_root / f"segmentation_validation_audit_{stamp}_{suffix}"; suffix += 1
    out.mkdir(parents=True)
    viz_dir = out / "visualizations"; viz_dir.mkdir()

    checkpoint_path = RUN / "best_checkpoint.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    model = reconstruct_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if tuple(checkpoint.get("validation_patient_ids", ())) != tuple(validation_ids):
        raise RuntimeError("Run configuration and checkpoint validation patient IDs differ")

    # Construct only the validation dataset. Dataset split validation checks the
    # manifest assignments but does not load test MRI volumes or masks.
    dataset = BraTSSegmentationDataset(manifest, "validation", dataset_root=dataset_root, patient_ids=validation_ids)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for i in range(len(dataset)):
            image, target = dataset[i]
            probability = torch.sigmoid(model(image.unsqueeze(0))).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            patient_id, slice_index = dataset._index[i]
            rows.append({"patient_id": patient_id, "slice_index": int(slice_index), "image": image[0].numpy().copy(), "target": target.squeeze(0).numpy().copy(), "probability": probability})
    if len(rows) != 620 or len({r["patient_id"] for r in rows}) != 4:
        raise RuntimeError(f"Expected 620 slices from four validation patients; got {len(rows)} from {len(set(r['patient_id'] for r in rows))}")

    baseline = summarize(rows, BASELINE)
    per_slice = []
    for row in rows:
        m = slice_metrics(row["probability"], row["target"], BASELINE)
        per_slice.append({"patient_id": row["patient_id"], "slice_index": row["slice_index"], **{k: m[k] for k in ("dice", "iou", "precision", "recall", "tp", "fp", "fn", "predicted_pixels", "ground_truth_pixels", "target_positive", "prediction_positive")}})

    # Patient metrics use the same slice macro convention as the original run.
    patients = []
    for pid in validation_ids:
        s = summarize(rows, BASELINE, patient_id=pid)
        patients.append({"patient_id": pid, "slice_count": s["slice_count"], "empty_ground_truth_slices": s["empty_ground_truth_slices"], "tumor_containing_slices": s["tumor_containing_slices"], "dice": s["all_slice_macro_dice"], "tumor_containing_slice_dice": s["tumor_containing_slice_macro_dice"], "iou": s["macro_iou"], "pixel_precision": s["macro_slice_precision"], "pixel_recall": s["macro_slice_recall"], "global_voxel_dice": s["global_voxel_dice"], "global_voxel_iou": s["global_voxel_iou"], "global_pixel_precision": s["global_pixel_precision"], "global_pixel_recall": s["global_pixel_recall"], "false_positive_pixels": s["false_positive_pixels"], "false_negative_pixels": s["false_negative_pixels"]})
    with (out / "patient_level_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(patients[0])); writer.writeheader(); writer.writerows(patients)

    thresholds = []
    for t in THRESHOLDS:
        s = summarize(rows, t)
        thresholds.append({"threshold": t, "all_slice_dice": s["all_slice_macro_dice"], "tumor_containing_slice_dice": s["tumor_containing_slice_macro_dice"], "iou": s["macro_iou"], "precision": s["macro_slice_precision"], "recall": s["macro_slice_recall"], "global_voxel_dice": s["global_voxel_dice"], "global_voxel_iou": s["global_voxel_iou"], "global_pixel_precision": s["global_pixel_precision"], "global_pixel_recall": s["global_pixel_recall"], "false_positive_pixels": s["false_positive_pixels"], "false_negative_pixels": s["false_negative_pixels"]})
    with (out / "threshold_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(thresholds[0])); writer.writeheader(); writer.writerows(thresholds)

    empty_rows = [r for r in per_slice if not r["target_positive"]]
    positive_rows = [r for r in per_slice if r["target_positive"]]
    row_lookup = {(r["patient_id"], r["slice_index"]): r for r in rows}
    empty_fp = [r for r in empty_rows if r["fp"] > 0]
    empty_fp_regions = []
    for r in empty_fp:
        probability = row_lookup[(r["patient_id"], r["slice_index"])]["probability"]
        binary = (probability >= BASELINE).astype(np.uint8)
        region_count = int(ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))[1])
        empty_fp_regions.append({"patient_id": r["patient_id"], "slice_index": r["slice_index"], "false_positive_region_count_8_connected": region_count, "false_positive_pixels": r["fp"]})
    tumor_empty_pred = [r for r in positive_rows if not r["prediction_positive"]]
    low_dice = sorted((r for r in positive_rows if r["dice"] < .20), key=lambda x: (x["dice"], -x["fn"]))
    high_fp = sorted((r for r in positive_rows if r["fp"] > r["ground_truth_pixels"]), key=lambda x: (-x["fp"], x["dice"]))
    underseg = sorted((r for r in positive_rows if r["fn"] > r["tp"]), key=lambda x: (-x["fn"], x["dice"]))
    fp_report = {
        "threshold": BASELINE, "empty_ground_truth_slice_count": len(empty_rows),
        "empty_prediction_count": sum(not r["prediction_positive"] for r in empty_rows),
        "false_positive_slice_count": len(empty_fp), "fraction_with_false_positives": div(len(empty_fp), len(empty_rows)),
        "false_positive_connected_region_count_8_connected": sum(r["false_positive_region_count_8_connected"] for r in empty_fp_regions),
        "false_positive_pixels_on_empty_slices": sum(r["fp"] for r in empty_rows),
        "false_positive_regions_by_empty_slice": sorted(empty_fp_regions, key=lambda x: -x["false_positive_region_count_8_connected"]),
        "top_empty_slice_false_positives": sorted(empty_fp, key=lambda x: -x["fp"])[:20],
        "tumor_slices_with_excessive_false_positive_pixels_fp_gt_gt_pixels_count": len(high_fp),
        "tumor_slices_with_excessive_false_positive_pixels_fp_gt_gt_pixels": high_fp[:30],
    }
    fn_report = {
        "threshold": BASELINE, "tumor_containing_slice_count": len(positive_rows),
        "empty_prediction_count": len(tumor_empty_pred), "empty_prediction_slices": tumor_empty_pred,
        "poor_dice_definition": "tumor-containing-slice Dice < 0.20 (audit flag for inspection, not a clinical threshold)",
        "poor_dice_slice_count": len(low_dice), "poor_dice_slices": low_dice[:40],
        "under_segmentation_definition": "false-negative pixels exceed true-positive pixels on a tumor-containing slice",
        "under_segmented_slice_count": len(underseg), "under_segmented_slices": underseg[:40],
        "total_false_negative_pixels": baseline["false_negative_pixels"],
        "top_false_negative_slices": sorted(positive_rows, key=lambda x: -x["fn"])[:20],
    }
    (out / "false_positive_analysis.json").write_text(json.dumps(fp_report, indent=2), encoding="utf-8")
    (out / "false_negative_analysis.json").write_text(json.dumps(fn_report, indent=2), encoding="utf-8")

    # Deterministic representative cases, all unique where the cohort permits.
    chosen: list[tuple[str, dict[str, Any]]] = []
    def select(title: str, candidates: list[dict[str, Any]], key) -> None:
        used = {(r["patient_id"], r["slice_index"]) for _, r in chosen}
        available = [r for r in candidates if (r["patient_id"], r["slice_index"]) not in used]
        if available: chosen.append((title, key(available)))
    select("High overlap", [r for r in per_slice if r["target_positive"]], lambda x: max(x, key=lambda r: r["dice"]))
    select("Moderate overlap", [r for r in per_slice if r["target_positive"]], lambda x: min(x, key=lambda r: abs(r["dice"] - .5)))
    select("False positive", [r for r in empty_rows if r["fp"] > 0], lambda x: max(x, key=lambda r: r["fp"]))
    select("False negative", [r for r in positive_rows if r["fn"] > 0], lambda x: max(x, key=lambda r: r["fn"]))
    select("Empty ground truth", empty_rows, lambda x: min(x, key=lambda r: r["fp"]))
    visualization_cases = []
    for title, summary in chosen:
        key = (summary["patient_id"], summary["slice_index"])
        row = row_lookup[key]
        filename = title.lower().replace(" ", "_") + ".png"
        save_case(viz_dir / filename, row, title)
        visualization_cases.append({"category": title, "file": f"visualizations/{filename}", "patient_id": key[0], "slice_index": key[1], "dice": summary["dice"], "false_positive_pixels": summary["fp"], "false_negative_pixels": summary["fn"]})

    history = json.loads((RUN / "training_history.json").read_text(encoding="utf-8"))
    saved = json.loads((RUN / "segmentation_metrics.json").read_text(encoding="utf-8"))
    saved_metrics = saved["metrics"]
    comparisons = {k: {"reloaded": baseline[map_key], "saved": saved_metrics[saved_key], "absolute_difference": abs(baseline[map_key] - saved_metrics[saved_key])} for k, map_key, saved_key in (("all_slice_dice", "all_slice_macro_dice", "dice"), ("macro_iou", "macro_iou", "iou"), ("macro_slice_precision", "macro_slice_precision", "precision"), ("macro_slice_recall", "macro_slice_recall", "recall"))}
    comparisons["tumor_containing_slice_dice"] = {"reloaded": baseline["tumor_containing_slice_macro_dice"], "saved": saved["tumor_positive_slice_dice"], "absolute_difference": abs(baseline["tumor_containing_slice_macro_dice"] - saved["tumor_positive_slice_dice"])}
    if max(v["absolute_difference"] for v in comparisons.values()) > 1e-5:
        raise RuntimeError(f"Reloaded metrics did not reproduce saved metrics: {comparisons}")

    report = {
        "audit_timestamp_utc": stamp, "run_directory": str(RUN.resolve()), "audit_directory": str(out.resolve()),
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": "all-slice validation Dice", "threshold": BASELINE,
        "validation_patients": validation_ids, "evaluated_slice_count": len(rows),
        "preprocessing": checkpoint.get("normalization_config"), "architecture": checkpoint.get("model_architecture"),
        "metric_convention": EMPTY_CONVENTION,
        "baseline_metrics": baseline, "reload_reproduction": comparisons,
        "patient_metrics": patients, "threshold_comparison": thresholds,
        "training_history": history, "visualizations": visualization_cases,
        "audit_scope": "Validation MRI and masks for the four configured validation patients only; no test MRI or masks loaded; no training or checkpoint writes performed.",
    }
    (out / "validation_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    epochs = [h["epoch"] for h in history]
    valdicebest = max(history, key=lambda h: h["validation_dice"])
    worst = min(patients, key=lambda p: p["dice"])
    fpdominates = baseline["false_positive_pixels"] > baseline["false_negative_pixels"]
    threshold_best = max(thresholds, key=lambda x: x["tumor_containing_slice_dice"])
    md = f"""# Validation segmentation quality audit

Audit directory: `{out.name}`  
Checkpoint: `{checkpoint_path.relative_to(ROOT)}` (epoch {checkpoint.get('epoch')})  
Validation cohort: {len(validation_ids)} patients, {len(rows)} slices. Threshold 0.5 is the official baseline.

## Measured baseline results

| Metric | Value |
|---|---:|
| All-slice macro Dice | {baseline['all_slice_macro_dice']:.4f} |
| Tumor-containing-slice macro Dice | {baseline['tumor_containing_slice_macro_dice']:.4f} |
| Macro IoU | {baseline['macro_iou']:.4f} |
| Global voxel Dice | {baseline['global_voxel_dice']:.4f} |
| Global voxel IoU | {baseline['global_voxel_iou']:.4f} |
| Global pixel precision / recall | {baseline['global_pixel_precision']:.4f} / {baseline['global_pixel_recall']:.4f} |
| Macro slice precision / recall | {baseline['macro_slice_precision']:.4f} / {baseline['macro_slice_recall']:.4f} |
| False-positive / false-negative pixels | {baseline['false_positive_pixels']} / {baseline['false_negative_pixels']} |

Macro metrics average slice scores; global voxel metrics pool pixel counts across all slices. Empty-mask conventions: {EMPTY_CONVENTION}

Reloaded values match saved run metrics within 1e-5: **yes**. Differences: `{json.dumps(comparisons, sort_keys=True)}`.

## Training history and checkpoint choice

Train loss by epoch: {', '.join(f"{h['train_loss']:.4f}" for h in history)} (overall {history[0]['train_loss']:.4f} → {history[-1]['train_loss']:.4f}); it does **not** decrease monotonically because it rises from epoch 3 to 4. Validation loss by epoch: {', '.join(f"{h['validation_loss']:.4f}" for h in history)}. Validation loss is lowest at epoch {min(history,key=lambda h:h['validation_loss'])['epoch']} and rises for epochs 4–5 while final training loss is lower. This is an overfitting signal after epoch 3, though the short run and small cohort limit certainty.

All-slice Dice by epoch: {', '.join(f"{h['validation_dice']:.4f}" for h in history)}. Tumor-only Dice: {', '.join(f"{h['validation_positive_slice_dice']:.4f}" for h in history)}. Macro IoU: {', '.join(f"{h['validation_iou']:.4f}" for h in history)}. Macro precision: {', '.join(f"{h['validation_precision']:.4f}" for h in history)}; macro recall: {', '.join(f"{h['validation_recall']:.4f}" for h in history)}. The best all-slice Dice is epoch {valdicebest['epoch']}; it was selected because the training code checkpoints on validation all-slice Dice, whose maximum was {valdicebest['validation_dice']:.4f}. Epoch 2 has the highest validation IoU ({max(history,key=lambda h:h['validation_iou'])['validation_iou']:.4f}), while epoch 3 has the highest all-slice Dice and lowest validation loss. The tumor-only score does not move in lockstep: empty slices earn Dice 1 when prediction is also empty, so 384/620 empty slices can substantially buoy all-slice Dice. All-slice Dice alone is incomplete for model selection.

Validation measures are unstable across epochs: tumor-only Dice changes substantially, and precision/recall tradeoffs move. Four patients are too few for strong population-level claims.

## Patient-level results

See `patient_level_metrics.csv`. Lowest all-slice macro Dice is **{worst['patient_id']}** ({worst['dice']:.4f}; tumor-slice Dice {worst['tumor_containing_slice_dice']:.4f}). This is a description of these four validation patients only.

## Error analysis

Among {len(empty_rows)} empty-GT slices, {fp_report['empty_prediction_count']} have empty predictions and {len(empty_fp)} have false-positive pixels ({fp_report['fraction_with_false_positives']:.1%}); these comprise {fp_report['false_positive_connected_region_count_8_connected']} separate 8-connected components. Among {len(positive_rows)} tumor slices, {len(tumor_empty_pred)} have empty predictions. False {('positives' if fpdominates else 'negatives')} dominate pooled pixel errors ({baseline['false_positive_pixels']} FP vs {baseline['false_negative_pixels']} FN pixels). Tumor slices flagged with Dice <0.20: {len(low_dice)}; under-segmentation flags (FN > TP): {len(underseg)}. Full slice lists are in the FP/FN JSON files.

## Threshold sensitivity

See `threshold_comparison.csv`. Best validation tumor-slice Dice among the tested values occurs at {threshold_best['threshold']:.2f} ({threshold_best['tumor_containing_slice_dice']:.4f}); this is a candidate for investigation only. Selecting a threshold on this validation cohort makes those threshold-specific validation metrics selection-dependent. Keep 0.5 as the official baseline until a predeclared selection protocol is established; do not infer test performance from this sweep.

## Visual review

Generated cases (all from actual validation MRI arrays, unchanged ground truth masks, and reloaded-model probabilities): {', '.join(f"{v['category']}—{v['patient_id']} slice {v['slice_index']}" for v in visualization_cases)}. Green denotes ground truth and red prediction in overlays. The five original run visualizations remain untouched in the source run directory.

## Interpretation and next experiment

The model has learned nontrivial tumor localization on this validation cohort: tumor-containing-slice Dice is above zero and the overlap examples show predicted/target agreement, but the tumor-only macro Dice indicates substantial slice-level errors. The current results do not establish clinically reliable boundaries. Overfitting is suggested by lower final training loss while validation loss worsens after epoch 3; training loss itself has an epoch 4 uptick. Threshold 0.5 is a defensible fixed baseline; the sweep can guide a separately planned validation threshold study, with selection dependence documented.

For future experiments, predeclare a checkpoint criterion that reports and considers both tumor-containing-slice Dice and all-slice Dice, with validation loss as a diagnostic. A documented composite or a lexicographic rule (e.g. maximize tumor-only Dice subject to a minimum empty-slice specificity/false-positive constraint, with all-slice Dice as secondary) is more aligned to the imbalance than all-slice Dice alone. Pick the rule before training and use patient-level aggregation; do not claim universal superiority. Next: run a controlled, patient-grouped experiment with more training epochs/early stopping, retain 0.5 as the primary threshold, log the same stratified metrics, and evaluate final generalization once on the held-out test cohort only after model and threshold choices are frozen.
"""
    (out / "segmentation_quality_summary.md").write_text(md, encoding="utf-8")
    print(json.dumps({"audit_dir": str(out), "baseline": baseline, "worst_patient": worst, "thresholds": thresholds, "visualizations": visualization_cases}, indent=2))
    return out


if __name__ == "__main__":
    main()
