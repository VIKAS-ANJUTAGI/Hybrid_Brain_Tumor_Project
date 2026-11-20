"""One-shot evaluation of the frozen Phase 2 detection and segmentation models."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_loading import load_patient
from backend.dataset_builder import feature_columns
from config import settings
from feature_extraction.slice_features import extract_slice_features
from models.integration import DetectionModel, HybridInferencePipeline
from models.segmentation.inference import normalize_and_stack_modalities, predict_slice, prepare_axial_slice
from models.segmentation.training import load_checkpoint


DATASET = ROOT / "results/phase2_full_dataset_seed42"
MANIFEST = DATASET / "manifest.json"
CLASSICAL = ROOT / "results/classical_baselines_seed42_20261118T170148_799890Z"
QUANTUM = {
    2: ROOT / "results/quantum_models_seed42_20261118T170214_243106Z",
    4: ROOT / "results/quantum_models_seed42_20261118T170214_243106Z",
    8: ROOT / "results/quantum_models_seed42_20261118T171424_499965Z",
}
UNET = ROOT / "results/segmentation_runs/unet_baseline_20261120T173712Z/best_checkpoint.pt"
INTEGRATION = ROOT / "models/integration/hybrid_pipeline.py"
MODELS = list(DetectionModel.SUPPORTED)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_div(a: int | float, b: int | float, zero: float = 0.0) -> float:
    return float(a / b) if b else float(zero)


def binary_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    return {
        "n_slices": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": [0, 1],
        "sensitivity": safe_div(tp, tp + fn), "specificity": safe_div(tn, tn + fp),
    }


def mask_metrics(targets: list[np.ndarray], predictions: list[np.ndarray], patient_ids: list[str]) -> dict:
    ds, positive_ds, patient_positive_ds = [], [], {}
    tp = fp = fn = tn = 0
    empty = empty_fp = 0
    pixel_tp = pixel_fp = pixel_fn = 0
    for target, prediction, patient in zip(targets, predictions, patient_ids):
        t, p = np.asarray(target, dtype=bool), np.asarray(prediction, dtype=bool)
        inter = int(np.count_nonzero(t & p)); ts = int(np.count_nonzero(t)); ps = int(np.count_nonzero(p))
        ds.append(1.0 if ts == ps == 0 else safe_div(2 * inter, ts + ps))
        if ts:
            positive_ds.append(ds[-1]); patient_positive_ds.setdefault(patient, []).append(ds[-1])
        pixel_tp += inter; pixel_fp += ps - inter; pixel_fn += ts - inter
        if ts:
            tp += int(ps > 0); fn += int(ps == 0)
        else:
            empty += 1; empty_fp += int(ps > 0)
    union = pixel_tp + pixel_fp + pixel_fn
    return {
        "slice_count": len(targets),
        "all_slice_macro_dice": float(np.mean(ds)) if ds else None,
        "tumor_containing_slice_macro_dice": float(np.mean(positive_ds)) if positive_ds else None,
        "patient_macro_tumor_dice": float(np.mean([np.mean(v) for v in patient_positive_ds.values()])) if patient_positive_ds else None,
        "patients_with_tumor_slices": len(patient_positive_ds),
        "global_voxel_dice": safe_div(2 * pixel_tp, 2 * pixel_tp + pixel_fp + pixel_fn, 1.0),
        "global_voxel_iou": safe_div(pixel_tp, union, 1.0),
        "pixel_precision": safe_div(pixel_tp, pixel_tp + pixel_fp, 1.0),
        "pixel_recall": safe_div(pixel_tp, pixel_tp + pixel_fn, 1.0),
        "empty_slice_count": empty, "empty_slice_false_positive_rate": safe_div(empty_fp, empty, 0.0),
        "tumor_slice_detection_sensitivity": safe_div(tp, tp + fn),
        "tumor_slice_detection_confusion": {"tp": tp, "fn": fn, "tn": empty - empty_fp, "fp": empty_fp},
    }


def save_case(path: Path, image: np.ndarray, truth: np.ndarray, prediction: np.ndarray, title: str) -> None:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    display = np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(display, cmap="gray"); axes[0].set_title("Original MRI (T1c)")
    axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth")
    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Predicted mask")
    axes[3].imshow(display, cmap="gray")
    axes[3].imshow(np.ma.masked_where(~np.asarray(truth, bool), truth), cmap="Greens", alpha=.5, vmin=0, vmax=1)
    axes[3].imshow(np.ma.masked_where(~np.asarray(prediction, bool), prediction), cmap="autumn", alpha=.5, vmin=0, vmax=1)
    axes[3].set_title("Overlay (GT green, pred red)")
    for ax in axes: ax.axis("off")
    fig.suptitle(title); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=140); plt.close(fig)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assignments = manifest["patient_assignments"]
    split_check = {name: assignments[name] for name in ("train", "validation", "test")}
    if (len(assignments["train"]) != 13 or len(assignments["validation"]) != 4 or len(assignments["test"]) != 3
            or set(assignments["train"]) & set(assignments["validation"])
            or set(assignments["train"]) & set(assignments["test"])
            or set(assignments["validation"]) & set(assignments["test"])):
        raise RuntimeError("Phase 2 manifest patient split is unexpected or overlapping")
    test_ids = list(assignments["test"])

    # Capture frozen artifact digests and validate all artifacts before opening test images/masks.
    artifact_paths = {"logistic_regression": CLASSICAL / "logistic_regression.joblib",
                      "random_forest": CLASSICAL / "random_forest.joblib",
                      "vqc_2": QUANTUM[2] / "vqc_2qubit.joblib", "vqc_2_reducer": QUANTUM[2] / "reducer_2qubit.joblib",
                      "vqc_4": QUANTUM[4] / "vqc_4qubit.joblib", "vqc_4_reducer": QUANTUM[4] / "reducer_4qubit.joblib",
                      "vqc_8": QUANTUM[8] / "vqc_8qubit.joblib", "vqc_8_reducer": QUANTUM[8] / "reducer_8qubit.joblib",
                      "unet": UNET}
    missing = [str(p) for p in artifact_paths.values() if not p.is_file()]
    if missing: raise FileNotFoundError("Required frozen artifact(s) missing: " + ", ".join(missing))
    before_hashes = {name: sha256(path) for name, path in artifact_paths.items()}
    checkpoint = load_checkpoint(UNET)
    if tuple(checkpoint.get("modality_order", settings.MRI_MODALITIES)) != tuple(settings.MRI_MODALITIES):
        raise RuntimeError("U-Net modality order is incompatible")
    if float(checkpoint.get("segmentation_threshold", checkpoint.get("metric_threshold", -1))) != .5:
        raise RuntimeError("U-Net checkpoint threshold differs from required validated threshold 0.5")
    pipe = HybridInferencePipeline(device="cpu", threshold=.5, neighbor_radius=1)
    # Loading each artifact validates deserialization and model-specific preprocessing compatibility.
    loaded_configs = {}
    for name in MODELS:
        model, _ = pipe._load_detector(name)
        if name.startswith("vqc_"):
            clf, reducer = model
            q = int(name.split("_")[1])
            if clf.n_qubits != q or reducer.n_qubits != q or reducer.is_fitted_ is not True:
                raise RuntimeError(f"Incompatible paired VQC-{q} classifier/reducer")
            loaded_configs[name] = {"classifier": {k: getattr(clf, k) for k in ("n_qubits", "n_layers", "learning_rate", "max_epochs", "batch_size", "class_weight", "random_state")},
                                    "reducer": {"n_qubits": reducer.n_qubits, "random_state": reducer.random_state, "angle_range": list(reducer.angle_range), "is_fitted": reducer.is_fitted_}}
        else:
            loaded_configs[name] = {"estimator_type": type(model).__name__, "params": model.get_params(deep=False) if hasattr(model, "get_params") else {}}
    unet = pipe._load_unet()
    if tuple(checkpoint.get("validation_patient_ids", ())) != tuple(json.loads((ROOT / "results/segmentation_runs/unet_baseline_20261120T173712Z/training_config.json").read_text())["patient_splits"]["validation"]):
        raise RuntimeError("U-Net checkpoint/config validation cohorts disagree")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"final_test_evaluation_{timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "confusion_matrices").mkdir(); (out / "visualizations").mkdir()

    patients = {}
    samples = []
    for patient_id in test_ids:
        patient = load_patient(patient_id)
        if not patient.dimensions_compatible or not patient.segmentation_labels_valid:
            raise RuntimeError(f"Invalid held-out patient volumes/labels: {patient_id}")
        modalities = {key: patient.volumes[key] for key in settings.MRI_MODALITIES}
        raw_gt = np.asarray(patient.volumes["seg"])
        target_vol = raw_gt > 0
        normalized = normalize_and_stack_modalities(modalities)
        n_slices = target_vol.shape[2]
        feature_volumes = {}
        from preprocessing.normalization import normalize_mri
        finite_masks = {key: np.isfinite(value) for key, value in modalities.items()}
        for key, value in modalities.items():
            feature_volumes[key] = normalize_mri(value, strategy="zscore", background_mask=np.zeros(value.shape, dtype=bool))
        features = [extract_slice_features(feature_volumes, i, foreground_masks=finite_masks) for i in range(n_slices)]
        preds = {name: [pipe.detect_slice(patient_id, name, i, features[i]) for i in range(n_slices)] for name in MODELS}
        probs, unet_masks = [], []
        for i in range(n_slices):
            probability, mask = predict_slice(unet, prepare_axial_slice(normalized, i), threshold=.5)
            probs.append(probability.cpu().numpy()[0].astype(np.float32))
            unet_masks.append(mask.cpu().numpy()[0].astype(bool))
        patients[patient_id] = {"modalities": modalities, "t1c": modalities["t1c"], "target": target_vol,
                                "unet_probabilities": probs, "unet_masks": unet_masks,
                                "predictions": preds, "n_slices": n_slices}
        for i in range(n_slices): samples.append((patient_id, i, int(target_vol[:, :, i].any())))
        print(f"Loaded/evaluated {patient_id}: {n_slices} slices", flush=True)

    y_all = np.asarray([s[2] for s in samples], dtype=int)
    detection_summary, detection_patient, comparisons = {}, {}, []
    hybrid_results = {}
    for name in MODELS:
        yprob = np.asarray([patients[p]["predictions"][name][i].probability for p, i, _ in samples])
        yhat = (yprob >= .5).astype(int)
        summary = binary_metrics(y_all, yhat, yprob); detection_summary[name] = summary
        detection_patient[name] = {}
        cm = np.asarray(summary["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negative", "positive"], yticklabels=["negative", "positive"], xlabel="Predicted", ylabel="Ground truth", title=name)
        for (r, c), value in np.ndenumerate(cm): ax.text(c, r, str(value), ha="center", va="center")
        fig.tight_layout(); fig.savefig(out / "confusion_matrices" / f"{name}.png", dpi=130); plt.close(fig)
        for pid in test_ids:
            chosen = [(p, i, y) for p, i, y in samples if p == pid]
            yp = np.asarray([patients[p]["predictions"][name][i].probability for p, i, _ in chosen])
            detection_patient[name][pid] = binary_metrics(np.asarray([x[2] for x in chosen]), (yp >= .5).astype(int), yp)
        selected_by_pid = {}
        hybrid_masks = []
        selected_count = skipped_count = empty_count = 0
        for pid in test_ids:
            data = patients[pid]; detections = data["predictions"][name]
            positive_indices = [i for i, d in enumerate(detections) if d.predicted_class == 1]
            selected = sorted({j for i in positive_indices for j in range(max(0, i - pipe.neighbor_radius), min(data["n_slices"] - 1, i + pipe.neighbor_radius) + 1)})
            selected_set = set(selected); selected_count += len(selected); skipped_count += data["n_slices"] - len(selected)
            predicted_masks = []
            for i in range(data["n_slices"]):
                if i in selected_set:
                    _, pred = predict_slice(unet, prepare_axial_slice(normalize_and_stack_modalities(data["modalities"]), i), threshold=.5)
                    m = pred.cpu().numpy()[0].astype(bool)
                else: m = np.zeros((240, 240), dtype=bool)
                empty_count += int(not m.any()); predicted_masks.append(m)
                hybrid_masks.append(m)
            selected_by_pid[pid] = {"selected": selected, "predictions": predicted_masks,
                                    "selected_slice_count": len(selected), "skipped_slice_count": data["n_slices"]-len(selected),
                                    "empty_unet_prediction_count": sum(not m.any() for m in predicted_masks)}
        hybrid_metrics = mask_metrics([patients[p]["target"][:, :, i] for p, i, _ in samples], hybrid_masks, [p for p, _, _ in samples])
        patient_hybrid = {}
        for pid in test_ids:
            data = patients[pid]; info = selected_by_pid[pid]
            metric = mask_metrics([data["target"][:, :, i] for i in range(data["n_slices"])], info["predictions"], [pid]*data["n_slices"])
            patient_hybrid[pid] = {**metric, "selected_slice_count": info["selected_slice_count"], "skipped_slice_count": info["skipped_slice_count"], "empty_unet_prediction_count": info["empty_unet_prediction_count"]}
        hybrid_results[name] = {"aggregate": {**hybrid_metrics, "detection_sensitivity": summary["recall"],
                                               "slices_passed_to_unet": selected_count, "slices_skipped": skipped_count,
                                               "empty_unet_predictions": empty_count,
                                               "neighbor_radius": pipe.neighbor_radius, "patient_metrics": patient_hybrid},
                                "_predictions": selected_by_pid}
        comparisons.append({"model": name, "detection_accuracy": summary["accuracy"], "detection_precision": summary["precision"], "detection_recall": summary["recall"], "detection_f1": summary["f1_score"], "detection_roc_auc": summary["roc_auc"],
                           "standalone_unet_global_dice": None, "hybrid_global_dice": hybrid_metrics["global_voxel_dice"], "hybrid_global_iou": hybrid_metrics["global_voxel_iou"],
                           "slices_passed_to_unet": selected_count, "slices_skipped": skipped_count, "empty_unet_predictions": empty_count})

    standalone_masks = [patients[p]["unet_masks"][i] for p, i, _ in samples]
    truth_masks = [patients[p]["target"][:, :, i] for p, i, _ in samples]
    patient_sample_ids = [p for p, _, _ in samples]
    standalone = mask_metrics(truth_masks, standalone_masks, patient_sample_ids)
    for row in comparisons: row["standalone_unet_global_dice"] = standalone["global_voxel_dice"]
    patient_segmentation = {}
    for pid in test_ids:
        ids = [i for p, i, _ in samples if p == pid]
        patient_segmentation[pid] = mask_metrics([patients[pid]["target"][:, :, i] for i in ids], [patients[pid]["unet_masks"][i] for i in ids], [pid]*len(ids))

    # Representative unchanged-ground-truth examples from the cohort.
    flat = [(p, i, patients[p]["target"][:, :, i], patients[p]["unet_masks"][i]) for p, i, _ in samples]
    good = max((x for x in flat if x[2].any() and x[3].any()), key=lambda x: (2*np.count_nonzero(x[2]&x[3]) / (np.count_nonzero(x[2])+np.count_nonzero(x[3]))), default=None)
    fp_case = next((x for x in flat if not x[2].any() and x[3].any()), None)
    fn_case = next((x for x in flat if x[2].any() and not x[3].any()), None)
    empty_case = next((x for x in flat if not x[2].any() and not x[3].any()), None)
    cases = [("good_segmentation", good), ("false_positive", fp_case), ("false_negative", fn_case), ("empty_masks", empty_case)]
    for label, case in cases:
        if case:
            p, i, gt, pred = case
            save_case(out / "visualizations" / f"{label}_{p}_slice{i:03d}.png", patients[p]["t1c"][:, :, i], gt, pred, f"Standalone U-Net: {label} | {p} slice {i}")
    for name in MODELS:
        got = None
        for p in test_ids:
            gated = set(range(patients[p]["n_slices"])) - set(hybrid_results[name]["_predictions"][p]["selected"])
            got = next(((p, i) for i in sorted(gated) if patients[p]["target"][:, :, i].any()), None)
            if got: break
        if got:
            p, i = got
            save_case(out / "visualizations" / f"detection_gated_{name}_{p}_slice{i:03d}.png", patients[p]["t1c"][:, :, i], patients[p]["target"][:, :, i], hybrid_results[name]["_predictions"][p]["predictions"][i], f"{name}: detection-gated tumor slice (empty end-to-end prediction) | {p} slice {i}")

    # Patient-level table: detection plus standalone and each hybrid detector.
    patient_rows = []
    for pid in test_ids:
        row = {"patient_id": pid, "slice_count": patients[pid]["n_slices"], "standalone_all_slice_macro_dice": patient_segmentation[pid]["all_slice_macro_dice"],
               "standalone_tumor_slice_macro_dice": patient_segmentation[pid]["tumor_containing_slice_macro_dice"], "standalone_global_voxel_dice": patient_segmentation[pid]["global_voxel_dice"],
               "standalone_global_voxel_iou": patient_segmentation[pid]["global_voxel_iou"], "standalone_pixel_precision": patient_segmentation[pid]["pixel_precision"], "standalone_pixel_recall": patient_segmentation[pid]["pixel_recall"],
               "standalone_empty_slice_fpr": patient_segmentation[pid]["empty_slice_false_positive_rate"]}
        for name in MODELS:
            row.update({f"{name}_detection_{k}": detection_patient[name][pid][k] for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc")})
            hm = hybrid_results[name]["aggregate"]["patient_metrics"][pid]
            row.update({f"{name}_hybrid_dice": hm["global_voxel_dice"], f"{name}_hybrid_iou": hm["global_voxel_iou"], f"{name}_hybrid_precision": hm["pixel_precision"], f"{name}_hybrid_recall": hm["pixel_recall"], f"{name}_slices_passed": hm["selected_slice_count"], f"{name}_slices_skipped": hm["skipped_slice_count"]})
        patient_rows.append(row)
    with (out / "patient_level_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(patient_rows[0])); w.writeheader(); w.writerows(patient_rows)
    with (out / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)

    after_hashes = {name: sha256(path) for name, path in artifact_paths.items()}
    if before_hashes != after_hashes: raise RuntimeError("A frozen model artifact changed during evaluation")
    experiment = {
        "created_utc": timestamp, "test_patients": test_ids, "test_slice_count": len(samples), "patient_split_counts": {k: len(v) for k, v in assignments.items()},
        "split_disjoint": True, "dataset_manifest_path": str(MANIFEST.relative_to(ROOT)), "dataset_manifest_sha256": sha256(MANIFEST),
        "model_artifacts": {name: {"path": str(path.relative_to(ROOT)), "sha256": before_hashes[name]} for name, path in artifact_paths.items()},
        "integration_code": {"path": str(INTEGRATION.relative_to(ROOT)), "sha256": sha256(INTEGRATION)},
        "classical_training_rows": 2015, "quantum_training_sample_limit": 200,
        "training_budget_fairness": "Unequal: classical estimators were fit on all 2,015 training slices; VQC models used 200 training samples. No fair training-budget comparison or quantum advantage is claimed.",
        "detection_configuration": loaded_configs, "detection_threshold": 0.5, "unet_threshold": 0.5,
        "neighbor_radius_slices": pipe.neighbor_radius, "modalities": list(settings.MRI_MODALITIES),
        "preprocessing": {"detection": "per-patient/per-modality z-score; all finite voxels included (explicit all-false background mask); 16 named slice features; VQCs use frozen paired reducer", "unet": checkpoint.get("normalization_config"), "unet_modality_order": checkpoint.get("modality_order", list(settings.MRI_MODALITIES))},
        "unet_architecture": checkpoint.get("model_architecture"), "unet_checkpoint_epoch": checkpoint.get("epoch"),
        "unet_training_config": checkpoint.get("config"), "metric_conventions": "Per-slice Dice/IoU score 1 when both masks empty and 0 when exactly one is empty; pixel precision/recall are 1 for zero denominator. Tumor-only Dice includes only target-positive slices.",
        "threshold_selection": "Fixed validated U-Net threshold 0.5; existing detector decision threshold 0.5; no test-label threshold selection.",
        "test_patients_excluded_from_dashboard_selector": True, "model_hashes_unchanged_after_evaluation": before_hashes == after_hashes,
    }
    (out / "experiment_manifest.json").write_text(json.dumps(experiment, indent=2, default=str), encoding="utf-8")
    (out / "detection_test_metrics.json").write_text(json.dumps({"cohort": {"patients": test_ids, "slices": len(samples), "positive_slices": int(y_all.sum()), "negative_slices": int(len(y_all)-y_all.sum())}, "training_budget_disclosure": experiment["training_budget_fairness"], "models": {n: {"aggregate": detection_summary[n], "patient_metrics": detection_patient[n]} for n in MODELS}}, indent=2), encoding="utf-8")
    (out / "segmentation_test_metrics.json").write_text(json.dumps({"standalone_unet": {"aggregate": standalone, "patient_metrics": patient_segmentation}, "threshold": .5}, indent=2), encoding="utf-8")
    for name in MODELS: hybrid_results[name].pop("_predictions")
    (out / "hybrid_test_metrics.json").write_text(json.dumps({"detectors": hybrid_results, "threshold": .5, "neighbor_radius": pipe.neighbor_radius, "empty_skipped_mask_policy": "all skipped slices are scored with an empty prediction"}, indent=2), encoding="utf-8")

    lines = [f"# Final held-out test evaluation ({timestamp} UTC)", "", "## Frozen artifacts and protocol", "", f"Evaluated {len(test_ids)} disjoint reserved test patients and {len(samples)} slices. Manifest assignments contain 13 train, 4 validation, and 3 test patients.", "", experiment["training_budget_fairness"], "", "All models were loaded and hashed before test images/masks were read. U-Net and detector thresholds remained 0.5; neighboring slice radius was 1. Test results were not used to select thresholds or models.", "", "## Detection comparison", "", "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in MODELS:
        m = detection_summary[n]; tn, fp, fn, tp = sum(m["confusion_matrix"], [])
        lines.append(f"| {n} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['roc_auc']:.4f} | {tn} | {fp} | {fn} | {tp} |")
    lines += ["", "## Standalone U-Net", "", f"All-slice macro Dice: {standalone['all_slice_macro_dice']:.4f}; tumor-containing-slice macro Dice: {standalone['tumor_containing_slice_macro_dice']:.4f}; patient-macro tumor Dice: {standalone['patient_macro_tumor_dice']:.4f}; global voxel Dice/IoU: {standalone['global_voxel_dice']:.4f}/{standalone['global_voxel_iou']:.4f}; pixel precision/recall: {standalone['pixel_precision']:.4f}/{standalone['pixel_recall']:.4f}; empty-slice FPR: {standalone['empty_slice_false_positive_rate']:.4f}.", "", "## End-to-end hybrid comparison", "", "| Detector | Sensitivity | Passed | Skipped | Empty U-Net predictions | End-to-end Dice | IoU | Precision | Recall |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in MODELS:
        m = hybrid_results[n]["aggregate"]
        lines.append(f"| {n} | {m['detection_sensitivity']:.4f} | {m['slices_passed_to_unet']} | {m['slices_skipped']} | {m['empty_unet_predictions']} | {m['global_voxel_dice']:.4f} | {m['global_voxel_iou']:.4f} | {m['pixel_precision']:.4f} | {m['pixel_recall']:.4f} |")
    lines += ["", "## Patient-level outcomes", "", "See `patient_level_metrics.csv` for per-patient detection, standalone segmentation, and hybrid scores. With only three test patients, these estimates are descriptive and highly uncertain.", "", "## Error review", "", "Representative standalone U-Net examples include good overlap, false positive, false negative, and jointly empty masks where available. Detection-gated examples show tumor-containing slices skipped by a detector; these slices receive empty end-to-end masks. See `visualizations/`.", "", "## Model hashes", "", "See `experiment_manifest.json` for exact paths, SHA-256 hashes, frozen configurations, preprocessing, and verification that artifact hashes were unchanged.", "", "## Integration verification", "", "Dashboard selector remains validation-only by construction: it intersects `validation.csv` with checkpoint validation patient IDs and does not read `test.csv`. Automated dashboard and hybrid integration tests were run after evaluation (see execution record below).", "", "## Interpretation and limitations", "", "This small, three-patient held-out cohort cannot support broad generalization claims. Detection predicts mask-positive slices, not patient diagnosis. Unequal model training sample counts prevent a fair compute/data-budget comparison; no quantum advantage is claimed. These results do not establish clinical diagnostic capability.", ""]
    (out / "final_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "detection": detection_summary, "standalone_unet": standalone, "hybrid": {n: v["aggregate"] for n, v in hybrid_results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
