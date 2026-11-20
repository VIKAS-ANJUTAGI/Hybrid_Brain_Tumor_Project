# Final experimental results

## Provenance and evaluation scope

The authoritative source is `results/final_test_evaluation_20261120T190221Z/final_evaluation_report.md`, supported by `detection_test_metrics.json`, `segmentation_test_metrics.json`, `hybrid_test_metrics.json`, `model_comparison.csv`, `patient_level_metrics.csv`, and `experiment_manifest.json`. These saved outputs are the source for the numbers below; no MRI volumes or masks are needed to read this report.

The final evaluation contains 465 axial slices from three held-out patients. The experiment manifest records 13 train, 4 validation, and 3 test patients and disjoint splits. The current experimental setup has 20 patients total. Thresholds were fixed at 0.5 and the hybrid neighbor radius at one slice. The saved manifest records model artifact hashes and states that hashes remained unchanged after evaluation.

Three scopes must be kept separate:

1. **Detection** classifies every axial slice as mask-positive or zero-mask.
2. **Standalone segmentation** evaluates U-Net predictions over all slices without detector gating.
3. **Conditional hybrid segmentation** evaluates outputs after a detector selects slices and neighboring slices for U-Net inference. A skipped slice has an empty hybrid output by definition.

## Detection: held-out test set

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7247 | 0.5701 | 0.7722 | 0.6559 | 0.8644 |
| Random Forest | 0.7290 | 0.5714 | 0.8101 | 0.6702 | 0.8635 |
| VQC-2 | 0.3398 | 0.3398 | 1.0000 | 0.5072 | 0.5546 |
| VQC-4 | 0.7527 | 0.5947 | 0.8544 | 0.7013 | 0.7709 |
| VQC-8 | 0.7462 | 0.5901 | 0.8291 | 0.6895 | 0.7618 |

VQC-2 predicted all 465 slices positive: 307 false positives and 158 true positives, with no true negatives or false negatives. The classical models were trained with 2,015 samples; the quantum models were trained with 200 samples. This was not a matched-budget comparison. The results do not show quantum advantage.

## Standalone U-Net segmentation: held-out test set

| Metric | Result |
|---|---:|
| All-slice macro Dice | 0.7414 |
| Tumor-slice macro Dice | 0.3970 |
| Patient-macro tumor Dice | 0.3946 |
| Global voxel Dice | 0.5862 |
| Global voxel IoU | 0.4147 |
| Pixel precision | 0.9654 |
| Pixel recall | 0.4209 |
| Empty-slice false-positive rate | 0.0814 |

The all-slice macro score includes empty slices, so it should be read alongside tumor-slice Dice and patient-macro tumor Dice. Pixel recall of 0.4209 indicates that many tumor pixels are missed despite high pixel precision.

## Conditional hybrid segmentation: held-out test set

These scores include the detection gate and are not standalone U-Net scores. The `passed` and `skipped` counts sum to 465 slices. `Empty U-Net predictions` includes empty outputs on skipped slices as well as selected slices where the predicted mask is empty.

| Detector | Detector sensitivity | Passed | Skipped | Empty U-Net outputs | End-to-end Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7722 | 225 | 240 | 341 | 0.5808 | 0.4093 | 0.9754 | 0.4135 |
| Random Forest | 0.8101 | 232 | 233 | 326 | 0.5770 | 0.4055 | 0.9650 | 0.4115 |
| VQC-2 | 1.0000 | 465 | 0 | 311 | 0.5862 | 0.4147 | 0.9654 | 0.4209 |
| VQC-4 | 0.8544 | 233 | 232 | 316 | 0.5857 | 0.4141 | 0.9657 | 0.4203 |
| VQC-8 | 0.8291 | 230 | 235 | 318 | 0.5795 | 0.4080 | 0.9652 | 0.4141 |

Because VQC-2 labels every slice positive, its hybrid segmentation does not skip any slice and matches standalone U-Net pooled test results. Other detectors skip some slices; detector false negatives therefore reduce segmentation coverage.

## Training and validation records

- Patient-level split and feature definitions: `results/phase2_full_dataset_seed42/manifest.json`.
- Classical model training and validation comparison: `results/classical_baselines_seed42_20261118T170148_799890Z/validation_results.json`.
- Quantum training settings and validation comparison: `results/quantum_models_seed42_20261118T170214_243106Z/quantum_validation_results.json` and `results/quantum_models_seed42_20261118T171424_499965Z/quantum_validation_results.json`.
- U-Net architecture and training parameters: `results/segmentation_runs/unet_baseline_20261120T173712Z/training_config.json`; history: `training_history.json`; metrics: `segmentation_metrics.json`; checkpoint: `best_checkpoint.pt`.
- Validation error audit: `results/segmentation_validation_audit_20261120T181719Z/segmentation_quality_summary.md`, with threshold, false-positive, false-negative, and patient-level CSV/JSON outputs.

Validation U-Net visualizations are in `results/segmentation_runs/unet_baseline_20261120T173712Z/visualizations/` (`validation_tumor_presence.png`, `validation_highest_tumor_dice.png`, `validation_empty_ground_truth.png`, `validation_largest_false_positive.png`, and `validation_largest_false_negative.png`). Test evaluation visualizations, including good segmentation, false-positive, false-negative, empty-mask, and detection-gated cases, are retained in `results/final_test_evaluation_20261120T190221Z/visualizations/`.

## Artifact integrity

The final evaluation manifest `experiment_manifest.json` records SHA-256 entries for the dataset manifest, classical estimators, VQC models, their paired reducers, and U-Net checkpoint. Its `model_hashes_unchanged_after_evaluation` field is `true`. The existing model and preprocessing artifacts are frozen; this documentation work does not require rewriting them. Recompute and compare hashes only as a read-only audit if desired.

## Interpretation and limitations

These are descriptive results from a small held-out cohort of three patients. Patient-level estimates are unstable and do not establish generalization. Quantum training budgets differ from classical budgets, and VQC-16 was deferred. The U-Net has low tumor-pixel recall. Detection is mask-positive slice classification, not patient diagnosis; segmentation outputs are experimental masks. The system is a research/educational prototype, not a clinical diagnostic device.
