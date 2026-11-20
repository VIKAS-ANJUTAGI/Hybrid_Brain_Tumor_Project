# Installation and execution guide

## Environment

The repository's README setup targets Python 3.11. From PowerShell at the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Python launcher is unavailable, use the installed Python 3.11 executable to create the environment. The requirements include NumPy, NiBabel, pandas, scikit-learn, joblib, matplotlib, OpenCV, Synapse client, PennyLane, and PyTorch. Dependency versions are not pinned, so this file does not promise an identical environment across machines.

## Dataset location and preparation

MRI data are not bundled. Obtain authorized BraTS-GLI data, then extract each patient's five NIfTI volumes under one root with folders named `BraTS-GLI-<patient-id>`. Each folder must include suffixes `t1c`, `t1n`, `t2w`, `t2f`, and `seg` (the loader resolves `.nii.gz` or `.nii`). Do not commit data or credentials.

`config/settings.py` selects `data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` if it exists; otherwise it uses the configured external Windows path. Edit `_LOCAL_EXTRACTED` / `_EXTERNAL_EXTRACTED` for the machine in use. The archive setting is similarly local/external. `scripts/download_dataset.py` is a Synapse-authenticated helper with fixed paths in the script and no command-line arguments; inspect/configure it before use.

The frozen experiment already has a patient-level dataset manifest and CSV files at `results/phase2_full_dataset_seed42/`. If creating a fresh dataset from configured MRI volumes, choose a new output directory because the builder requires that it not exist:

```powershell
python -m backend.dataset_builder results\phase2_full_dataset_seed42_reproduction --seed 42
```

This writes `train.csv`, `validation.csv`, `test.csv`, and `manifest.json`; it reads configured patient data. Never use a test cohort for training or dashboard demonstration. The current frozen split is 13 train, 4 validation, and 3 test patients.

## Classical model training

The following supported command fits Logistic Regression and Random Forest using the generated CSVs, evaluates on validation, and writes a new timestamped result directory. It does not load `test.csv`:

```powershell
python -m models.classical.train_baselines `
  --dataset-dir results\phase2_full_dataset_seed42_reproduction `
  --results-root results `
  --seed 42
```

This is the reproducible training workflow, not a required setup step for opening the dashboard when the saved artifacts already exist.

## Quantum model training

The supported CLI accepts an explicit qubit list. Restrict integrated experiments to 2, 4, and 8 qubits; do not use the CLI's default list because it includes deferred VQC-16. The command below uses the experiment's 200-sample cap and writes new artifacts:

```powershell
python -m models.quantum.train_quantum `
  --dataset-dir results\phase2_full_dataset_seed42_reproduction `
  --results-root results `
  --seed 42 `
  --qubits 2 4 8 `
  --sample-limit 200
```

The CLI also accepts `--layers`, `--epochs`, and `--batch-size`. The saved experiments are in the result directories listed below. The command is documented for reproducibility; it is not needed for inference with existing artifacts.

## U-Net training

The repository's actual U-Net training entry point is `scripts/train_unet.py`, with `--manifest` and `--dataset-root` arguments. It trains a new CPU run (five epochs, batch size two, seed 42 as configured in the script) and writes a timestamped checkpoint, metrics, history, and validation PNGs under `results/segmentation_runs/`:

```powershell
python scripts/train_unet.py `
  --manifest results\phase2_full_dataset_seed42\manifest.json `
  --dataset-root data\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
```

Training is not necessary for the dashboard when its saved checkpoint is present. The current frozen U-Net checkpoint is at `results/segmentation_runs/unet_baseline_20261120T173712Z/best_checkpoint.pt`.

## Loading existing artifacts and starting the dashboard

The dashboard loads these existing artifacts by default:

| Model | Required files |
|---|---|
| Logistic Regression | `results/classical_baselines_seed42_20261118T170148_799890Z/logistic_regression.joblib` |
| Random Forest | `results/classical_baselines_seed42_20261118T170148_799890Z/random_forest.joblib` |
| VQC-2 | `results/quantum_models_seed42_20261118T170214_243106Z/vqc_2qubit.joblib` and `reducer_2qubit.joblib` in the same directory |
| VQC-4 | `results/quantum_models_seed42_20261118T170214_243106Z/vqc_4qubit.joblib` and `reducer_4qubit.joblib` in the same directory |
| VQC-8 | `results/quantum_models_seed42_20261118T171424_499965Z/vqc_8qubit.joblib` and `reducer_8qubit.joblib` in the same directory |
| U-Net | `results/segmentation_runs/unet_baseline_20261120T173712Z/best_checkpoint.pt` |

The dashboard also needs `results/phase2_full_dataset_seed42/validation.csv`, the segmentation checkpoint's validation-patient metadata, and accessible MRI volumes for permitted validation patients. Saved comparison displays additionally use classical and quantum validation JSON plus U-Net `segmentation_metrics.json`. If required artifacts or validation data are missing, the dashboard reports an unavailable-artifact or validation-data error; it does not train replacement models. It intersects validation CSV patients with checkpoint validation IDs and keeps the held-out test cohort out of the interactive selector.

Start from the repository root:

```powershell
python -m dashboard.app
```

Open <http://127.0.0.1:8000>. Stop the local server with Ctrl+C.

## Verification commands

Run the project's regression suite and source checks from the root:

```powershell
python -m pytest -q
python -m compileall -q backend config dashboard feature_extraction models preprocessing scripts tests
node --check dashboard/static/dashboard.js
git diff --check
```

These commands do not train models. Dashboard demonstrations should select a validation patient only. The final test evaluation script is not part of setup and must not be rerun for this frozen submission.
