"""Train and evaluate Variational Quantum Classifiers (2, 4, 8, and 16 qubits)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.dataset_builder import feature_columns
from models.classical.train_baselines import (
    DEFAULT_DATASET_DIR,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SEED,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    _read_feature_columns,
    _read_split,
)
from models.quantum.dim_reduction import (
    SUPPORTED_QUBIT_COUNTS,
    QuantumDimensionalityReducer,
)
from models.quantum.vqc_model import VariationalQuantumClassifier


def _evaluate_quantum(
    model: VariationalQuantumClassifier,
    features: np.ndarray,
    targets: pd.Series,
) -> dict[str, Any]:
    """Evaluate VQC model on features using standard classification metrics."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    positive_probs = probabilities[:, 1]

    roc_auc = None
    if targets.nunique() == 2:
        try:
            roc_auc = float(roc_auc_score(targets, positive_probs))
        except ValueError:
            roc_auc = None

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1_score": float(f1_score(targets, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=(0, 1)
        ).tolist(),
        "confusion_matrix_labels": [0, 1],
        "roc_auc": roc_auc,
    }


def _reserve_quantum_output_dir(results_root: Path, seed: int) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    base_name = f"quantum_models_seed{seed}_{timestamp}"
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        output_dir = results_root / name
        try:
            output_dir.mkdir()
        except FileExistsError:
            continue
        return output_dir
    raise FileExistsError("Unable to reserve a unique quantum output directory")


def train_quantum_pipeline(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    seed: int = DEFAULT_SEED,
    qubit_configs: Sequence[int] = SUPPORTED_QUBIT_COUNTS,
    n_layers: int = 2,
    max_epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 0.05,
    sample_limit: int | None = None,
) -> Path:
    """Train independent VQC models for 2, 4, 8, and 16 qubits on train.csv and evaluate on validation.csv.

    Args:
        dataset_dir: Path to folder containing train.csv, validation.csv, and manifest.json.
        results_root: Base results destination.
        seed: Random seed for reproducibility.
        qubit_configs: List/tuple of qubit configurations (subset of [2, 4, 8, 16]).
        n_layers: Number of parameterized ansatz layers.
        max_epochs: Number of training epochs per quantum model.
        batch_size: Mini-batch size for parameter updates.
        learning_rate: Adam optimizer learning rate.
        sample_limit: Optional cap on training samples for rapid simulation / tests.

    Returns:
        Path to output directory containing serialized models, reducers, and results JSON.
    """
    for n_q in qubit_configs:
        if n_q not in SUPPORTED_QUBIT_COUNTS:
            raise ValueError(f"Unsupported qubit configuration: {n_q}")

    dataset_path = Path(dataset_dir)
    selected_features = _read_feature_columns(dataset_path)

    x_train_df, y_train, train_summary = _read_split(
        dataset_path / "train.csv", selected_features
    )
    x_val_df, y_val, val_summary = _read_split(
        dataset_path / "validation.csv", selected_features
    )

    x_train_arr = x_train_df.to_numpy(dtype=np.float64)
    y_train_arr = y_train.to_numpy(dtype=np.int64)
    x_val_arr = x_val_df.to_numpy(dtype=np.float64)
    y_val_arr = y_val.to_numpy(dtype=np.int64)

    if sample_limit is not None and sample_limit < len(x_train_arr):
        rng = np.random.RandomState(seed)
        sample_indices = rng.choice(len(x_train_arr), size=sample_limit, replace=False)
        x_train_arr = x_train_arr[sample_indices]
        y_train_arr = y_train_arr[sample_indices]

    output_dir = _reserve_quantum_output_dir(Path(results_root), seed)
    quantum_validation_metrics: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, str] = {}

    for n_qubits in qubit_configs:
        config_name = f"vqc_{n_qubits}qubit"

        # 1. Fit dimensionality reduction strictly on training data
        reducer = QuantumDimensionalityReducer(
            n_qubits=n_qubits,
            random_state=seed,
        )
        x_train_angles = reducer.fit_transform(x_train_arr)
        x_val_angles = reducer.transform(x_val_arr)

        reducer_path = output_dir / f"reducer_{n_qubits}qubit.joblib"
        reducer.save(reducer_path)
        artifact_paths[f"reducer_{n_qubits}qubit"] = reducer_path.name

        # 2. Fit Variational Quantum Classifier
        vqc = VariationalQuantumClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            class_weight="balanced",
            random_state=seed,
        )
        vqc.fit(x_train_angles, y_train_arr)

        # 3. Evaluate on validation split
        metrics = _evaluate_quantum(vqc, x_val_angles, y_val)
        quantum_validation_metrics[config_name] = metrics

        model_path = output_dir / f"{config_name}.joblib"
        vqc.save(str(model_path))
        artifact_paths[config_name] = model_path.name

    # Select best quantum configuration by (F1, ROC-AUC, Accuracy)
    selected_quantum_model = max(
        quantum_validation_metrics,
        key=lambda name: (
            quantum_validation_metrics[name]["f1_score"],
            quantum_validation_metrics[name]["roc_auc"]
            if quantum_validation_metrics[name]["roc_auc"] is not None
            else -1.0,
            quantum_validation_metrics[name]["accuracy"],
        ),
    )

    report = {
        "random_seed": seed,
        "dataset_dir": str(dataset_path.resolve()),
        "feature_columns": list(selected_features),
        "excluded_columns": list(METADATA_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_definition": (
            "1 means the axial slice contains any nonzero segmentation-mask voxel; "
            "0 means it contains none."
        ),
        "qubit_configurations": list(qubit_configs),
        "ansatz_layers": n_layers,
        "training_hyperparameters": {
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "class_weight": "balanced",
            "sample_limit": sample_limit,
        },
        "data_usage": {
            "fit_split": "train.csv",
            "comparison_split": "validation.csv",
            "test_split_loaded": False,
        },
        "split_summaries": {
            "train": train_summary,
            "validation": val_summary,
        },
        "quantum_validation_metrics": quantum_validation_metrics,
        "selection": {
            "metric_order": ["f1_score", "roc_auc", "accuracy"],
            "selected_quantum_model": selected_quantum_model,
        },
        "artifacts": artifact_paths,
    }

    results_file = output_dir / "quantum_validation_results.json"
    results_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="List of qubit configurations (subset of 2 4 8 16)",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args(argv)

    output_dir = train_quantum_pipeline(
        dataset_dir=args.dataset_dir,
        results_root=args.results_root,
        seed=args.seed,
        qubit_configs=args.qubits,
        n_layers=args.layers,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
    )
    print(f"Quantum artifacts and metrics written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
