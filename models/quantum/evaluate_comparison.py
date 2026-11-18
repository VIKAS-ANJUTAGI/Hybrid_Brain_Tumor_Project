"""Benchmark and compare classical (LR, RF) and quantum (VQC 2, 4, 8, 16) classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

from models.classical.train_baselines import (
    DEFAULT_DATASET_DIR,
    DEFAULT_RESULTS_ROOT,
    _read_feature_columns,
    _read_split,
)
from models.quantum.dim_reduction import QuantumDimensionalityReducer
from models.quantum.vqc_model import VariationalQuantumClassifier


def evaluate_estimator(
    estimator: Any,
    features: np.ndarray,
    targets: pd.Series,
    *,
    is_vqc: bool = False,
) -> dict[str, Any]:
    """Compute standard classification metrics for a classical or quantum estimator."""
    predictions = estimator.predict(features)
    probabilities = estimator.predict_proba(features)

    roc_auc = None
    if targets.nunique() == 2:
        try:
            positive_col = 1
            if not is_vqc and hasattr(estimator, "classes_"):
                positive_col = list(estimator.classes_).index(1)
            roc_auc = float(roc_auc_score(targets, probabilities[:, positive_col]))
        except Exception:
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


def compare_models(
    classical_results_dir: str | Path,
    quantum_results_dir: str | Path,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compare all classical and quantum models on the validation split.

    Args:
        classical_results_dir: Directory containing logistic_regression.joblib and random_forest.joblib.
        quantum_results_dir: Directory containing vqc_*qubit.joblib and reducer_*qubit.joblib.
        dataset_dir: Directory containing validation.csv and manifest.json.
        output_dir: Where to write benchmark_comparison_report.json.

    Returns:
        Structured dictionary with comparative metrics and model rankings.
    """
    dataset_path = Path(dataset_dir)
    classical_dir = Path(classical_results_dir)
    quantum_dir = Path(quantum_results_dir)

    selected_features = _read_feature_columns(dataset_path)
    x_val_df, y_val, val_summary = _read_split(
        dataset_path / "validation.csv", selected_features
    )
    x_val_arr = x_val_df.to_numpy(dtype=np.float64)

    comparison_results: dict[str, dict[str, Any]] = {}

    # 1. Classical Models
    for name in ("logistic_regression", "random_forest"):
        model_file = classical_dir / f"{name}.joblib"
        if model_file.is_file():
            model = joblib.load(model_file)
            comparison_results[name] = evaluate_estimator(
                model, x_val_df, y_val, is_vqc=False
            )

    # 2. Quantum Models (2, 4, 8, 16 qubits)
    for n_qubits in (2, 4, 8, 16):
        vqc_name = f"vqc_{n_qubits}qubit"
        vqc_file = quantum_dir / f"{vqc_name}.joblib"
        reducer_file = quantum_dir / f"reducer_{n_qubits}qubit.joblib"

        if vqc_file.is_file() and reducer_file.is_file():
            reducer = QuantumDimensionalityReducer.load(reducer_file)
            vqc = VariationalQuantumClassifier.load(str(vqc_file))

            x_val_angles = reducer.transform(x_val_arr)
            comparison_results[vqc_name] = evaluate_estimator(
                vqc, x_val_angles, y_val, is_vqc=True
            )

    if not comparison_results:
        raise ValueError("No classical or quantum model artifacts found in provided directories.")

    # Sort all models by (F1-score, ROC-AUC, Accuracy)
    ranked_models = sorted(
        comparison_results.keys(),
        key=lambda m: (
            comparison_results[m]["f1_score"],
            comparison_results[m]["roc_auc"] if comparison_results[m]["roc_auc"] is not None else -1.0,
            comparison_results[m]["accuracy"],
        ),
        reverse=True,
    )

    report = {
        "dataset_dir": str(dataset_path.resolve()),
        "validation_samples": val_summary["rows"],
        "class_distribution": val_summary["class_distribution"],
        "model_rankings": ranked_models,
        "best_overall_model": ranked_models[0],
        "metrics_summary": comparison_results,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        report_file = out_path / "benchmark_comparison_report.json"
        report_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical-dir", type=Path, required=True)
    parser.add_argument("--quantum-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    report = compare_models(
        classical_results_dir=args.classical_dir,
        quantum_results_dir=args.quantum_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir or args.quantum_dir,
    )

    print("\n================== BENCHMARK COMPARISON ==================")
    print(f"{'Model':<24} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 74)
    for model_name in report["model_rankings"]:
        m = report["metrics_summary"][model_name]
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        print(
            f"{model_name:<24} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
            f"{m['recall']:<10.4f} {m['f1_score']:<10.4f} {auc_str:<10}"
        )
    print("=" * 74)
    print(f"Top-Ranked Model: {report['best_overall_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
