"""Slice-wise detection and conditional independent-slice U-Net inference."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from backend.dataset_builder import feature_columns
from config import settings
from feature_extraction.slice_features import extract_slice_features
from preprocessing.normalization import normalize_mri
from models.quantum.dim_reduction import QuantumDimensionalityReducer
from models.quantum.vqc_model import VariationalQuantumClassifier
from models.segmentation.inference import (
    predict_slice,
    prepare_axial_slice,
)
from models.segmentation.training import load_checkpoint, reconstruct_model

from .result_types import DetectionResult, HybridPatientResult, SegmentedSliceResult


class ModelArtifactError(FileNotFoundError):
    """An expected trained model or matching preprocessing artifact is absent."""


class DetectionModel:
    """Names accepted by :class:`HybridInferencePipeline`."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    VQC_2 = "vqc_2"
    VQC_4 = "vqc_4"
    VQC_8 = "vqc_8"
    SUPPORTED = (LOGISTIC_REGRESSION, RANDOM_FOREST, VQC_2, VQC_4, VQC_8)


_MODEL_CACHE: dict[tuple[str, str], object] = {}
_UNET_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}
_CACHE_LOCK = threading.RLock()


def clear_model_cache() -> None:
    """Clear loaded model instances (primarily useful for isolated tests)."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _UNET_CACHE.clear()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HybridInferencePipeline:
    """Load existing artifacts once and run conditional segmentation per patient.

    The default paths point to this repository's trained artifacts. VQC models
    always use their paired dimensionality reducer. Slice detection results are
    cached in memory by patient, model artifact digest, and slice index.
    """

    def __init__(
        self,
        *,
        classical_artifact_dir: str | Path | None = None,
        quantum_artifact_dirs: Mapping[int, str | Path] | None = None,
        unet_checkpoint: str | Path | None = None,
        device: str | torch.device = "cpu",
        threshold: float = 0.5,
        neighbor_radius: int = 1,
        overlay_dir: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.classical_artifact_dir = Path(classical_artifact_dir or root / "results/classical_baselines_seed42_20261118T170148_799890Z")
        default_quantum = {
            2: root / "results/quantum_models_seed42_20261118T170214_243106Z",
            4: root / "results/quantum_models_seed42_20261118T170214_243106Z",
            8: root / "results/quantum_models_seed42_20261118T171424_499965Z",
        }
        if quantum_artifact_dirs:
            default_quantum.update({int(k): Path(v) for k, v in quantum_artifact_dirs.items()})
        self.quantum_artifact_dirs = default_quantum
        self.unet_checkpoint = Path(unet_checkpoint or root / "results/segmentation_runs/unet_baseline_20261120T173712Z/best_checkpoint.pt")
        self.device = torch.device(device)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if isinstance(neighbor_radius, bool) or not isinstance(neighbor_radius, int) or neighbor_radius < 0:
            raise ValueError("neighbor_radius must be a non-negative integer")
        self.threshold = float(threshold)
        self.neighbor_radius = neighbor_radius
        self.overlay_dir = Path(overlay_dir) if overlay_dir is not None else None
        self._detection_cache: dict[tuple[str, str, int], DetectionResult] = {}
        self._model_name: str | None = None
        self._model = None
        self._model_version: str | None = None
        self._unet = None

    @staticmethod
    def _require(path: Path, kind: str) -> Path:
        if not path.is_file():
            raise ModelArtifactError(f"Missing {kind} artifact: {path}")
        return path

    def _load_detector(self, model_name: str):
        if model_name not in DetectionModel.SUPPORTED:
            raise ValueError(f"Unsupported detection model {model_name!r}; choose from {DetectionModel.SUPPORTED}")
        if self._model_name == model_name and self._model is not None:
            return self._model, self._model_version
        if model_name in (DetectionModel.LOGISTIC_REGRESSION, DetectionModel.RANDOM_FOREST):
            path = self.classical_artifact_dir / f"{model_name}.joblib"
            self._require(path, "classical detection model")
            key = ("classical", str(path.resolve()))
            with _CACHE_LOCK:
                model = _MODEL_CACHE.get(key)
                if model is None:
                    model = joblib.load(path)
                    _MODEL_CACHE[key] = model
            artifact_version = _file_digest(path)
        else:
            qubits = int(model_name.split("_")[1])
            directory = Path(self.quantum_artifact_dirs[qubits])
            model_path = self._require(directory / f"vqc_{qubits}qubit.joblib", f"VQC-{qubits}")
            reducer_path = self._require(directory / f"reducer_{qubits}qubit.joblib", f"VQC-{qubits} preprocessing")
            key = ("vqc", str(model_path.resolve()) + ":" + _file_digest(reducer_path))
            with _CACHE_LOCK:
                cached = _MODEL_CACHE.get(key)
                if cached is None:
                    cached = (VariationalQuantumClassifier.load(model_path), QuantumDimensionalityReducer.load(reducer_path))
                    _MODEL_CACHE[key] = cached
            model = cached
            path = model_path
            artifact_version = hashlib.sha256(
                (_file_digest(model_path) + _file_digest(reducer_path)).encode("ascii")
            ).hexdigest()
        self._model_name, self._model = model_name, model
        self._model_version = artifact_version
        return model, self._model_version

    def _load_unet(self):
        if self._unet is None:
            path = self._require(self.unet_checkpoint, "U-Net checkpoint")
            key = (str(path.resolve()), _file_digest(path), str(self.device))
            with _CACHE_LOCK:
                model = _UNET_CACHE.get(key)
                if model is None:
                    checkpoint = load_checkpoint(path)
                    model = reconstruct_model(checkpoint)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(self.device).eval()
                    _UNET_CACHE[key] = model
                self._unet = model
        return self._unet

    def detect_slice(self, patient_id: str, model_name: str, slice_index: int, features: np.ndarray) -> DetectionResult:
        model, version = self._load_detector(model_name)
        cache_key = (patient_id, version, int(slice_index))
        cached = self._detection_cache.get(cache_key)
        if cached is not None:
            return cached
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError(f"slice features must be 16 finite values, got {values.shape}")
        started = time.perf_counter()
        if model_name.startswith("vqc_"):
            classifier, reducer = model
            prediction_input = reducer.transform(values.reshape(1, 16))
            probabilities = classifier.predict_proba(prediction_input)
        else:
            # The classical artifacts were fitted with named manifest columns.
            prediction_input = pd.DataFrame([values], columns=feature_columns())
            probabilities = model.predict_proba(prediction_input)
        probability = float(probabilities[0, list(model[0].classes_).index(1)] if model_name.startswith("vqc_") else probabilities[0, list(model.classes_).index(1)])
        result = DetectionResult(patient_id, model_name, int(slice_index), probability, int(probability >= 0.5), time.perf_counter() - started)
        self._detection_cache[cache_key] = result
        return result

    @staticmethod
    def _save_overlay(path: Path, background: np.ndarray, mask: np.ndarray) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.asarray(background, dtype=np.float32)
        low, high = np.percentile(image, (1, 99))
        display = np.clip((image - low) / (high - low), 0, 1) if high > low else np.zeros_like(image)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(display, cmap="gray")
        if np.any(mask):
            ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.55, vmin=0, vmax=1)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def run_patient(
        self,
        patient_id: str,
        modalities: Mapping[str, np.ndarray],
        model_name: str,
        *,
        slice_indices: Sequence[int] | None = None,
        neighbor_radius: int | None = None,
    ) -> HybridPatientResult:
        """Analyze slices and segment every positive slice plus clamped neighbors."""
        started = time.perf_counter()
        if set(modalities) != set(settings.MRI_MODALITIES):
            raise ValueError(f"modalities must contain exactly {settings.MRI_MODALITIES}")
        arrays = {name: np.asarray(modalities[name]) for name in settings.MRI_MODALITIES}
        shape = arrays[settings.MRI_MODALITIES[0]].shape
        if len(shape) != 3 or shape[:2] != (240, 240) or any(value.shape != shape for value in arrays.values()):
            raise ValueError("all MRI modalities must have identical native 240x240xZ shapes")
        for name, value in arrays.items():
            if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
                raise TypeError(f"{name} must contain real numeric intensities")
        finite_masks = {name: np.isfinite(value) for name, value in arrays.items()}
        feature_volumes = {
            name: normalize_mri(
                value,
                strategy="zscore",
                background_mask=np.zeros(shape, dtype=bool),
            )
            for name, value in arrays.items()
        }
        preprocessed = np.stack(
            [feature_volumes[name] for name in settings.MRI_MODALITIES], axis=0
        ).astype(np.float32, copy=False)
        indices = tuple(range(shape[2])) if slice_indices is None else tuple(int(i) for i in slice_indices)
        if not indices or any(i < 0 or i >= shape[2] for i in indices):
            raise ValueError("slice_indices must be nonempty and within volume boundaries")
        detections = tuple(
            self.detect_slice(
                patient_id,
                model_name,
                i,
                extract_slice_features(feature_volumes, i, foreground_masks=finite_masks),
            )
            for i in indices
        )
        positives = [item.slice_index for item in detections if item.predicted_class == 1]
        if not positives:
            return HybridPatientResult(patient_id, model_name, detections, "skipped_no_positive_slice", inference_seconds=time.perf_counter() - started)
        radius = self.neighbor_radius if neighbor_radius is None else neighbor_radius
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise ValueError("neighbor_radius must be a non-negative integer")
        selected = sorted({j for i in positives for j in range(max(0, i - radius), min(shape[2] - 1, i + radius) + 1)})
        model = self._load_unet()
        segments = []
        for index in selected:
            slice_started = time.perf_counter()
            probability, mask = predict_slice(model, prepare_axial_slice(preprocessed, index), threshold=self.threshold)
            probability_array = probability.detach().cpu().numpy()[0].astype(np.float32, copy=False)
            mask_array = mask.detach().cpu().numpy()[0].astype(bool, copy=False)
            overlay_path = None
            if self.overlay_dir is not None:
                destination = self.overlay_dir / patient_id / model_name / f"slice_{index:03d}.png"
                self._save_overlay(destination, arrays["t1c"][:, :, index], mask_array)
                overlay_path = str(destination)
            segments.append(SegmentedSliceResult(index, probability_array, mask_array, overlay_path, time.perf_counter() - slice_started))
        return HybridPatientResult(patient_id, model_name, detections, "completed", tuple(segments), tuple(selected), time.perf_counter() - started)
