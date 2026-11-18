"""Shared MRI preparation and slice-wise inference for the 2D U-Net."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

from config import settings

from .unet import UNet2D

NORMALIZATION_CONFIG = {
    "strategy": "zscore",
    "background_mask_policy": "explicit_all_false_includes_all_finite_voxels",
}
_AFFINE_TOLERANCE = 1e-5


def _normalizer():
    """Load the Phase 2 normalizer without package-name ambiguity."""
    name = "segmentation_phase2_normalization"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached.normalize_mri
    path = Path(__file__).resolve().parents[2] / "preprocessing" / "normalization.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Phase 2 normalization module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.normalize_mri


@dataclass(frozen=True)
class PatientVolumes:
    """Four MRI volumes plus their geometry, in native BraTS orientation."""

    modalities: Mapping[str, np.ndarray]
    affines: Mapping[str, np.ndarray]
    spacings: Mapping[str, Sequence[float]]
    orientations: Mapping[str, Sequence[str | None]]


def _validate_modalities(modalities: Mapping[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    expected = tuple(settings.MRI_MODALITIES)
    if set(modalities) != set(expected):
        raise ValueError(f"modalities must contain exactly {expected}")
    values: list[np.ndarray] = []
    reference_shape: tuple[int, ...] | None = None
    for name in expected:
        volume = np.asarray(modalities[name])
        if volume.ndim != 3 or volume.shape[:2] != (240, 240):
            raise ValueError(f"{name}: expected a 3D native 240x240 volume, got {volume.shape}")
        if not np.issubdtype(volume.dtype, np.number) or np.iscomplexobj(volume):
            raise TypeError(f"{name}: volume must contain real numeric intensities")
        if not np.isfinite(volume).all():
            raise ValueError(f"{name}: volume contains non-finite values")
        if reference_shape is None:
            reference_shape = volume.shape
        elif volume.shape != reference_shape:
            raise ValueError(f"{name}: shape {volume.shape} does not match {reference_shape}")
        values.append(volume)
    return tuple(values)


def validate_patient_volumes(patient: PatientVolumes) -> tuple[np.ndarray, ...]:
    """Validate content and geometry before model preprocessing."""
    values = _validate_modalities(patient.modalities)
    expected = tuple(settings.MRI_MODALITIES)
    if set(patient.affines) != set(expected) or set(patient.spacings) != set(expected) or set(patient.orientations) != set(expected):
        raise ValueError("geometry must be provided for every MRI modality")
    reference = expected[0]
    reference_affine = np.asarray(patient.affines[reference], dtype=np.float64)
    reference_spacing = np.asarray(patient.spacings[reference], dtype=np.float64)
    reference_orientation = tuple(patient.orientations[reference])
    if reference_affine.shape != (4, 4) or reference_spacing.shape != (3,):
        raise ValueError("reference geometry has invalid dimensions")
    if not np.isfinite(reference_affine).all() or not np.isfinite(reference_spacing).all():
        raise ValueError("reference geometry contains non-finite values")
    for name in expected[1:]:
        affine = np.asarray(patient.affines[name], dtype=np.float64)
        spacing = np.asarray(patient.spacings[name], dtype=np.float64)
        orientation = tuple(patient.orientations[name])
        if affine.shape != (4, 4) or spacing.shape != (3,):
            raise ValueError(f"{name}: geometry has invalid dimensions")
        if not np.isfinite(affine).all() or not np.isfinite(spacing).all():
            raise ValueError(f"{name}: geometry contains non-finite values")
        if not np.allclose(affine, reference_affine, rtol=0.0, atol=_AFFINE_TOLERANCE):
            raise ValueError(f"{name}: affine does not match {reference}")
        if not np.allclose(spacing, reference_spacing, rtol=0.0, atol=_AFFINE_TOLERANCE):
            raise ValueError(f"{name}: spacing does not match {reference}")
        if orientation != reference_orientation:
            raise ValueError(f"{name}: orientation does not match {reference}")
    return values


def load_patient_volumes(modality_paths: Mapping[str, str | Path]) -> PatientVolumes:
    """Load four NIfTI modalities and retain header geometry for validation."""
    expected = tuple(settings.MRI_MODALITIES)
    if set(modality_paths) != set(expected):
        raise ValueError(f"modality_paths must contain exactly {expected}")
    modalities: dict[str, np.ndarray] = {}
    affines: dict[str, np.ndarray] = {}
    spacings: dict[str, tuple[float, ...]] = {}
    orientations: dict[str, tuple[str | None, ...]] = {}
    for name in expected:
        image = nib.load(str(modality_paths[name]))
        modalities[name] = np.asanyarray(image.dataobj)
        affines[name] = np.asarray(image.affine, dtype=np.float64)
        spacings[name] = tuple(float(value) for value in image.header.get_zooms()[:3])
        orientations[name] = tuple(nib.aff2axcodes(image.affine))
    return PatientVolumes(modalities, affines, spacings, orientations)


def normalize_and_stack_modalities(modalities: Mapping[str, np.ndarray]) -> np.ndarray:
    """Apply the exact training normalization and return ``(4, X, Y, Z)``."""
    values = _validate_modalities(modalities)
    normalize_mri = _normalizer()
    return np.stack(
        [
            normalize_mri(value, strategy="zscore", background_mask=np.zeros(value.shape, dtype=bool))
            for value in values
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def preprocess_patient(patient: PatientVolumes) -> np.ndarray:
    """Validate then normalize a full patient volume for U-Net inference."""
    validate_patient_volumes(patient)
    return normalize_and_stack_modalities(patient.modalities)


def prepare_axial_slice(preprocessed: np.ndarray, slice_index: int) -> torch.Tensor:
    """Return one native axial slice as contiguous ``float32 (4, 240, 240)``."""
    values = np.asarray(preprocessed)
    if values.ndim != 4 or values.shape[:3] != (4, 240, 240):
        raise ValueError(f"expected preprocessed volume shape (4, 240, 240, Z), got {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("preprocessed volume must contain only finite numeric values")
    if not 0 <= slice_index < values.shape[3]:
        raise IndexError(f"slice_index {slice_index} is outside [0, {values.shape[3]})")
    return torch.from_numpy(np.ascontiguousarray(values[:, :, :, slice_index], dtype=np.float32))


def predict_slice(model: UNet2D, image: torch.Tensor, *, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict one prepared slice and return ``(probability_map, binary_mask)``."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if tuple(image.shape) != (4, 240, 240):
        raise ValueError(f"expected image shape (4, 240, 240), got {tuple(image.shape)}")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(image.unsqueeze(0).to(dtype=torch.float32))).squeeze(0)
    model.train(was_training)
    return probabilities, probabilities >= threshold


def predict_volume(model: UNet2D, preprocessed: np.ndarray, *, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Predict a patient one axial slice at a time, returning ``(1, X, Y, Z)`` arrays."""
    slices = [predict_slice(model, prepare_axial_slice(preprocessed, index), threshold=threshold) for index in range(preprocessed.shape[3])]
    probabilities = np.stack([item[0].cpu().numpy() for item in slices], axis=-1)
    masks = np.stack([item[1].cpu().numpy() for item in slices], axis=-1)
    return probabilities.astype(np.float32, copy=False), masks.astype(bool, copy=False)
