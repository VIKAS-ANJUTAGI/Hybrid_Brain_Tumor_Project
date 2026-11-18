"""Lazy, geometry-checked BraTS axial-slice dataset for binary segmentation."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from config import settings

from .inference import normalize_and_stack_modalities

Split = Literal["train", "validation"]
_SUPPORTED_LABELS = frozenset({0, 1, 2, 3})
_EXTENSIONS = (".nii.gz", ".nii")


class ManifestSplitError(ValueError):
    """Raised when an operation could expose manifest test patients."""


@dataclass(frozen=True)
class _PatientRecord:
    patient_id: str
    paths: Mapping[str, Path]
    slice_count: int


def validate_patient_assignments(manifest: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Validate the manifest's patient-level split boundary without reading data."""
    assignments = manifest.get("patient_assignments")
    if not isinstance(assignments, dict):
        raise ManifestSplitError("manifest must contain patient_assignments")
    validated: dict[str, tuple[str, ...]] = {}
    for split in ("train", "validation", "test"):
        ids = assignments.get(split)
        if not isinstance(ids, list) or not all(isinstance(patient_id, str) for patient_id in ids):
            raise ManifestSplitError(f"manifest {split!r} assignment must be a list of patient IDs")
        if any(not patient_id or Path(patient_id).name != patient_id or patient_id in {".", ".."} for patient_id in ids):
            raise ManifestSplitError(f"manifest {split!r} contains an invalid patient ID")
        if len(ids) != len(set(ids)):
            raise ManifestSplitError(f"manifest {split!r} contains duplicate patient IDs")
        validated[split] = tuple(ids)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(validated[left]) & set(validated[right])
        if overlap:
            raise ManifestSplitError(f"manifest {left!r} IDs overlap {right!r} IDs: {sorted(overlap)}")
    return validated


def _volume_path(patient_dir: Path, suffix: str) -> Path:
    matches = [patient_dir / f"{patient_dir.name}-{suffix}{ext}" for ext in _EXTENSIONS]
    found = [path for path in matches if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {suffix!r} NIfTI file for {patient_dir.name}; "
            f"found {len(found)}"
        )
    return found[0]


def _header_geometry(image: nib.spatialimages.SpatialImage) -> tuple[tuple[int, ...], np.ndarray, tuple[float, ...], tuple[str | None, ...]]:
    shape = tuple(int(value) for value in image.shape)
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    orientation = tuple(nib.aff2axcodes(affine))
    return shape, affine, spacing, orientation


def _validate_headers(patient_id: str, paths: Mapping[str, Path]) -> int:
    """Validate geometry using headers only; voxel data remains lazy."""
    reference_name = settings.MRI_MODALITIES[0]
    try:
        reference = nib.load(str(paths[reference_name]))
    except Exception as exc:  # nibabel has several format-specific exceptions
        raise OSError(f"Unable to read NIfTI header for {paths[reference_name]}: {exc}") from exc
    reference_shape, reference_affine, reference_spacing, reference_orientation = _header_geometry(reference)
    if len(reference_shape) != 3 or reference_shape[:2] != (240, 240):
        raise ValueError(f"{patient_id}: expected 3D native 240x240 volumes, got {reference_shape}")
    if not np.isfinite(reference_affine).all() or not np.isfinite(reference_spacing).all():
        raise ValueError(f"{patient_id}: reference geometry contains non-finite values")

    for name, path in paths.items():
        try:
            image = nib.load(str(path))
        except Exception as exc:
            raise OSError(f"Unable to read NIfTI header for {path}: {exc}") from exc
        shape, affine, spacing, orientation = _header_geometry(image)
        if len(shape) != 3 or shape != reference_shape:
            raise ValueError(f"{patient_id}: {name} shape {shape} does not match {reference_shape}")
        if not np.isfinite(affine).all() or not np.isfinite(spacing).all():
            raise ValueError(f"{patient_id}: {name} geometry contains non-finite values")
        if not np.allclose(affine, reference_affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"{patient_id}: {name} affine does not match {reference_name}")
        if not np.allclose(spacing, reference_spacing, rtol=0.0, atol=1e-5):
            raise ValueError(f"{patient_id}: {name} spacing does not match {reference_name}")
        if orientation != reference_orientation:
            raise ValueError(f"{patient_id}: {name} orientation does not match {reference_name}")
    return reference_shape[2]


class BraTSSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Patient-grouped lazy axial slices with four MRI channels and binary masks.

    Images are normalized independently per patient and modality with the exact
    Phase 2 z-score policy: all finite voxels, including zeros, are included.
    The only cached objects are normalized volumes for recently requested
    patients; no patient data is read while the dataset index is constructed.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Split,
        *,
        dataset_root: str | Path | None = None,
        cache_size: int = 1,
        patient_ids: Iterable[str] | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ManifestSplitError("Segmentation accepts only 'train' or 'validation', never 'test'")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assignments = validate_patient_assignments(manifest)
        allowed = assignments[split]
        test_ids = set(assignments["test"])
        selected = list(allowed) if patient_ids is None else list(patient_ids)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("patient_ids must be a non-empty, duplicate-free sequence")
        prohibited = set(selected) & test_ids
        outside_split = set(selected) - set(allowed)
        if prohibited or outside_split:
            raise ManifestSplitError("patient_ids must be drawn exclusively from the selected non-test manifest split")

        root = Path(settings.DATASET_ROOT) if dataset_root is None else Path(dataset_root)
        self.split = split
        self.manifest_path = manifest_file
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._records: dict[str, _PatientRecord] = {}
        self._index: list[tuple[str, int]] = []
        suffixes = (*settings.MRI_MODALITIES, settings.SEGMENTATION_MASK_SUFFIX)
        for patient_id in selected:
            if Path(patient_id).name != patient_id or patient_id in {".", ".."}:
                raise ValueError(f"invalid patient ID: {patient_id!r}")
            patient_dir = root / patient_id
            if not patient_dir.is_dir():
                raise FileNotFoundError(f"patient directory not found: {patient_dir}")
            paths = {suffix: _volume_path(patient_dir, suffix) for suffix in suffixes}
            depth = _validate_headers(patient_id, paths)
            self._records[patient_id] = _PatientRecord(patient_id, paths, depth)
            self._index.extend((patient_id, slice_index) for slice_index in range(depth))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def _load_patient(self, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
        if patient_id in self._cache:
            self._cache.move_to_end(patient_id)
            return self._cache[patient_id]
        record = self._records[patient_id]
        modalities = [np.asanyarray(nib.load(str(record.paths[name])).dataobj) for name in settings.MRI_MODALITIES]
        segmentation = np.asanyarray(nib.load(str(record.paths[settings.SEGMENTATION_MASK_SUFFIX])).dataobj)
        if not np.isfinite(segmentation).all():
            raise ValueError(f"{patient_id}: segmentation contains non-finite values")
        labels = np.unique(segmentation)
        if (not np.all(np.equal(labels, np.floor(labels)))) or not set(labels.astype(np.int64)).issubset(_SUPPORTED_LABELS):
            raise ValueError(f"{patient_id}: segmentation contains unsupported labels {labels.tolist()}")
        for modality, name in zip(modalities, settings.MRI_MODALITIES):
            if not np.issubdtype(modality.dtype, np.number) or np.iscomplexobj(modality):
                raise TypeError(f"{patient_id}: {name} must contain real numeric intensities")
        normalized = normalize_and_stack_modalities(dict(zip(settings.MRI_MODALITIES, modalities)))
        mask = (segmentation > 0).astype(np.float32, copy=False)[np.newaxis, ...]
        if self.cache_size:
            self._cache[patient_id] = (normalized, mask)
            self._cache.move_to_end(patient_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return normalized, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_id, slice_index = self._index[index]
        images, mask = self._load_patient(patient_id)
        # BraTS arrays are (x, y, z); slice the native axial z axis.
        return (
            torch.from_numpy(np.ascontiguousarray(images[:, :, :, slice_index])),
            torch.from_numpy(np.ascontiguousarray(mask[:, :, :, slice_index])),
        )
