"""Unit tests for patient volume loading and integrity validation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np

from backend.data_loading import (
    EXPECTED_SEGMENTATION_LABELS,
    LoadedPatient,
    _resolve_volume_path,
    load_patient,
)
from config import settings


class MockImageFileError(Exception):
    pass


def _create_mock_nibabel() -> ModuleType:
    mock_nib = ModuleType("nibabel")
    mock_filebased = ModuleType("nibabel.filebasedimages")
    mock_filebased.ImageFileError = MockImageFileError
    mock_nib.filebasedimages = mock_filebased
    return mock_nib


class DataLoadingTests(unittest.TestCase):
    def test_resolve_volume_path_finds_single_nii_gz(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            patient_dir = Path(temp_dir) / "BraTS-GLI-00001-000"
            patient_dir.mkdir()
            target_file = patient_dir / "BraTS-GLI-00001-000-t1c.nii.gz"
            target_file.touch()

            resolved = _resolve_volume_path(patient_dir, "t1c")
            self.assertEqual(resolved, target_file)

    def test_resolve_volume_path_raises_on_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            patient_dir = Path(temp_dir) / "BraTS-GLI-00001-000"
            patient_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                _resolve_volume_path(patient_dir, "t1c")

    def test_resolve_volume_path_raises_on_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            patient_dir = Path(temp_dir) / "BraTS-GLI-00001-000"
            patient_dir.mkdir()
            (patient_dir / "BraTS-GLI-00001-000-t1c.nii.gz").touch()
            (patient_dir / "BraTS-GLI-00001-000-t1c.nii").touch()

            with self.assertRaises(ValueError):
                _resolve_volume_path(patient_dir, "t1c")

    def test_load_patient_rejects_invalid_patient_id(self) -> None:
        mock_nib = _create_mock_nibabel()
        with patch.dict(sys.modules, {"nibabel": mock_nib, "nibabel.filebasedimages": mock_nib.filebasedimages}):
            for invalid_id in ("", "..", ".", "foo/bar"):
                with self.subTest(invalid_id=invalid_id):
                    with self.assertRaises(ValueError):
                        load_patient(invalid_id)

    def test_load_patient_rejects_missing_patient_dir(self) -> None:
        mock_nib = _create_mock_nibabel()
        with patch.dict(sys.modules, {"nibabel": mock_nib, "nibabel.filebasedimages": mock_nib.filebasedimages}):
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch.object(settings, "DATASET_ROOT", Path(temp_dir)):
                    with self.assertRaises(FileNotFoundError):
                        load_patient("BraTS-GLI-99999-000")

    def test_load_patient_successful_load(self) -> None:
        mock_nib = _create_mock_nibabel()
        shape = (10, 10, 10)
        dummy_data = np.zeros(shape, dtype=np.float32)

        def mock_load(path_str: str) -> MagicMock:
            img = MagicMock()
            img.dataobj = dummy_data
            return img

        mock_nib.load = mock_load

        with patch.dict(sys.modules, {"nibabel": mock_nib, "nibabel.filebasedimages": mock_nib.filebasedimages}):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                patient_id = "BraTS-GLI-00001-000"
                patient_dir = root / patient_id
                patient_dir.mkdir()
                for suffix in (*settings.MRI_MODALITIES, settings.SEGMENTATION_MASK_SUFFIX):
                    (patient_dir / f"{patient_id}-{suffix}.nii.gz").touch()

                with patch.object(settings, "DATASET_ROOT", root):
                    patient = load_patient(patient_id)
                    self.assertEqual(patient.patient_id, patient_id)
                    self.assertTrue(patient.dimensions_compatible)
                    self.assertTrue(patient.segmentation_labels_valid)
                    self.assertEqual(patient.segmentation_labels, (0,))

    def test_loaded_patient_properties_valid(self) -> None:
        patient = LoadedPatient(
            patient_id="BraTS-GLI-00001-000",
            volumes={"t1c": np.zeros((10, 10, 10))},
            source_paths={"t1c": Path("/fake/t1c.nii.gz")},
            shape_mismatches={},
            non_3d_volumes=(),
            segmentation_labels=(0, 1, 2, 3),
            invalid_segmentation_labels=(),
            segmentation_has_non_finite_values=False,
        )
        self.assertTrue(patient.dimensions_compatible)
        self.assertTrue(patient.segmentation_labels_valid)

    def test_loaded_patient_properties_invalid_shape_or_labels(self) -> None:
        patient_mismatch = LoadedPatient(
            patient_id="BraTS-GLI-00001-000",
            volumes={},
            source_paths={},
            shape_mismatches={"t2w": (10, 10, 5)},
            non_3d_volumes=(),
            segmentation_labels=(0, 1),
            invalid_segmentation_labels=(),
            segmentation_has_non_finite_values=False,
        )
        self.assertFalse(patient_mismatch.dimensions_compatible)

        patient_invalid_labels = LoadedPatient(
            patient_id="BraTS-GLI-00001-000",
            volumes={},
            source_paths={},
            shape_mismatches={},
            non_3d_volumes=(),
            segmentation_labels=(0, 1, 4),
            invalid_segmentation_labels=(4,),
            segmentation_has_non_finite_values=False,
        )
        self.assertFalse(patient_invalid_labels.segmentation_labels_valid)

        patient_non_finite = LoadedPatient(
            patient_id="BraTS-GLI-00001-000",
            volumes={},
            source_paths={},
            shape_mismatches={},
            non_3d_volumes=(),
            segmentation_labels=(0, 1),
            invalid_segmentation_labels=(),
            segmentation_has_non_finite_values=True,
        )
        self.assertFalse(patient_non_finite.segmentation_labels_valid)


if __name__ == "__main__":
    unittest.main()
