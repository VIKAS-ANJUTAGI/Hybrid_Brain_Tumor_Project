"""Synthetic regression tests for the isolated 2D segmentation baseline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from models.segmentation import (
    BCEDiceLoss, BraTSSegmentationDataset, ManifestSplitError, PatientVolumes,
    UNet2D, load_patient_volumes, predict_slice, predict_volume,
    prepare_axial_slice, preprocess_patient, segmentation_metrics,
)
from models.segmentation.training import TrainingConfig, build_datasets, load_checkpoint, reconstruct_model, train_segmentation


MODALITIES = ("t1c", "t1n", "t2w", "t2f")


def _write_patient(root: Path, patient_id: str, *, invalid_label: bool = False, mismatch: bool = False) -> None:
    directory = root / patient_id
    directory.mkdir()
    affine = np.eye(4)
    for number, suffix in enumerate(MODALITIES, start=1):
        shape = (239, 240, 2) if mismatch and suffix == "t2w" else (240, 240, 2)
        values = np.full(shape, number, dtype=np.float32)
        values[0, 0, 0] = number + 1
        nib.save(nib.Nifti1Image(values, affine), directory / f"{patient_id}-{suffix}.nii.gz")
    segmentation = np.zeros((240, 240, 2), dtype=np.int16)
    segmentation[2, 3, 0] = 9 if invalid_label else 3
    nib.save(nib.Nifti1Image(segmentation, affine), directory / f"{patient_id}-seg.nii.gz")


def _manifest(train: list[str], validation: list[str], test: list[str] | None = None) -> dict[str, object]:
    return {"patient_assignments": {"train": train, "validation": validation, "test": test or []}}


def _logits_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask > 0, torch.full_like(mask, 100.0), torch.full_like(mask, -100.0))


class SegmentationMetricsTests(unittest.TestCase):
    def test_pixel_metrics_perfect_incorrect_and_empty_conventions(self) -> None:
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        self.assertEqual(segmentation_metrics(_logits_from_mask(target), target), {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0})
        self.assertEqual(segmentation_metrics(-_logits_from_mask(target), target), {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0})
        empty = torch.zeros_like(target)
        self.assertEqual(segmentation_metrics(torch.full_like(empty, -100.0), empty), {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0})
        self.assertEqual(segmentation_metrics(torch.full_like(empty, 100.0), empty), {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 1.0})
        self.assertEqual(segmentation_metrics(torch.full_like(target, -100.0), target), {"dice": 0.0, "iou": 0.0, "precision": 1.0, "recall": 0.0})

    def test_pixel_metrics_partial_batch_and_stability(self) -> None:
        target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
        predicted = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
        metrics = segmentation_metrics(_logits_from_mask(predicted), target)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.75)
        self.assertAlmostEqual(metrics["dice"], (0.5 + 1.0) / 2)
        self.assertTrue(all(np.isfinite(value) for value in segmentation_metrics(torch.tensor([[[[-1e30]]]]), torch.zeros(1, 1, 1, 1)).values()))


class SegmentationDatasetTests(unittest.TestCase):
    def test_manifest_validation_rejects_overlap_duplicates_and_malformed_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = _manifest(["train"], ["validation"], ["test"])
            for name, manifest in {
                "train_validation": _manifest(["same"], ["same"]),
                "train_test": _manifest(["same"], ["validation"], ["same"]),
                "validation_test": _manifest(["train"], ["same"], ["same"]),
                "duplicates": _manifest(["same", "same"], ["validation"]),
                "malformed": {"patient_assignments": {"train": ["train"], "validation": "bad", "test": []}},
                "missing": {"patient_assignments": {"train": ["train"], "validation": ["validation"]}},
            }.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ManifestSplitError):
                    build_datasets(path, dataset_root=root)
            for patient_id in ("train", "validation"):
                _write_patient(root, patient_id)
            path = root / "valid.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            train, validation = build_datasets(path, dataset_root=root)
            self.assertEqual(train.patient_ids, ("train",))
            self.assertEqual(validation.patient_ids, ("validation",))

    def test_lazy_slices_convert_masks_and_geometry_errors_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patient_id, validation_id = "BraTS-GLI-00001-000", "BraTS-GLI-00002-000"
            _write_patient(root, patient_id)
            _write_patient(root, validation_id)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest([patient_id], [validation_id], ["reserved"])), encoding="utf-8")
            dataset = BraTSSegmentationDataset(manifest, "train", dataset_root=root, cache_size=1)
            image, mask = dataset[0]
            self.assertEqual(tuple(image.shape), (4, 240, 240))
            self.assertEqual(tuple(mask.shape), (1, 240, 240))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(float(mask[0, 2, 3]), 1.0)
            with self.assertRaises(ManifestSplitError):
                BraTSSegmentationDataset(manifest, "test", dataset_root=root)
            bad = "BraTS-GLI-00003-000"
            _write_patient(root, bad, mismatch=True)
            bad_manifest = root / "bad.json"
            bad_manifest.write_text(json.dumps(_manifest([bad], [validation_id])), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shape"):
                BraTSSegmentationDataset(bad_manifest, "train", dataset_root=root)


class SegmentationInferenceTests(unittest.TestCase):
    def test_training_and_inference_preprocessing_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, patient_id, validation_id = Path(temporary), "patient", "validation"
            _write_patient(root, patient_id)
            _write_patient(root, validation_id)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest([patient_id], [validation_id])), encoding="utf-8")
            dataset = BraTSSegmentationDataset(manifest, "train", dataset_root=root)
            paths = {name: root / patient_id / f"{patient_id}-{name}.nii.gz" for name in MODALITIES}
            preprocessed = preprocess_patient(load_patient_volumes(paths))
            expected = prepare_axial_slice(preprocessed, 0)
            actual, _ = dataset[0]
            self.assertTrue(torch.equal(actual, expected))

    def test_inference_validates_geometry_and_returns_expected_dimensions(self) -> None:
        values = {name: np.ones((240, 240, 2), dtype=np.float32) for name in MODALITIES}
        affine = np.eye(4)
        patient = PatientVolumes(values, {name: affine for name in MODALITIES}, {name: (1.0, 1.0, 1.0) for name in MODALITIES}, {name: ("R", "A", "S") for name in MODALITIES})
        preprocessed = preprocess_patient(patient)
        model = UNet2D(base_width=1)
        probability, mask = predict_slice(model, prepare_axial_slice(preprocessed, 0))
        self.assertEqual(tuple(probability.shape), (1, 240, 240))
        self.assertEqual(tuple(mask.shape), (1, 240, 240))
        probabilities, masks = predict_volume(model, preprocessed)
        self.assertEqual(probabilities.shape, (1, 240, 240, 2))
        self.assertEqual(masks.shape, (1, 240, 240, 2))
        bad_affines = dict(patient.affines)
        bad_affines["t2w"] = np.diag([2.0, 1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "affine"):
            preprocess_patient(PatientVolumes(values, bad_affines, patient.spacings, patient.orientations))
        nonfinite = dict(values)
        nonfinite["t1c"] = values["t1c"].copy()
        nonfinite["t1c"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            preprocess_patient(PatientVolumes(nonfinite, patient.affines, patient.spacings, patient.orientations))


class SegmentationCheckpointTests(unittest.TestCase):
    def test_best_last_history_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_patient(root, "train")
            _write_patient(root, "validation")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest(["train"], ["validation"])), encoding="utf-8")
            best = root / "best_checkpoint.pt"
            first = train_segmentation(manifest, best, dataset_root=root, config=TrainingConfig(epochs=1, batch_size=2, patience=2))
            self.assertTrue(best.is_file())
            self.assertTrue(Path(first["last_checkpoint_path"]).is_file())
            self.assertTrue(Path(first["history_path"]).is_file())
            checkpoint = load_checkpoint(best)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(reconstruct_model(checkpoint).in_channels, 4)
            self.assertEqual(len(json.loads(Path(first["history_path"]).read_text(encoding="utf-8"))), 1)
            resumed = train_segmentation(manifest, best, dataset_root=root, config=TrainingConfig(epochs=2, batch_size=2, patience=2), resume_checkpoint=first["last_checkpoint_path"])
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertEqual(load_checkpoint(resumed["last_checkpoint_path"])["epoch"], 2)


class SegmentationModelTests(unittest.TestCase):
    def test_unet_forward_backward(self) -> None:
        model = UNet2D(base_width=1)
        images, targets = torch.randn(1, 4, 240, 240), torch.zeros(1, 1, 240, 240)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (1, 1, 240, 240))
        BCEDiceLoss()(logits, targets).backward()


if __name__ == "__main__":
    unittest.main()
