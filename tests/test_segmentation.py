"""Synthetic regression tests for the isolated 2D segmentation baseline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from unittest.mock import patch

from models.segmentation import (
    BCEDiceLoss, BraTSSegmentationDataset, ManifestSplitError, PatientVolumes,
    UNet2D, load_patient_volumes, predict_slice, predict_volume,
    prepare_axial_slice, preprocess_patient, segmentation_metrics,
)
from models.segmentation.training import (EMPTY_SLICE_FP_CONSTRAINT, PatientBatchSampler, TrainingConfig, _run_epoch,
    _select_checkpoint, build_datasets, load_checkpoint, reconstruct_model, train_segmentation)


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

    def test_cache_counters_reset_and_count_hits_misses_and_patient_switches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for patient in ("train-a", "train-b", "validation"):
                _write_patient(root, patient)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest(["train-a", "train-b"], ["validation"])), encoding="utf-8")
            dataset = BraTSSegmentationDataset(manifest, "train", dataset_root=root, cache_size=1)
            dataset[0]
            dataset[1]
            dataset[2]
            self.assertEqual((dataset.cache_misses, dataset.cache_hits), (2, 1))
            self.assertEqual(dataset.volume_load_count, 10)
            self.assertEqual(dataset.patient_switches, 1)
            self.assertGreaterEqual(dataset.data_loading_seconds, 0)
            dataset.reset_counters()
            self.assertEqual((dataset.cache_hits, dataset.cache_misses, dataset.volume_load_count, dataset.patient_switches), (0, 0, 0, 0))
            self.assertEqual(dataset.data_loading_seconds, 0)


class PatientSamplerTests(unittest.TestCase):
    def test_patient_sampler_covers_each_slice_once_and_keeps_incomplete_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for patient in ("train-a", "train-b", "validation"):
                _write_patient(root, patient)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest(["train-a", "train-b"], ["validation"])), encoding="utf-8")
            dataset = BraTSSegmentationDataset(manifest, "train", dataset_root=root)
            sampler = PatientBatchSampler(dataset, batch_size=3, seed=77)
            batches = list(sampler)
            flat = [index for batch in batches for index in batch]
            self.assertEqual(sorted(flat), list(range(len(dataset))))
            self.assertEqual(len(flat), len(set(flat)))
            self.assertEqual(len(batches), len(sampler))
            self.assertIn(2, [len(batch) for batch in batches])
            for batch in batches:
                self.assertEqual(len({dataset._index[index][0] for index in batch}), 1)

    def test_patient_order_is_reproducible_and_epoch_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patients = ["train-a", "train-b", "train-c", "validation"]
            for patient in patients:
                _write_patient(root, patient)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest(patients[:3], [patients[3]])), encoding="utf-8")
            dataset = BraTSSegmentationDataset(manifest, "train", dataset_root=root)
            first, second = PatientBatchSampler(dataset, 2, 93), PatientBatchSampler(dataset, 2, 93)
            first.set_epoch(4)
            second.set_epoch(4)
            epoch_four = list(first)
            self.assertEqual(epoch_four, list(second))
            first.set_epoch(5)
            self.assertNotEqual(epoch_four, list(first))


class _TinyBatchDataset(Dataset):
    def __init__(self):
        self.values = (1.0, 3.0, 9.0)
    def __len__(self):
        return len(self.values)
    def __getitem__(self, index):
        return torch.tensor([[[[self.values[index]]]]]), torch.zeros((1, 1, 1, 1))


class _ScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, images):
        return images * self.scale


class _MeanLoss(nn.Module):
    def forward(self, logits, targets):
        return logits.mean()


class BatchAccountingTests(unittest.TestCase):
    def test_batch_sampler_loader_uses_actual_sizes_for_weighted_average(self) -> None:
        dataset = _TinyBatchDataset()
        loader = DataLoader(dataset, batch_sampler=[[0, 1], [2]])
        self.assertIsNone(loader.batch_size)
        model = _ScaleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        statuses = []
        result = _run_epoch(model, loader, _MeanLoss(), optimizer=optimizer, threshold=0.5, status=lambda *args: statuses.append(args), epoch=1)
        self.assertAlmostEqual(result["loss"], 13.0 / 3.0)
        self.assertAlmostEqual(statuses[-1][3], 13.0 / 3.0)
        self.assertEqual(statuses[-1][:2], (2, 2))


class _PatientMetricDataset(Dataset):
    def __init__(self):
        # p1 contributes three tumor slices, p2 one tumor slice and p3 only empty slices.
        self.patient_ids = ("p1", "p2", "p3")
        self._index = [("p1", i) for i in range(3)] + [("p2", 0)] + [("p3", i) for i in range(2)]
        self.targets = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        self.predictions = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    def __len__(self): return len(self._index)
    def __getitem__(self, index):
        return torch.tensor([[[self.predictions[index] * 100 - 50]]]), torch.tensor([[[self.targets[index]]]])


class _ZeroLoss(nn.Module):
    def forward(self, logits, targets): return logits.sum() * 0


class PatientAwareSelectionTests(unittest.TestCase):
    def test_patient_macro_dice_weights_patients_not_slices_and_reports_counts(self):
        dataset = _PatientMetricDataset()
        result = _run_epoch(_ScaleModel(), DataLoader(dataset, batch_size=2), _ZeroLoss(), optimizer=None, threshold=0.5)
        # p1 mean=1, p2 mean=0; p3 has no tumor and is excluded: macro=(1+0)/2.
        self.assertEqual(result["patient_tumor_dice"], {"p1": 1.0, "p2": 0.0, "p3": None})
        self.assertEqual(result["patient_tumor_slice_counts"], {"p1": 3, "p2": 1, "p3": 0})
        self.assertEqual(result["patient_empty_slice_counts"], {"p1": 0, "p2": 0, "p3": 2})
        self.assertEqual(result["patients_without_tumor_slices"], ["p3"])
        self.assertEqual(result["patient_macro_tumor_dice"], 0.5)
        self.assertEqual(result["positive_slice_dice"], 0.75)

    def test_empty_fp_rate_and_constraint_boundary(self):
        dataset = _PatientMetricDataset()
        result = _run_epoch(_ScaleModel(), DataLoader(dataset, batch_size=2), _ZeroLoss(), optimizer=None, threshold=0.5)
        self.assertEqual(result["empty_slice_count"], 2)
        self.assertEqual(result["empty_slice_false_positive_count"], 1)
        self.assertEqual(result["empty_slice_false_positive_rate"], 0.5)
        self.assertFalse(result["constraint_satisfied"])
        self.assertEqual(EMPTY_SLICE_FP_CONSTRAINT.numerator, 85)
        self.assertEqual(EMPTY_SLICE_FP_CONSTRAINT.denominator, 384)
        self.assertLessEqual(85 / 384, float(EMPTY_SLICE_FP_CONSTRAINT))

    @staticmethod
    def _epoch(epoch, *, dice, fp, loss, eligible=None):
        satisfied = fp <= float(EMPTY_SLICE_FP_CONSTRAINT) if eligible is None else eligible
        return {"epoch": epoch, "validation_patient_macro_tumor_dice": dice,
            "validation_empty_slice_false_positive_rate": fp, "validation_constraint_satisfied": satisfied,
            "validation_loss": loss, "validation_dice": 0.6 + epoch / 100,
            "validation_patient_tumor_dice": {"patient-a": dice}, "segmentation_threshold": 0.5}

    def test_equal_patient_dice_uses_lower_validation_loss(self):
        chosen, metadata = _select_checkpoint([self._epoch(1, dice=0.7, fp=0.1, loss=0.3), self._epoch(2, dice=0.7, fp=0.1, loss=0.2)])
        self.assertEqual(chosen["epoch"], 2)
        self.assertTrue(metadata["constraint_satisfied"])
        self.assertFalse(metadata["fallback_used"])

    def test_no_eligible_epoch_falls_back_to_lowest_fp_then_dice(self):
        chosen, metadata = _select_checkpoint([self._epoch(1, dice=0.9, fp=0.30, loss=0.3), self._epoch(2, dice=0.6, fp=0.25, loss=0.2), self._epoch(3, dice=0.8, fp=0.25, loss=0.4)])
        self.assertEqual(chosen["epoch"], 3)
        self.assertTrue(metadata["fallback_used"])
        self.assertFalse(metadata["constraint_satisfied"])

    def test_checkpoint_selection_metadata_and_legacy_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            model = UNet2D(base_width=1)
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            record = self._epoch(4, dice=0.75, fp=0.1, loss=0.2)
            record.update({"validation_patient_tumor_slice_counts": {"patient-a": 3},
                           "validation_empty_slice_false_positive_count": 2,
                           "validation_iou": 0.6, "validation_precision": 0.8,
                           "validation_recall": 0.7})
            _, metadata = _select_checkpoint([record])
            # Exercise serialization by writing the actual metadata structure alongside a legacy-shaped payload.
            torch.save({"model_state_dict": model.state_dict(), "selection_metadata": metadata,
                        "history": [record], "epoch": 4, "metric_threshold": 0.5}, path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["selection_metadata"]["epoch"], 4)
            self.assertEqual(loaded["selection_metadata"]["patient_level_dice_values"], {"patient-a": 0.75})
            legacy = Path(temporary) / "legacy.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, legacy)
            self.assertEqual(load_checkpoint(legacy)["epoch"], 1)

    def test_existing_baseline_experiment_is_preserved(self):
        baseline = Path(__file__).resolve().parents[1] / "results" / "segmentation_runs" / "unet_baseline_20261120T173712Z"
        self.assertTrue(baseline.is_dir())
        self.assertTrue((baseline / "best_checkpoint.pt").is_file())


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
            history = json.loads(Path(first["history_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(history), 1)
            for key in ("validation_dice", "validation_patient_macro_tumor_dice", "validation_patient_tumor_dice",
                        "validation_empty_slice_false_positive_count", "validation_empty_slice_false_positive_rate",
                        "validation_constraint_satisfied", "validation_loss", "validation_iou",
                        "validation_precision", "validation_recall"):
                self.assertIn(key, history[0])
            for key in ("selection_criterion", "patient_macro_tumor_dice", "empty_slice_false_positive_rate",
                        "constraint_value", "constraint_satisfied", "all_slice_dice", "validation_loss",
                        "epoch", "patient_level_dice_values", "segmentation_threshold"):
                self.assertIn(key, checkpoint["selection_metadata"])
            last_metadata = load_checkpoint(first["last_checkpoint_path"])["selection_metadata"]
            self.assertEqual(last_metadata, checkpoint["selection_metadata"])
            status_path = root / "training_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "COMPLETED")
            self.assertTrue(status["run_id"])
            self.assertEqual(status["pid"], status["process_id"])
            self.assertTrue(status["last_successful_checkpoint"])
            self.assertIn("last_updated_time", status)
            self.assertFalse(status_path.with_suffix(".json.tmp").exists())
            resumed = train_segmentation(manifest, best, dataset_root=root, config=TrainingConfig(epochs=2, batch_size=2, patience=2), resume_checkpoint=first["last_checkpoint_path"])
            self.assertEqual(resumed["epochs_completed"], 2)
            self.assertEqual(load_checkpoint(resumed["last_checkpoint_path"])["epoch"], 2)

    def test_training_exception_persists_failed_status_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_patient(root, "train")
            _write_patient(root, "validation")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(_manifest(["train"], ["validation"])), encoding="utf-8")
            with patch("models.segmentation.training._run_epoch", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    train_segmentation(manifest, root / "best.pt", dataset_root=root, config=TrainingConfig(epochs=1, batch_size=2))
            status_path = root / "training_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertEqual(status["error_message"], "synthetic failure")
            self.assertFalse(status_path.with_suffix(".json.tmp").exists())


class SegmentationModelTests(unittest.TestCase):
    def test_unet_forward_backward(self) -> None:
        model = UNet2D(base_width=1)
        images, targets = torch.randn(1, 4, 240, 240), torch.zeros(1, 1, 240, 240)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (1, 1, 240, 240))
        BCEDiceLoss()(logits, targets).backward()


if __name__ == "__main__":
    unittest.main()
