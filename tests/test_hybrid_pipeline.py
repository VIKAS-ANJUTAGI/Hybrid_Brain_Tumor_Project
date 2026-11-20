"""Synthetic checks for detection and conditional segmentation integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.dataset_builder import feature_columns
from models.integration.hybrid_pipeline import (
    DetectionModel,
    HybridInferencePipeline,
    ModelArtifactError,
)
from models.integration.result_types import DetectionResult


class HybridPipelineTests(unittest.TestCase):
    def setUp(self):
        self.modalities = {name: np.zeros((240, 240, 5), dtype=np.float32) for name in ("t1c", "t1n", "t2w", "t2f")}
        self.pipeline = HybridInferencePipeline(neighbor_radius=1)
        self.pipeline._load_unet = Mock(return_value=object())

    def _run(self, labels, indices=(0, 1, 2, 3, 4), radius=1):
        self.pipeline.detect_slice = Mock(side_effect=lambda patient, model, index, features: DetectionResult(patient, model, index, float(labels[index]), labels[index], 0.0))
        prediction_calls = []

        def segmentation(model, image, *, threshold):
            index = int(image[0, 0, 0].item())
            prediction_calls.append(index)
            probabilities = torch.full((1, 240, 240), 0.0)
            if index == 2:
                probabilities[:, 0, 0] = 0.9
            return probabilities, probabilities >= threshold

        with patch("models.integration.hybrid_pipeline.extract_slice_features", side_effect=lambda vols, index, foreground_masks: np.full(16, index, dtype=np.float64)), patch(
            "models.integration.hybrid_pipeline.normalize_mri",
            side_effect=lambda volume, **kwargs: volume,
        ) as normalization, patch(
            "models.integration.hybrid_pipeline.prepare_axial_slice",
            side_effect=lambda volume, index: torch.full((4, 240, 240), float(index)),
        ), patch("models.integration.hybrid_pipeline.predict_slice", side_effect=segmentation):
            result = self.pipeline.run_patient("validation-patient", self.modalities, DetectionModel.LOGISTIC_REGRESSION, slice_indices=indices, neighbor_radius=radius)
        return result, prediction_calls, normalization

    def test_positive_detection_invokes_unet_for_all_selected_slices(self):
        result, calls, _ = self._run([0, 0, 1, 0, 0], radius=1)
        self.assertEqual(result.segmentation_status, "completed")
        self.assertEqual(result.selected_neighbor_slices, (1, 2, 3))
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual([x.slice_index for x in result.segmented_slices], [1, 2, 3])

    def test_negative_detection_skips_unet(self):
        result, calls, _ = self._run([0, 0, 0, 0, 0])
        self.assertEqual(result.segmentation_status, "skipped_no_positive_slice")
        self.assertEqual(result.segmented_slices, ())
        self.assertEqual(calls, [])
        self.pipeline._load_unet.assert_not_called()

    def test_neighbors_are_deduplicated_and_clamped_at_both_boundaries(self):
        result, calls, _ = self._run([1, 0, 0, 0, 1], radius=2)
        self.assertEqual(result.selected_neighbor_slices, (0, 1, 2, 3, 4))
        self.assertEqual(calls, [0, 1, 2, 3, 4])

    def test_each_neighbor_gets_independent_unet_prediction(self):
        result, calls, _ = self._run([0, 1, 0, 0, 0], radius=2)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(len({item.slice_index for item in result.segmented_slices}), 4)
        self.assertFalse(result.segmented_slices[0].binary_mask.any())
        self.assertTrue(result.segmented_slices[2].binary_mask[0, 0])

    def test_training_normalization_is_used_and_empty_masks_are_preserved(self):
        result, _, normalization = self._run([0, 1, 0, 0, 0], radius=0)
        self.assertEqual(normalization.call_count, 4)
        self.assertTrue(all(call.kwargs["strategy"] == "zscore" for call in normalization.call_args_list))
        self.assertTrue(all(not call.kwargs["background_mask"].any() for call in normalization.call_args_list))
        self.assertFalse(result.segmented_slices[0].binary_mask.any())

    def test_vqc_uses_its_model_specific_reducer(self):
        classifier = Mock()
        classifier.classes_ = np.array([0, 1])
        classifier.predict_proba.return_value = np.array([[0.25, 0.75]])
        reducer = Mock()
        reducer.transform.return_value = np.array([[4.0, 4.0]])
        self.pipeline._load_detector = Mock(return_value=((classifier, reducer), "model-digest"))
        result = self.pipeline.detect_slice("p", DetectionModel.VQC_2, 3, np.arange(16))
        reducer.transform.assert_called_once()
        classifier.predict_proba.assert_called_once_with(reducer.transform.return_value)
        self.assertEqual(result.predicted_class, 1)
        self.assertAlmostEqual(result.probability, 0.75)

    def test_classical_artifact_pipeline_applies_its_fitted_scaler(self):
        rng = np.random.RandomState(12)
        x = rng.normal(size=(30, 16))
        y = np.asarray([0, 1] * 15)
        feature_frame = pd.DataFrame(x, columns=feature_columns())
        trained = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(random_state=3))]).fit(feature_frame, y)
        with tempfile.TemporaryDirectory() as temporary:
            joblib.dump(trained, Path(temporary) / "logistic_regression.joblib")
            pipeline = HybridInferencePipeline(classical_artifact_dir=temporary)
            features = rng.normal(size=16)
            result = pipeline.detect_slice("synthetic-patient", DetectionModel.LOGISTIC_REGRESSION, 0, features)
            expected = trained.predict_proba(pd.DataFrame([features], columns=feature_columns()))[0, 1]
            self.assertAlmostEqual(result.probability, float(expected))
            self.assertIsInstance(pipeline._model.named_steps["scaler"], StandardScaler)

    def test_detection_cache_uses_patient_model_version_and_slice(self):
        classifier = Mock()
        classifier.classes_ = np.array([0, 1])
        classifier.predict_proba.return_value = np.array([[0.8, 0.2]])
        self.pipeline._load_detector = Mock(return_value=(classifier, "version-a"))
        features = np.zeros(16)
        self.pipeline.detect_slice("patient-a", DetectionModel.RANDOM_FOREST, 3, features)
        self.pipeline.detect_slice("patient-a", DetectionModel.RANDOM_FOREST, 3, features)
        self.assertEqual(classifier.predict_proba.call_count, 1)

    def test_missing_artifact_error_names_expected_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = HybridInferencePipeline(classical_artifact_dir=temporary)
            with self.assertRaisesRegex(ModelArtifactError, "logistic_regression.joblib"):
                pipeline._load_detector(DetectionModel.LOGISTIC_REGRESSION)

    def test_supported_model_set_excludes_postponed_vqc16(self):
        self.assertEqual(DetectionModel.SUPPORTED, ("logistic_regression", "random_forest", "vqc_2", "vqc_4", "vqc_8"))


if __name__ == "__main__":
    unittest.main()
