"""Dashboard API and visualization checks using synthetic validation data."""

from __future__ import annotations

import csv
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from dashboard import app
from models.integration import DetectionModel
from models.integration.result_types import DetectionResult, HybridPatientResult, SegmentedSliceResult


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.patient_id = "validation-patient"
        shape = (240, 240, 3)
        volumes = {name: np.zeros(shape, dtype=np.float32) for name in ("t1c", "t1n", "t2w", "t2f")}
        volumes["seg"] = np.zeros(shape, dtype=np.uint8)
        volumes["seg"][100:110, 100:110, 1] = 1
        self.patient = SimpleNamespace(volumes=volumes, dimensions_compatible=True, segmentation_labels_valid=True)
        self.loader = Mock(return_value=self.patient)
        self.pipeline = Mock()
        self.service = app.DashboardService(self.pipeline, self.loader)
        self.service.allowed_patients = [self.patient_id]
        detections = tuple(DetectionResult(self.patient_id, DetectionModel.LOGISTIC_REGRESSION, i, 0.8 if i == 1 else 0.1, int(i == 1), 0) for i in range(3))
        probability = np.zeros((240, 240), dtype=np.float32)
        probability[101:106, 101:106] = .9
        segment = SegmentedSliceResult(1, probability, probability >= .5)
        self.result = HybridPatientResult(self.patient_id, DetectionModel.LOGISTIC_REGRESSION, detections, "completed", (segment,), (1,))
        self.pipeline.threshold = .5
        self.pipeline.run_patient.return_value = self.result

    def test_patient_allowlist_uses_only_validation_csv_checkpoint_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "validation.csv").write_text("patient_id,slice_index\nval-a,0\nval-b,0\n", encoding="utf-8")
            (root / "test.csv").write_text("patient_id,slice_index\nsecret-test,0\n", encoding="utf-8")
            (root / "checkpoint.pt").touch()
            with patch.object(app, "load_checkpoint", return_value={"validation_patient_ids": ["val-b", "secret-test"]}):
                self.assertEqual(app.validation_patients(root, root / "checkpoint.pt"), ["val-b"])

    def test_model_selection_and_patient_loading(self):
        response = self.service.analyze(self.patient_id, DetectionModel.LOGISTIC_REGRESSION)
        self.assertEqual(response["model_label"], "Logistic Regression")
        self.assertEqual(response["split"], "Validation")
        self.assertEqual((response["positive_slices"], response["negative_slices"]), (1, 2))
        self.pipeline.run_patient.assert_called_once()
        self.assertEqual(self.loader.call_args.args, (self.patient_id,))
        with self.assertRaisesRegex(ValueError, "supported detection model"):
            self.service.analyze(self.patient_id, "vqc_16")

    def test_slice_navigation_visualizes_modality_and_rejects_invalid_index(self):
        png = self.service.image(self.patient_id, "t2f", 2)
        self.assertTrue(png.startswith(b"\x89PNG"))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.service.image(self.patient_id, "t1c", 3)
        with self.assertRaisesRegex(ValueError, "Unknown MRI modality"):
            self.service.image(self.patient_id, "flair", 0)

    def test_detection_prediction_and_conditional_segmentation_visualizations(self):
        response = self.service.analyze(self.patient_id, DetectionModel.LOGISTIC_REGRESSION)
        self.assertEqual(response["detections"][1]["prediction"], 1)
        for kind in ("overlay", "prediction", "probability", "ground_truth"):
            png = self.service.segmentation_image(self.patient_id, DetectionModel.LOGISTIC_REGRESSION, 1, kind)
            self.assertTrue(png.startswith(b"\x89PNG"), kind)
        with self.assertRaisesRegex(ValueError, "not selected"):
            self.service.segmentation_image(self.patient_id, DetectionModel.LOGISTIC_REGRESSION, 0, "prediction")

    def test_negative_detection_skips_segmentation(self):
        negative = HybridPatientResult(self.patient_id, DetectionModel.RANDOM_FOREST, tuple(DetectionResult(self.patient_id, DetectionModel.RANDOM_FOREST, i, .1, 0, 0) for i in range(3)), "skipped_no_positive_slice")
        self.pipeline.run_patient.return_value = negative
        response = self.service.analyze(self.patient_id, DetectionModel.RANDOM_FOREST)
        self.assertEqual(response["segmentation_status"], "skipped_no_positive_slice")
        self.assertEqual(response["segmented_slice_indices"], [])

    def test_invalid_patient_is_rejected_before_disk_loader(self):
        with self.assertRaisesRegex(ValueError, "permitted validation cohort"):
            self.service.analyze("test-patient", DetectionModel.LOGISTIC_REGRESSION)
        self.loader.assert_not_called()

    def test_missing_artifact_is_reported_without_absolute_path(self):
        with patch.object(app, "service", self.service):
            self.pipeline.run_patient.side_effect = FileNotFoundError("C:/private/checkpoint.pt")
            server = app.create_server(port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}/api/analyze/{self.patient_id}/logistic_regression"
                with self.assertRaises(HTTPError) as caught:
                    urlopen(url)
                body = caught.exception.read().decode()
                self.assertEqual(caught.exception.code, 503)
                self.assertIn("artifact is unavailable", body)
                self.assertNotIn("private", body)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_dashboard_startup_and_static_entry_point(self):
        with patch.object(app, "service", self.service):
            server = app.create_server(port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_address[1]}/") as response:
                    page = response.read().decode()
                    self.assertEqual(response.status, 200)
                    self.assertIn("Patient review", page)
                    self.assertIn("VQC-16 · unavailable", page)
                with urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/patients") as response:
                    self.assertIn("validation-patient", response.read().decode())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
