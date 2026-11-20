"""Small standard-library HTTP dashboard backed by the existing ML pipeline."""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_loading import load_patient
from config import settings
from models.integration import DetectionModel, HybridInferencePipeline
from models.segmentation.training import load_checkpoint

RESULTS = ROOT / "results"
DATASET_DIR = RESULTS / "phase2_full_dataset_seed42"
CLASSICAL_REPORT = RESULTS / "classical_baselines_seed42_20261118T170148_799890Z" / "validation_results.json"
QUANTUM_REPORTS = (
    RESULTS / "quantum_models_seed42_20261118T170214_243106Z" / "quantum_validation_results.json",
    RESULTS / "quantum_models_seed42_20261118T171424_499965Z" / "quantum_validation_results.json",
)
SEGMENTATION_REPORT = RESULTS / "segmentation_runs/unet_baseline_20261120T173712Z/segmentation_metrics.json"
UNET_CHECKPOINT = RESULTS / "segmentation_runs/unet_baseline_20261120T173712Z/best_checkpoint.pt"
MODEL_LABELS = {
    DetectionModel.LOGISTIC_REGRESSION: "Logistic Regression",
    DetectionModel.RANDOM_FOREST: "Random Forest",
    DetectionModel.VQC_2: "VQC-2",
    DetectionModel.VQC_4: "VQC-4",
    DetectionModel.VQC_8: "VQC-8",
}
MODALITIES = {"t1c": "T1C", "t1n": "T1N", "t2w": "T2W", "t2f": "T2F"}


def validation_patients(dataset_dir: Path = DATASET_DIR, checkpoint: Path = UNET_CHECKPOINT) -> list[str]:
    """Use only validation.csv and the checkpoint's validation cohort as an allowlist."""
    validation_csv = dataset_dir / "validation.csv"
    if not validation_csv.is_file():
        raise FileNotFoundError("Validation patient list is unavailable")
    with validation_csv.open(newline="", encoding="utf-8") as stream:
        csv_patients = {row["patient_id"] for row in csv.DictReader(stream)}
    if not checkpoint.is_file():
        raise FileNotFoundError("Segmentation checkpoint is unavailable")
    saved = load_checkpoint(checkpoint)
    checkpoint_patients = set(saved.get("validation_patient_ids", ()))
    allowed = sorted(csv_patients & checkpoint_patients)
    if not allowed:
        raise RuntimeError("No patients are shared by the detection and segmentation validation cohorts")
    return allowed


def _json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def comparison_results() -> dict:
    """Return recorded validation values only; absent metrics remain null."""
    models: dict[str, dict] = {}
    classical = _json_file(CLASSICAL_REPORT).get("validation_metrics", {})
    for name in (DetectionModel.LOGISTIC_REGRESSION, DetectionModel.RANDOM_FOREST):
        metric = classical.get(name)
        models[name] = _standard_metrics(metric)
    training_sizes = {
        DetectionModel.LOGISTIC_REGRESSION: _json_file(CLASSICAL_REPORT).get("split_summaries", {}).get("train", {}).get("rows"),
        DetectionModel.RANDOM_FOREST: _json_file(CLASSICAL_REPORT).get("split_summaries", {}).get("train", {}).get("rows"),
    }
    for report_path in QUANTUM_REPORTS:
        report = _json_file(report_path)
        for key, metric in report.get("quantum_validation_metrics", {}).items():
            qubits = key.removeprefix("vqc_").removesuffix("qubit")
            name = f"vqc_{qubits}"
            models[name] = _standard_metrics(metric)
            training_sizes[name] = report.get("training_hyperparameters", {}).get("sample_limit") or report.get("split_summaries", {}).get("train", {}).get("rows")
    counts = _json_file(CLASSICAL_REPORT).get("split_summaries", {})
    for name, result in models.items():
        result["validation_slices"] = counts.get("validation", {}).get("rows")
        result["training_slices"] = training_sizes.get(name)
        result["experiment"] = "preliminary"
    return {
        "models": models,
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "notes": ["Preliminary experiments; training sample sizes differ between model runs.", "All reported scores use validation.csv; no test split was read."],
    }


def _standard_metrics(values: dict | None) -> dict:
    if not values:
        return {key: None for key in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    return {"accuracy": values.get("accuracy"), "precision": values.get("precision"), "recall": values.get("recall"), "f1": values.get("f1_score"), "roc_auc": values.get("roc_auc")}


def segmentation_metrics() -> dict:
    data = _json_file(SEGMENTATION_REPORT)
    values = data.get("metrics", {})
    return {
        "scope": "validation",
        "aggregation": data.get("metric_aggregation"),
        "slices": data.get("validation_slice_count"),
        "metrics": {"dice": values.get("dice"), "iou": values.get("iou"), "pixel_precision": values.get("precision"), "pixel_recall": values.get("recall")},
        "training": None,
        "final_test": None,
    }


def png_data(image: np.ndarray, mask: np.ndarray | None = None, *, gray: bool = True) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(image)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(image, cmap="gray" if gray else None)
    if mask is not None and np.any(mask):
        ax.imshow(np.ma.masked_where(~np.asarray(mask, dtype=bool), mask), cmap="autumn", alpha=0.58, vmin=0, vmax=1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buffer.getvalue()


class DashboardService:
    def __init__(self, pipeline: HybridInferencePipeline | None = None, patient_loader=load_patient):
        self.pipeline = pipeline or HybridInferencePipeline(unet_checkpoint=UNET_CHECKPOINT)
        self.patient_loader = patient_loader
        self.allowed_patients: list[str] | None = None
        self.patients: dict[str, object] = {}
        self.results: dict[tuple[str, str], object] = {}

    def patient_ids(self) -> list[str]:
        if self.allowed_patients is None:
            self.allowed_patients = validation_patients()
        return self.allowed_patients

    def get_patient(self, patient_id: str):
        if patient_id not in self.patient_ids():
            raise ValueError("Patient is not in the permitted validation cohort")
        if patient_id not in self.patients:
            patient = self.patient_loader(patient_id)
            if not patient.dimensions_compatible or not patient.segmentation_labels_valid:
                raise ValueError("Patient MRI volumes or evaluation mask failed validation")
            self.patients[patient_id] = patient
        return self.patients[patient_id]

    def analyze(self, patient_id: str, model_name: str) -> dict:
        if model_name not in DetectionModel.SUPPORTED:
            raise ValueError("Choose a supported detection model")
        patient = self.get_patient(patient_id)
        key = (patient_id, model_name)
        if key not in self.results:
            modalities = {name: patient.volumes[name] for name in settings.MRI_MODALITIES}
            self.results[key] = self.pipeline.run_patient(patient_id, modalities, model_name)
        result = self.results[key]
        detections = [{"slice_index": d.slice_index, "probability": d.probability, "prediction": d.predicted_class} for d in result.detections]
        segments = {item.slice_index: item for item in result.segmented_slices}
        return {
            "patient_id": patient_id,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "split": "Validation",
            "slice_count": len(detections),
            "detections": detections,
            "positive_slices": sum(d["prediction"] == 1 for d in detections),
            "negative_slices": sum(d["prediction"] == 0 for d in detections),
            "segmentation_status": result.segmentation_status,
            "selected_neighbor_slices": list(result.selected_neighbor_slices),
            "segmented_slices": [
                {
                    "slice_index": index,
                    "max_probability": float(np.max(segment.probability_map)),
                    "mask_nonempty": bool(np.any(segment.binary_mask)),
                    "selected_as_neighbor": not bool(next((d.predicted_class for d in result.detections if d.slice_index == index), 0)),
                }
                for index, segment in segments.items()
            ],
            "segmented_slice_indices": list(segments),
            "threshold": self.pipeline.threshold,
        }

    def image(self, patient_id: str, modality: str, index: int) -> bytes:
        if modality not in MODALITIES:
            raise ValueError("Unknown MRI modality")
        patient = self.get_patient(patient_id)
        volume = patient.volumes[modality]
        if index < 0 or index >= volume.shape[2]:
            raise ValueError("Slice index is outside this MRI volume")
        return png_data(volume[:, :, index])

    def segmentation_image(self, patient_id: str, model_name: str, index: int, kind: str) -> bytes:
        patient = self.get_patient(patient_id)
        self.analyze(patient_id, model_name)
        result = self.results[(patient_id, model_name)]
        segment = next((s for s in result.segmented_slices if s.slice_index == index), None)
        if segment is None:
            raise ValueError("This slice was not selected for segmentation")
        if kind == "prediction":
            return png_data(segment.binary_mask.astype(np.uint8))
        if kind == "probability":
            return png_data(segment.probability_map, gray=False)
        if kind == "overlay":
            return png_data(patient.volumes["t1c"][:, :, index], segment.binary_mask)
        if kind == "ground_truth":
            return png_data(patient.volumes["t1c"][:, :, index], patient.volumes["seg"][:, :, index] > 0)
        raise ValueError("Unknown segmentation image type")


service = DashboardService()


class Handler(BaseHTTPRequestHandler):
    server_version = "ResearchDashboard/1.0"

    def log_message(self, fmt, *args):
        # Suppress request URLs that can contain patient IDs from console logs.
        pass

    def _json(self, data: dict, status: int = 200):
        encoded = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _bytes(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "private, max-age=60")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.strip("/").split("/") if p]
        try:
            if parsed.path == "/api/patients":
                self._json({"patients": self.server.service.patient_ids(), "split": "Validation"})
            elif parsed.path == "/api/comparison":
                self._json(comparison_results())
            elif parsed.path == "/api/segmentation-metrics":
                self._json(segmentation_metrics())
            elif len(parts) == 4 and parts[:2] == ["api", "analyze"]:
                self._json(self.server.service.analyze(parts[2], parts[3]))
            elif len(parts) == 5 and parts[:2] == ["api", "image"]:
                content = self.server.service.image(parts[2], parts[3], int(parts[4]))
                self._bytes(content, "image/png")
            elif len(parts) == 6 and parts[:2] == ["api", "segment"]:
                kind = parse_qs(parsed.query).get("kind", ["overlay"])[0]
                content = self.server.service.segmentation_image(parts[2], parts[3], int(parts[4]), parts[5] if parts[5] else kind)
                self._bytes(content, "image/png")
            elif parsed.path == "/" or parsed.path == "/index.html":
                path = ROOT / "dashboard" / "static" / "index.html"
                self._bytes(path.read_bytes(), "text/html; charset=utf-8")
            elif parsed.path.startswith("/static/"):
                path = ROOT / "dashboard" / "static" / Path(*parts[1:])
                if not path.is_file() or ROOT / "dashboard" / "static" not in path.resolve().parents:
                    self._json({"error": "Not found"}, 404)
                else:
                    self._bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            else:
                self._json({"error": "Not found"}, 404)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except FileNotFoundError:
            self._json({"error": "Required validation data or model artifact is unavailable. Check the configured saved artifacts."}, 503)
        except Exception:
            self._json({"error": "Unable to complete this request. Check that the configured validation data and model artifacts are available."}, 500)


def create_server(host: str = "127.0.0.1", port: int = 8000, dashboard_service: DashboardService | None = None):
    server = ThreadingHTTPServer((host, port), Handler)
    server.service = dashboard_service or service
    return server


def main():
    server = create_server()
    print("Research dashboard: http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
