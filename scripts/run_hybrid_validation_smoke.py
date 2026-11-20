"""Run the integrated pipeline on one patient from the checkpoint validation split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from models.segmentation.inference import load_patient_volumes
from models.segmentation.training import load_checkpoint
from models.integration import HybridInferencePipeline


def main(argv: list[str] | None = None) -> int:
    root = ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-id", help="Optional patient ID; must be in checkpoint validation IDs")
    parser.add_argument("--model", default="logistic_regression", choices=("logistic_regression", "random_forest", "vqc_2", "vqc_4", "vqc_8"))
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=root / "results/hybrid_validation_smoke")
    args = parser.parse_args(argv)
    checkpoint_path = root / "results/segmentation_runs/unet_baseline_20261120T173712Z/best_checkpoint.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    validation_ids = tuple(checkpoint.get("validation_patient_ids", ()))
    if not validation_ids:
        raise RuntimeError("U-Net checkpoint does not identify its validation patients")
    patient_id = args.patient_id or validation_ids[0]
    if patient_id not in validation_ids:
        raise ValueError(f"{patient_id!r} is not in the checkpoint validation cohort")

    modality_paths = {
        name: settings.DATASET_ROOT / patient_id / f"{patient_id}-{name}.nii.gz"
        for name in settings.MRI_MODALITIES
    }
    patient = load_patient_volumes(modality_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = HybridInferencePipeline(
        unet_checkpoint=checkpoint_path,
        overlay_dir=args.output_dir / "overlays",
        neighbor_radius=args.neighbor_radius,
    )
    start = time.perf_counter()
    result = pipeline.run_patient(patient_id, patient.modalities, args.model)
    elapsed = time.perf_counter() - start
    report = {
        "patient_id": patient_id,
        "patient_source": "U-Net checkpoint validation split",
        "test_patients_loaded": False,
        "detection_model": args.model,
        "analyzed_slice_count": len(result.detections),
        "detected_slices": [d.slice_index for d in result.detections if d.predicted_class == 1],
        "detected_slice_probabilities": {str(d.slice_index): d.probability for d in result.detections if d.predicted_class == 1},
        "segmentation_status": result.segmentation_status,
        "segmented_slices": [segment.slice_index for segment in result.segmented_slices],
        "empty_predicted_mask_slices": [segment.slice_index for segment in result.segmented_slices if not segment.binary_mask.any()],
        "neighbor_radius": args.neighbor_radius,
        "selected_neighbor_slices": list(result.selected_neighbor_slices),
        "inference_seconds": elapsed,
        "generated_overlays": [segment.overlay_path for segment in result.segmented_slices if segment.overlay_path],
        "checkpoint": str(checkpoint_path),
    }
    report_path = args.output_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "patient_id": patient_id,
        "detection_model": args.model,
        "analyzed_slice_count": report["analyzed_slice_count"],
        "detected_slice_count": len(report["detected_slices"]),
        "detected_slice_range": [report["detected_slices"][0], report["detected_slices"][-1]] if report["detected_slices"] else [],
        "segmentation_status": report["segmentation_status"],
        "segmented_slice_count": len(report["segmented_slices"]),
        "empty_predicted_mask_count": len(report["empty_predicted_mask_slices"]),
        "inference_seconds": elapsed,
        "overlay_count": len(report["generated_overlays"]),
        "report_path": str(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
