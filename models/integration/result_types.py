"""Typed results returned by the hybrid inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    patient_id: str
    model_name: str
    slice_index: int
    probability: float
    predicted_class: int
    inference_seconds: float


@dataclass(frozen=True)
class SegmentedSliceResult:
    slice_index: int
    probability_map: np.ndarray
    binary_mask: np.ndarray
    overlay_path: str | None = None
    inference_seconds: float = 0.0


@dataclass(frozen=True)
class HybridPatientResult:
    patient_id: str
    detection_model: str
    detections: tuple[DetectionResult, ...]
    segmentation_status: Literal["completed", "skipped_no_positive_slice"]
    segmented_slices: tuple[SegmentedSliceResult, ...] = field(default_factory=tuple)
    selected_neighbor_slices: tuple[int, ...] = field(default_factory=tuple)
    inference_seconds: float = 0.0
