"""CPU-safe 2D whole-tumor segmentation components.

This package deliberately consumes only the train and validation assignments
from the Phase 2 manifest.  The manifest test partition is never a training
or validation input.
"""

from .dataset import BraTSSegmentationDataset, ManifestSplitError, validate_patient_assignments
from .inference import PatientVolumes, load_patient_volumes, predict_slice, predict_volume, prepare_axial_slice, preprocess_patient
from .losses import BCEDiceLoss, dice_loss
from .metrics import segmentation_metrics
from .unet import UNet2D

__all__ = [
    "BCEDiceLoss",
    "BraTSSegmentationDataset",
    "ManifestSplitError",
    "PatientVolumes",
    "UNet2D",
    "dice_loss",
    "segmentation_metrics",
    "validate_patient_assignments",
    "load_patient_volumes",
    "preprocess_patient",
    "prepare_axial_slice",
    "predict_slice",
    "predict_volume",
]
