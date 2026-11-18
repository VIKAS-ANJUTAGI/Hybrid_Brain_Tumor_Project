from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LOCAL_RAW = PROJECT_ROOT / "data" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
_EXTERNAL_RAW = Path(
    r"D:\BrainTumorDataset\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
)
DATASET_ARCHIVE_PATH = _LOCAL_RAW if _LOCAL_RAW.exists() else _EXTERNAL_RAW

_LOCAL_EXTRACTED = PROJECT_ROOT / "data" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
_EXTERNAL_EXTRACTED = Path(
    r"D:\BrainTumorDataset\extracted\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
DATASET_ROOT = _LOCAL_EXTRACTED if _LOCAL_EXTRACTED.exists() else _EXTERNAL_EXTRACTED

MRI_MODALITIES = ("t1c", "t1n", "t2w", "t2f")
SEGMENTATION_MASK_SUFFIX = "seg"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
QUANTUM_MODELS_DIR = MODELS_DIR / "quantum"
