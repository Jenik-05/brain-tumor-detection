from pathlib import Path

# Project root = the folder that contains "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset locations (official Training / Testing split is preserved)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = RAW_DATA_DIR / "Training"
TEST_DIR = RAW_DATA_DIR / "Testing"

# Where plots are saved
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ----------settings ----------
IMAGE_SIZE = (224, 224)        # every image is resized to this (height, width)
BATCH_SIZE = 32                # images processed together in one step
SEED = 42                      # fixed seed -> same split every time
VALIDATION_FRACTION = 0.2      # 20% of the cleaned Training images
NUM_CLASSES = len(CLASS_NAMES)

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"