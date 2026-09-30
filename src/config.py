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