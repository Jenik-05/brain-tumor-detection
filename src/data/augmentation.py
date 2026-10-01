"""Random changes applied to TRAINING images only (switched off for validation/test)."""
from tensorflow.keras import layers, Sequential


def build_augmentation():
    return Sequential(
        [
            layers.RandomFlip("horizontal"),                      # left-right mirror
            layers.RandomRotation(0.04, fill_mode="constant"),    # about +-14 degrees
            layers.RandomZoom(0.1, fill_mode="constant"),         # zoom in/out up to 10%
            layers.RandomTranslation(0.05, 0.05, fill_mode="constant"),  # shift up to 5%
            layers.RandomBrightness(0.1, value_range=(0, 255)),   # small brightness change
            layers.RandomContrast(0.1),                           # small contrast change
        ],
        name="augmentation",
    )