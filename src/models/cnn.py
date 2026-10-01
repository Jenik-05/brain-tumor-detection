"""Model 1: a small CNN built from scratch."""
from tensorflow import keras
from tensorflow.keras import layers

from src import config
from src.data.augmentation import build_augmentation


def build_custom_cnn(use_augmentation=True):
    inputs = keras.Input(shape=(*config.IMAGE_SIZE, 3))
    x = inputs
    if use_augmentation:
        x = build_augmentation()(x)          # only active while training
    x = layers.Rescaling(1.0 / 255)(x)       # pixel values 0-255 -> 0-1

    # Three blocks: Conv (find patterns) -> ReLU -> MaxPool (shrink the image)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)   # each feature map -> 1 number
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)               # randomly switch off 50% of units
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")