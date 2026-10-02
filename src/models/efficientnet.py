"""Model 3: EfficientNetB0 with ImageNet weights (transfer learning)."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from src import config
from src.data.augmentation import build_augmentation


def build_efficientnet(use_augmentation=True, weights="imagenet"):
    inputs = keras.Input(shape=(*config.IMAGE_SIZE, 3))
    x = inputs
    if use_augmentation:
        x = build_augmentation()(x)          # only active while training

    # EfficientNetB0 already contains its own preprocessing layers and expects
    # pixel values 0-255, which is exactly what our pipeline gives it.
    base_model = EfficientNetB0(include_top=False, weights=weights,
                                input_shape=(*config.IMAGE_SIZE, 3))
    base_model.trainable = False             # freeze: keep the pretrained knowledge
    x = base_model(x, training=False)        # training=False keeps BatchNorm layers fixed

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="efficientnet")