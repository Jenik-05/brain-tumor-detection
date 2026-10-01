"""Model 2: ResNet50 with ImageNet weights (transfer learning)."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

from src import config
from src.data.augmentation import build_augmentation


def build_resnet50(use_augmentation=True, weights="imagenet"):
    inputs = keras.Input(shape=(*config.IMAGE_SIZE, 3))
    x = inputs
    if use_augmentation:
        x = build_augmentation()(x)          # only active while training

    # ResNet50 was trained on images with the ImageNet average colour subtracted.
    # Our 3 channels are identical (gray), so this matches ResNet50's own
    # preprocess_input exactly.
    x = layers.Normalization(
        mean=[103.939, 116.779, 123.68], variance=[1.0, 1.0, 1.0], name="resnet_preprocess"
    )(x)

    base_model = ResNet50(include_top=False, weights=weights,
                          input_shape=(*config.IMAGE_SIZE, 3))
    base_model.trainable = False             # freeze: keep the pretrained knowledge
    x = base_model(x, training=False)        # training=False keeps BatchNorm layers fixed

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet50")