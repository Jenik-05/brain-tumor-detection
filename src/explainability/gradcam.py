"""Grad-CAM: shows which image regions most influenced the model's prediction.

IMPORTANT: this is a model-explanation picture. It does NOT prove where a
tumor really is - it only shows where the model "looked".
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def _find_target_layer(model):
    """The layer just before GlobalAveragePooling2D = the last feature maps."""
    layers = model.layers
    for i, layer in enumerate(layers):
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            return layers[i - 1].name
    raise ValueError("No GlobalAveragePooling2D layer found in the model")


def make_gradcam_heatmap(model, image, class_index=None):
    """image: array shaped (224, 224, 3) with values 0-255.
    Returns (heatmap with values 0-1, class index used, class probabilities)."""
    target_name = _find_target_layer(model)
    x = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps = None
        # Run the model layer by layer so we can grab the last feature maps
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            x = layer(x, training=False)       # training=False: no random augmentation, no dropout
            if layer.name == target_name:
                feature_maps = x
                tape.watch(feature_maps)      # we want gradients w.r.t. these maps
        probabilities = x                     # final softmax output, shape (1, 4)
        if class_index is None:
            class_index = int(tf.argmax(probabilities[0]))
        score = probabilities[0, class_index]

    # How much does each feature map influence the score?
    grads = tape.gradient(score, feature_maps)
    channel_weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)             # keep only positive influence
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)   # scale to 0-1
    return heatmap.numpy(), class_index, probabilities[0].numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Returns (coloured heatmap, overlay on the MRI) as uint8 images."""
    height, width = image.shape[:2]
    heatmap_big = tf.image.resize(heatmap[..., np.newaxis], (height, width)).numpy()[..., 0]
    coloured = matplotlib.colormaps["jet"](heatmap_big)[..., :3]      # values 0-1
    original = image.astype("float32") / 255.0
    overlay = (1 - alpha) * original + alpha * coloured
    return (coloured * 255).astype("uint8"), (np.clip(overlay, 0, 1) * 255).astype("uint8")


def plot_gradcam(image, heatmap, title="", save_path=None):
    """Original MRI | Grad-CAM heatmap | overlay."""
    heatmap_rgb, overlay = overlay_heatmap(image, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    axes[0].imshow(image.astype("uint8")); axes[0].set_title("Original MRI")
    axes[1].imshow(heatmap_rgb);           axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay);               axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()