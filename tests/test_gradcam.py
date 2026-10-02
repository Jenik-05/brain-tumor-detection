"""Tests for Grad-CAM (uses an untrained model, so no files are needed)."""
import numpy as np

from src import config
from src.data.preprocessing import preprocess_bytes
from src.explainability.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.models.cnn import build_custom_cnn


def test_heatmap_shape_and_range(make_image_bytes):
    model = build_custom_cnn(use_augmentation=False)
    image = preprocess_bytes(make_image_bytes("RGB")).numpy()
    heatmap, class_index, _ = make_gradcam_heatmap(model, image)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6
    assert 0 <= class_index < config.NUM_CLASSES


def test_gradcam_agrees_with_normal_prediction(make_image_bytes):
    # Regression test: Grad-CAM must not apply random augmentation
    model = build_custom_cnn(use_augmentation=True)
    image = preprocess_bytes(make_image_bytes("RGB")).numpy()
    _, _, probabilities = make_gradcam_heatmap(model, image)
    expected = model.predict(image[None, ...], verbose=0)[0]
    assert np.allclose(probabilities, expected, atol=1e-4)


def test_overlay_keeps_the_image_size():
    image = np.random.randint(0, 256, (*config.IMAGE_SIZE, 3)).astype("float32")
    heatmap = np.random.rand(7, 7).astype("float32")
    coloured, overlay = overlay_heatmap(image, heatmap)
    assert coloured.shape == (*config.IMAGE_SIZE, 3)
    assert overlay.shape == (*config.IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8