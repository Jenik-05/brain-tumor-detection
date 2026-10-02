"""Tests for the prediction pipeline (uses an untrained model, so no files are needed)."""
from src import config
from src.models.cnn import build_custom_cnn
from src.predict import predict_image


def test_prediction_output_is_consistent(make_image_bytes):
    model = build_custom_cnn(use_augmentation=False)
    result = predict_image(model, make_image_bytes("RGB"), config.CLASS_NAMES)

    probabilities = list(result["probabilities"].values())
    assert result["predicted_class"] in config.CLASS_NAMES
    assert len(probabilities) == config.NUM_CLASSES
    assert abs(sum(probabilities) - 1.0) < 1e-3          # probabilities add up to 100%
    assert result["confidence"] == max(probabilities)    # confidence = highest probability
    assert result["image"].shape == (*config.IMAGE_SIZE, 3)


def test_same_image_gives_same_prediction(make_image_bytes):
    model = build_custom_cnn(use_augmentation=True)      # augmentation must be OFF when predicting
    first = predict_image(model, make_image_bytes("L"), config.CLASS_NAMES)
    second = predict_image(model, make_image_bytes("L"), config.CLASS_NAMES)
    assert first["probabilities"] == second["probabilities"]