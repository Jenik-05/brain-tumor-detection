"""Reusable prediction pipeline: image bytes -> class, confidence, probabilities."""
import json

from tensorflow import keras

from src import config
from src.data.preprocessing import preprocess_bytes


def load_best_model():
    return keras.models.load_model(config.MODELS_DIR / "best_model.keras", compile=False)


def load_class_names():
    with open(config.MODELS_DIR / "class_names.json") as f:
        return json.load(f)


def predict_image(model, file_bytes, class_names):
    """file_bytes = the content of an image file (e.g. read_bytes() or an upload)."""
    image = preprocess_bytes(file_bytes).numpy()            # (224, 224, 3), values 0-255
    probabilities = model.predict(image[None, ...], verbose=0)[0]
    best = int(probabilities.argmax())
    return {
        "image": image,
        "predicted_class": class_names[best],
        "predicted_index": best,
        "confidence": float(probabilities[best]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probabilities)},
    }