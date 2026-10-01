"""Callbacks = helpers that run automatically during training."""
from tensorflow import keras

from src import config


def get_callbacks(model_name):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        # Stop when validation loss stops improving; go back to the best weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        # Save the BEST model seen so far (not just the last epoch)
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / f"{model_name}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        # If stuck for 2 epochs, halve the learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]