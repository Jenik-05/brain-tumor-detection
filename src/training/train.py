"""Trains a model and records how long it took."""
import time

import pandas as pd
from tensorflow import keras

from src import config
from src.training.callbacks import get_callbacks


def train_model(model, train_ds, val_ds, model_name, epochs=None):
    """Compile + train. Returns (history_df, training_time_in_seconds)."""
    epochs = epochs or config.EPOCHS

    model.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",   # labels are whole numbers 0-3
        metrics=["accuracy"],
    )

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(model_name),
    )
    training_time = time.time() - start

    history_df = pd.DataFrame(history.history)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(config.METRICS_DIR / f"{model_name}_history.csv", index=False)
    return history_df, training_time