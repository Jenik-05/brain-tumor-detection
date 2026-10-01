"""Training curves and saving results to a JSON file."""
import json

import matplotlib.pyplot as plt

from src import config


def plot_training_curves(history_df, model_name):
    epochs = range(1, len(history_df) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history_df["accuracy"], label="Training")
    ax1.plot(epochs, history_df["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history_df["loss"], label="Training")
    ax2.plot(epochs, history_df["val_loss"], label="Validation")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()

    fig.suptitle(f"Training curves - {model_name}")
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / f"training_curves_{model_name}.png", dpi=150)
    plt.show()


def save_results(model_name, val_metrics, test_metrics, training_time, history_df, num_params):
    """Save everything we will need later for the model comparison table."""
    results = {
        "model": model_name,
        "parameters": int(num_params),
        "training_time_seconds": round(float(training_time), 1),
        "epochs_run": int(len(history_df)),
        "best_epoch": int(history_df["val_loss"].idxmin() + 1),
        "validation": val_metrics,
        "test": test_metrics,
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / f"{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results