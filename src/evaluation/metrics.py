"""Predictions and numbers that describe how good a model is."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support, roc_auc_score)

from src import config


def predict_dataset(model, dataset):
    """Return (true labels, predicted labels, class probabilities).
    The dataset must NOT be shuffled, so labels and predictions line up."""
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    y_prob = model.predict(dataset, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob):
    # "macro" = calculate per class, then average (every class counts equally)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")),
    }


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=config.CLASS_NAMES,
                                 digits=3, zero_division=0)