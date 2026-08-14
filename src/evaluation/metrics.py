"""
Test-set evaluation: accuracy, macro precision/recall/F1, AUC-ROC, and the
confusion matrix, computed identically for every (dataset, architecture)
combination (Section III-B/IV).
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)


def evaluate_model(model, test_ds, num_classes: int) -> Dict:
    y_true, y_pred_probs = [], []
    for images, labels in test_ds:
        y_pred_probs.append(model.predict(images, verbose=0))
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if num_classes == 2:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_probs[:, 1])
        else:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
    except ValueError as exc:
        print(f"[warning] AUC-ROC could not be computed: {exc}")
        metrics["auc_roc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_pred_probs"] = y_pred_probs
    return metrics
