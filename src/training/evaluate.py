"""
Report generation for one already-trained (dataset, architecture) run:
training curves, confusion matrix, ROC curve, Grad-CAM examples, and the
summary metrics dict used to build the results tables.
"""

from pathlib import Path
from typing import List

import pandas as pd
import tensorflow as tf

from .. import config
from ..evaluation.metrics import evaluate_model
from ..evaluation.reporting import plot_confusion_matrix, plot_roc_curves, plot_training_curves
from ..gradcam.gradcam import run_gradcam_examples


def find_best_model_path(run_name: str) -> Path:
    """Looks for the final saved model for `run_name`
    (e.g. "BreakHis_resnet50"), preferring the fully fine-tuned version but
    falling back to earlier checkpoints if training stopped early."""
    candidates = [
        config.MODELS_DIR / f"{run_name}_final.keras",
        config.MODELS_DIR / f"{run_name}_phase2_final.keras",
        config.MODELS_DIR / f"{run_name}_phase2_best.keras",
        config.MODELS_DIR / f"{run_name}_phase1_final.keras",
        config.MODELS_DIR / f"{run_name}_phase1_best.keras",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No saved model found for '{run_name}'. Checked:\n" + "\n".join(str(p) for p in candidates)
    )


def reconstruct_history(run_name: str):
    """Rebuilds a combined training history from the CSVLogger files written
    by src.training.callbacks.build_callbacks during phase 1 and phase 2."""
    phase1_csv = config.LOGS_DIR / f"{run_name}_phase1_history.csv"
    phase2_csv = config.LOGS_DIR / f"{run_name}_phase2_history.csv"

    phase1_exists, phase2_exists = phase1_csv.exists(), phase2_csv.exists()
    if not phase1_exists and not phase2_exists:
        print(f"[warning] No training history found for {run_name}.")
        return None

    df1 = pd.read_csv(phase1_csv) if phase1_exists else None
    df2 = pd.read_csv(phase2_csv) if phase2_exists else None

    if phase1_exists and phase2_exists:
        return {
            "loss": df1["loss"].tolist() + df2["loss"].tolist(),
            "val_loss": df1["val_loss"].tolist() + df2["val_loss"].tolist(),
            "accuracy": df1["accuracy"].tolist() + df2["accuracy"].tolist(),
            "val_accuracy": df1["val_accuracy"].tolist() + df2["val_accuracy"].tolist(),
            "phase_boundary": len(df1),
        }
    if phase1_exists:
        return {"loss": df1["loss"].tolist(), "val_loss": df1["val_loss"].tolist(),
                "accuracy": df1["accuracy"].tolist(), "val_accuracy": df1["val_accuracy"].tolist(),
                "phase_boundary": len(df1)}
    return {"loss": df2["loss"].tolist(), "val_loss": df2["val_loss"].tolist(),
            "accuracy": df2["accuracy"].tolist(), "val_accuracy": df2["val_accuracy"].tolist(),
            "phase_boundary": 0}


def evaluate_run(dataset_name: str, architecture: str, test_ds, class_names: List[str], num_classes: int) -> dict:
    """Loads the trained model for (dataset_name, architecture), generates the
    training-curve/confusion-matrix/ROC/Grad-CAM figures, and returns the
    summary metrics row used by evaluation.reporting.results_table."""
    run_name = f"{dataset_name}_{architecture}"
    model = tf.keras.models.load_model(find_best_model_path(run_name))

    history = reconstruct_history(run_name)
    if history is not None:
        plot_training_curves(history, run_name, config.FIGURES_RESULTS_DIR / f"{run_name}_training_curves.png")
    else:
        print("Skipping training curve because no history was found.")

    metrics = evaluate_model(model, test_ds, num_classes)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, run_name,
                           config.FIGURES_RESULTS_DIR / f"{run_name}_confusion_matrix.png")
    plot_roc_curves(metrics["y_true"], metrics["y_pred_probs"], class_names, run_name,
                     config.FIGURES_RESULTS_DIR / f"{run_name}_roc_curve.png")
    run_gradcam_examples(model, test_ds, architecture, class_names, run_name)

    result = {
        "dataset": dataset_name,
        "architecture": architecture,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "auc_roc": metrics["auc_roc"],
    }
    tf.keras.backend.clear_session()
    return result
