"""
Figure and table generation for the methodology and results sections of the
project report. Every function here saves to disk and also returns the
figure/DataFrame it produced, so it can be reused interactively (notebook)
or from a script (scripts/generate_methodology_report.py,
scripts/evaluate_model.py).
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


def _save_fig(fig, save_path: Path):
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"[figure] saved -> {save_path}")


def _write_table(df: pd.DataFrame, save_path: Path, index: bool = False):
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path.with_suffix(".csv"), index=index)
    with open(save_path.with_suffix(".md"), "w") as f:
        f.write(df.to_markdown(index=index))
    print(f"[table] saved -> {save_path.with_suffix('.csv')} and {save_path.with_suffix('.md')}")


def plot_class_distribution(df, dataset_name, save_path, split="train"):
    counts = df.loc[df["split"] == split, "label"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_title(f"{dataset_name}: Class Distribution ({split} split)")
    ax.set_xlabel("Class"); ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels(): label.set_ha("right")
    fig.tight_layout(); _save_fig(fig, save_path); return fig


def plot_sample_images(df, dataset_name, save_path, n_per_class=1, max_classes=8):
    classes = df["label"].unique()[:max_classes]
    n_cols = len(classes)
    fig, axes = plt.subplots(n_per_class, n_cols, figsize=(2.0 * n_cols, 2.2 * n_per_class))
    axes = np.atleast_2d(axes)
    for col, cls in enumerate(classes):
        rows = df[df["label"] == cls].sample(min(n_per_class, len(df[df["label"] == cls])), random_state=42)
        for row_idx, (_, row) in enumerate(rows.iterrows()):
            ax = axes[row_idx, col]
            try:
                ax.imshow(Image.open(row["filepath"]).convert("RGB"))
            except Exception as exc:
                ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
                print(f"[warning] could not load '{row['filepath']}': {exc}")
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0: ax.set_title(str(cls), fontsize=9)
    fig.suptitle(f"{dataset_name}: Example Images by Class", y=1.02)
    fig.tight_layout(); _save_fig(fig, save_path); return fig


def plot_model_complexity(params_by_architecture, save_path):
    archs = list(params_by_architecture.keys())
    totals = [params_by_architecture[a]["total"] / 1e6 for a in archs]
    trainables = [params_by_architecture[a]["trainable"] / 1e6 for a in archs]
    x = np.arange(len(archs)); width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width/2, totals, width, label="Total parameters", color="#4C72B0")
    ax.bar(x + width/2, trainables, width, label="Trainable (phase 2)", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([a.upper() if a != "densenet121" else "DenseNet121" for a in archs])
    ax.set_ylabel("Parameters (millions)"); ax.set_title("Model Complexity by Architecture")
    ax.legend(); fig.tight_layout(); _save_fig(fig, save_path); return fig


def plot_training_curves(history, run_name, save_path):
    epochs = range(1, len(history["loss"]) + 1)
    boundary = history.get("phase_boundary")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()
    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()
    if boundary:
        for ax in axes:
            ax.axvline(boundary + 0.5, color="grey", linestyle="--", linewidth=1)
            ax.text(boundary + 0.6, ax.get_ylim()[1]*0.95, "fine-tuning\nstarts", fontsize=7, va="top", color="grey")
    fig.suptitle(f"Training Curves: {run_name}"); fig.tight_layout(); _save_fig(fig, save_path); return fig


def plot_confusion_matrix(cm_array, class_names, run_name, save_path):
    fig, ax = plt.subplots(figsize=(0.6*len(class_names)+2, 0.6*len(class_names)+2))
    im = ax.imshow(cm_array, cmap="Blues"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(f"Confusion Matrix: {run_name}")
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, format(cm_array[i, j], "d"), ha="center", va="center",
                     color="white" if cm_array[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout(); _save_fig(fig, save_path); return fig


def plot_roc_curves(y_true, y_pred_probs, class_names, run_name, save_path):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes)) if n_classes > 2 else None
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc(fpr, tpr):.3f})")
    else:
        all_fpr = np.linspace(0, 1, 200); mean_tpr = np.zeros_like(all_fpr)
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
            ax.plot(fpr, tpr, alpha=0.35, linewidth=1, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
        mean_tpr /= n_classes
        ax.plot(all_fpr, mean_tpr, color="black", linewidth=2, label=f"Macro-average (AUC={auc(all_fpr, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {run_name}"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); _save_fig(fig, save_path); return fig


def plot_gradcam_grid(originals, overlays, titles, save_path):
    n = len(originals)
    fig, axes = plt.subplots(2, n, figsize=(2.2*n, 4.6)); axes = np.atleast_2d(axes)
    for i in range(n):
        axes[0, i].imshow(originals[i]); axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[0, i].set_title(titles[i], fontsize=8)
        axes[1, i].imshow(overlays[i]); axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    axes[0, 0].set_ylabel("Original", fontsize=9); axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    fig.tight_layout(); _save_fig(fig, save_path); return fig


def plot_architecture_comparison(results_df, metric, save_path):
    datasets_ = results_df["dataset"].unique(); architectures_ = results_df["architecture"].unique()
    x = np.arange(len(datasets_)); width = 0.8 / len(architectures_)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, arch in enumerate(architectures_):
        values = [results_df[(results_df["dataset"] == d) & (results_df["architecture"] == arch)][metric].values
                  for d in datasets_]
        values = [v[0] if len(v) else np.nan for v in values]
        ax.bar(x + i*width - 0.4 + width/2, values, width, label=arch)
    ax.set_xticks(x); ax.set_xticklabels(datasets_, rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Architecture Comparison: {metric.replace('_', ' ').title()}")
    ax.legend(fontsize=8); fig.tight_layout(); _save_fig(fig, save_path); return fig


def dataset_summary_table(df, dataset_name, save_path):
    summary = df.groupby(["label", "split"]).size().unstack(fill_value=0)
    for col in ("train", "val", "test"):
        if col not in summary.columns: summary[col] = 0
    summary = summary[["train", "val", "test"]]
    summary["total"] = summary.sum(axis=1)
    summary = summary.reset_index().rename(columns={"label": "Class"})
    summary.insert(0, "Dataset", dataset_name)
    _write_table(summary, save_path); return summary


def hyperparameters_table(save_path):
    from .. import config
    rows = [
        ("Input size (all architectures)", "224 x 224 x 3"),
        ("Batch size", config.BATCH_SIZE),
        ("Train / val / test split", "70% / 15% / 15% (stratified)"),
        ("Phase 1 epochs (head only, base frozen)", config.FREEZE_BASE_EPOCHS),
        ("Phase 2 epochs (fine-tuning)", config.FINE_TUNE_EPOCHS),
        ("Phase 2 unfrozen fraction of base", f"{config.FINE_TUNE_UNFREEZE_FRACTION:.0%}"),
        ("Phase 1 learning rate", config.BASE_LEARNING_RATE),
        ("Phase 2 learning rate", config.FINE_TUNE_LEARNING_RATE),
        ("Optimizer", "Adam"),
        ("Loss function", "Sparse categorical cross-entropy"),
        ("Dropout rate", config.DROPOUT_RATE),
        ("Dense head units", config.DENSE_UNITS),
        ("Early stopping patience", f"{config.EARLY_STOPPING_PATIENCE} epochs (val_loss)"),
        ("LR reduction", f"factor {config.REDUCE_LR_FACTOR}, patience {config.REDUCE_LR_PATIENCE} epochs (val_loss)"),
        ("Class imbalance handling", "Class weighting (balanced) / SMOTE (BreakHis only)"),
        ("Augmentation (train split only)", "Flip, rotation (10%), zoom (10%), contrast jitter (10%)"),
        ("Random seed", config.RANDOM_SEED),
    ]
    df = pd.DataFrame(rows, columns=["Setting", "Value"])
    _write_table(df, save_path); return df


def model_complexity_table(params_by_architecture, save_path):
    rows = [{"Architecture": arch,
             "Total parameters": f"{c['total']:,}",
             "Trainable (phase 2)": f"{c['trainable']:,}",
             "Frozen (phase 2)": f"{c['total'] - c['trainable']:,}"}
            for arch, c in params_by_architecture.items()]
    df = pd.DataFrame(rows); _write_table(df, save_path); return df


def results_table(results, save_path):
    df = pd.DataFrame(results)[["dataset", "architecture", "accuracy", "precision_macro",
                                  "recall_macro", "f1_macro", "auc_roc"]].copy()
    for col in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_roc"):
        df[col] = (df[col] * 100).round(2)
    df = df.rename(columns={"dataset": "Dataset", "architecture": "Architecture",
                              "accuracy": "Accuracy (%)", "precision_macro": "Precision (%)",
                              "recall_macro": "Recall (%)", "f1_macro": "F1-score (%)",
                              "auc_roc": "AUC-ROC (%)"})
    _write_table(df, save_path); return df
