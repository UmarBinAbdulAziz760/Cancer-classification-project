"""
Stratified train/val/test splitting and class-imbalance handling.

Two imbalance strategies are supported, matching the methodology described
in the project report (Section III-A):
    - class_weight: compute_class_weights() - used by default for all
      three datasets.
    - smote: smote_oversample_index() - used only for BreakHis, since
      SMOTE's feature-space interpolation on flattened thumbnails becomes
      computationally impractical at the scale of the other two datasets.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .. import config


def stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                      test_ratio: float, label_col: str = "label",
                      random_state: int = 42) -> pd.DataFrame:
    """Adds a `split` column ('train'/'val'/'test') via two stratified splits."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df[label_col], random_state=random_state)
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val, stratify=temp_df[label_col], random_state=random_state)

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy();     val_df["split"] = "val"
    test_df = test_df.copy();   test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def compute_class_weights(df: pd.DataFrame, label_col: str = "label") -> Dict[int, float]:
    """Balanced class weights keyed by the same integer index LabelEncoder would assign
    (i.e. classes sorted alphabetically)."""
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df[label_col].map(class_to_idx).values
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=y)
    return {class_to_idx[c]: float(w) for c, w in zip(classes, weights)}


def smote_oversample_index(df: pd.DataFrame, label_col: str = "label",
                            thumbnail_size=(32, 32), max_samples_per_class: int = 1500,
                            random_state: int = 42) -> pd.DataFrame:
    """SMOTE on flattened thumbnails - only tractable for the smaller BreakHis split.

    Synthetic thumbnails are written to disk under
    `config.OUTPUT_ROOT / "smote_synthetic"` and referenced by filepath just
    like real images, so the rest of the pipeline doesn't need to
    special-case them.
    """
    from imblearn.over_sampling import SMOTE

    thumbnails, labels_list = [], []
    for _, row in df.iterrows():
        try:
            img = Image.open(row["filepath"]).convert("RGB").resize(thumbnail_size)
        except Exception as exc:
            print(f"[warning] Skipping unreadable image '{row['filepath']}': {exc}")
            continue
        thumbnails.append(np.asarray(img).flatten())
        labels_list.append(row[label_col])

    X = np.stack(thumbnails)
    y = np.array(labels_list)
    class_counts = pd.Series(y).value_counts()
    sampling_strategy = {cls: min(max_samples_per_class, int(class_counts.max()))
                          for cls in class_counts.index}

    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    n_original = len(X)
    n_synthetic = len(X_resampled) - n_original
    print(f"[info] SMOTE generated {n_synthetic} synthetic samples ({n_original} -> {len(X_resampled)}).")

    synthetic_records = []
    if n_synthetic > 0:
        synth_dir = config.OUTPUT_ROOT / "smote_synthetic"
        synth_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_original, len(X_resampled)):
            pixels = X_resampled[i].reshape(*thumbnail_size, 3).astype(np.uint8)
            out_path = synth_dir / f"synthetic_{i}.png"
            Image.fromarray(pixels).save(out_path)
            synthetic_records.append({"filepath": str(out_path), "label": y_resampled[i], "split": "train"})

    synthetic_df = pd.DataFrame(synthetic_records)
    return pd.concat([df, synthetic_df], ignore_index=True) if len(synthetic_df) else df


def preprocess_dataset(name: str, df: pd.DataFrame,
                        imbalance_strategy: str = None) -> pd.DataFrame:
    """Splits `df`, applies the configured imbalance strategy, and writes
    `index.csv` (+ `class_weights.json` if applicable) to
    `config.PROCESSED_ROOT / name`."""
    imbalance_strategy = imbalance_strategy or config.IMBALANCE_STRATEGY
    df = stratified_split(df, config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO,
                           random_state=config.RANDOM_SEED)

    print(f"\n[{name}] split sizes:")
    print(df["split"].value_counts().to_string())
    print(f"[{name}] class distribution (train split):")
    print(df.loc[df["split"] == "train", "label"].value_counts().to_string())

    train_df = df[df["split"] == "train"]
    class_weights = None

    if imbalance_strategy == "smote":
        train_df = smote_oversample_index(train_df, thumbnail_size=config.SMOTE_THUMBNAIL_SIZE,
                                           max_samples_per_class=config.SMOTE_MAX_SAMPLES_PER_CLASS,
                                           random_state=config.RANDOM_SEED)
        df = pd.concat([train_df, df[df["split"] != "train"]], ignore_index=True)
    elif imbalance_strategy == "class_weight":
        class_weights = compute_class_weights(train_df)

    out_dir = config.PROCESSED_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "index.csv", index=False)
    if class_weights is not None:
        with open(out_dir / "class_weights.json", "w") as f:
            json.dump(class_weights, f, indent=2)

    print(f"[{name}] preprocessing complete -> '{out_dir}'")
    return df
