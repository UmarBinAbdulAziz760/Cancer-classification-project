"""
tf.data pipeline: turns a preprocessed index.csv into batched, architecture-
specific train/val/test tf.data.Dataset objects.

Images are decoded and resized lazily rather than loaded into memory
upfront, which matters most for NCT-CRC-HE-100K's 100,000 images
(Section III-A).
"""

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

from .. import config
from ..augmentation.augment import augment_image, decode_and_resize

PREPROCESS_FUNCTIONS = {
    "vgg16": vgg16_preprocess,
    "resnet50": resnet50_preprocess,
    "mobilenet": mobilenet_preprocess,
    "densenet121": densenet_preprocess,
}


def build_dataset(filepaths, labels, architecture: str, target_size, batch_size: int,
                   shuffle: bool = False, augment_fn: Optional[Callable] = None) -> tf.data.Dataset:
    """Builds one batched, prefetched tf.data.Dataset for a single split."""
    preprocess_fn = PREPROCESS_FUNCTIONS[architecture]
    ds = tf.data.Dataset.from_tensor_slices((list(filepaths), list(labels)))

    if shuffle:
        ds = ds.shuffle(buffer_size=max(1, len(filepaths)), seed=config.RANDOM_SEED,
                         reshuffle_each_iteration=True)

    def _load(fp, lbl):
        return decode_and_resize(fp, target_size), lbl
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_fn is not None:
        ds = ds.map(lambda img, lbl: (augment_fn(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)

    def _preprocess(img, lbl):
        return preprocess_fn(img), lbl
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_index(dataset_name: str) -> pd.DataFrame:
    """Loads the preprocessed index.csv for `dataset_name` (see
    src.data.splitting.preprocess_dataset)."""
    index_path = config.PROCESSED_ROOT / dataset_name / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"'{index_path}' not found - run scripts/build_dataset_indices.py first."
        )
    return pd.read_csv(index_path)


def get_datasets(dataset_name: str, architecture: str):
    """Returns (train_ds, val_ds, test_ds, label_encoder, num_classes)."""
    df = load_index(dataset_name)
    label_encoder = LabelEncoder()
    label_encoder.fit(df.loc[df["split"] == "train", "label"])
    df = df[df["label"].isin(label_encoder.classes_)].copy()
    df["label_idx"] = label_encoder.transform(df["label"])

    target_size = config.TARGET_SIZE[architecture]

    def subset(split):
        sub = df[df["split"] == split]
        return sub["filepath"].tolist(), sub["label_idx"].tolist()

    train_paths, train_labels = subset("train")
    val_paths, val_labels = subset("val")
    test_paths, test_labels = subset("test")

    train_ds = build_dataset(train_paths, train_labels, architecture, target_size, config.BATCH_SIZE,
                              shuffle=True, augment_fn=augment_image)
    val_ds = build_dataset(val_paths, val_labels, architecture, target_size, config.BATCH_SIZE, shuffle=False)
    test_ds = build_dataset(test_paths, test_labels, architecture, target_size, config.BATCH_SIZE, shuffle=False)

    return train_ds, val_ds, test_ds, label_encoder, len(label_encoder.classes_)


def get_class_weight_dict(dataset_name: str) -> Optional[Dict[int, float]]:
    path = config.PROCESSED_ROOT / dataset_name / "class_weights.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}
