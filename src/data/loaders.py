"""
Dataset loaders.

Each loader walks a dataset's folder structure and returns a flat
pandas.DataFrame with (at minimum) `filepath` and `label` columns. This is
the only part of the pipeline that needs to know how a specific dataset is
laid out on disk - everything downstream (splitting, class weights, the
tf.data pipeline) works off the resulting DataFrame and doesn't care which
dataset it came from.
"""

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

VALID_MAGNIFICATIONS = {"40X", "100X", "200X", "400X"}


def load_breakhis_index(root_dir: Path, label_mode: str = "binary") -> pd.DataFrame:
    """BreakHis: benign|malignant/SOB/<tumour_type>/<patient_id>/<magnification>/*.png

    Args:
        root_dir: path to the BreakHis 'breast' directory (contains benign/ and malignant/).
        label_mode: "binary" (benign/malignant) or "subtype" (8-way tumour type).
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"BreakHis root not found at '{root_dir}'.")

    records = []
    for png_path in root_dir.rglob("*.png"):
        parts = png_path.parts
        class_dir = magnification = patient_id = tumour_type = None
        for i, part in enumerate(parts):
            if part in ("benign", "malignant"):
                class_dir = part
            if part in VALID_MAGNIFICATIONS:
                magnification = part
                patient_id = parts[i - 1]
                tumour_type = parts[i - 2]
        label = class_dir if label_mode == "binary" else tumour_type
        records.append({
            "filepath": str(png_path),
            "label": label or "unknown",
            "magnification": magnification or "unknown",
            "patient_id": patient_id or "unknown",
        })

    if not records:
        raise RuntimeError(f"No .png files found under '{root_dir}'.")
    return pd.DataFrame(records)


def load_folder_labelled_index(
    root_dir: Path,
    expected_classes: Optional[set] = None,
    extensions: Sequence[str] = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
) -> pd.DataFrame:
    """
    Generic loader for datasets already sorted into one subfolder per class.

    Used for both NCT-CRC-HE-100K (ADI/BACK/DEB/LYM/MUC/MUS/NORM/STR/TUM) and
    the ISIC 2019 mirror sorted into one subfolder per class
    (MEL/NV/BCC/AK/BKL/DF/VASC/SCC).
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found at '{root_dir}'.")

    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    found = {d.name for d in class_dirs}
    if expected_classes:
        missing = expected_classes - found
        if missing:
            print(f"[warning] Expected class folders not found under '{root_dir}': {sorted(missing)}")

    records = []
    for class_dir in class_dirs:
        for ext in extensions:
            for img_path in class_dir.glob(ext):
                records.append({"filepath": str(img_path), "label": class_dir.name})

    if not records:
        raise RuntimeError(f"No images found under '{root_dir}'.")
    return pd.DataFrame(records)
