"""
Fixed classification-type -> deployed-model mapping for the web app.

Each of the three classification types below has exactly one deployed
model, already chosen ahead of time from the project's experimental
results (see notebooks/ and outputs/tables/results/). This module
intentionally contains NO auto-detection, comparison, or "best model"
selection logic - it only defines:

  1. which folder under `models/` each classification type's files live in
     (see backend/app/registry.py, which just checks that folder for a
     fixed `final.keras` + `classes.json` pair), and
  2. how to translate a model's raw training-time class labels (e.g. "TUM",
     "MEL") into short, doctor-readable names for the web interface, while
     the raw labels themselves stay exactly what the model was trained on.

The CNN architecture used for each classification type is a research/
experimental detail and is deliberately not part of this mapping - it is
never surfaced to the end user (see backend/app/model_utils.py for how the
backend figures out the right preprocessing on its own).
"""

from typing import Dict, List, NamedTuple, Optional


class ClassificationTypeConfig(NamedTuple):
    key: str                       # stable identifier used in the API, e.g. "breast"
    label: str                     # user-facing name, e.g. "Breast Cancer"
    folder: str                    # subfolder of MODELS_DIR holding this type's model
    dataset_label: str             # short description of the expected image type
    dataset_hint: str              # one more sentence of detail
    display_names: Dict[str, str]  # raw class label (lowercased) -> friendly name


CLASSIFICATION_TYPES: List[ClassificationTypeConfig] = [
    ClassificationTypeConfig(
        key="breast",
        label="Breast Cancer",
        folder="breast",
        dataset_label="Breast tissue biopsy (histopathology)",
        dataset_hint="A microscope image of H&E-stained breast tissue.",
        display_names={
            "benign": "Benign",
            "malignant": "Malignant",
        },
    ),
    ClassificationTypeConfig(
        key="colorectal",
        label="Colorectal Cancer",
        folder="colorectal",
        dataset_label="Colorectal tissue biopsy (histopathology)",
        dataset_hint="A microscope image of an H&E-stained colorectal tissue patch.",
        display_names={
            "tum": "Tumour Tissue",
            "muc": "Mucus",
            "str": "Cancer-Associated Stroma",
            "adi": "Fat Tissue",
            "lym": "Lymphocytes",
            "mus": "Muscle Tissue",
            "deb": "Tissue Debris",
            "back": "Background",
            "norm": "Normal Tissue",
        },
    ),
    ClassificationTypeConfig(
        key="skin",
        label="Skin Lesion",
        folder="skin",
        dataset_label="Skin lesion photo (dermoscopy)",
        dataset_hint="A close-up dermoscopic photo of a skin lesion.",
        display_names={
            "mel": "Melanoma",
            "nv": "Mole (Melanocytic Nevus)",
            "bcc": "Basal Cell Carcinoma",
            "scc": "Squamous Cell Carcinoma",
            "df": "Dermatofibroma",
            "vasc": "Vascular Lesion",
            "bkl": "Benign Keratosis",
            "ak": "Actinic Keratosis",
        },
    ),
]

CLASSIFICATION_TYPES_BY_KEY: Dict[str, ClassificationTypeConfig] = {
    c.key: c for c in CLASSIFICATION_TYPES
}


def get_type_config(key: str) -> Optional[ClassificationTypeConfig]:
    return CLASSIFICATION_TYPES_BY_KEY.get(key)


def friendly_class_name(classification_key: str, raw_class: str) -> str:
    """Maps a raw training-time class label to its short, user-friendly
    display name for the given classification type. Falls back to the raw
    label unchanged if the type or label isn't recognised, so an unexpected
    class never disappears from the UI."""
    type_config = CLASSIFICATION_TYPES_BY_KEY.get(classification_key)
    if type_config is None:
        return raw_class
    return type_config.display_names.get(raw_class.strip().lower(), raw_class)
