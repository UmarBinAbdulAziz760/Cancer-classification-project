"""
Architecture auto-detection, dataset/domain auto-detection (from a model's
class list), and a generic Grad-CAM last-conv-layer fallback - so the
backend can work with any dropped-in model file without requiring a
specific naming or folder convention.
"""

import re
from typing import List, Optional, Tuple

from src import config

# Order doesn't matter here - the four names are mutually exclusive substrings.
_ARCHITECTURE_KEYWORDS = config.ARCHITECTURES  # ["vgg16", "resnet50", "mobilenet", "densenet121"]

# Common shorthand people actually use in filenames (dropping the numeric
# suffix) - checked after the full names above.
_ARCHITECTURE_ALIASES = {
    "vgg": "vgg16",
    "resnet": "resnet50",
    "densenet": "densenet121",
    # "mobilenet" has no numeric suffix to drop - already matches directly.
}


def detect_architecture_from_name(model_name: str) -> Optional[str]:
    """Looks for one of the known architecture names (or a common shorthand
    alias) anywhere in `model_name` (case-insensitive, ignoring separators):
        "resnet50"                       -> "resnet50"
        "BreakHis_resnet50"              -> "resnet50"
        "my-best-MobileNet-model"        -> "mobilenet"
        "NCT-CRC-HE-100K_densenet_final" -> "densenet121"  (alias, no "121" in the filename)
        "final_model_v3"                 -> None
    """
    normalized = re.sub(r"[^a-z0-9]", "", model_name.lower())
    for arch in _ARCHITECTURE_KEYWORDS:
        if arch in normalized:
            return arch
    for alias, arch in _ARCHITECTURE_ALIASES.items():
        if alias in normalized:
            return arch
    return None


def detect_architecture_from_model(model) -> Optional[str]:
    """Fallback for when the architecture can't be read from the filename
    (e.g. deployed model files are simply called "final.keras" - see
    backend/app/registry.py). Identifies which of the four known CNN
    architectures a *loaded* model is by checking for each architecture's
    known Grad-CAM layer name (config.GRADCAM_LAST_CONV_LAYER) - since that
    layer name only exists on a model actually built with that backbone,
    it doubles as a reliable fingerprint. Returns None if none match, in
    which case the caller falls back to generic preprocessing / Grad-CAM
    layer detection, exactly as it already does for an unrecognised
    filename."""
    for arch, layer_name in config.GRADCAM_LAST_CONV_LAYER.items():
        try:
            model.get_layer(layer_name)
            return arch
        except ValueError:
            continue
    return None


_NICE_LABELS = {
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
    "mobilenet": "MobileNet",
    "densenet121": "DenseNet121",
}


def pretty_display_name(model_name: str, architecture: Optional[str]) -> str:
    """Human-friendly label for the frontend model picker. If an architecture
    was detected, show its canonical name directly (e.g. "DenseNet121")
    rather than title-casing the whole filename, which is often cluttered
    with a dataset prefix or run identifier the user doesn't need to see.
    Otherwise, title-cases the filename as a reasonable fallback."""
    if architecture and architecture in _NICE_LABELS:
        return _NICE_LABELS[architecture]
    pretty = model_name.replace("_", " ").replace("-", " ").strip()
    words = [w.capitalize() for w in pretty.split()]
    return " ".join(words) if words else model_name


# ---------------------------------------------------------------------------
# Dataset/domain auto-detection.
#
# IMPORTANT: this is what stops a model trained on one dataset from silently
# producing a meaningless prediction for an image from a completely
# different domain (e.g. a BreakHis breast-tissue image run through the
# NCT-CRC-HE-100K colorectal-tissue model). A softmax classifier has no
# built-in "none of the above" - it always forces its input into one of its
# trained classes. Detecting which dataset a model belongs to (from its own
# class list, not filename guesswork) lets the app tell the user what kind
# of image the model actually expects *before* they upload one.
# ---------------------------------------------------------------------------

_KNOWN_DATASET_CLASS_SETS = {
    "BreakHis": frozenset({"benign", "malignant"}),
    "NCT-CRC-HE-100K": frozenset({"adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"}),
    "ISIC_2019": frozenset({"ak", "bcc", "bkl", "df", "mel", "nv", "scc", "vasc"}),
}

_DATASET_DESCRIPTIONS = {
    "BreakHis": {
        "label": "Breast tissue biopsy (histopathology)",
        "hint": "Microscope images of H&E-stained breast tissue, classified as benign or malignant.",
    },
    "NCT-CRC-HE-100K": {
        "label": "Colorectal tissue biopsy (histopathology)",
        "hint": "Microscope images of H&E-stained colorectal tissue patches, classified into 9 tissue types.",
    },
    "ISIC_2019": {
        "label": "Skin lesion photo (dermoscopy)",
        "hint": "Close-up dermoscopic photos of skin lesions, classified into 8 diagnostic categories.",
    },
}


def detect_dataset_from_classes(classes: List[str]) -> Optional[str]:
    """Matches a model's exact class set against the three known datasets'
    class sets (case-insensitive). Returns None if it doesn't match any of
    them (e.g. a model trained on a custom/unseen dataset) - the model still
    works, it just won't get a friendly domain description."""
    normalized = frozenset(c.strip().lower() for c in classes)
    for dataset_name, known_classes in _KNOWN_DATASET_CLASS_SETS.items():
        if normalized == known_classes:
            return dataset_name
    return None


def dataset_description(dataset: Optional[str], classes: List[str]) -> Tuple[str, str]:
    """Returns (label, hint) describing what kind of image a model expects,
    for display before the user uploads anything. Falls back to a generic
    description (still listing the actual classes) when the dataset
    couldn't be identified."""
    if dataset and dataset in _DATASET_DESCRIPTIONS:
        info = _DATASET_DESCRIPTIONS[dataset]
        return info["label"], info["hint"]
    class_preview = ", ".join(classes[:6]) + ("..." if len(classes) > 6 else "")
    return "Custom / unrecognised dataset", f"Classifies into: {class_preview}"


def find_last_conv_layer_name(model) -> str:
    """Generic fallback: walks the model's layers in reverse and returns the
    name of the last one whose output has a 4D shape (batch, H, W, C) - i.e.
    the last layer that still preserves a spatial feature map. Works for any
    CNN, regardless of architecture, and is only used when the architecture
    couldn't be auto-detected from the filename (see detect_architecture_from_name)."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError(
        "Could not find a convolutional (4D output) layer in this model to run Grad-CAM against."
    )


def resolve_last_conv_layer(model, architecture: Optional[str]) -> str:
    """Prefers the known, validated layer name for `architecture`
    (config.GRADCAM_LAST_CONV_LAYER) if it actually exists on this model;
    otherwise falls back to find_last_conv_layer_name()."""
    if architecture is not None:
        candidate = config.GRADCAM_LAST_CONV_LAYER.get(architecture)
        if candidate is not None:
            try:
                model.get_layer(candidate)
                return candidate
            except ValueError:
                pass  # named layer not present on this particular model - fall through
    return find_last_conv_layer_name(model)
