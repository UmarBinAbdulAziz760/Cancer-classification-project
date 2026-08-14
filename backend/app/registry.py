"""
Classification-type registry.

Each of the three supported classification types (Breast Cancer,
Colorectal Cancer, Skin Lesion) has exactly one fixed deployed model - see
classification_types.py for the type -> folder mapping. This module does
NOT scan for arbitrary models, compare them, or pick a "best" one; it only
checks whether the expected `final.keras` + `classes.json` pair for each
classification type is present in its dedicated folder under `models/`,
and exposes it for loading. See models/README.md for the expected file
layout.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config  # noqa: E402

from .classification_types import CLASSIFICATION_TYPES

MODEL_FILENAME = "final.keras"
CLASSES_FILENAME = "classes.json"


@dataclass
class ClassificationEntry:
    key: str                       # e.g. "breast"
    label: str                     # e.g. "Breast Cancer"
    dataset_label: str             # short description of the expected image type
    dataset_hint: str              # one more sentence of detail
    model_path: Path
    classes: List[str]             # raw training-time labels, in model output order
    display_names: Dict[str, str]  # raw class label (lowercased) -> friendly name
    # Detected lazily from the loaded model on first use (see inference.py) -
    # deliberately NOT derived from a filename convention, since every
    # deployed model file is just called "final.keras".
    architecture: Optional[str] = None

    def friendly_class(self, raw_class: str) -> str:
        return self.display_names.get(raw_class.strip().lower(), raw_class)

    @property
    def friendly_classes(self) -> List[str]:
        return [self.friendly_class(c) for c in self.classes]


def _discover() -> Dict[str, ClassificationEntry]:
    found: Dict[str, ClassificationEntry] = {}

    for type_config in CLASSIFICATION_TYPES:
        folder = config.MODELS_DIR / type_config.folder
        model_path = folder / MODEL_FILENAME
        classes_path = folder / CLASSES_FILENAME

        if not model_path.exists() or not classes_path.exists():
            print(
                f"[registry] '{type_config.label}' not available yet: expected "
                f"'{model_path}' and '{classes_path}'. See models/README.md."
            )
            continue

        with open(classes_path) as f:
            classes = json.load(f)

        found[type_config.key] = ClassificationEntry(
            key=type_config.key,
            label=type_config.label,
            dataset_label=type_config.dataset_label,
            dataset_hint=type_config.dataset_hint,
            model_path=model_path,
            classes=classes,
            display_names=type_config.display_names,
        )

    return found


class ModelRegistry:
    """Thin wrapper so the registry can be refreshed without restarting the
    server (call `.refresh()` after copying in new model files)."""

    def __init__(self):
        self.refresh()

    def refresh(self) -> None:
        self._entries = _discover()

    def all(self) -> List[ClassificationEntry]:
        return list(self._entries.values())

    def get(self, key: str) -> ClassificationEntry:
        if key in self._entries:
            return self._entries[key]

        valid_keys = {c.key for c in CLASSIFICATION_TYPES}
        if key not in valid_keys:
            valid = ", ".join(sorted(valid_keys))
            raise KeyError(f"Unknown classification type '{key}'. Valid types: {valid}.")

        type_config = next(c for c in CLASSIFICATION_TYPES if c.key == key)
        raise KeyError(
            f"No deployed model found for '{type_config.label}'. Expected "
            f"'{config.MODELS_DIR / type_config.folder / MODEL_FILENAME}' and "
            f"'{config.MODELS_DIR / type_config.folder / CLASSES_FILENAME}'. "
            f"See models/README.md."
        )

    def is_empty(self) -> bool:
        return len(self._entries) == 0


registry = ModelRegistry()

# Kept as an alias so modules written against the old name (inference.py,
# gradcam_service.py) don't need to change their imports.
ModelEntry = ClassificationEntry
