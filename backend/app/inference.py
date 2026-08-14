"""
Inference: loads a trained model (cached in memory) and runs a single
uploaded image through the appropriate preprocessing for its architecture
(src.data.pipeline.PREPROCESS_FUNCTIONS), so predictions match what the
model actually saw during training.

If the model's architecture couldn't be auto-detected from its filename
(see backend/app/model_utils.py), a generic [-1, 1] rescaling is used
instead - reasonable for any ImageNet-style CNN, though renaming the file
to include the architecture (e.g. "resnet50_final.keras") is recommended
for exact parity with training-time preprocessing.

No image is ever written to disk here - everything happens on the decoded
bytes in memory and is discarded once the response is returned (see the
deployment note in the root README: this app has no database and no
persistent storage of uploads by design).
"""

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.pipeline import PREPROCESS_FUNCTIONS

from .model_utils import detect_architecture_from_model, detect_architecture_from_name
from .registry import ModelEntry

DEFAULT_TARGET_SIZE = (224, 224)


def _generic_preprocess(raw: np.ndarray) -> np.ndarray:
    """Rescales to [-1, 1] - a common default across ImageNet-style CNNs,
    used only when the architecture couldn't be identified."""
    return (raw / 127.5) - 1.0


@lru_cache(maxsize=16)
def _load_keras_model(model_path_str: str):
    """Cached so the same model isn't re-loaded from disk on every request."""
    print(f"[inference] loading model from {model_path_str}")
    return tf.keras.models.load_model(model_path_str, compile=False)


def get_model(entry: ModelEntry):
    model = _load_keras_model(str(entry.model_path))
    if entry.architecture is None:
        # Deployed model files are always just "final.keras" (see registry.py),
        # so filename-based detection won't match anything - fall back to
        # identifying the architecture from the loaded model itself. Detected
        # once per model path and cached on the entry, not re-derived per request.
        entry.architecture = (
            detect_architecture_from_name(entry.model_path.stem)
            or detect_architecture_from_model(model)
        )
    return model


def _target_size_for(model, architecture) -> Tuple[int, int]:
    if architecture is not None:
        from src import config
        return config.TARGET_SIZE.get(architecture, DEFAULT_TARGET_SIZE)
    try:
        shape = model.inputs[0].shape  # (None, H, W, 3)
        if shape[1] and shape[2]:
            return (int(shape[1]), int(shape[2]))
    except Exception:
        pass
    return DEFAULT_TARGET_SIZE


def preprocess_for_inference(image: Image.Image, model, architecture) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: a PIL image, any size/mode.
        model: the loaded Keras model (used to infer input size when the
            architecture is unknown).
        architecture: one of config.ARCHITECTURES, or None if it couldn't
            be auto-detected from the filename.

    Returns:
        (model_input, display_image):
            model_input: float32 array, shape (1, H, W, 3), ready for
                model.predict() / Grad-CAM.
            display_image: uint8 array, shape (H, W, 3), min-max rescaled
                for on-screen display - matches the rescaling used for the
                project's Grad-CAM figures, so the web app's overlay looks
                the same style as the report's.
    """
    target_size = _target_size_for(model, architecture)
    image = image.convert("RGB").resize(target_size)
    raw = np.asarray(image).astype("float32")

    preprocess_fn = PREPROCESS_FUNCTIONS.get(architecture, _generic_preprocess) if architecture else _generic_preprocess
    preprocessed = preprocess_fn(raw.copy())
    model_input = np.expand_dims(preprocessed, axis=0)

    display = raw - raw.min()
    display = (display / (display.max() + 1e-8) * 255).astype("uint8")

    return model_input, display


def predict(entry: ModelEntry, image: Image.Image) -> Dict:
    """Runs the full predict step for one uploaded image against one
    registered model."""
    model = get_model(entry)
    model_input, display_image = preprocess_for_inference(image, model, entry.architecture)

    probs = model.predict(model_input, verbose=0)[0]
    pred_index = int(np.argmax(probs))

    return {
        "predicted_class": entry.classes[pred_index],
        "pred_index": pred_index,
        "confidence": float(probs[pred_index]),
        "probabilities": {cls: float(p) for cls, p in zip(entry.classes, probs)},
        "model_input": model_input,
        "display_image": display_image,
    }
