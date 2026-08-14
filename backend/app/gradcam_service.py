"""
Wraps src.gradcam (the same code used to generate the project report's
Grad-CAM figures) to produce a base64-encoded overlay PNG for a single
uploaded image.
"""

import base64
import io

import numpy as np
from PIL import Image

from src.gradcam.gradcam import make_gradcam_heatmap, overlay_heatmap

from .model_utils import resolve_last_conv_layer
from .registry import ModelEntry

_last_conv_layer_cache = {}


def _to_data_url(image_array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _last_conv_layer_for(model, entry: ModelEntry) -> str:
    """Resolved once per model path and cached, since scanning a model's
    layers is small but pointless to repeat on every single request."""
    key = str(entry.model_path)
    if key not in _last_conv_layer_cache:
        _last_conv_layer_cache[key] = resolve_last_conv_layer(model, entry.architecture)
    return _last_conv_layer_cache[key]


def generate_gradcam_overlay(model, entry: ModelEntry, model_input: np.ndarray,
                              display_image: np.ndarray, pred_index: int) -> str:
    """Runs Grad-CAM against the model's final convolutional layer for the
    class the model actually predicted, and returns the overlay as a base64
    PNG data URL."""
    last_conv_layer = _last_conv_layer_for(model, entry)
    heatmap, _ = make_gradcam_heatmap(model_input, model, last_conv_layer, pred_index=pred_index)
    overlay = overlay_heatmap(display_image, heatmap, alpha=0.4)
    return _to_data_url(overlay)


def image_to_data_url(image_array: np.ndarray) -> str:
    return _to_data_url(image_array)
