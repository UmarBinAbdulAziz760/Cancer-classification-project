"""
Grad-CAM heatmap generation and overlay (Selvaraju et al., 2017).

`make_gradcam_heatmap` and `overlay_heatmap` are the two functions the
FastAPI backend calls directly (backend/app/gradcam_service.py) to produce
the overlay shown in the web app for a single user-uploaded image - the
same code path used to generate the report's Grad-CAM figures.
"""

from typing import List, Optional

import matplotlib
import numpy as np
import tensorflow as tf

from .. import config
from ..evaluation.reporting import plot_gradcam_grid


def make_gradcam_heatmap(img_array, model, last_conv_layer_name: str, pred_index: Optional[int] = None):
    """
    Args:
        img_array: preprocessed image batch, shape (1, H, W, 3), already run
            through the architecture's own preprocess_input.
        model: a compiled/loaded Keras model.
        last_conv_layer_name: name of the final convolutional layer to read
            activations from (see config.GRADCAM_LAST_CONV_LAYER).
        pred_index: class index to explain; defaults to the model's own
            top prediction.

    Returns:
        (heatmap, pred_index): heatmap is a 2D numpy array in [0, 1].
    """
    grad_model = tf.keras.models.Model(inputs=model.inputs,
                                        outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index


def overlay_heatmap(image_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resizes `heatmap` to match `image_uint8` and blends it on top using the
    'jet' colormap, returning a uint8 (H, W, 3) array."""
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], (image_uint8.shape[0], image_uint8.shape[1])).numpy().squeeze()
    try:
        jet = matplotlib.colormaps["jet"]  # matplotlib >= 3.7
    except AttributeError:  # pragma: no cover - older matplotlib fallback
        jet = matplotlib.cm.get_cmap("jet")
    jet_colors = (jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    return (jet_colors * alpha + image_uint8 * (1 - alpha)).astype(np.uint8)


def run_gradcam_examples(model, test_ds, architecture: str, class_names: List[str], run_name: str):
    """Report-generation helper: draws one batch from `test_ds`, runs Grad-CAM
    on the first `config.GRADCAM_NUM_EXAMPLES` images, and saves a grid
    figure under config.FIGURES_RESULTS_DIR."""
    last_conv_layer = config.GRADCAM_LAST_CONV_LAYER[architecture]
    originals, overlays, titles = [], [], []
    for images, labels in test_ds.take(1):
        images = images.numpy(); labels = labels.numpy()
        for i in range(min(config.GRADCAM_NUM_EXAMPLES, len(images))):
            heatmap, pred_idx = make_gradcam_heatmap(images[i:i+1], model, last_conv_layer)
            display_img = images[i] - images[i].min()
            display_img = (display_img / (display_img.max() + 1e-8) * 255).astype(np.uint8)
            overlays.append(overlay_heatmap(display_img, heatmap))
            originals.append(display_img)
            titles.append(f"true={class_names[labels[i]]}\npred={class_names[pred_idx]}")
        break
    if originals:
        plot_gradcam_grid(originals, overlays, titles, config.FIGURES_RESULTS_DIR / f"{run_name}_gradcam.png")
