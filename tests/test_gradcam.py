"""Unit tests for src.gradcam, using a tiny toy CNN instead of a real trained
architecture so the test runs in well under a second."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from src.gradcam.gradcam import make_gradcam_heatmap, overlay_heatmap


def _tiny_model(num_classes=3):
    inputs = layers.Input(shape=(16, 16, 3))
    x = layers.Conv2D(4, 3, padding="same", activation="relu", name="last_conv")(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def test_make_gradcam_heatmap_output_shape_matches_conv_layer():
    model = _tiny_model()
    img_array = tf.random.uniform((1, 16, 16, 3))

    heatmap, pred_index = make_gradcam_heatmap(img_array, model, "last_conv")

    assert heatmap.shape == (16, 16)
    assert 0 <= pred_index < 3


def test_make_gradcam_heatmap_values_in_unit_range():
    model = _tiny_model()
    img_array = tf.random.uniform((1, 16, 16, 3))

    heatmap, _ = make_gradcam_heatmap(img_array, model, "last_conv")

    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_make_gradcam_heatmap_respects_explicit_pred_index():
    model = _tiny_model(num_classes=3)
    img_array = tf.random.uniform((1, 16, 16, 3))

    _, pred_index = make_gradcam_heatmap(img_array, model, "last_conv", pred_index=1)

    assert pred_index == 1


def test_overlay_heatmap_returns_uint8_image_of_original_size():
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    heatmap = np.random.rand(16, 16).astype("float32")

    overlay = overlay_heatmap(image, heatmap, alpha=0.4)

    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8
