"""Unit tests for src.models.builders, in isolation from any real trained
weights - build_transfer_model's ImageNet-weights loading is tested only for
its failure path (missing file), since downloading real weights isn't
appropriate in a unit test."""

import pytest
import tensorflow as tf
from tensorflow.keras import layers

from src.models.builders import build_transfer_model, unfreeze_top_layers


def _dummy_layers(n, batchnorm_indices=()):
    made = []
    for i in range(n):
        if i in batchnorm_indices:
            layer = layers.BatchNormalization()
        else:
            layer = layers.Dense(4)
        layer.build((None, 4))
        made.append(layer)
    return made


def test_unfreeze_top_layers_freezes_bottom_and_unfreezes_top():
    layer_list = _dummy_layers(10)
    unfreeze_top_layers(layer_list, fraction=0.30)

    n_frozen_expected = int(10 * 0.70)
    for layer in layer_list[:n_frozen_expected]:
        assert layer.trainable is False
    for layer in layer_list[n_frozen_expected:]:
        assert layer.trainable is True


def test_unfreeze_top_layers_keeps_batchnorm_frozen_even_in_unfrozen_region():
    # BatchNorm layer placed inside the "top fraction" that would otherwise
    # be unfrozen - it must stay frozen (Section III-B: BatchNorm layers are
    # kept frozen throughout phase 2 regardless of position).
    layer_list = _dummy_layers(10, batchnorm_indices=(8,))
    unfreeze_top_layers(layer_list, fraction=0.30)

    assert layer_list[8].trainable is False


def test_build_transfer_model_raises_clear_error_when_weights_missing(monkeypatch, tmp_path):
    from src import config
    missing_path = tmp_path / "does_not_exist.h5"
    monkeypatch.setitem(config.WEIGHTS_PATHS, "mobilenet", missing_path)

    with pytest.raises(FileNotFoundError, match="ImageNet weights not found"):
        build_transfer_model("mobilenet", input_shape=(224, 224, 3), num_classes=2)
