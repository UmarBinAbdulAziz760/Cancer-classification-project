"""Unit tests for backend.app.model_utils - architecture auto-detection and
the generic Grad-CAM last-conv-layer fallback."""

import pytest
from tensorflow.keras import layers, models

from backend.app.model_utils import (
    detect_architecture_from_name,
    find_last_conv_layer_name,
    pretty_display_name,
    resolve_last_conv_layer,
)


@pytest.mark.parametrize("model_name,expected", [
    ("resnet50", "resnet50"),
    ("BreakHis_resnet50", "resnet50"),
    ("my-best-MobileNet-model", "mobilenet"),
    ("VGG16_final_v2", "vgg16"),
    ("densenet121", "densenet121"),
    ("final_model_v3", None),
    ("best_model", None),
    ("NCT-CRC-HE-100K_densenet", "densenet121"),  # real-world case: shorthand, no numeric suffix
    ("BreakHis_resnet", "resnet50"),
    ("my_vgg_model", "vgg16"),
])
def test_detect_architecture_from_name(model_name, expected):
    assert detect_architecture_from_name(model_name) == expected


def test_pretty_display_name_uses_canonical_casing_for_known_architecture():
    assert pretty_display_name("resnet50", "resnet50") == "ResNet50"
    assert pretty_display_name("mobilenet", "mobilenet") == "MobileNet"


def test_pretty_display_name_falls_back_to_title_case_for_unknown():
    assert pretty_display_name("best_model", None) == "Best Model"


def _tiny_model(last_conv_name="last_conv"):
    inputs = layers.Input(shape=(16, 16, 3))
    x = layers.Conv2D(4, 3, padding="same", activation="relu", name=last_conv_name)(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return models.Model(inputs, outputs)


def test_find_last_conv_layer_name_finds_the_conv_layer():
    model = _tiny_model(last_conv_name="my_conv")
    assert find_last_conv_layer_name(model) == "my_conv"


def test_resolve_last_conv_layer_prefers_known_architecture_name_when_present():
    model = _tiny_model(last_conv_name="conv_pw_13_relu")  # matches mobilenet's known layer name
    assert resolve_last_conv_layer(model, "mobilenet") == "conv_pw_13_relu"


def test_resolve_last_conv_layer_falls_back_when_known_name_absent():
    # architecture says "mobilenet" but this particular model doesn't actually
    # have a layer called "conv_pw_13_relu" - should fall back gracefully.
    model = _tiny_model(last_conv_name="some_other_conv")
    assert resolve_last_conv_layer(model, "mobilenet") == "some_other_conv"


def test_resolve_last_conv_layer_falls_back_when_architecture_unknown():
    model = _tiny_model(last_conv_name="some_other_conv")
    assert resolve_last_conv_layer(model, None) == "some_other_conv"


# --- Dataset/domain auto-detection (the actual bug this was added for:
# a model trained on one dataset silently "classifying" an image from a
# completely different domain into one of its own classes) -----------------

from backend.app.model_utils import dataset_description, detect_dataset_from_classes


def test_detect_dataset_from_classes_breakhis():
    assert detect_dataset_from_classes(["benign", "malignant"]) == "BreakHis"


def test_detect_dataset_from_classes_nct_crc():
    classes = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    assert detect_dataset_from_classes(classes) == "NCT-CRC-HE-100K"


def test_detect_dataset_from_classes_isic():
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    assert detect_dataset_from_classes(classes) == "ISIC_2019"


def test_detect_dataset_from_classes_is_case_and_order_insensitive():
    assert detect_dataset_from_classes(["Malignant", "BENIGN"]) == "BreakHis"
    assert detect_dataset_from_classes(["tum", "str", "adi", "back", "deb", "lym", "muc", "mus", "norm"]) == "NCT-CRC-HE-100K"


def test_detect_dataset_from_classes_unknown_returns_none():
    assert detect_dataset_from_classes(["cat", "dog"]) is None


def test_dataset_description_known_dataset_gives_plain_language_label():
    label, hint = dataset_description("NCT-CRC-HE-100K", ["ADI", "BACK", "DEB"])
    assert "colorectal" in label.lower()
    assert "histopathology" in label.lower()


def test_dataset_description_unknown_dataset_falls_back_to_class_list():
    label, hint = dataset_description(None, ["cat", "dog"])
    assert "custom" in label.lower() or "unrecognised" in label.lower()
    assert "cat" in hint and "dog" in hint
