"""
Integration tests for the FastAPI backend, using tiny throwaway Keras
models (built in fixtures) instead of any of the real trained architectures,
so these run in seconds without needing the actual trained models.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tensorflow.keras import layers, models

from src import config
from backend.app.main import app
from backend.app.registry import registry, MODEL_FILENAME, CLASSES_FILENAME


def _build_tiny_model(folder, classes, conv_layer_name="conv_pw_13_relu"):
    folder.mkdir(parents=True, exist_ok=True)
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(4, 3, padding="same", activation="relu", name=conv_layer_name)(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(len(classes), activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.save(folder / MODEL_FILENAME)
    with open(folder / CLASSES_FILENAME, "w") as f:
        json.dump(classes, f)


def _dummy_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def client_with_breast_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    _build_tiny_model(tmp_path / "breast", ["benign", "malignant"])
    registry.refresh()
    yield TestClient(app)
    registry.refresh()


@pytest.fixture
def client_with_colorectal_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    classes = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    _build_tiny_model(tmp_path / "colorectal", classes, conv_layer_name="relu")
    registry.refresh()
    yield TestClient(app)
    registry.refresh()


@pytest.fixture
def client_with_no_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    registry.refresh()
    yield TestClient(app)
    registry.refresh()


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classification_types_endpoint_lists_all_three_types_always(client_with_no_models):
    response = client_with_no_models.get("/api/classification-types")
    assert response.status_code == 200
    types = response.json()["classification_types"]
    keys = {t["key"] for t in types}
    assert keys == {"breast", "colorectal", "skin"}
    # No models deployed in this fixture - none should be marked available,
    # and no architecture/model-file details should be exposed anywhere.
    assert all(t["available"] is False for t in types)
    body_text = json.dumps(types).lower()
    for leaked_term in ("vgg16", "resnet50", "densenet121", "mobilenet", "final.keras", "architecture"):
        assert leaked_term not in body_text


def test_classification_types_endpoint_marks_available_type(client_with_breast_model):
    response = client_with_breast_model.get("/api/classification-types")
    assert response.status_code == 200
    types = {t["key"]: t for t in response.json()["classification_types"]}
    assert types["breast"]["available"] is True
    assert types["breast"]["label"] == "Breast Cancer"
    assert set(types["breast"]["classes"]) == {"Benign", "Malignant"}
    assert types["colorectal"]["available"] is False
    assert types["skin"]["available"] is False


def test_predict_with_breast_classification_type(client_with_breast_model):
    response = client_with_breast_model.post(
        "/api/predict",
        data={"classification_type": "breast"},
        files={"file": ("test.png", _dummy_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["classification_type"] == "breast"
    assert body["classification_label"] == "Breast Cancer"
    assert body["predicted_class"] in ("Benign", "Malignant")
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == {"Benign", "Malignant"}
    assert body["gradcam_image"].startswith("data:image/png;base64,")
    assert body["original_image"].startswith("data:image/png;base64,")
    assert "not a clinical diagnosis" in body["disclaimer"]


def test_predict_translates_raw_labels_to_friendly_names(client_with_colorectal_model):
    response = client_with_colorectal_model.post(
        "/api/predict",
        data={"classification_type": "colorectal"},
        files={"file": ("test.png", _dummy_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    friendly_names = {
        "Tumour Tissue", "Mucus", "Cancer-Associated Stroma", "Fat Tissue", "Lymphocytes",
        "Muscle Tissue", "Tissue Debris", "Background", "Normal Tissue",
    }
    assert body["predicted_class"] in friendly_names
    assert set(body["probabilities"]) == friendly_names
    # Raw abbreviations should not leak into the API response.
    assert "TUM" not in body["probabilities"]


def test_predict_requires_classification_type(client_with_breast_model):
    response = client_with_breast_model.post(
        "/api/predict",
        files={"file": ("test.png", _dummy_png_bytes(), "image/png")},
    )
    assert response.status_code == 422  # missing required form field


def test_predict_returns_404_for_unknown_classification_type(client_with_breast_model):
    response = client_with_breast_model.post(
        "/api/predict",
        data={"classification_type": "does-not-exist"},
        files={"file": ("test.png", _dummy_png_bytes(), "image/png")},
    )
    assert response.status_code == 404


def test_predict_returns_404_for_unavailable_classification_type(client_with_breast_model):
    response = client_with_breast_model.post(
        "/api/predict",
        data={"classification_type": "skin"},
        files={"file": ("test.png", _dummy_png_bytes(), "image/png")},
    )
    assert response.status_code == 404
    assert "skin" in response.json()["detail"].lower() or "Skin" in response.json()["detail"]


def test_predict_rejects_non_image_file(client_with_breast_model):
    response = client_with_breast_model.post(
        "/api/predict",
        data={"classification_type": "breast"},
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 415


def test_index_page_is_served_at_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Cancer Image Classification" in response.text


def test_static_assets_are_served():
    client = TestClient(app)
    response = client.get("/assets/app.js")
    assert response.status_code == 200
    response = client.get("/assets/styles.css")
    assert response.status_code == 200


# --- Confidence warning surfaced through the API ---------------------------

def test_predict_flags_low_confidence_predictions(client_with_colorectal_model):
    # An untrained tiny model on a solid-colour image will not produce a
    # sharply peaked softmax over 9 classes - this should trip the
    # low-confidence warning rather than silently reporting a class.
    response = client_with_colorectal_model.post(
        "/api/predict",
        data={"classification_type": "colorectal"},
        files={"file": ("test.png", _dummy_png_bytes(), "image/png")},
    )
    body = response.json()
    if body["confidence"] < config.LOW_CONFIDENCE_THRESHOLD:
        assert body["low_confidence"] is True
        assert body["low_confidence_warning"] is not None
        assert "Colorectal Cancer" in body["low_confidence_warning"]
    else:
        assert body["low_confidence"] is False
        assert body["low_confidence_warning"] is None
