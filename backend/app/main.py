"""
FastAPI backend for the cancer-image-classification demo.

This single process serves BOTH the API and the frontend (backend/static/) -
there's nothing else to install or build. Run it with:

    uvicorn backend.app.main:app --reload --port 8000

then open http://localhost:8000 in a browser.

Deliberately lightweight, matching the project report's deployment scope:
no user authentication and no database - an uploaded image is decoded in
memory, classified, explained with Grad-CAM, and the result is returned to
the browser. Nothing is written to disk and nothing persists between
requests. This is a translational demo of the trained models, not a
production clinical system.
"""

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src import config

from .classification_types import CLASSIFICATION_TYPES
from .gradcam_service import generate_gradcam_overlay, image_to_data_url
from .inference import get_model, predict
from .registry import registry
from .schemas import ClassificationTypeInfo, ClassificationTypesResponse, PredictionResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Cancer Image Classification - Demo API",
    description=(
        "Lightweight demo backend: upload a histopathology or dermoscopy image, "
        "get a prediction from a fine-tuned CNN and a Grad-CAM overlay. "
        "Not a medical device; no auth; no persistent storage. See project README."
    ),
    version="1.0.0",
)

# Permissive CORS - this is a local single-user demo, not a multi-tenant service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "classification_types_available": len(registry.all())}


@app.get("/api/classification-types", response_model=ClassificationTypesResponse)
def list_classification_types():
    """Returns the three supported classification types (Breast Cancer,
    Colorectal Cancer, Skin Lesion) and whether each one's deployed model
    is currently present in models/ (see backend/app/registry.py +
    models/README.md). No CNN architecture or model-file details are
    included here - those are implementation details, not something the
    end user chooses."""
    registry.refresh()  # cheap file-existence check - picks up newly copied-in models without a restart
    available_by_key = {e.key: e for e in registry.all()}
    types = [
        ClassificationTypeInfo(
            key=type_config.key,
            label=type_config.label,
            available=type_config.key in available_by_key,
            dataset_label=type_config.dataset_label,
            dataset_hint=type_config.dataset_hint,
            classes=available_by_key[type_config.key].friendly_classes
            if type_config.key in available_by_key
            else [],
        )
        for type_config in CLASSIFICATION_TYPES
    ]
    return ClassificationTypesResponse(classification_types=types)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    classification_type: str = Form(
        ..., description="Which classification type to run: 'breast', 'colorectal', or 'skin'."
    ),
):
    registry.refresh()

    try:
        entry = registry.get(classification_type)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415,
                             detail=f"Unsupported file type '{file.content_type}'. Upload a PNG/JPEG/WebP image.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded file as an image.")

    result = predict(entry, image)
    model = get_model(entry)
    gradcam_data_url = generate_gradcam_overlay(
        model, entry, result["model_input"], result["display_image"], result["pred_index"]
    )
    original_data_url = image_to_data_url(result["display_image"])

    # Raw training-time labels (e.g. "TUM", "MEL") are used for the actual
    # prediction above; only the API response is translated to short,
    # doctor-readable names (see classification_types.py).
    friendly_predicted_class = entry.friendly_class(result["predicted_class"])
    friendly_probabilities = {
        entry.friendly_class(raw_class): prob for raw_class, prob in result["probabilities"].items()
    }

    low_confidence = result["confidence"] < config.LOW_CONFIDENCE_THRESHOLD
    low_confidence_warning = None
    if low_confidence:
        low_confidence_warning = (
            f"This prediction was only {result['confidence']*100:.0f}% confident. That often means "
            f"the uploaded image doesn't clearly resemble anything expected for {entry.label} "
            f"({entry.dataset_label}). Treat this result with caution."
        )

    return PredictionResponse(
        classification_type=entry.key,
        classification_label=entry.label,
        dataset_label=entry.dataset_label,
        predicted_class=friendly_predicted_class,
        confidence=result["confidence"],
        probabilities=friendly_probabilities,
        gradcam_image=gradcam_data_url,
        original_image=original_data_url,
        low_confidence=low_confidence,
        low_confidence_warning=low_confidence_warning,
    )


# --- Static frontend (no Node/npm - plain HTML/CSS/JS served directly) -----
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="static-assets")

    @app.get("/")
    def serve_index():
        return FileResponse(STATIC_DIR / "index.html")
