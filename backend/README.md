# Backend (FastAPI) - also serves the frontend

This one process is the entire web app: FastAPI serves both the JSON API
and the plain HTML/CSS/JS frontend in `static/`. There's no separate
frontend server, no Node.js, and no build step.

## Setup

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

If you have a GPU and want to use it for inference, replace `tensorflow-cpu`
with `tensorflow` in `backend/requirements.txt` before installing.

Make sure at least one trained model is in place first - see
`../models/README.md`.

## Run

```bash
# from the project root
uvicorn backend.app.main:app --reload --port 8000
```

Open **http://localhost:8000** - that's the whole app.

Interactive API docs (Swagger UI) are auto-generated at
`http://localhost:8000/docs`, if you want to call the API directly.

## Endpoints

| Method | Path                       | Description                                                |
|--------|----------------------------|--------------------------------------------------------------|
| GET    | `/`                        | The web app (serves `static/index.html`)                    |
| GET    | `/api/health`              | Liveness check + count of available classification types   |
| GET    | `/api/classification-types`| Lists the 3 supported classification types and whether each has a deployed model |
| POST   | `/api/predict`             | multipart form: `file` (image), `classification_type` (`breast`/`colorectal`/`skin`) -> prediction + Grad-CAM overlay |

`classification_type` in `/api/predict` is required - there is no
model/architecture picker. See `../models/README.md` for where each
type's deployed model needs to live.

`/api/predict` response shape:

```json
{
  "classification_type": "breast",
  "classification_label": "Breast Cancer",
  "predicted_class": "Malignant",
  "confidence": 0.94,
  "probabilities": {"Benign": 0.06, "Malignant": 0.94},
  "gradcam_image": "data:image/png;base64,...",
  "original_image": "data:image/png;base64,...",
  "disclaimer": "This is a Model Prediction generated for research / decision-support purposes..."
}
```

## How the model registry works

`backend/app/classification_types.py` defines the three supported
classification types, each with a fixed subfolder under `models/` and a
map from raw training-time class labels to short, doctor-readable display
names. `backend/app/registry.py` just checks whether `models/<type>/final.keras`
+ `models/<type>/classes.json` exist for each type - there is no scanning,
comparing, or "best model" selection. The CNN architecture (VGG16/ResNet50/
MobileNet/DenseNet121) is never part of this mapping and is never shown to
the user; `backend/app/model_utils.py` figures out the right preprocessing
and Grad-CAM layer directly from the loaded model.

`GET /api/classification-types` re-checks the folders on every call, so
dropping in a newly deployed model is picked up without restarting the
server.

## Tests

```bash
pytest backend/tests/ -v
```

These use tiny throwaway Keras models built in test fixtures, so they run
in seconds and don't need any of the real trained models. They also check
that the static frontend routes (`/`, `/assets/app.js`, `/assets/styles.css`)
are served correctly.

## Notes

- Models are loaded lazily and cached in memory on first request; the first
  prediction for a given model will be slower than subsequent ones.
- CORS is wide open (`allow_origins=["*"]`) since this is a local,
  single-user demo, not a multi-tenant service - tighten this in
  `backend/app/main.py` if you ever deploy it somewhere more exposed.
