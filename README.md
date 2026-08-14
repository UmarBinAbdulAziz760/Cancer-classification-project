# Cancer Image Classification - Web App

A comparative study of four CNN architectures (VGG16, ResNet50, MobileNet,
DenseNet121) for cancer image classification across three datasets
(BreakHis, NCT-CRC-HE-100K, ISIC 2019), plus a lightweight web app that
serves a trained model: upload an image, get a prediction and a Grad-CAM
overlay.

The web app is one Python process:
FastAPI serves both the API and a plain HTML/CSS/JS page directly. If you
can `pip install` and run one `uvicorn` command, you can run this.

This repo is the code behind the project report - training was carried out
on Kaggle (see `notebooks/training_notebook.ipynb`); this repo re-organises
that notebook into independently testable modules and adds the FastAPI
deployment described in the report's Methodology section (dataset loaders,
augmentation, model builders, evaluation, and Grad-CAM as separate,
unit-tested components).

## Repository structure

```
.
├── notebooks/
│   └── training_notebook.ipynb    Original Kaggle notebook (kept for provenance)
├── src/                            Reusable, independently testable modules
│   ├── config.py                   All paths/constants (Kaggle paths replaced with
│   │                                local, environment-overridable ones - see below)
│   ├── data/                        Dataset loaders, stratified splitting, class
│   │                                weights/SMOTE, tf.data pipeline
│   ├── augmentation/                Training-time image augmentation
│   ├── models/                      Transfer-learning model builders
│   ├── evaluation/                  Metrics + all figure/table generation
│   ├── gradcam/                     Grad-CAM heatmap generation and overlay
│   └── training/                    Two-phase fine-tuning loop + report generation
│                                     for a trained run
├── scripts/                         CLI entry points that call into src/
│   ├── build_dataset_indices.py     Preprocess all three datasets once
│   ├── train_model.py               Train one (dataset, architecture) pair
│   ├── evaluate_model.py            Generate figures/metrics for a trained pair
│   └── generate_methodology_report.py
├── tests/                           Unit tests for src/ (24 tests, no dataset/model needed)
├── models/                          Where trained .keras files go (see models/README.md)
├── data/                            raw/ (datasets), processed/ (index.csv per dataset),
│                                     imagenet_weights/ (backbone .h5 files)
├── outputs/                         Generated figures/tables (methodology + results)
└── backend/                         The web app - FastAPI + a plain static frontend
    ├── app/
    │   ├── main.py                    FastAPI app; also serves static/ - this is the
    │   │                              only thing you need to run
    │   ├── registry.py                Discovers whatever model file(s) are in models/
    │   ├── model_utils.py             Architecture auto-detection + generic Grad-CAM fallback
    │   ├── inference.py               Preprocessing + prediction
    │   └── gradcam_service.py         Grad-CAM overlay for one uploaded image
    ├── static/                        The frontend: index.html, styles.css, app.js
    │                                  (plain JS - no framework, no build step)
    └── tests/                         Integration tests (uses throwaway models)
```

## Kaggle -> local path changes

The original notebook hardcodes `/kaggle/input/...` (datasets, ImageNet
weights) and `/kaggle/working/...` (processed data, models, figures).
`src/config.py` replaces all of these with paths under the project root,
each overridable via an environment variable:

| Purpose                  | Kaggle path (original)                                   | Local default              | Override with        |
|---------------------------|-----------------------------------------------------------|-----------------------------|------------------------|
| BreakHis raw images       | `/kaggle/input/datasets/.../BreaKHis_v1/BreaKHis_v1/...`  | `data/raw/BreaKHis_v1/...`  | `BREAKHIS_DIR`         |
| NCT-CRC-HE-100K raw images| `/kaggle/input/datasets/.../NCT-CRC-HE-100K/NCT-CRC-HE-100K` | `data/raw/NCT-CRC-HE-100K/` | `NCT_CRC_DIR`          |
| ISIC 2019 raw images       | `/kaggle/input/datasets/salviohexia/...`                  | `data/raw/ISIC_2019/`       | `ISIC_DIR`             |
| ImageNet backbone weights  | `/kaggle/input/datasets/.../existing-weights/*.h5`        | `data/imagenet_weights/`    | `WEIGHTS_DIR`          |
| Processed dataset indices  | `/kaggle/working/processed/`                              | `data/processed/`           | `PROCESSED_ROOT`       |
| Trained models              | `/kaggle/working/outputs/models/`                         | `models/`                   | `MODELS_DIR`           |
| Figures/tables/logs        | `/kaggle/working/outputs/{figures,tables,logs}/`          | `outputs/`                  | `OUTPUT_ROOT`          |

You only need to set these if you're re-running preprocessing or training
locally. **The web app itself only reads from `models/`.**

## Quick start - just run the web app

You already have a trained model from Kaggle - you don't need the raw
datasets or ImageNet weights, Node.js, or anything besides Python, to run
the demo.

```bash
# 1. Copy your best trained model in (any name works - see models/README.md)
cp /path/to/kaggle/output/resnet50_final.keras models/
cp /path/to/kaggle/output/resnet50_classes.json models/

# 2. Install and run - one process, one command
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Open **http://localhost:8000** in a browser. That's it - upload an image
and click Classify.

If you drop in more than one model (e.g. `resnet50_final.keras` and
`vgg16_final.keras`), the page automatically shows a picker so you can
compare them; with only one, it skips straight to the upload form. See
`models/README.md` for the exact file-naming details.

## Re-running preprocessing/training/evaluation locally

Only needed if you want to reproduce or extend the training itself, not to
run the web app.

```bash
pip install -r requirements.txt

# Point these at wherever you've placed the raw datasets, if not under data/raw/
export BREAKHIS_DIR=/path/to/BreaKHis_v1/histology_slides/breast
export NCT_CRC_DIR=/path/to/NCT-CRC-HE-100K
export ISIC_DIR=/path/to/ISIC_2019          # one subfolder per class

python scripts/build_dataset_indices.py
python scripts/generate_methodology_report.py
python scripts/train_model.py --dataset BreakHis --architecture resnet50
python scripts/evaluate_model.py --dataset BreakHis --architecture resnet50
```

## Testing

```bash
pip install -r requirements.txt -r backend/requirements.txt
pytest              # runs tests/ and backend/tests/ (48 tests total)
```

Every module is covered without needing the real datasets or trained
models - loaders are tested against tiny synthetic files, the model builder
against a toy Keras model, Grad-CAM against a tiny CNN, and the backend API
end-to-end (including the static frontend routes) against throwaway models
built in test fixtures.

## Design notes

**Why no dataset picker in the UI.** BreakHis, NCT-CRC-HE-100K, and ISIC
2019 have entirely different class sets (binary vs. 9-class vs. 8-class).
Letting a user pick a dataset changes what the prediction *means*, and
someone who doesn't understand that distinction could walk away with a
genuinely misleading result. The app instead just uses whichever model(s)
you've placed in `models/` - see `models/README.md`.

**Why an architecture picker is fine.** Switching between VGG16 / ResNet50
/ MobileNet / DenseNet121 on the *same* model file's classes doesn't change
what the output means - just which model's opinion you're looking at. This
is also literally the comparison the underlying research project is about,
so showing it in the demo (when more than one architecture's model is
present) is a feature, not a complication.

## Deployment notes (read before treating this as more than a demo)

Matching the project report's Methodology section: **this deployment is
intentionally lightweight**. It has:

- **No user authentication.** Anyone who can reach the API can use it.
- **No database.** Uploaded images are decoded in memory, classified, and
  discarded - nothing is persisted, and there's no history/audit trail.
- **No production hardening** (rate limiting, HTTPS termination, model
  versioning, monitoring, etc.).

The purpose of this app is to demonstrate a practical integration of a
trained model and Grad-CAM, not to provide a production-ready clinical
system. **Do not use its output for actual diagnosis** - every prediction
response includes a disclaimer to this effect, and the page displays it
too.

If you were to extend this toward a real deployment, adding authentication
and persistent storage for uploaded images would introduce their own
privacy considerations (handling of patient-derived images) that are
explicitly out of scope here and would need separate review.
