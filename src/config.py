"""
Central configuration for the cancer-image-classification project.

This module replaces the hardcoded /kaggle/input and /kaggle/working paths
used in the original Kaggle notebook (notebooks/training_notebook.ipynb) with
paths relative to the project root, all overridable via environment
variables. Nothing else in `src/` should hardcode a path directly - import
the constants from here instead, so the whole codebase works unchanged
whether it's running on Kaggle, a laptop, or a server.

Directory-naming note (Kaggle -> local):
    Kaggle mirror                                                  Local default (this file)
    /kaggle/input/.../BreaKHis_v1/BreaKHis_v1/histology_slides/... -> data/raw/BreaKHis_v1/histology_slides/...
    /kaggle/input/.../NCT-CRC-HE-100K/NCT-CRC-HE-100K/...          -> data/raw/NCT-CRC-HE-100K/...
    /kaggle/input/.../ISIC_2019_Training_Input/... (per-class dirs) -> data/raw/ISIC_2019/...
    /kaggle/working/...                                            -> outputs/, data/processed/, models/

If your local copy of a dataset still has the doubled Kaggle folder name
(e.g. `BreaKHis_v1/BreaKHis_v1/...`), either rename it on disk or point the
relevant environment variable (BREAKHIS_DIR / NCT_CRC_DIR / ISIC_DIR)
straight at wherever it actually lives - you don't need to move files to
match the defaults below.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Raw dataset locations (only needed for re-running preprocessing/training;
# the web app itself only needs MODELS_DIR below).
# ---------------------------------------------------------------------------
DATA_RAW_ROOT = Path(os.environ.get("DATA_RAW_ROOT", PROJECT_ROOT / "data" / "raw"))

BREAKHIS_DIR = Path(os.environ.get(
    "BREAKHIS_DIR",
    DATA_RAW_ROOT / "BreaKHis_v1" / "histology_slides" / "breast",
))
NCT_CRC_DIR = Path(os.environ.get(
    "NCT_CRC_DIR",
    DATA_RAW_ROOT / "NCT-CRC-HE-100K",
))
# Expects one subfolder per class directly under this directory:
# MEL/ NV/ BCC/ AK/ BKL/ DF/ VASC/ SCC/
ISIC_DIR = Path(os.environ.get("ISIC_DIR", DATA_RAW_ROOT / "ISIC_2019"))

# ---------------------------------------------------------------------------
# ImageNet backbone weights (.h5, no-top). Only needed to train/fine-tune
# from scratch - NOT needed to run the web app, which loads already
# fine-tuned .keras files from MODELS_DIR.
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", PROJECT_ROOT / "data" / "imagenet_weights"))
WEIGHTS_PATHS = {
    "vgg16": WEIGHTS_DIR / "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    "resnet50": WEIGHTS_DIR / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
    "mobilenet": WEIGHTS_DIR / "mobilenet_1_0_224_tf_no_top.h5",
    "densenet121": WEIGHTS_DIR / "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
}

# ---------------------------------------------------------------------------
# Outputs (processed indices, trained models, figures, tables)
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
PROCESSED_ROOT = Path(os.environ.get("PROCESSED_ROOT", PROJECT_ROOT / "data" / "processed"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = OUTPUT_ROOT / "logs"
FIGURES_METHODOLOGY_DIR = OUTPUT_ROOT / "figures" / "methodology"
FIGURES_RESULTS_DIR = OUTPUT_ROOT / "figures" / "results"
TABLES_METHODOLOGY_DIR = OUTPUT_ROOT / "tables" / "methodology"
TABLES_RESULTS_DIR = OUTPUT_ROOT / "tables" / "results"

for _d in (PROCESSED_ROOT, MODELS_DIR, LOGS_DIR, FIGURES_METHODOLOGY_DIR,
           FIGURES_RESULTS_DIR, TABLES_METHODOLOGY_DIR, TABLES_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility & split ratios
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# ---------------------------------------------------------------------------
# Architectures & datasets
# ---------------------------------------------------------------------------
ARCHITECTURES = ["vgg16", "resnet50", "mobilenet", "densenet121"]
DATASETS = ["BreakHis", "NCT-CRC-HE-100K", "ISIC_2019"]



TARGET_SIZE = {
    "vgg16": (224, 224),
    "resnet50": (224, 224),
    "mobilenet": (224, 224),
    "densenet121": (224, 224),
}
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Two-phase fine-tuning protocol
# ---------------------------------------------------------------------------
FREEZE_BASE_EPOCHS = 5
FINE_TUNE_EPOCHS = 15
FINE_TUNE_UNFREEZE_FRACTION = 0.30
BASE_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.30
DENSE_UNITS = 256
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# ---------------------------------------------------------------------------
# Dataset-specific options
# ---------------------------------------------------------------------------
# BreakHis label granularity: "binary" (benign/malignant) or "subtype" (8-way)
BREAKHIS_LABEL_MODE = "binary"

IMBALANCE_STRATEGY = "class_weight"  # "class_weight" | "smote" | "none"
SMOTE_MAX_SAMPLES_PER_CLASS = 1500
SMOTE_THUMBNAIL_SIZE = (32, 32)

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
GRADCAM_LAST_CONV_LAYER = {
    "vgg16": "block5_conv3",
    "resnet50": "conv5_block3_out",
    "mobilenet": "conv_pw_13_relu",
    "densenet121": "relu",
}
GRADCAM_NUM_EXAMPLES = 6

# ---------------------------------------------------------------------------
# Web app: low-confidence warning
#
# A softmax classifier always forces its input into one of its trained
# classes, even if the image is nothing like anything it was trained on
# (e.g. a skin-lesion photo run through a tissue-histology model). This
# threshold is a coarse, imperfect safety net: if the model's own top-class
# probability is below it, the app shows a warning that the prediction may
# not be reliable. It will NOT catch every out-of-domain image (a
# confidently wrong prediction on an unfamiliar image is a known failure
# mode of softmax classifiers) - the dataset/domain description shown
# before upload (see backend/app/model_utils.py) is the primary safeguard;
# this is a second line of defence, not a substitute for it.
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.60

