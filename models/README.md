# Trained model artifacts

This folder is where the web app looks for trained models. `.keras` files
are large binaries that don't belong in version control (see
`.gitignore`) - copy your own trained models in here from wherever your
Kaggle notebook saved them (`/kaggle/working/outputs/models/`).

## Fixed layout: one folder per classification type

The web app supports three independent classification tasks. Each one has
its own folder, and each folder holds exactly one deployed model - the one
already chosen from the project's experimental results for that task.
There is **no** automatic best-model detection, comparison, or selection:
whatever is in a classification type's folder is what gets used for it.

```text
models/
├── breast/
│   ├── final.keras       <- deployed model for Breast Cancer (BreakHis)
│   └── classes.json      <- ["benign", "malignant"]
│
├── colorectal/
│   ├── final.keras       <- deployed model for Colorectal Cancer (NCT-CRC-HE-100K)
│   └── classes.json      <- ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
│
└── skin/
    ├── final.keras       <- deployed model for Skin Lesion (ISIC 2019)
    └── classes.json      <- ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
```

Each `classes.json` is the ordered list of class names used during
training for that model - exactly what `src/training/train.py` /
`scripts/train_model.py` produces. Don't mix and match: a model trained on
one dataset must never be paired with another dataset's `classes.json`.

`{folder}/final.keras` and `{folder}/classes.json` must have matching base
names (`final` / `classes`, not e.g. `best` or `latest`) - see the
`_best.keras` / `_latest.keras` note at the bottom.

## Where these files come from

Produced automatically by `src/training/train.py` /
`scripts/train_model.py`, e.g.:

```bash
python scripts/train_model.py --dataset BreakHis --architecture resnet50
```

which saves `BreakHis_resnet50_final.keras` and
`BreakHis_resnet50_classes.json` under `outputs/models/`. Once you've
picked the deployed model for a classification type from your experiment
results, copy (and rename) those two files into the matching folder above:

```bash
cp outputs/models/BreakHis_resnet50_final.keras    models/breast/final.keras
cp outputs/models/BreakHis_resnet50_classes.json   models/breast/classes.json
```

If you already trained on Kaggle, copy the equivalent two files per run
from `/kaggle/working/outputs/models/`.

## Architecture is not part of this layout on purpose

The four CNN architectures explored during experimentation (VGG16,
ResNet50, MobileNet, DenseNet121) are research details. The web app never
asks the user to pick one, and the deployed file is always just
`final.keras` regardless of which architecture it is - the backend
figures out the right preprocessing and Grad-CAM layer on its own by
inspecting the loaded model (see `backend/app/model_utils.py`).

## `_best.keras` / `_latest.keras` files

You may also have `{run_name}_phase1_best.keras`,
`{run_name}_phase2_latest.keras`, etc. from checkpointing during training -
the web app only looks for `final.keras`. These are safe to leave out of
the `models/<type>/` folders.

## If a classification type's files are missing

The app still starts up fine - `GET /api/classification-types` reports
that type as unavailable, and the UI disables it in the selector rather
than erroring out. Predicting against an unavailable type returns a clear
404 pointing at the expected file paths.
