"""
Two-phase fine-tuning training loop.

This is a cleaned-up, environment-agnostic version of the training loop
used in the Kaggle notebook: the notebook's `run_stage`/`run_training`
resumable-session machinery was written specifically to survive Kaggle's
per-session time limits (manually re-entering "completed_epochs" between
sessions, falling back to hardcoded dataset-version paths for one specific
run that got interrupted, etc.) and isn't something you want to carry into
a normal long-running environment. `train_model` below does the same two
phases in one call; if you need to stop and resume on a machine with its
own session limits, wrap it with your own checkpoint-reload logic using
`config.MODELS_DIR` and the `*_latest.keras` / CSV log files this already
writes.
"""

import json
from typing import Optional

from tensorflow.keras import optimizers

from .. import config
from ..models.builders import build_transfer_model, unfreeze_top_layers
from .callbacks import build_callbacks


def train_model(dataset_name: str, architecture: str, train_ds, val_ds, num_classes: int,
                 class_weight: Optional[dict] = None, class_names: Optional[list] = None):
    """Runs phase 1 (head-only) then phase 2 (fine-tuning) for one
    (dataset, architecture) combination, saving the final model and a
    combined training history.

    Returns (model, combined_history).
    """
    run_name = f"{dataset_name}_{architecture}"
    input_shape = config.TARGET_SIZE[architecture] + (3,)

    if class_names is not None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.MODELS_DIR / f"{run_name}_classes.json", "w") as f:
            json.dump(list(class_names), f, indent=2)

    model, base_model = build_transfer_model(
        architecture, input_shape=input_shape, num_classes=num_classes,
        dense_units=config.DENSE_UNITS, dropout_rate=config.DROPOUT_RATE,
        learning_rate=config.BASE_LEARNING_RATE,
    )

    print(f"\n=== [{run_name}] Phase 1: training classifier head (base frozen) ===")
    history_1 = model.fit(train_ds, validation_data=val_ds, epochs=config.FREEZE_BASE_EPOCHS,
                           class_weight=class_weight, callbacks=build_callbacks(f"{run_name}_phase1"))

    print(f"\n=== [{run_name}] Phase 2: fine-tuning top {config.FINE_TUNE_UNFREEZE_FRACTION:.0%} of base layers ===")
    unfreeze_top_layers(base_model.layers, fraction=config.FINE_TUNE_UNFREEZE_FRACTION)
    model.compile(optimizer=optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_2 = model.fit(train_ds, validation_data=val_ds, epochs=config.FINE_TUNE_EPOCHS,
                           class_weight=class_weight, callbacks=build_callbacks(f"{run_name}_phase2"))

    combined_history = {key: list(history_1.history[key]) + list(history_2.history[key])
                         for key in history_1.history}
    combined_history["phase_boundary"] = len(history_1.history["loss"])

    final_path = config.MODELS_DIR / f"{run_name}_final.keras"
    model.save(final_path)
    print(f"[{run_name}] Saved final model -> {final_path}")

    return model, combined_history
