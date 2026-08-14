"""Keras callback construction shared by both fine-tuning phases."""

from typing import Optional

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .. import config


def build_callbacks(run_name: str, csv_append: bool = False, csv_log_name: Optional[str] = None):
    """
    run_name is the checkpoint-file base name for this run
    (e.g. "BreakHis_vgg16_phase1" or "BreakHis_vgg16_phase2").
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = config.MODELS_DIR / f"{run_name}_best.keras"
    latest_checkpoint_path = config.MODELS_DIR / f"{run_name}_latest.keras"
    log_path = config.LOGS_DIR / (csv_log_name or f"{run_name}_history.csv")

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(latest_checkpoint_path),
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
            verbose=1,
        ),
        CSVLogger(
            str(log_path),
            append=csv_append,
        ),
    ]
