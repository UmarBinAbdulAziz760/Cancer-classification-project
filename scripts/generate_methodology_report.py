#!/usr/bin/env python3
"""
Generates the methodology-section figures and tables (class distribution,
example images, dataset summary, hyperparameters, model complexity) for all
three datasets. Requires scripts/build_dataset_indices.py to have been run
first.

    python scripts/generate_methodology_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf

from src import config
from src.data.pipeline import load_index
from src.models.builders import build_transfer_model, count_total_params, count_trainable_params, unfreeze_top_layers
from src.evaluation.reporting import (
    dataset_summary_table, hyperparameters_table, model_complexity_table,
    plot_class_distribution, plot_model_complexity, plot_sample_images,
)


def main():
    print("=== Generating methodology-section figures/tables ===")
    loaders = {
        "BreakHis": lambda: load_index("BreakHis"),
        "NCT-CRC-HE-100K": lambda: load_index("NCT-CRC-HE-100K"),
        "ISIC_2019": lambda: load_index("ISIC_2019"),
    }
    for dataset_name, loader in loaders.items():
        df = loader()
        plot_class_distribution(df, dataset_name, config.FIGURES_METHODOLOGY_DIR / f"{dataset_name}_class_distribution.png")
        plot_sample_images(df, dataset_name, config.FIGURES_METHODOLOGY_DIR / f"{dataset_name}_sample_images.png")
        dataset_summary_table(df, dataset_name, config.TABLES_METHODOLOGY_DIR / f"{dataset_name}_dataset_summary")

    hyperparameters_table(config.TABLES_METHODOLOGY_DIR / "hyperparameters")

    params_by_arch = {}
    for arch in config.ARCHITECTURES:
        input_shape = config.TARGET_SIZE[arch] + (3,)
        model, base_model = build_transfer_model(arch, input_shape=input_shape, num_classes=2,
                                                   dense_units=config.DENSE_UNITS, dropout_rate=config.DROPOUT_RATE)
        total = count_total_params(model)
        unfreeze_top_layers(base_model.layers, fraction=config.FINE_TUNE_UNFREEZE_FRACTION)
        trainable = count_trainable_params(model)
        params_by_arch[arch] = {"total": total, "trainable": trainable}
        del model, base_model
        tf.keras.backend.clear_session()

    plot_model_complexity(params_by_arch, config.FIGURES_METHODOLOGY_DIR / "model_complexity.png")
    model_complexity_table(params_by_arch, config.TABLES_METHODOLOGY_DIR / "model_complexity")


if __name__ == "__main__":
    main()
