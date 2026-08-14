#!/usr/bin/env python3
"""
Trains one (dataset, architecture) combination end to end (phase 1 + phase 2)
and saves the final model + class list to config.MODELS_DIR.

Example:
    python scripts/train_model.py --dataset BreakHis --architecture resnet50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.pipeline import get_class_weight_dict, get_datasets
from src.training.train import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=config.DATASETS)
    parser.add_argument("--architecture", required=True, choices=config.ARCHITECTURES)
    args = parser.parse_args()

    train_ds, val_ds, test_ds, label_encoder, num_classes = get_datasets(args.dataset, args.architecture)
    class_weight = get_class_weight_dict(args.dataset)

    train_model(
        dataset_name=args.dataset,
        architecture=args.architecture,
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=num_classes,
        class_weight=class_weight,
        class_names=list(label_encoder.classes_),
    )


if __name__ == "__main__":
    main()
