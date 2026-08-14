#!/usr/bin/env python3
"""
Evaluates one already-trained (dataset, architecture) combination: generates
the training-curve, confusion-matrix, ROC-curve, and Grad-CAM figures, and
appends a row to the results table.

Example:
    python scripts/evaluate_model.py --dataset BreakHis --architecture resnet50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.pipeline import get_datasets
from src.training.evaluate import evaluate_run
from src.evaluation.reporting import results_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=config.DATASETS)
    parser.add_argument("--architecture", required=True, choices=config.ARCHITECTURES)
    args = parser.parse_args()

    _, _, test_ds, label_encoder, num_classes = get_datasets(args.dataset, args.architecture)
    class_names = list(label_encoder.classes_)

    result = evaluate_run(args.dataset, args.architecture, test_ds, class_names, num_classes)
    results_table([result], config.TABLES_RESULTS_DIR / f"{args.dataset}_{args.architecture}_results")
    print(result)


if __name__ == "__main__":
    main()
