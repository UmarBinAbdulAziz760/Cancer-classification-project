#!/usr/bin/env python3
"""
Builds the preprocessed index.csv (+ class_weights.json) for all three
datasets from their raw folders (config.BREAKHIS_DIR / NCT_CRC_DIR / ISIC_DIR).

Run this once before training:
    python scripts/build_dataset_indices.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.loaders import load_breakhis_index, load_folder_labelled_index
from src.data.splitting import preprocess_dataset


def main():
    print(f"BreakHis raw dir:  {config.BREAKHIS_DIR}")
    print(f"NCT-CRC raw dir:   {config.NCT_CRC_DIR}")
    print(f"ISIC 2019 raw dir: {config.ISIC_DIR}\n")

    breakhis_df = load_breakhis_index(config.BREAKHIS_DIR, label_mode=config.BREAKHIS_LABEL_MODE)
    preprocess_dataset("BreakHis", breakhis_df)

    nct_crc_df = load_folder_labelled_index(
        config.NCT_CRC_DIR,
        expected_classes={"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"},
    )
    preprocess_dataset("NCT-CRC-HE-100K", nct_crc_df)

    isic_df = load_folder_labelled_index(
        config.ISIC_DIR,
        expected_classes={"MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"},
    )
    preprocess_dataset("ISIC_2019", isic_df)


if __name__ == "__main__":
    main()
