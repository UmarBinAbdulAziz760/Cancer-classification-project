"""Unit tests for src.data.splitting - stratified splitting and class weights."""

import pandas as pd
import pytest

from src.data.splitting import compute_class_weights, stratified_split


def _synthetic_df(n_per_class=40):
    rows = []
    for label, count in (("benign", n_per_class), ("malignant", n_per_class * 3)):
        for i in range(count):
            rows.append({"filepath": f"{label}_{i}.png", "label": label})
    return pd.DataFrame(rows)


def test_stratified_split_produces_all_three_splits():
    df = _synthetic_df()
    out = stratified_split(df, 0.70, 0.15, 0.15, random_state=42)
    assert set(out["split"]) == {"train", "val", "test"}
    assert len(out) == len(df)


def test_stratified_split_preserves_class_ratio_in_train_split():
    df = _synthetic_df(n_per_class=100)  # 100 benign, 300 malignant -> 25%/75%
    out = stratified_split(df, 0.70, 0.15, 0.15, random_state=42)
    train = out[out["split"] == "train"]
    fraction_malignant = (train["label"] == "malignant").mean()
    assert fraction_malignant == pytest.approx(0.75, abs=0.03)


def test_stratified_split_rejects_ratios_that_dont_sum_to_one():
    df = _synthetic_df()
    with pytest.raises(AssertionError):
        stratified_split(df, 0.5, 0.3, 0.3)


def test_compute_class_weights_gives_higher_weight_to_minority_class():
    df = _synthetic_df(n_per_class=25)  # 25 benign, 75 malignant
    weights = compute_class_weights(df)
    # alphabetical order -> benign=0, malignant=1
    assert weights[0] > weights[1]


def test_compute_class_weights_balanced_classes_are_equal():
    df = _synthetic_df(n_per_class=50)
    # use only "benign" duplicated under a different name to force balance
    df2 = pd.DataFrame({
        "filepath": [f"a_{i}.png" for i in range(50)] + [f"b_{i}.png" for i in range(50)],
        "label": ["a"] * 50 + ["b"] * 50,
    })
    weights = compute_class_weights(df2)
    assert weights[0] == pytest.approx(weights[1])
