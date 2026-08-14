"""Unit tests for src.data.loaders - dataset indexing, in isolation from any
real dataset (everything runs against tiny synthetic files in a tmp_path)."""

import pytest
from PIL import Image

from src.data.loaders import load_breakhis_index, load_folder_labelled_index


def _make_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)


def test_load_breakhis_index_parses_class_magnification_and_patient(tmp_path):
    root = tmp_path / "breast"
    _make_png(root / "benign" / "SOB" / "adenosis" / "SOB_B_A-14-1234" / "40X" / "img1.png")
    _make_png(root / "malignant" / "SOB" / "ductal_carcinoma" / "SOB_M_DC-14-5678" / "100X" / "img2.png")

    df = load_breakhis_index(root, label_mode="binary")

    assert set(df["label"]) == {"benign", "malignant"}
    assert set(df["magnification"]) == {"40X", "100X"}
    row = df[df["label"] == "benign"].iloc[0]
    assert row["patient_id"] == "SOB_B_A-14-1234"


def test_load_breakhis_index_subtype_mode_uses_tumour_type(tmp_path):
    root = tmp_path / "breast"
    _make_png(root / "benign" / "SOB" / "adenosis" / "SOB_B_A-14-1234" / "40X" / "img1.png")

    df = load_breakhis_index(root, label_mode="subtype")
    assert df.iloc[0]["label"] == "adenosis"


def test_load_breakhis_index_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_breakhis_index(tmp_path / "does_not_exist")


def test_load_breakhis_index_empty_root_raises(tmp_path):
    root = tmp_path / "breast"
    root.mkdir()
    with pytest.raises(RuntimeError):
        load_breakhis_index(root)


def test_load_folder_labelled_index_reads_class_subfolders(tmp_path):
    root = tmp_path / "NCT-CRC-HE-100K"
    _make_png(root / "TUM" / "a.png")
    _make_png(root / "TUM" / "b.png")
    _make_png(root / "NORM" / "c.png")

    df = load_folder_labelled_index(root, expected_classes={"TUM", "NORM"})

    assert len(df) == 3
    assert df["label"].value_counts()["TUM"] == 2
    assert df["label"].value_counts()["NORM"] == 1


def test_load_folder_labelled_index_warns_on_missing_expected_class(tmp_path, capsys):
    root = tmp_path / "ISIC_2019"
    _make_png(root / "MEL" / "a.png")

    load_folder_labelled_index(root, expected_classes={"MEL", "NV"})

    captured = capsys.readouterr()
    assert "NV" in captured.out
