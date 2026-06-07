from pathlib import Path

import pandas as pd


def test_normalize_subject_id_preserves_adni_site_and_subject_code():
    from src.data.subject_index import normalize_subject_id

    assert normalize_subject_id("002_S_0413") == "sub-002_S_0413"
    assert normalize_subject_id("sub-002_S_0413") == "sub-002_S_0413"
    assert normalize_subject_id("sub-002S0413") == "sub-002_S_0413"
    assert normalize_subject_id(1) == "sub-001"


def test_load_subject_index_uses_excel_mapping_and_subject_splits():
    from src.data.subject_index import load_subject_index

    df = load_subject_index(
        excel_path="data/data_information.xlsx",
        split_json="data/processed/dataset_splits.json",
    )

    assert list(df.columns) == [
        "subject_id",
        "group_id",
        "group_name",
        "gender",
        "age",
        "edu",
        "MMSE",
        "split",
    ]
    assert df.shape == (248, 8)
    assert df["subject_id"].str.match(r"sub-\d{3}$").all()
    assert df.groupby("group_name").size().to_dict() == {
        "AD": 30,
        "CN": 92,
        "MCI": 70,
        "SCD": 56,
    }
    assert df.groupby("split").size().to_dict() == {
        "test": 38,
        "train": 173,
        "val": 37,
    }


def test_save_subject_index_roundtrip(tmp_path):
    from src.data.subject_index import load_subject_index, save_subject_index

    output_path = tmp_path / "subject_index.csv"
    df = load_subject_index(
        excel_path="data/data_information.xlsx",
        split_json="data/processed/dataset_splits.json",
    )

    save_subject_index(df, output_path)

    reloaded = pd.read_csv(output_path)
    assert output_path.exists()
    assert reloaded.shape == (248, 8)
    assert reloaded.loc[0, "subject_id"].startswith("sub-")
