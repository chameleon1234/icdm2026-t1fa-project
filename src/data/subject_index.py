import json
import re
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd


GROUP_MAPPING: Dict[int, str] = {
    1: "CN",
    2: "SCD",
    3: "MCI",
    4: "AD",
}

SUBJECT_INDEX_COLUMNS = [
    "subject_id",
    "group_id",
    "group_name",
    "gender",
    "age",
    "edu",
    "MMSE",
    "split",
]


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse subject id from {value!r}")
    return f"sub-{int(match.group(1)):03d}"


def _load_split_lookup(split_json: str | Path) -> Dict[str, str]:
    with open(split_json, "r", encoding="utf-8") as handle:
        splits: Mapping[str, list[str]] = json.load(handle)

    lookup: Dict[str, str] = {}
    for split_name, subject_ids in splits.items():
        for subject_id in subject_ids:
            normalized = normalize_subject_id(subject_id)
            if normalized in lookup:
                raise ValueError(f"Subject {normalized} appears in multiple splits")
            lookup[normalized] = split_name
    return lookup


def load_subject_index(
    excel_path: str | Path,
    split_json: str | Path,
    sheet_name: str = "re_order",
) -> pd.DataFrame:
    excel_path = Path(excel_path)
    split_json = Path(split_json)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    required_columns = {"New_order", "groups", "gender", "age", "edu", "MMSE"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {excel_path}: {sorted(missing)}")

    split_lookup = _load_split_lookup(split_json)
    result = pd.DataFrame(
        {
            "subject_id": df["New_order"].map(normalize_subject_id),
            "group_id": df["groups"].astype(int),
            "gender": df["gender"],
            "age": df["age"],
            "edu": df["edu"],
            "MMSE": df["MMSE"],
        }
    )
    result["group_name"] = result["group_id"].map(GROUP_MAPPING)
    result["split"] = result["subject_id"].map(split_lookup)

    if result["group_name"].isna().any():
        bad_groups = sorted(result.loc[result["group_name"].isna(), "group_id"].unique().tolist())
        raise ValueError(f"Unknown group ids: {bad_groups}")
    if result["split"].isna().any():
        missing_subjects = result.loc[result["split"].isna(), "subject_id"].tolist()
        raise ValueError(f"Subjects missing from split file: {missing_subjects[:10]}")
    if result["subject_id"].duplicated().any():
        duplicated = result.loc[result["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Duplicated subjects in Excel file: {duplicated[:10]}")

    result = result[SUBJECT_INDEX_COLUMNS].sort_values("subject_id").reset_index(drop=True)
    if len(result) != 248:
        raise ValueError(f"Expected 248 subjects, found {len(result)}")
    return result


def save_subject_index(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

