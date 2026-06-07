import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GROUP_MAP = {
    "CN": "CN",
    "AD": "AD",
    "MCI": "MCI_spectrum",
    "EMCI": "MCI_spectrum",
    "LMCI": "MCI_spectrum",
    "_S_MC": "EXCLUDE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ADNI T1-FA subject manifest and subject-level split files.")
    parser.add_argument("--archive", default="data/ADNI_data.7z")
    parser.add_argument("--labels", default="data/subject_group_cleaned_filtered.csv")
    parser.add_argument("--output_root", default="outputs/icdm2026")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--inventory_csv", default="", help="Optional precomputed archive inventory CSV.")
    return parser.parse_args()


def normalize_adni_subject_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    name = Path(text).name
    match = re.search(r"sub-([A-Za-z0-9]+)", name)
    if match:
        text = match.group(1)
    if re.match(r"^\d{3}_S_\d+$", text):
        return text
    compact = re.match(r"^(\d{3})S(\d+)$", text)
    if compact:
        return f"{compact.group(1)}_S_{compact.group(2)}"
    return text


def normalize_group(group: Any) -> str:
    if group is None or pd.isna(group):
        return "UNLABELED"
    key = str(group).strip()
    if not key:
        return "UNLABELED"
    return GROUP_MAP.get(key, "EXCLUDE")


def scan_adni_archive(archive_path: str | Path) -> pd.DataFrame:
    import libarchive

    rows: list[dict[str, Any]] = []
    with libarchive.file_reader(str(archive_path)) as entries:
        for entry in entries:
            path = entry.pathname.replace("\\", "/")
            if not path.endswith(".nii.gz"):
                continue
            if "/FA/" in path:
                modality = "FA"
            elif "/T1/" in path:
                modality = "T1"
            else:
                continue
            rows.append(
                {
                    "subject": normalize_adni_subject_id(path),
                    "modality": modality,
                    "path": path,
                    "size": int(entry.size),
                }
            )
    if not rows:
        raise ValueError(f"No ADNI T1/FA NIfTI files found in archive: {archive_path}")
    return pd.DataFrame(rows).sort_values(["subject", "modality", "path"]).reset_index(drop=True)


def load_labels(labels_csv: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    required = {"Subject", "Group"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label CSV is missing columns: {sorted(missing)}")
    labels = labels[["Subject", "Group"]].copy()
    labels["subject"] = labels["Subject"].map(normalize_adni_subject_id)
    labels["raw_group"] = labels["Group"].astype(str).str.strip()
    labels["normalized_group"] = labels["Group"].map(normalize_group)
    return labels[["subject", "raw_group", "normalized_group"]].drop_duplicates("subject")


def build_subject_manifest(inventory: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        inventory.pivot_table(index="subject", columns="modality", values="path", aggfunc="first")
        .reset_index()
        .rename(columns={"T1": "t1_path", "FA": "fa_path"})
    )
    for column in ("t1_path", "fa_path"):
        if column not in pivot.columns:
            pivot[column] = pd.NA
    pivot["has_t1"] = pivot["t1_path"].notna()
    pivot["has_fa"] = pivot["fa_path"].notna()
    pivot["is_paired"] = pivot["has_t1"] & pivot["has_fa"]

    labels = labels.copy()
    if "Subject" in labels.columns or "Group" in labels.columns:
        labels = load_labels_from_frame(labels)
    manifest = pivot.merge(labels, on="subject", how="left")
    manifest["raw_group"] = manifest["raw_group"].fillna("UNLABELED")
    manifest["normalized_group"] = manifest["normalized_group"].fillna("UNLABELED")
    manifest["is_labeled"] = ~manifest["normalized_group"].isin(["UNLABELED", "EXCLUDE"])
    manifest["is_downstream_eligible"] = manifest["is_paired"] & manifest["is_labeled"]
    manifest["split_group"] = manifest["normalized_group"].where(manifest["is_labeled"], "UNLABELED")
    return manifest.sort_values("subject").reset_index(drop=True)


def load_labels_from_frame(labels: pd.DataFrame) -> pd.DataFrame:
    frame = labels.copy()
    if "subject" not in frame.columns:
        frame["subject"] = frame["Subject"].map(normalize_adni_subject_id)
    if "raw_group" not in frame.columns:
        frame["raw_group"] = frame["Group"].astype(str).str.strip()
    if "normalized_group" not in frame.columns:
        frame["normalized_group"] = frame["Group"].map(normalize_group)
    return frame[["subject", "raw_group", "normalized_group"]].drop_duplicates("subject")


def _stratified_train_val_test(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(split=pd.Series(dtype=str))
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must leave a positive test ratio.")
    stratify = frame["split_group"] if frame["split_group"].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        frame,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df["split_group"] if temp_df["split_group"].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        random_state=seed + 1,
        shuffle=True,
        stratify=stratify_temp,
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def assign_subject_splits(
    manifest: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not manifest["is_paired"].all():
        raise ValueError("assign_subject_splits expects paired subjects only.")
    if manifest["subject"].duplicated().any():
        raise ValueError("Subject IDs must be unique before splitting.")

    split_df = _stratified_train_val_test(
        manifest.copy(),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    ).sort_values("subject")
    split_json = {
        split: sorted(split_df.loc[split_df["split"].eq(split), "subject"].tolist())
        for split in ("train", "val", "test")
    }
    return split_df.reset_index(drop=True), split_json


def build_label_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ["all", "train", "val", "test"]:
        subset = manifest if split == "all" else manifest[manifest["split"].eq(split)]
        for group, count in subset["normalized_group"].value_counts(dropna=False).sort_index().items():
            rows.append({"split": split, "normalized_group": group, "n_subjects": int(count)})
    return pd.DataFrame(rows)


def write_outputs(manifest: pd.DataFrame, split_json: dict[str, list[str]], output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "adni_subject_manifest.csv"
    summary_path = root / "adni_label_summary.csv"
    split_path = root / "adni_split_subjects.json"
    manifest.to_csv(manifest_path, index=False)
    build_label_summary(manifest).to_csv(summary_path, index=False)
    split_payload = {
        "splits": split_json,
        "n_subjects": {split: len(subjects) for split, subjects in split_json.items()},
        "label_policy": {
            "MCI": "MCI_spectrum",
            "EMCI": "MCI_spectrum",
            "LMCI": "MCI_spectrum",
            "_S_MC": "EXCLUDE",
            "missing": "UNLABELED",
        },
    }
    split_path.write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest_path, "summary": summary_path, "splits": split_path}


def main() -> None:
    args = parse_args()
    if args.inventory_csv:
        inventory = pd.read_csv(args.inventory_csv)
    else:
        inventory = scan_adni_archive(args.archive)
    labels = load_labels(args.labels)
    manifest = build_subject_manifest(inventory, labels)
    paired = manifest[manifest["is_paired"]].copy()
    split_df, split_json = assign_subject_splits(
        paired,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    manifest = manifest.merge(split_df[["subject", "split"]], on="subject", how="left")
    paths = write_outputs(manifest, split_json, args.output_root)

    print(f"ADNI files: {len(inventory)} | subjects: {manifest['subject'].nunique()} | paired: {len(paired)}")
    print("Normalized groups:")
    print(manifest["normalized_group"].value_counts(dropna=False).to_string())
    print("Splits:")
    print(manifest["split"].value_counts(dropna=False).to_string())
    for key, path in paths.items():
        print(f"Saved {key}: {path}")


if __name__ == "__main__":
    main()
