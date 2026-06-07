import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ADNI paired T1/FA NIfTI volumes into paired PNG slices.")
    parser.add_argument("--archive", default="data/ADNI_data.7z")
    parser.add_argument("--manifest", default="outputs/icdm2026/adni_subject_manifest.csv")
    parser.add_argument("--output_root", default="data/adni_processed")
    parser.add_argument("--extract_root", default="data/adni_raw_extracted")
    parser.add_argument("--slice_manifest", default="", help="Defaults to <output_root>/adni_slice_manifest.csv")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to preprocess.")
    parser.add_argument("--subject_limit", type=int, default=0, help="Optional total paired subject limit for smoke tests.")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--slice_start", type=int, default=20)
    parser.add_argument("--slice_end", type=int, default=72)
    parser.add_argument("--min_brain_fraction", type=float, default=0.01)
    parser.add_argument("--clean", action="store_true", help="Delete output_root before preprocessing.")
    parser.add_argument("--force_extract", action="store_true", help="Overwrite cached extracted NIfTI files.")
    return parser.parse_args()


def make_slice_filename(subject: str, z_index: int) -> str:
    return f"sub-{subject}_z{int(z_index):03d}.png"


def select_slice_indices(
    volume_shape: tuple[int, ...],
    *,
    axis: int,
    slice_start: int,
    slice_end: int,
) -> list[int]:
    if axis < 0:
        axis += len(volume_shape)
    if axis < 0 or axis >= len(volume_shape):
        raise ValueError(f"axis={axis} is outside volume shape {volume_shape}")
    depth = int(volume_shape[axis])
    start = max(0, int(slice_start))
    end = min(depth, int(slice_end))
    if end <= start:
        raise ValueError(f"Empty slice range after clipping: start={start}, end={end}, depth={depth}")
    return list(range(start, end))


def normalize_t1_slice(image: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = data[data > 0]
    if nonzero.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.percentile(nonzero, 0.5))
    hi = float(np.percentile(nonzero, 99.5))
    if hi <= lo:
        hi = float(nonzero.max())
        lo = float(nonzero.min())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo)
    data[data < 0] = 0
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def normalize_fa_slice(image: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def resize_and_pad_uint8(image: np.ndarray, target_size: int) -> np.ndarray:
    data = np.clip(image.astype(np.float32), 0.0, 1.0)
    height, width = data.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image.")
    scale = float(target_size) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(data, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"Failed to save PNG: {path}")
    encoded.tofile(str(path))


def _take_slice(volume: np.ndarray, index: int, axis: int) -> np.ndarray:
    image = np.take(volume, int(index), axis=int(axis))
    return np.rot90(image)


def _brain_fraction(t1_slice: np.ndarray, fa_slice: np.ndarray) -> float:
    mask = (np.nan_to_num(t1_slice) > 0) | (np.nan_to_num(fa_slice) > 0)
    return float(mask.mean())


def write_subject_slice_pairs(
    *,
    subject_row: pd.Series,
    t1_volume: np.ndarray,
    fa_volume: np.ndarray,
    output_root: str | Path,
    axis: int,
    slice_start: int,
    slice_end: int,
    target_size: int,
    min_brain_fraction: float,
) -> list[dict[str, Any]]:
    if tuple(t1_volume.shape) != tuple(fa_volume.shape):
        raise ValueError(f"T1/FA shape mismatch for {subject_row['subject']}: {t1_volume.shape} vs {fa_volume.shape}")
    output_root = Path(output_root)
    split = str(subject_row["split"])
    subject = str(subject_row["subject"])
    rows: list[dict[str, Any]] = []

    for z_index in select_slice_indices(t1_volume.shape, axis=axis, slice_start=slice_start, slice_end=slice_end):
        t1_raw = _take_slice(t1_volume, z_index, axis)
        fa_raw = _take_slice(fa_volume, z_index, axis)
        brain_fraction = _brain_fraction(t1_raw, fa_raw)
        if brain_fraction < min_brain_fraction:
            continue

        filename = make_slice_filename(subject, z_index)
        t1_png = resize_and_pad_uint8(normalize_t1_slice(t1_raw), target_size)
        fa_png = resize_and_pad_uint8(normalize_fa_slice(fa_raw), target_size)
        t1_path = output_root / split / "t1_slices" / filename
        fa_path = output_root / split / "fa_slices" / filename
        _save_png(t1_path, t1_png)
        _save_png(fa_path, fa_png)
        rows.append(
            {
                "subject": subject,
                "split": split,
                "filename": filename,
                "z_index": int(z_index),
                "normalized_group": subject_row.get("normalized_group", "UNLABELED"),
                "raw_group": subject_row.get("raw_group", "UNLABELED"),
                "brain_fraction": brain_fraction,
                "t1_png": str(t1_path),
                "fa_png": str(fa_path),
            }
        )
    return rows


def _extract_needed_archive_members(
    *,
    archive_path: str | Path,
    member_paths: set[str],
    extract_root: str | Path,
    force_extract: bool,
) -> None:
    import libarchive

    extract_root = Path(extract_root)
    missing = {
        member.replace("\\", "/")
        for member in member_paths
        if force_extract or not (extract_root / member.replace("\\", "/")).exists()
    }
    if not missing:
        return
    extracted = 0
    with libarchive.file_reader(str(archive_path)) as entries:
        for entry in tqdm(entries, desc="Extracting ADNI NIfTI files"):
            member = entry.pathname.replace("\\", "/")
            if member not in missing:
                continue
            dest = extract_root / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as handle:
                for block in entry.get_blocks():
                    handle.write(block)
            extracted += 1
            if extracted == len(missing):
                break
    remaining = [member for member in missing if not (extract_root / member).exists()]
    if remaining:
        raise FileNotFoundError(f"Archive did not contain required members, first missing: {remaining[:5]}")


def load_nifti_volume(path: str | Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)


def load_preprocess_subjects(manifest_path: str | Path, splits: set[str], subject_limit: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"subject", "split", "t1_path", "fa_path", "is_paired"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"ADNI manifest is missing columns: {sorted(missing)}")
    frame = manifest[manifest["is_paired"].astype(bool) & manifest["split"].isin(splits)].copy()
    frame = frame.dropna(subset=["t1_path", "fa_path", "split"])
    frame = frame.sort_values(["split", "subject"]).reset_index(drop=True)
    if subject_limit > 0:
        frame = frame.head(subject_limit).copy()
    if frame.empty:
        raise ValueError(f"No paired ADNI subjects found for splits={sorted(splits)}")
    return frame


def preprocess_adni_slices(args: argparse.Namespace) -> pd.DataFrame:
    output_root = Path(args.output_root)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    subjects = load_preprocess_subjects(args.manifest, splits=splits, subject_limit=args.subject_limit)
    member_paths = set(subjects["t1_path"].astype(str)) | set(subjects["fa_path"].astype(str))
    _extract_needed_archive_members(
        archive_path=args.archive,
        member_paths=member_paths,
        extract_root=args.extract_root,
        force_extract=args.force_extract,
    )

    rows: list[dict[str, Any]] = []
    extract_root = Path(args.extract_root)
    for _, row in tqdm(list(subjects.iterrows()), desc="Writing ADNI PNG slices"):
        t1_volume = load_nifti_volume(extract_root / str(row["t1_path"]))
        fa_volume = load_nifti_volume(extract_root / str(row["fa_path"]))
        rows.extend(
            write_subject_slice_pairs(
                subject_row=row,
                t1_volume=t1_volume,
                fa_volume=fa_volume,
                output_root=output_root,
                axis=args.axis,
                slice_start=args.slice_start,
                slice_end=args.slice_end,
                target_size=args.target_size,
                min_brain_fraction=args.min_brain_fraction,
            )
        )

    slice_manifest = pd.DataFrame(rows)
    manifest_path = Path(args.slice_manifest) if args.slice_manifest else output_root / "adni_slice_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    slice_manifest.to_csv(manifest_path, index=False)
    summary = {
        "n_subjects": int(subjects["subject"].nunique()),
        "n_slices": int(len(slice_manifest)),
        "splits": slice_manifest["split"].value_counts().sort_index().to_dict() if not slice_manifest.empty else {},
        "slice_start": int(args.slice_start),
        "slice_end": int(args.slice_end),
        "axis": int(args.axis),
        "target_size": int(args.target_size),
        "min_brain_fraction": float(args.min_brain_fraction),
        "slice_manifest": str(manifest_path),
    }
    (output_root / "adni_preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"ADNI preprocessing: subjects={summary['n_subjects']} slices={summary['n_slices']}")
    print(f"Saved slice manifest: {manifest_path}")
    print(f"Saved summary: {output_root / 'adni_preprocess_summary.json'}")
    return slice_manifest


def main() -> None:
    preprocess_adni_slices(parse_args())


if __name__ == "__main__":
    main()
