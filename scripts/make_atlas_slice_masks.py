from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import yaml
from nibabel.processing import resample_from_to


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    if array.max(initial=0) > np.iinfo(np.uint8).max:
        out = array.astype(np.uint16)
    else:
        out = array.astype(np.uint8)
    ok, encoded = cv2.imencode(".png", out)
    if not ok:
        raise ValueError(f"Failed to encode atlas slice: {path}")
    encoded.tofile(str(path))


def resize_and_pad_mask(mask: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Match preprocessing/preprocess_new.py resize+pad geometry for label masks."""
    height, width = mask.shape
    if height > width:
        new_height = int(target_size)
        new_width = int(width * target_size / height)
    else:
        new_height = int(height * target_size / width)
        new_width = int(target_size)
    resized = cv2.resize(mask.astype(np.int32), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    pad_top = (target_size - new_height) // 2
    pad_bottom = target_size - new_height - pad_top
    pad_left = (target_size - new_width) // 2
    pad_right = target_size - new_width - pad_left
    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    ).astype(np.int32)


def export_atlas_slices(
    atlas_nii: str | Path,
    reference_nii: str | Path,
    output_dir: str | Path,
    axis: int = 2,
    z_start: int = 0,
    z_end: int = 0,
    target_size: int = 224,
    prefix: str = "atlas",
    rotate90: bool = True,
    nonzero_only: bool = False,
) -> dict[str, Any]:
    atlas_img = nib.load(str(atlas_nii))
    reference_img = nib.load(str(reference_nii))
    if atlas_img.shape[:3] != reference_img.shape[:3] or not np.allclose(atlas_img.affine, reference_img.affine):
        atlas_img = resample_from_to(atlas_img, reference_img, order=0)

    labels = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int32)
    labels[labels < 0] = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axis = int(axis)
    if axis not in {0, 1, 2}:
        raise ValueError("--axis must be 0, 1, or 2")
    n_slices = int(labels.shape[axis])
    start = max(0, int(z_start))
    end = n_slices if int(z_end) <= 0 else min(n_slices, int(z_end))
    if start >= end:
        raise ValueError(f"Invalid z range: start={start}, end={end}, n_slices={n_slices}")
    written = 0
    for slice_idx in range(start, end):
        slicer: list[slice | int] = [slice(None), slice(None), slice(None)]
        slicer[axis] = slice_idx
        mask = np.asarray(labels[tuple(slicer)])
        if rotate90:
            mask = np.rot90(mask)
        mask = resize_and_pad_mask(mask, target_size=target_size)
        if nonzero_only and not np.any(mask > 0):
            continue
        _write_png(output_dir / f"{prefix}_z{slice_idx:03d}.png", mask)
        written += 1

    manifest = {
        "atlas_nii": str(atlas_nii),
        "reference_nii": str(reference_nii),
        "output_dir": str(output_dir),
        "axis": axis,
        "z_start": start,
        "z_end": end,
        "target_size": int(target_size),
        "prefix": prefix,
        "rotate90": bool(rotate90),
        "n_slices": n_slices,
        "n_written": written,
        "unique_labels": [int(item) for item in sorted(np.unique(labels).tolist()) if int(item) > 0],
    }
    with open(output_dir / "atlas_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(manifest), handle, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an anatomical atlas label NIfTI into PNG slice masks for final.pdf-style ROI evaluation."
    )
    parser.add_argument("--atlas_nii", required=True, help="Label atlas NIfTI, for example AAL/Neuromorphometrics in MNI space.")
    parser.add_argument("--reference_nii", required=True, help="Reference subject/image NIfTI defining shape and affine.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_config", default="configs/data_config.yaml", help="Optional preprocessing config for axis/z range/target size.")
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--z_start", type=int, default=-1)
    parser.add_argument("--z_end", type=int, default=-1)
    parser.add_argument("--target_size", type=int, default=-1)
    parser.add_argument("--prefix", default="atlas")
    parser.add_argument("--no_rotate90", action="store_true")
    parser.add_argument("--nonzero_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    axis = args.axis
    z_start = args.z_start
    z_end = args.z_end
    target_size = args.target_size
    if args.data_config and Path(args.data_config).exists():
        with open(args.data_config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        axis = int(config.get("slice_axis", axis))
        if z_start < 0:
            z_start = int(config.get("slice_z_start", 0))
        if z_end < 0:
            z_end = int(config.get("slice_z_end", 0))
        if target_size < 0:
            target_size = int(config.get("target_size", 224))
    if z_start < 0:
        z_start = 0
    if z_end < 0:
        z_end = 0
    if target_size < 0:
        target_size = 224
    manifest = export_atlas_slices(
        atlas_nii=args.atlas_nii,
        reference_nii=args.reference_nii,
        output_dir=args.output_dir,
        axis=axis,
        z_start=z_start,
        z_end=z_end,
        target_size=target_size,
        prefix=args.prefix,
        rotate90=not args.no_rotate90,
        nonzero_only=args.nonzero_only,
    )
    print(f"Exported {manifest['n_written']} atlas slices to: {manifest['output_dir']}")
    print(f"Saved manifest to: {Path(manifest['output_dir']) / 'atlas_manifest.json'}")


if __name__ == "__main__":
    main()
