from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
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


def export_atlas_slices(
    atlas_nii: str | Path,
    reference_nii: str | Path,
    output_dir: str | Path,
    axis: int = 2,
    prefix: str = "atlas",
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
    written = 0
    for slice_idx in range(n_slices):
        slicer: list[slice | int] = [slice(None), slice(None), slice(None)]
        slicer[axis] = slice_idx
        mask = np.asarray(labels[tuple(slicer)])
        if axis == 0:
            mask = np.rot90(mask)
        elif axis == 1:
            mask = np.rot90(mask)
        if nonzero_only and not np.any(mask > 0):
            continue
        _write_png(output_dir / f"{prefix}_z{slice_idx:03d}.png", mask)
        written += 1

    manifest = {
        "atlas_nii": str(atlas_nii),
        "reference_nii": str(reference_nii),
        "output_dir": str(output_dir),
        "axis": axis,
        "prefix": prefix,
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
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--prefix", default="atlas")
    parser.add_argument("--nonzero_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_atlas_slices(
        atlas_nii=args.atlas_nii,
        reference_nii=args.reference_nii,
        output_dir=args.output_dir,
        axis=args.axis,
        prefix=args.prefix,
        nonzero_only=args.nonzero_only,
    )
    print(f"Exported {manifest['n_written']} atlas slices to: {manifest['output_dir']}")
    print(f"Saved manifest to: {Path(manifest['output_dir']) / 'atlas_manifest.json'}")


if __name__ == "__main__":
    main()
