from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_METHODS = [
    ("T1", "data/adni_processed/test/t1_slices"),
    ("FA_GT", "data/adni_processed/test/fa_slices"),
    ("Fidelity Flow", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Stage1 LPIPS+GAN", "outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL"),
    ("DBM", "outputs/icdm2026/predictions/ADNI_DBM_E100_K40_PRETRAINED"),
    ("DDIM", "outputs/icdm2026/predictions/ADNI_DDIM_E100_K50_PRETRAINED"),
    ("CycleGAN", "outputs/icdm2026/predictions/ADNI_CycleGAN_E50"),
    ("Pix2Pix", "outputs/icdm2026/predictions/ADNI_Pix2Pix_E50"),
    ("U-Net", "outputs/icdm2026/predictions/ADNI_UNet_E50"),
]

DEFAULT_SLICES = ["sub-006_S_6651_z042", "sub-019_S_6186_z034"]
DEFAULT_ROIS = {
    # x0, y0, x1, y1 in image-relative coordinates.
    "sub-006_S_6651_z042": (0.37, 0.30, 0.64, 0.61),
    "sub-019_S_6186_z034": (0.34, 0.32, 0.63, 0.63),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build horizontal ADNI method panels for selected slices.")
    parser.add_argument("--output_dir", default="outputs/icdm2026/figures/adni_method_slice_panels")
    parser.add_argument("--slice_id", action="append", default=DEFAULT_SLICES)
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        help="Override ROI as SLICE_ID:x0,y0,x1,y1 in relative coordinates, e.g. sub-006_S_6651_z042:0.35,0.3,0.65,0.6",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _read_gray(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("L")


def _roi_overrides(items: list[str]) -> dict[str, tuple[float, float, float, float]]:
    rois = dict(DEFAULT_ROIS)
    for item in items:
        if ":" not in item:
            raise ValueError(f"ROI override must be SLICE:x0,y0,x1,y1, got {item!r}")
        slice_id, coords = item.split(":", 1)
        values = tuple(float(part) for part in coords.split(","))
        if len(values) != 4:
            raise ValueError(f"ROI override needs four coordinates, got {item!r}")
        rois[slice_id] = values  # type: ignore[assignment]
    return rois


def _draw_roi(image: Image.Image, roi: tuple[float, float, float, float]) -> Image.Image:
    out = image.convert("RGB")
    width, height = out.size
    x0, y0, x1, y1 = roi
    box = [int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)]
    draw = ImageDraw.Draw(out)
    line_width = max(2, round(width / 96))
    for offset in range(line_width):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=(255, 0, 0),
        )
    return out


def _make_panel(
    slice_id: str,
    methods: list[tuple[str, str]],
    output_path: Path,
    *,
    roi: tuple[float, float, float, float] | None = None,
    dpi: int = 220,
) -> None:
    images: list[tuple[str, Image.Image]] = []
    filename = f"{slice_id}.png"
    for label, directory in methods:
        image = _read_gray(Path(directory) / filename)
        if roi is not None:
            image = _draw_roi(image, roi)
        images.append((label, image))

    fig_width = max(10.5, 1.45 * len(images))
    fig, axes = plt.subplots(1, len(images), figsize=(fig_width, 1.8), dpi=dpi)
    if len(images) == 1:
        axes = [axes]
    for axis, (label, image) in zip(axes, images):
        axis.imshow(np.asarray(image), cmap=None if image.mode == "RGB" else "gray", vmin=0, vmax=255)
        axis.set_title(label, fontsize=7, pad=3)
        axis.axis("off")
    fig.suptitle(slice_id, fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.78, bottom=0.02, wspace=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rois = _roi_overrides(args.roi)
    for slice_id in args.slice_id:
        normal_path = output_dir / f"{slice_id}_adni_methods_panel.png"
        roi_path = output_dir / f"{slice_id}_adni_methods_panel_roi.png"
        _make_panel(slice_id, DEFAULT_METHODS, normal_path, dpi=args.dpi)
        _make_panel(slice_id, DEFAULT_METHODS, roi_path, roi=rois.get(slice_id), dpi=args.dpi)
        print(f"Saved: {normal_path}")
        print(f"Saved: {roi_path}")


if __name__ == "__main__":
    main()
