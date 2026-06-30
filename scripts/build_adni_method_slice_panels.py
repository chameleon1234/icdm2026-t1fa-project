from __future__ import annotations

import argparse
import re
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

COMPACT_ADNI_METHODS = [
    ("T1", "data/adni_processed/test/t1_slices"),
    ("FA_GT", "data/adni_processed/test/fa_slices"),
    ("Fidelity Flow", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Stage1 LPIPS+GAN", "outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL"),
    ("DBM", "outputs/icdm2026/predictions/ADNI_DBM_E100_K40_PRETRAINED"),
    ("Pix2Pix", "outputs/icdm2026/predictions/ADNI_Pix2Pix_E50"),
    ("U-Net", "outputs/icdm2026/predictions/ADNI_UNet_E50"),
]

PRIVATE_SINGLE_STAGE2_METHODS = [
    ("T1", "data/processed/test/t1_slices"),
    ("FA_GT", "data/processed/test/fa_slices"),
    ("Stage1 E030", "outputs/icdm2026/predictions/PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030"),
    ("DS Hybrid", "outputs/icdm2026/predictions/PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL"),
]

ADNI_SINGLE_STAGE2_SMOKE_METHODS = [
    ("T1", "data/adni_processed/test/t1_slices"),
    ("FA_GT", "data/adni_processed/test/fa_slices"),
    ("Stage1 Single", "outputs/icdm2026/predictions/ADNI_STAGE1_SINGLE_SHARP_STRIPE_SMOKE_4096_E5"),
    ("DS Hybrid", "outputs/icdm2026/predictions/ADNI_DS_HYBRID_SINGLE_SMOKE_4096_E5"),
    ("Old Fidelity Flow", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
]

ADNI_SINGLE_STAGE2_MULTIHEAD_METHODS = [
    ("T1", "data/adni_processed/test/t1_slices"),
    ("FA_GT", "data/adni_processed/test/fa_slices"),
    ("Stage1 Single", "outputs/icdm2026/predictions/ADNI_STAGE1_SINGLE_SHARP_STRIPE_SMOKE_4096_E5"),
    ("DS Hybrid", "outputs/icdm2026/predictions/ADNI_DS_HYBRID_SINGLE_SMOKE_4096_E5"),
    ("DS Multihead", "outputs/icdm2026/predictions/ADNI_DS_MULTIHEAD_SINGLE_PROBE_2048_E3"),
    ("Old Fidelity Flow", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
]

ADNI_SINGLE_SHARP_STAGE2_METHODS = [
    ("T1", "data/adni_processed/test/t1_slices"),
    ("FA_GT", "data/adni_processed/test/fa_slices"),
    ("Stage1 Single Sharp", "outputs/icdm2026/predictions/ADNI_STAGE1_SINGLE_SHARP_FULL_E30"),
    ("Single Fidelity Flow", "outputs/icdm2026/predictions/ADNI_SINGLE_FIDELITY_FLOW_SHARP_STAGE1_PROBE_4096_E5"),
    ("Single DS Multihead", "outputs/icdm2026/predictions/ADNI_SINGLE_DS_MULTIHEAD_SHARP_STAGE1_PROBE_4096_E5"),
    ("DS Multihead Balanced", "outputs/icdm2026/predictions/ADNI_SINGLE_DS_MULTIHEAD_BALANCED_8192_E2"),
    ("DS Multihead Final", "outputs/icdm2026/predictions/ADNI_SINGLE_DS_MULTIHEAD_FINAL_12000_E8"),
    ("Old 5-slice Flow", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
]

ADNI_TEXTURE_PROBE_METHODS = [
    ("T1", "data/adni_processed/test/t1_slices"),
    ("FA_GT", "data/adni_processed/test/fa_slices"),
    ("LowGuard", "outputs/icdm2026/predictions/ADNI_FIDELITY_FLOW_ARTIFACT_LOWGUARD_TEMPLATE_STRONG"),
    ("LightGuard", "outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_4096_E8_BEST_K8"),
    ("Old Fidelity", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Low+Light HF", "outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A035"),
    ("Low+Light A060", "outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A060"),
    ("Low+Light A080", "outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080"),
    ("A080+DS", "outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_FAIR_4096_E8"),
    ("A080+DS ROIBoost", "outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_ROIBOOST_4096_E12"),
    ("A080+DS DiseaseROI", "outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_4096_E8"),
    ("A080+DS HFKeep", "outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6"),
    ("Low+Old HF", "outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_OLDFIDELITY_HF_A035"),
    ("Gated MB", "outputs/icdm2026/predictions/ADNI_STAGE1_GATED_MB_FULLBASE_A025_CAP0065_SMOKE_K10"),
]

ADNI_FINAL_A080_DS_METHODS = [
    ("T1", "data/adni_processed/test/t1_slices"),
    ("FA_GT", "data/adni_processed/test/fa_slices"),
    ("A080+DS Full", "outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST"),
    ("A080 Base", "outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080"),
    ("Old Fidelity Flow", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Pix2Pix", "outputs/icdm2026/predictions/ADNI_Pix2Pix_E50"),
    ("U-Net", "outputs/icdm2026/predictions/ADNI_UNet_E50"),
    ("DDIM", "outputs/icdm2026/predictions/ADNI_DDIM_E100_K50_PRETRAINED"),
    ("DBM", "outputs/icdm2026/predictions/ADNI_DBM_E100_K40_PRETRAINED"),
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
    parser.add_argument("--slice_id", action="append", default=[])
    parser.add_argument(
        "--method_preset",
        default="full_adni",
        choices=[
            "full_adni",
            "adni_compact",
            "private_single_stage2",
            "adni_single_stage2_smoke",
            "adni_single_stage2_multihead",
            "adni_single_sharp_stage2",
            "adni_texture_probe",
            "adni_final_a080_ds",
        ],
        help=(
            "full_adni includes all available ADNI comparison methods; adni_compact uses the requested "
            "7-column paper review order; private_single_stage2 compares the private single-slice Stage1 "
            "and disease-sensitive Stage2 outputs; adni_single_stage2_multihead compares the new multi-head "
            "Stage2 probe against the prior single-slice Stage2 smoke output; adni_single_sharp_stage2 "
            "compares the stronger single-slice sharp Stage1 with its Flow and multi-head Stage2 probes; "
            "adni_texture_probe compares texture-source and high-frequency teacher probes."
        ),
    )
    parser.add_argument(
        "--all_slices",
        action="store_true",
        help="Build panels for every slice found in the FA_GT directory.",
    )
    parser.add_argument(
        "--slice_source_dir",
        default=None,
        help="Directory of existing method-slice figures. Slice ids are inferred from PNG names, e.g. sub-006_S_6651_z042_*.png.",
    )
    parser.add_argument(
        "--gt_dir",
        default="data/adni_processed/test/fa_slices",
        help="Directory used to enumerate slices when --all_slices is enabled.",
    )
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        help="Override ROI as SLICE_ID:x0,y0,x1,y1 in relative coordinates, e.g. sub-006_S_6651_z042:0.35,0.3,0.65,0.6",
    )
    parser.add_argument(
        "--default_roi",
        default="0.34,0.30,0.66,0.64",
        help="Default red-box ROI for slices without an explicit --roi override.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_roi", action="store_true", help="Only write the plain horizontal method panel.")
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


def _parse_roi(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(part) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError(f"ROI needs four coordinates, got {value!r}")
    return parts  # type: ignore[return-value]


def _slice_ids_from_gt(gt_dir: str | Path) -> list[str]:
    paths = sorted(Path(gt_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG slices found in {gt_dir}")
    return [path.stem for path in paths]


def _slice_ids_from_source(source_dir: str | Path) -> list[str]:
    paths = sorted(Path(source_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG slices found in {source_dir}")
    slice_ids: list[str] = []
    seen: set[str] = set()
    for path in paths:
        match = re.search(r"(sub-\d+_S_\d+_z\d{3})", path.stem)
        if match is None:
            match = re.search(r"(.+_z\d{3})", path.stem)
        if match is None:
            raise ValueError(f"Could not infer slice id from {path.name}")
        slice_id = match.group(1)
        if slice_id not in seen:
            seen.add(slice_id)
            slice_ids.append(slice_id)
    return slice_ids


def _method_list(preset: str) -> list[tuple[str, str]]:
    if preset == "adni_final_a080_ds":
        return ADNI_FINAL_A080_DS_METHODS
    if preset == "adni_compact":
        return COMPACT_ADNI_METHODS
    if preset == "private_single_stage2":
        return PRIVATE_SINGLE_STAGE2_METHODS
    if preset == "adni_single_stage2_smoke":
        return ADNI_SINGLE_STAGE2_SMOKE_METHODS
    if preset == "adni_single_stage2_multihead":
        return ADNI_SINGLE_STAGE2_MULTIHEAD_METHODS
    if preset == "adni_single_sharp_stage2":
        return ADNI_SINGLE_SHARP_STAGE2_METHODS
    if preset == "adni_texture_probe":
        return ADNI_TEXTURE_PROBE_METHODS
    return DEFAULT_METHODS


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
    default_roi = _parse_roi(args.default_roi)
    if args.slice_source_dir:
        slice_ids = _slice_ids_from_source(args.slice_source_dir)
    elif args.all_slices:
        slice_ids = _slice_ids_from_gt(args.gt_dir)
    elif args.slice_id:
        slice_ids = args.slice_id
    else:
        slice_ids = DEFAULT_SLICES
    methods = _method_list(args.method_preset)
    print(f"Building {len(slice_ids)} slice panel(s) with method_preset={args.method_preset}")
    saved = 0
    for index, slice_id in enumerate(slice_ids, start=1):
        normal_path = output_dir / f"{slice_id}_adni_methods_panel.png"
        roi_path = output_dir / f"{slice_id}_adni_methods_panel_roi.png"
        _make_panel(slice_id, methods, normal_path, dpi=args.dpi)
        saved += 1
        if not args.skip_roi:
            _make_panel(slice_id, methods, roi_path, roi=rois.get(slice_id, default_roi), dpi=args.dpi)
            saved += 1
        if not args.all_slices or index % 100 == 0 or index == len(slice_ids):
            print(f"Saved {saved} panel(s), latest slice: {slice_id}")


if __name__ == "__main__":
    main()
