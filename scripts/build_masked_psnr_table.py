import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.image_metrics import build_brain_mask, build_wm_mask, compute_psnr, masked_psnr


DEFAULT_METHODS = [
    ("U-Net", "ADNI_UNet_E50"),
    ("Pix2Pix", "ADNI_Pix2Pix_E50"),
    ("CycleGAN", "ADNI_CycleGAN_E50"),
    ("DDIM", "ADNI_DDIM_E100_K50_PRETRAINED"),
    ("DBM", "ADNI_DBM_E100_K40_PRETRAINED"),
    ("StackUNet5", "ADNI_StackUNet5_E12"),
    ("StackUNet7", "ADNI_StackUNet7_E50"),
    ("Old5SliceFlow", "ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Stage1SingleSharp", "ADNI_STAGE1_SINGLE_SHARP_FULL_E30"),
    ("SingleFidelityFlow", "ADNI_SINGLE_FIDELITY_FLOW_SHARP_STAGE1_PROBE_4096_E5"),
    ("OursFinal", "ADNI_SINGLE_DS_MULTIHEAD_FINAL_12000_E8"),
]

CORE_METHODS = [
    ("U-Net", "ADNI_UNet_E50"),
    ("Pix2Pix", "ADNI_Pix2Pix_E50"),
    ("StackUNet5", "ADNI_StackUNet5_E12"),
    ("Old5SliceFlow", "ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Stage1SingleSharp", "ADNI_STAGE1_SINGLE_SHARP_FULL_E30"),
    ("OursFinal", "ADNI_SINGLE_DS_MULTIHEAD_FINAL_12000_E8"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full/brain/WM PSNR table from existing prediction folders.")
    parser.add_argument("--test_t1_dir", default="data/adni_processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/adni_processed/test/fa_slices")
    parser.add_argument("--prediction_root", default="outputs/icdm2026/predictions")
    parser.add_argument("--output_csv", default="outputs/icdm2026/metrics/adni_masked_psnr_table.csv")
    parser.add_argument("--output_md", default="outputs/icdm2026/metrics/adni_masked_psnr_table.md")
    parser.add_argument("--preset", choices=["core", "all"], default="core")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    return parser.parse_args()


def read_png01(path: str | Path) -> torch.Tensor:
    array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise FileNotFoundError(path)
    return torch.from_numpy(array.astype(np.float32) / 255.0).view(1, 1, *array.shape)


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def evaluate_method(method_name: str, folder_name: str, args: argparse.Namespace) -> dict[str, Any]:
    t1_dir = Path(args.test_t1_dir)
    fa_dir = Path(args.test_fa_dir)
    pred_dir = Path(args.prediction_root) / folder_name
    if not pred_dir.exists():
        return {"method": method_name, "folder": folder_name, "missing": True}

    full_psnr: list[float] = []
    brain_psnr: list[float] = []
    wm_psnr: list[float] = []
    brain_fracs: list[float] = []
    fa_paths = sorted(fa_dir.glob("*.png"))
    if args.limit > 0:
        fa_paths = fa_paths[: args.limit]
    for fa_path in tqdm(fa_paths, desc=method_name, leave=False):
        pred_path = pred_dir / fa_path.name
        t1_path = t1_dir / fa_path.name
        if not pred_path.exists() or not t1_path.exists():
            continue
        pred = read_png01(pred_path)
        target = read_png01(fa_path)
        t1 = read_png01(t1_path)
        brain = build_brain_mask(t1, target, args.brain_t1_threshold, args.brain_fa_threshold)
        wm = build_wm_mask(target, brain, args.wm_quantile, args.wm_min_threshold)
        full_psnr.append(compute_psnr(pred, target))
        brain_psnr.append(masked_psnr(pred, target, brain))
        wm_psnr.append(masked_psnr(pred, target, wm))
        brain_fracs.append(float(brain.float().mean().item()))

    return {
        "method": method_name,
        "folder": folder_name,
        "n_slices": len(full_psnr),
        "full_psnr": _mean(full_psnr),
        "brain_psnr": _mean(brain_psnr),
        "wm_psnr": _mean(wm_psnr),
        "brain_fraction": _mean(brain_fracs),
    }


def main() -> None:
    args = parse_args()
    methods = CORE_METHODS if args.preset == "core" else DEFAULT_METHODS
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, folder in methods:
        rows.append(evaluate_method(name, folder, args))
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        write_markdown(rows, output_md)
    print(output_md.read_text(encoding="utf-8"))
    print(f"Saved CSV to: {output_csv}")
    print(f"Saved Markdown to: {output_md}")


def write_markdown(rows: list[dict[str, Any]], output_md: Path) -> None:
    df = pd.DataFrame(rows)
    md_lines = [
        "| Method | Full PSNR | Brain PSNR | WM PSNR | Brain Fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("missing"):
            md_lines.append(f"| {row['method']} | missing | missing | missing | missing |")
            continue
        md_lines.append(
            f"| {row['method']} | {row['full_psnr']:.3f} | {row['brain_psnr']:.3f} | "
            f"{row['wm_psnr']:.3f} | {row['brain_fraction']:.3f} |"
        )
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
