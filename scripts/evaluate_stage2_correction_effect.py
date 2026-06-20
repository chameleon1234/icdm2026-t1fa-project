from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.image_metrics import (
    build_brain_mask,
    build_wm_mask,
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_ssim,
    masked_laplacian_variance,
    masked_mae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage1/coarse and Stage2/refined predictions with metrics focused on "
            "low-frequency correction, WM fidelity, sharpness retention, and stripe artifacts."
        )
    )
    parser.add_argument("--coarse_dir", required=True, help="Stage1/coarse prediction PNG folder.")
    parser.add_argument("--refined_dir", required=True, help="Stage2/refined prediction PNG folder.")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--method", default="STAGE2_CORRECTION")
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lowpass_kernel", type=int, default=13)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    return parser.parse_args()


def read_image(path: Path, target_shape: tuple[int, int] | None = None) -> torch.Tensor:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if target_shape is not None and image.shape != target_shape:
        image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(image).float().view(1, 1, image.shape[0], image.shape[1]) / 255.0


def lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return x
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def masked_mean_abs(x: torch.Tensor, mask: torch.Tensor) -> float:
    vals = torch.abs(x)[mask.bool()]
    return float("nan") if vals.numel() == 0 else float(vals.mean().item())


def stripe_bias(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    residual = (pred - target) * mask.float()
    row_bias = residual.mean(dim=2, keepdim=True).abs().mean()
    col_bias = residual.mean(dim=3, keepdim=True).abs().mean()
    return float((row_bias + col_bias).item())


def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(values)
    summary: dict[str, Any] = {"n_slices": int(len(frame))}
    numeric = frame.select_dtypes(include=[np.number])
    for column in numeric.columns:
        summary[column] = float(numeric[column].mean())
    return summary


def main() -> None:
    args = parse_args()
    coarse_dir = Path(args.coarse_dir)
    refined_dir = Path(args.refined_dir)
    t1_dir = Path(args.test_t1_dir)
    fa_dir = Path(args.test_fa_dir)
    metric_root = Path("outputs/icdm2026/metrics")
    out_csv = Path(args.out_csv) if args.out_csv else metric_root / f"{args.method}_correction_effect_slice_metrics.csv"
    out_json = Path(args.out_json) if args.out_json else metric_root / f"{args.method}_correction_effect_summary.json"

    rows: list[dict[str, Any]] = []
    target_paths = sorted(fa_dir.glob("*.png"))
    if args.limit > 0:
        target_paths = target_paths[: args.limit]

    for target_path in target_paths:
        name = target_path.name
        coarse_path = coarse_dir / name
        refined_path = refined_dir / name
        t1_path = t1_dir / name
        if not coarse_path.exists() or not refined_path.exists() or not t1_path.exists():
            missing = [str(p) for p in (coarse_path, refined_path, t1_path) if not p.exists()]
            raise FileNotFoundError(f"Missing paired files for {name}: {missing}")

        target = read_image(target_path)
        shape = tuple(target.shape[-2:])
        t1 = read_image(t1_path, target_shape=shape)
        coarse = read_image(coarse_path, target_shape=shape)
        refined = read_image(refined_path, target_shape=shape)
        brain = build_brain_mask(t1, target, args.brain_t1_threshold, args.brain_fa_threshold)
        wm = build_wm_mask(target, brain, args.wm_quantile, args.wm_min_threshold)

        coarse_low = lowpass(coarse, args.lowpass_kernel)
        refined_low = lowpass(refined, args.lowpass_kernel)
        target_low = lowpass(target, args.lowpass_kernel)
        coarse_high = coarse - coarse_low
        refined_high = refined - refined_low
        target_high = target - target_low

        coarse_sharp = masked_laplacian_variance(coarse, brain)
        refined_sharp = masked_laplacian_variance(refined, brain)
        target_sharp = masked_laplacian_variance(target, brain)

        row = {
            "fname": name,
            "coarse_psnr": compute_psnr(coarse, target),
            "refined_psnr": compute_psnr(refined, target),
            "coarse_ssim": compute_ssim(coarse, target),
            "refined_ssim": compute_ssim(refined, target),
            "coarse_mse": compute_mse(coarse, target),
            "refined_mse": compute_mse(refined, target),
            "coarse_mae": compute_mae(coarse, target),
            "refined_mae": compute_mae(refined, target),
            "coarse_brain_mae": masked_mae(coarse, target, brain),
            "refined_brain_mae": masked_mae(refined, target, brain),
            "coarse_wm_mae": masked_mae(coarse, target, wm),
            "refined_wm_mae": masked_mae(refined, target, wm),
            "coarse_low_brain_mae": masked_mean_abs(coarse_low - target_low, brain),
            "refined_low_brain_mae": masked_mean_abs(refined_low - target_low, brain),
            "coarse_high_brain_mae": masked_mean_abs(coarse_high - target_high, brain),
            "refined_high_brain_mae": masked_mean_abs(refined_high - target_high, brain),
            "coarse_stripe_bias": stripe_bias(coarse, target, brain),
            "refined_stripe_bias": stripe_bias(refined, target, brain),
            "coarse_sharpness": coarse_sharp,
            "refined_sharpness": refined_sharp,
            "target_sharpness": target_sharp,
            "coarse_sharp_ratio": coarse_sharp / max(target_sharp, 1e-12),
            "refined_sharp_ratio": refined_sharp / max(target_sharp, 1e-12),
        }
        row.update(
            {
                "delta_psnr": row["refined_psnr"] - row["coarse_psnr"],
                "delta_ssim": row["refined_ssim"] - row["coarse_ssim"],
                "delta_mse": row["coarse_mse"] - row["refined_mse"],
                "delta_mae": row["coarse_mae"] - row["refined_mae"],
                "delta_brain_mae": row["coarse_brain_mae"] - row["refined_brain_mae"],
                "delta_wm_mae": row["coarse_wm_mae"] - row["refined_wm_mae"],
                "delta_low_brain_mae": row["coarse_low_brain_mae"] - row["refined_low_brain_mae"],
                "delta_high_brain_mae": row["coarse_high_brain_mae"] - row["refined_high_brain_mae"],
                "delta_stripe_bias": row["coarse_stripe_bias"] - row["refined_stripe_bias"],
                "delta_sharp_ratio": row["refined_sharp_ratio"] - row["coarse_sharp_ratio"],
                "sharp_retention": row["refined_sharp_ratio"] / max(row["coarse_sharp_ratio"], 1e-12),
            }
        )
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    summary = summarize(rows)
    summary["method"] = args.method
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"{args.method}: n={summary['n_slices']} "
        f"DeltaPSNR={summary['delta_psnr']:.4f} DeltaSSIM={summary['delta_ssim']:.4f} "
        f"DeltaLowBrainMAE={summary['delta_low_brain_mae']:.6f} "
        f"DeltaWMMAE={summary['delta_wm_mae']:.6f} "
        f"SharpRetention={summary['sharp_retention']:.4f} "
        f"DeltaStripe={summary['delta_stripe_bias']:.6f}"
    )
    print(f"Saved slice metrics to: {out_csv}")
    print(f"Saved summary to: {out_json}")


if __name__ == "__main__":
    main()
