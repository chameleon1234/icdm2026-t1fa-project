import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.models.pmrf_t1fa import prepare_stage1_input, reduce_rgb_to_single_channel
from pmrf_t1fa.train_pmrf_t1fa_stage2 import (
    build_training_masks,
    compute_psnr,
    laplacian_filter,
    laplacian_variance,
    load_stage1_model,
    make_slice_dataset,
    masked_l1_loss,
    maybe_limit_dataset,
    predict_stage1_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Stage1 residual distribution for uncertainty-aware residual flow.")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt")
    parser.add_argument("--t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--output_json", default="outputs/icdm2026/metrics/stage1_residual_distribution_val.json")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    stage1, prediction_mode, stage1_channels, detail_scale = load_stage1_model(args.stage1_ckpt, device)
    dataset = maybe_limit_dataset(make_slice_dataset(args.t1_dir, args.fa_dir, stage1_channels), args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows: dict[str, list[float]] = {
        "psnr": [],
        "residual_abs": [],
        "residual_std": [],
        "wm_residual_abs": [],
        "brain_residual_abs": [],
        "residual_hp_abs": [],
        "wm_residual_hp_abs": [],
        "stage1_sharp_ratio": [],
    }
    for batch in tqdm(loader, desc="Diagnosing Stage1 residuals", leave=False):
        t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
        target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
        coarse = predict_stage1_batch(stage1, t1_img, device, prediction_mode, detail_scale)
        residual = target - coarse
        brain_mask, wm_mask = build_training_masks(
            t1_img,
            target,
            brain_t1_threshold=args.brain_t1_threshold,
            brain_fa_threshold=args.brain_fa_threshold,
            wm_quantile=args.wm_quantile,
            wm_min_threshold=args.wm_min_threshold,
        )
        residual_hp = laplacian_filter(residual).abs()
        rows["psnr"].append(compute_psnr(coarse, target).item())
        rows["residual_abs"].append(residual.abs().mean().item())
        rows["residual_std"].append(residual.flatten(1).std(dim=1, unbiased=False).mean().item())
        rows["wm_residual_abs"].append(masked_l1_loss(coarse, target, wm_mask).item())
        rows["brain_residual_abs"].append(masked_l1_loss(coarse, target, brain_mask).item())
        rows["residual_hp_abs"].append(residual_hp.mean().item())
        wm_denom = wm_mask.sum().clamp_min(1.0)
        rows["wm_residual_hp_abs"].append(((residual_hp * wm_mask).sum() / wm_denom).item())
        target_sharp = laplacian_variance(target).item()
        rows["stage1_sharp_ratio"].append(laplacian_variance(coarse).item() / max(target_sharp, 1e-8))
    summary = {
        "stage1_ckpt": str(args.stage1_ckpt),
        "t1_dir": str(args.t1_dir),
        "fa_dir": str(args.fa_dir),
        "evaluated": len(dataset),
        "stage1_channels": stage1_channels,
        "stage1_prediction_mode": prediction_mode,
        "stage1_detail_scale": detail_scale,
        "psnr_mean": _mean(rows["psnr"]),
        "residual_abs_mean": _mean(rows["residual_abs"]),
        "residual_std_mean": _mean(rows["residual_std"]),
        "brain_residual_abs_mean": _mean(rows["brain_residual_abs"]),
        "wm_residual_abs_mean": _mean(rows["wm_residual_abs"]),
        "residual_hp_abs_mean": _mean(rows["residual_hp_abs"]),
        "wm_residual_hp_abs_mean": _mean(rows["wm_residual_hp_abs"]),
        "stage1_sharp_ratio_mean": _mean(rows["stage1_sharp_ratio"]),
        "interpretation": (
            "Non-zero residual and WM high-pass residual support a residual-space Stage2. "
            "Sharpness below target means deterministic Stage1 is a posterior-mean baseline, "
            "while residual flow should model uncertainty rather than force a single hallucinated detail map."
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    diagnose(parse_args())
