from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:  # pragma: no cover - optional metric dependency
    FrechetInceptionDistance = None

from pmrf_t1fa.models.pmrf_t1fa import center_channel, highpass_residual, prepare_stage1_input
from pmrf_t1fa.train_pmrf_t1fa_stage2 import (
    SSIMLoss,
    build_training_masks,
    compute_psnr,
    laplacian_variance,
    load_stage1_model,
    make_slice_dataset,
    masked_l1_loss,
    maybe_limit_dataset,
    predict_stage1_batch,
    roi_consistency_loss,
)
from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import (
    NAFBlock,
    frequency_highpass,
    frequency_lowpass,
)
from src.eval.disease_sensitive_roi import roi_weight_tensor_from_frame, weighted_grid_roi_l1


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


VARIANTS = ("metric", "ds_roi", "uncertainty", "atlas", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train single-slice Stage 2 disease-sensitive corrector variants."
    )
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage1_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run_name", default="pmrf_t1fa_stage2_ds_corrector_single")
    parser.add_argument("--variant", default="hybrid", choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--correction_scale", type=float, default=0.18)
    parser.add_argument("--lowpass_kernel", type=int, default=13)
    parser.add_argument("--hp_kernel", type=int, default=5)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--roi_rows", type=int, default=4)
    parser.add_argument("--roi_cols", type=int, default=4)
    parser.add_argument("--roi_min_pixels", type=int, default=16)
    parser.add_argument("--disease_roi_csv", default="")
    parser.add_argument("--final_l1_weight", type=float, default=0.25)
    parser.add_argument("--final_mse_weight", type=float, default=0.20)
    parser.add_argument("--final_ssim_weight", type=float, default=0.15)
    parser.add_argument("--wm_l1_weight", type=float, default=1.0)
    parser.add_argument("--roi_weight", type=float, default=0.50)
    parser.add_argument("--disease_roi_weight", type=float, default=1.50)
    parser.add_argument("--correction_l1_weight", type=float, default=0.50)
    parser.add_argument("--bounded_weight", type=float, default=0.04)
    parser.add_argument("--sharp_retention_weight", type=float, default=0.50)
    parser.add_argument("--stripe_weight", type=float, default=1.0)
    parser.add_argument("--hf_preserve_weight", type=float, default=1.0)
    parser.add_argument("--uncertainty_weight", type=float, default=0.25)
    parser.add_argument("--atlas_smooth_weight", type=float, default=0.15)
    parser.add_argument("--best_min_sharp_retention", type=float, default=0.95)
    parser.add_argument("--best_min_delta_disease_roi", type=float, default=0.0)
    parser.add_argument("--best_max_delta_stripe", type=float, default=0.002)
    parser.add_argument("--fid_eval_every", type=int, default=0)
    parser.add_argument("--preview_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


def ensure_dirs(run_name: str) -> tuple[Path, Path, Path]:
    root = Path("outputs") / run_name
    ckpt_dir = root / "checkpoints"
    preview_dir = root / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    return root, ckpt_dir, preview_dir


def _avg_pool_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def stripe_score(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    residual = (pred - target) * mask.to(pred)
    return residual.mean(dim=2, keepdim=True).abs().mean() + residual.mean(dim=3, keepdim=True).abs().mean()


def _coord_channels(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1).expand(batch, 1, height, width)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width).expand(batch, 1, height, width)
    return torch.cat([xx, yy], dim=1)


def roi_weight_map(
    roi_weights: torch.Tensor | None,
    ref: torch.Tensor,
    wm_mask: torch.Tensor,
) -> torch.Tensor:
    if roi_weights is None:
        return torch.zeros_like(ref)
    weights = roi_weights.to(device=ref.device, dtype=ref.dtype)
    rows, cols = int(weights.shape[0]), int(weights.shape[1])
    out = torch.zeros_like(ref)
    height, width = ref.shape[-2:]
    for r in range(rows):
        y0 = int(round(r * height / rows))
        y1 = int(round((r + 1) * height / rows))
        for c in range(cols):
            x0 = int(round(c * width / cols))
            x1 = int(round((c + 1) * width / cols))
            out[:, :, y0:y1, x0:x1] = weights[r, c]
    return out * wm_mask.to(out)


def load_disease_roi_weights(path: str, rows: int, cols: int, device: torch.device) -> torch.Tensor | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    return roi_weight_tensor_from_frame(frame, rows, cols).to(device)


class SingleSliceCorrector(nn.Module):
    def __init__(self, in_channels: int, width: int, num_blocks: int, uncertainty: bool = False):
        super().__init__()
        out_channels = 2 if uncertainty else 1
        self.uncertainty = bool(uncertainty)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            *[NAFBlock(width) for _ in range(num_blocks)],
            nn.Conv2d(width, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        y = self.net(x)
        if self.uncertainty:
            return y[:, :1], y[:, 1:2].clamp(-5.0, 5.0)
        return y, None


def build_condition(
    t1_img: torch.Tensor,
    coarse: torch.Tensor,
    wm_mask: torch.Tensor,
    roi_map: torch.Tensor,
    variant: str,
    lowpass_kernel: int,
    hp_kernel: int,
) -> torch.Tensor:
    coarse_low = _avg_pool_same(coarse, lowpass_kernel)
    coarse_high = coarse - _avg_pool_same(coarse, hp_kernel)
    t1_edge = highpass_residual(center_channel(t1_img), kernel_size=hp_kernel)
    uncertainty_proxy = (t1_edge - coarse_high).abs().detach()
    channels = [center_channel(t1_img), coarse, coarse_low, coarse_high, t1_edge]
    if variant in {"ds_roi", "hybrid", "atlas"}:
        channels.append(roi_map)
    else:
        channels.append(torch.zeros_like(coarse))
    if variant in {"uncertainty", "hybrid"}:
        channels.append(uncertainty_proxy)
    else:
        channels.append(torch.zeros_like(coarse))
    if variant in {"atlas", "hybrid"}:
        channels.append(_coord_channels(coarse.shape[0], coarse.shape[-2], coarse.shape[-1], coarse.device, coarse.dtype))
    else:
        channels.append(torch.zeros(coarse.shape[0], 2, coarse.shape[-2], coarse.shape[-1], device=coarse.device, dtype=coarse.dtype))
    return torch.cat(channels, dim=1)


def correction_gate(variant: str, brain_mask: torch.Tensor, wm_mask: torch.Tensor, roi_map: torch.Tensor) -> torch.Tensor:
    gate = brain_mask.to(wm_mask)
    if variant in {"ds_roi", "hybrid"}:
        gate = gate * (0.35 + 0.65 * torch.clamp(wm_mask.to(gate) + roi_map, 0.0, 1.0))
    elif variant == "atlas":
        gate = gate * (0.55 + 0.45 * wm_mask.to(gate))
    return gate


def build_loss(
    raw: torch.Tensor,
    log_sigma: torch.Tensor | None,
    coarse: torch.Tensor,
    target: torch.Tensor,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    roi_map: torch.Tensor,
    roi_weights: torch.Tensor | None,
    ssim_loss: SSIMLoss,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    gate = correction_gate(args.variant, brain_mask, wm_mask, roi_map)
    correction = float(args.correction_scale) * torch.tanh(raw.float()) * gate.float()
    refined = torch.clamp(coarse.float() + correction, -1.0, 1.0)
    target_fp32 = target.float()
    target_correction = target_fp32 - coarse.float()

    final_l1 = F.l1_loss(refined, target_fp32)
    final_mse = F.mse_loss(refined, target_fp32)
    final_ssim = ssim_loss(refined, target_fp32)
    wm_l1 = masked_l1_loss(refined, target_fp32, wm_mask)
    roi = roi_consistency_loss(refined, target_fp32, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels)
    disease_roi = refined.new_tensor(0.0)
    if roi_weights is not None and args.variant in {"ds_roi", "hybrid", "atlas"}:
        disease_roi = weighted_grid_roi_l1(refined, target_fp32, wm_mask, roi_weights, args.roi_min_pixels)
    correction_l1 = F.l1_loss(correction, target_correction * gate.float())
    bounded = correction.abs().mean()
    sharp_ratio = laplacian_variance(refined) / laplacian_variance(coarse.float()).clamp_min(1e-8)
    sharp_loss = F.relu(0.95 - sharp_ratio)
    stripe = stripe_score(refined, target_fp32, brain_mask)
    hf_preserve = F.l1_loss(frequency_highpass(refined, 0.12, 0.04), frequency_highpass(coarse.float(), 0.12, 0.04))
    uncertainty = refined.new_tensor(0.0)
    if log_sigma is not None and args.variant in {"uncertainty", "hybrid"}:
        abs_error = (refined - target_fp32).abs().detach()
        uncertainty = (torch.exp(-log_sigma.float()) * (refined - target_fp32).abs() + 0.05 * log_sigma.float()).mean()
        uncertainty = uncertainty + 0.25 * F.l1_loss(torch.sigmoid(log_sigma.float()), torch.clamp(abs_error * 6.0, 0.0, 1.0))
    atlas_smooth = refined.new_tensor(0.0)
    if args.variant in {"atlas", "hybrid"}:
        atlas_smooth = (correction[:, :, :, 1:] - correction[:, :, :, :-1]).abs().mean()
        atlas_smooth = atlas_smooth + (correction[:, :, 1:, :] - correction[:, :, :-1, :]).abs().mean()

    total = (
        args.final_l1_weight * final_l1
        + args.final_mse_weight * final_mse
        + args.final_ssim_weight * final_ssim
        + args.wm_l1_weight * wm_l1
        + args.roi_weight * roi
        + args.disease_roi_weight * disease_roi
        + args.correction_l1_weight * correction_l1
        + args.bounded_weight * bounded
        + args.sharp_retention_weight * sharp_loss
        + args.stripe_weight * stripe
        + args.hf_preserve_weight * hf_preserve
        + args.uncertainty_weight * uncertainty
        + args.atlas_smooth_weight * atlas_smooth
    )
    losses = {
        "total": total,
        "final_l1": final_l1,
        "final_mse": final_mse,
        "final_ssim": final_ssim,
        "wm_l1": wm_l1,
        "roi": roi,
        "disease_roi": disease_roi,
        "correction_l1": correction_l1,
        "bounded": bounded,
        "sharp_loss": sharp_loss,
        "stripe": stripe,
        "hf_preserve": hf_preserve,
        "uncertainty": uncertainty,
        "atlas_smooth": atlas_smooth,
    }
    return total, losses, refined


def metric_row(
    refined: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    roi_weights: torch.Tensor | None,
    ssim_loss: SSIMLoss,
    args: argparse.Namespace,
) -> dict[str, float]:
    psnr = float(compute_psnr(refined, target).item())
    coarse_psnr = float(compute_psnr(coarse, target).item())
    ssim = float(1.0 - ssim_loss(refined, target).item())
    coarse_ssim = float(1.0 - ssim_loss(coarse, target).item())
    wm_l1 = float(masked_l1_loss(refined, target, wm_mask).item())
    coarse_wm_l1 = float(masked_l1_loss(coarse, target, wm_mask).item())
    roi = float(roi_consistency_loss(refined, target, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels).item())
    coarse_roi = float(roi_consistency_loss(coarse, target, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels).item())
    disease_roi = 0.0
    coarse_disease_roi = 0.0
    if roi_weights is not None:
        disease_roi = float(weighted_grid_roi_l1(refined, target, wm_mask, roi_weights, args.roi_min_pixels).item())
        coarse_disease_roi = float(weighted_grid_roi_l1(coarse, target, wm_mask, roi_weights, args.roi_min_pixels).item())
    sharp = float(laplacian_variance(refined).item())
    coarse_sharp = float(laplacian_variance(coarse).item())
    stripe = float(stripe_score(refined, target, brain_mask).item())
    coarse_stripe = float(stripe_score(coarse, target, brain_mask).item())
    return {
        "psnr": psnr,
        "ssim": ssim,
        "wm_l1": wm_l1,
        "roi": roi,
        "disease_roi": disease_roi,
        "stripe": stripe,
        "sharp_ratio": sharp,
        "coarse_psnr": coarse_psnr,
        "coarse_ssim": coarse_ssim,
        "coarse_wm_l1": coarse_wm_l1,
        "coarse_roi": coarse_roi,
        "coarse_disease_roi": coarse_disease_roi,
        "coarse_stripe": coarse_stripe,
        "coarse_sharp_ratio": coarse_sharp,
        "sharp_retention": sharp / max(coarse_sharp, 1e-8),
        "delta_psnr": psnr - coarse_psnr,
        "delta_ssim": ssim - coarse_ssim,
        "delta_wm_l1": coarse_wm_l1 - wm_l1,
        "delta_roi": coarse_roi - roi,
        "delta_disease_roi": coarse_disease_roi - disease_roi,
        "delta_stripe": coarse_stripe - stripe,
        "correction_l1": float(F.l1_loss(refined, coarse).item()),
    }


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / max(len(rows), 1) for key in rows[0]} if rows else {}


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["delta_psnr"]
        + 10.0 * metrics["delta_ssim"]
        + 20.0 * metrics["delta_wm_l1"]
        + 20.0 * metrics["delta_roi"]
        + 30.0 * metrics["delta_disease_roi"]
        + 10.0 * metrics["delta_stripe"]
        - 5.0 * max(0.0, 0.95 - metrics["sharp_retention"])
    )


def passes_gate(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        metrics["sharp_retention"] >= args.best_min_sharp_retention
        and metrics["delta_disease_roi"] >= args.best_min_delta_disease_roi
        and metrics["delta_stripe"] >= -abs(args.best_max_delta_stripe)
    )


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_preview_panel(path: Path, t1: torch.Tensor, coarse: torch.Tensor, refined: torch.Tensor, target: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = torch.cat(
        [
            center_channel(t1[:4]).detach().cpu(),
            coarse[:4].detach().cpu(),
            refined[:4].detach().cpu(),
            target[:4].detach().cpu(),
            (refined[:4] - coarse[:4]).detach().cpu() * 4.0,
        ],
        dim=0,
    )
    save_image((panel + 1.0) / 2.0, path, nrow=4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_payload(
    model: nn.Module,
    args: argparse.Namespace,
    stage1_channels: int,
    prediction_mode: str,
    detail_scale: float,
    epoch: int,
    score: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "args": vars(args),
        "stage1_ckpt": args.stage1_ckpt,
        "stage1_channels": stage1_channels,
        "stage1_prediction_mode": prediction_mode,
        "stage1_detail_scale": detail_scale,
        "epoch": epoch,
        "score": score,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    root, ckpt_dir, preview_dir = ensure_dirs(args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_device = device if args.stage1_device == "auto" else torch.device(args.stage1_device)
    stage1, prediction_mode, stage1_channels, detail_scale = load_stage1_model(args.stage1_ckpt, stage1_device)
    if stage1_channels != 1:
        raise ValueError(
            f"This experiment is single-slice only, but stage1 checkpoint expects {stage1_channels} channels: {args.stage1_ckpt}"
        )
    train_ds = maybe_limit_dataset(make_slice_dataset(args.train_t1_dir, args.train_fa_dir, stage1_channels), args.train_limit)
    val_ds = maybe_limit_dataset(make_slice_dataset(args.val_t1_dir, args.val_fa_dir, stage1_channels), args.val_limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    roi_weights = load_disease_roi_weights(args.disease_roi_csv, args.roi_rows, args.roi_cols, device)
    model = SingleSliceCorrector(
        in_channels=9,
        width=args.width,
        num_blocks=args.num_blocks,
        uncertainty=args.variant in {"uncertainty", "hybrid"},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ssim_loss = SSIMLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    history: list[dict[str, Any]] = []
    best_gated_score = -1e9
    best_score = -1e9
    best_gated_metrics: dict[str, float] = {}
    best_score_metrics: dict[str, float] = {}
    print(
        f"DS Stage2 single-slice training | variant={args.variant} | train={len(train_ds)} val={len(val_ds)} "
        f"| stage1={args.stage1_ckpt} | roi_weights={'yes' if roi_weights is not None else 'no'} | device={device}"
    )
    for epoch in range(args.epochs):
        model.train()
        running: dict[str, float] = {}
        progress = tqdm(train_loader, desc=f"DSCorrector {args.variant} Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            t1 = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
            target = batch["fa_slice"].to(device)
            target = target.mean(dim=1, keepdim=True) if target.shape[1] > 1 else target
            with torch.no_grad():
                coarse = predict_stage1_batch(stage1, t1, stage1_device, prediction_mode, detail_scale).to(device)
            brain_mask, wm_mask = build_training_masks(
                t1,
                target,
                args.brain_t1_threshold,
                args.brain_fa_threshold,
                args.wm_quantile,
                args.wm_min_threshold,
            )
            rmap = roi_weight_map(roi_weights, coarse, wm_mask)
            condition = build_condition(t1, coarse, wm_mask, rmap, args.variant, args.lowpass_kernel, args.hp_kernel)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.mixed_precision):
                raw, log_sigma = model(condition)
                loss, losses, refined = build_loss(raw, log_sigma, coarse, target, brain_mask, wm_mask, rmap, roi_weights, ssim_loss, args)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach().item())
            progress.set_postfix(
                loss=f"{running['total'] / step:.4f}",
                roi=f"{running.get('roi', 0.0) / step:.4f}",
                stripe=f"{running.get('stripe', 0.0) / step:.4f}",
            )
            global_step = epoch * max(len(train_loader), 1) + step
            if args.preview_every > 0 and global_step % args.preview_every == 0:
                save_preview_panel(preview_dir / f"train_step_{global_step:06d}.png", t1, coarse, refined, target)

        model.eval()
        rows: list[dict[str, float]] = []
        fid_metric = None
        if args.fid_eval_every > 0 and (epoch + 1) % args.fid_eval_every == 0 and FrechetInceptionDistance is not None:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"DSCorrector {args.variant} Val {epoch + 1}/{args.epochs}"):
                t1 = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
                target = batch["fa_slice"].to(device)
                target = target.mean(dim=1, keepdim=True) if target.shape[1] > 1 else target
                coarse = predict_stage1_batch(stage1, t1, stage1_device, prediction_mode, detail_scale).to(device)
                brain_mask, wm_mask = build_training_masks(
                    t1,
                    target,
                    args.brain_t1_threshold,
                    args.brain_fa_threshold,
                    args.wm_quantile,
                    args.wm_min_threshold,
                )
                rmap = roi_weight_map(roi_weights, coarse, wm_mask)
                condition = build_condition(t1, coarse, wm_mask, rmap, args.variant, args.lowpass_kernel, args.hp_kernel)
                raw, log_sigma = model(condition)
                _, _, refined = build_loss(raw, log_sigma, coarse, target, brain_mask, wm_mask, rmap, roi_weights, ssim_loss, args)
                rows.append(metric_row(refined.float(), coarse.float(), target.float(), brain_mask, wm_mask, roi_weights, ssim_loss, args))
                if fid_metric is not None:
                    fid_metric.update(((target.float() + 1.0) / 2.0).repeat(1, 3, 1, 1).clamp(0.0, 1.0), real=True)
                    fid_metric.update(((refined.float() + 1.0) / 2.0).repeat(1, 3, 1, 1).clamp(0.0, 1.0), real=False)
        metrics = average_rows(rows)
        metrics["fid"] = float(fid_metric.compute().item()) if fid_metric is not None else float("inf")
        score = selection_score(metrics)
        gate = passes_gate(metrics, args)
        payload = checkpoint_payload(
            model,
            args,
            stage1_channels,
            prediction_mode,
            detail_scale,
            epoch + 1,
            score,
            metrics,
        )
        torch.save(payload, ckpt_dir / "latest_ds_corrector.pt")
        is_score_best = score > best_score
        if is_score_best:
            best_score = score
            best_score_metrics = dict(metrics)
            torch.save(payload, ckpt_dir / "best_score_ds_corrector.pt")
        is_gated_best = gate and score > best_gated_score
        if is_gated_best:
            best_gated_score = score
            best_gated_metrics = dict(metrics)
            torch.save(payload, ckpt_dir / "best_ds_corrector.pt")
        if (epoch + 1) % max(args.save_every, 1) == 0:
            torch.save(payload, ckpt_dir / f"epoch_{epoch + 1:03d}.pt")
        row = {
            "epoch": epoch + 1,
            "score": score,
            "gate": int(gate),
            "best": int(is_gated_best),
            "best_gated": int(is_gated_best),
            "best_score": int(is_score_best),
            **metrics,
        }
        history.append(row)
        write_csv(root / "training_history.csv", history)
        with open(root / "best_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_gated_score": best_gated_score,
                    "best_gated_metrics": best_gated_metrics,
                    "best_score": best_score,
                    "best_score_metrics": best_score_metrics,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(
            f"[DSCorrector][{args.variant}][Epoch {epoch + 1}] "
            f"PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"DeltaPSNR={metrics['delta_psnr']:.4f} DeltaSSIM={metrics['delta_ssim']:.4f} "
            f"DeltaWM={metrics['delta_wm_l1']:.6f} DeltaROI={metrics['delta_roi']:.6f} "
            f"DeltaDiseaseROI={metrics['delta_disease_roi']:.6f} DeltaStripe={metrics['delta_stripe']:.6f} "
            f"SharpRetention={metrics['sharp_retention']:.4f} FID={metrics['fid']:.4f} "
            f"Gate={int(gate)} BestGate={int(is_gated_best)} BestScore={int(is_score_best)}"
        )


if __name__ == "__main__":
    main()
