import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from pmrf_t1fa.models.pmrf_t1fa import center_channel, prepare_stage1_input, reduce_rgb_to_single_channel, single_channel_laplacian
from pmrf_t1fa.train_pmrf_t1fa_stage2 import (
    SSIMLoss,
    build_training_masks,
    clamp_to_image_range,
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
    parser = argparse.ArgumentParser(description="Train a high-pass residual Stage 2 refiner for PMRF-T1FA.")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt")
    parser.add_argument("--stage1_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run_name", default="pmrf_t1fa_stage2_hp_refiner_smoke")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--preview_batch_size", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=128)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--hp_kernel_size", type=int, default=5)
    parser.add_argument("--lp_kernel_size", type=int, default=13)
    parser.add_argument("--hp_residual_weight", type=float, default=1.0)
    parser.add_argument("--hp_image_weight", type=float, default=1.0)
    parser.add_argument("--wm_hp_weight", type=float, default=2.0)
    parser.add_argument("--wm_final_l1_weight", type=float, default=0.5)
    parser.add_argument("--wm_guard_weight", type=float, default=0.0)
    parser.add_argument("--wm_guard_margin", type=float, default=0.0)
    parser.add_argument("--lowpass_weight", type=float, default=2.0)
    parser.add_argument("--final_l1_weight", type=float, default=0.05)
    parser.add_argument("--final_ssim_weight", type=float, default=0.02)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


def ensure_dirs(run_name: str) -> tuple[Path, Path, Path]:
    root = Path("outputs") / run_name
    ckpt = root / "checkpoints"
    preview = root / "previews"
    ckpt.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    return root, ckpt, preview


def lowpass(x: torch.Tensor, kernel_size: int = 13) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def highpass(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return x - lowpass(x, kernel_size=kernel_size)


class NAFBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = channels * 2
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pw2 = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.pw2(y)
        return x + self.beta * y


class HighPassRefinerNet(nn.Module):
    def __init__(self, in_channels: int, width: int = 32, num_blocks: int = 8):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.blocks(self.in_proj(x)))


def build_hp_refiner_input(t1_stack: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
    t1_center = center_channel(t1_stack)
    t1_edge = single_channel_laplacian(t1_center)
    coarse_edge = single_channel_laplacian(coarse)
    return torch.cat([t1_stack, coarse, t1_edge, coarse_edge], dim=1)


def build_hp_refiner_loss(
    hp_pred: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    hp_kernel_size: int,
    lp_kernel_size: int,
    hp_residual_weight: float,
    hp_image_weight: float,
    wm_hp_weight: float,
    wm_final_l1_weight: float,
    wm_guard_weight: float,
    wm_guard_margin: float,
    lowpass_weight: float,
    final_l1_weight: float,
    final_ssim_weight: float,
    ssim_loss_fn: SSIMLoss | None,
) -> Dict[str, torch.Tensor]:
    final = coarse + hp_pred
    hp_target = highpass(target, hp_kernel_size) - highpass(coarse, hp_kernel_size)
    hp_final_target = highpass(target, hp_kernel_size)
    hp_final = highpass(final, hp_kernel_size)
    hp_error = torch.abs(hp_final - hp_final_target)
    hp_residual = F.l1_loss(hp_pred, hp_target)
    hp_image = F.l1_loss(hp_final, hp_final_target)
    wm_hp = (hp_error * wm_mask.to(dtype=hp_error.dtype, device=hp_error.device)).sum() / wm_mask.sum().clamp_min(1.0)
    wm_final_l1 = masked_l1_loss(final, target, wm_mask)
    coarse_wm_l1 = masked_l1_loss(coarse, target, wm_mask).detach()
    wm_guard = F.relu(wm_final_l1 - coarse_wm_l1 + float(wm_guard_margin))
    lowpass_consistency = masked_l1_loss(lowpass(final, lp_kernel_size), lowpass(coarse, lp_kernel_size), brain_mask)
    final_l1 = F.l1_loss(final, target)
    final_ssim = final.new_tensor(0.0) if ssim_loss_fn is None else ssim_loss_fn(final, target)
    total = (
        hp_residual_weight * hp_residual
        + hp_image_weight * hp_image
        + wm_hp_weight * wm_hp
        + wm_final_l1_weight * wm_final_l1
        + wm_guard_weight * wm_guard
        + lowpass_weight * lowpass_consistency
        + final_l1_weight * final_l1
        + final_ssim_weight * final_ssim
    )
    return {
        "total": total,
        "hp_residual": hp_residual,
        "hp_image": hp_image,
        "wm_hp": wm_hp,
        "wm_final_l1": wm_final_l1,
        "wm_guard": wm_guard,
        "lowpass": lowpass_consistency,
        "final_l1": final_l1,
        "final_ssim": final_ssim,
        "final": final,
    }


def save_preview(preview_dir: Path, step: int, t1_img: torch.Tensor, coarse: torch.Tensor, refined: torch.Tensor, target: torch.Tensor) -> None:
    t1_vis = torch.clamp((center_channel(t1_img) + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    coarse_vis = torch.clamp((coarse + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    refined_vis = torch.clamp((refined + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    target_vis = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    panel = torch.cat([t1_vis, coarse_vis, refined_vis, target_vis], dim=-1)
    save_image(panel, preview_dir / f"step_{step}.png", nrow=1)


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _metric_dict(refined: torch.Tensor, coarse: torch.Tensor, target: torch.Tensor, t1_img: torch.Tensor, ssim_loss_fn: SSIMLoss, args) -> Dict[str, float]:
    brain_mask, wm_mask = build_training_masks(
        t1_img,
        target,
        brain_t1_threshold=args.brain_t1_threshold,
        brain_fa_threshold=args.brain_fa_threshold,
        wm_quantile=args.wm_quantile,
        wm_min_threshold=args.wm_min_threshold,
    )
    coarse_ssim = 1.0 - ssim_loss_fn(coarse, target).item()
    refined_ssim = 1.0 - ssim_loss_fn(refined, target).item()
    sharp = laplacian_variance(refined).item()
    coarse_sharp = laplacian_variance(coarse).item()
    target_sharp = laplacian_variance(target).item()
    wm_l1 = masked_l1_loss(refined, target, wm_mask).item()
    coarse_wm_l1 = masked_l1_loss(coarse, target, wm_mask).item()
    return {
        "psnr": compute_psnr(refined, target).item(),
        "ssim": refined_ssim,
        "mse": F.mse_loss(refined, target).item(),
        "l1": F.l1_loss(refined, target).item(),
        "hp_l1": F.l1_loss(laplacian_filter(refined), laplacian_filter(target)).item(),
        "wm_l1": wm_l1,
        "coarse_psnr": compute_psnr(coarse, target).item(),
        "coarse_ssim": coarse_ssim,
        "coarse_wm_l1": coarse_wm_l1,
        "sharp_ratio": sharp / max(target_sharp, 1e-8),
        "coarse_sharp_ratio": coarse_sharp / max(target_sharp, 1e-8),
        "delta_psnr": compute_psnr(refined, target).item() - compute_psnr(coarse, target).item(),
        "delta_ssim": refined_ssim - coarse_ssim,
        "delta_sharp": (sharp - coarse_sharp) / max(target_sharp, 1e-8),
        "delta_wm_l1": coarse_wm_l1 - wm_l1,
    }


def _average_metrics(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main() -> None:
    args = parse_args()
    root_dir, ckpt_dir, preview_dir = ensure_dirs(args.run_name)
    log_path = root_dir / "train.log"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage1_device == "cpu":
        stage1_device = torch.device("cpu")
    elif args.stage1_device == "cuda":
        stage1_device = torch.device("cuda")
    else:
        stage1_device = device

    stage1, stage1_prediction_mode, stage1_channels, stage1_detail_scale = load_stage1_model(args.stage1_ckpt, stage1_device)
    train_dataset = maybe_limit_dataset(make_slice_dataset(args.train_t1_dir, args.train_fa_dir, stage1_channels), args.train_limit)
    val_dataset = maybe_limit_dataset(make_slice_dataset(args.val_t1_dir, args.val_fa_dir, stage1_channels), args.val_limit)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    model = HighPassRefinerNet(in_channels=stage1_channels + 3, width=args.width, num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    ssim_loss_fn = SSIMLoss().to(device)
    best_score = float("-inf")
    global_step = 0

    print(
        f"HP refiner training on {len(train_dataset)} train slices / {len(val_dataset)} val slices | "
        f"stage1_channels={stage1_channels} | device={device} | width={args.width} blocks={args.num_blocks} | "
        f"hp/lp kernels={args.hp_kernel_size}/{args.lp_kernel_size} | weights hp={args.hp_residual_weight}/{args.hp_image_weight} "
        f"wm_hp={args.wm_hp_weight} wm_final={args.wm_final_l1_weight} wm_guard={args.wm_guard_weight}@{args.wm_guard_margin} "
        f"lowpass={args.lowpass_weight} "
        f"final={args.final_l1_weight}/{args.final_ssim_weight}"
    )

    for epoch in range(args.epochs):
        model.train()
        train_totals: Dict[str, float] = {}
        for batch in tqdm(train_loader, desc=f"HPRefiner Epoch {epoch + 1}/{args.epochs}"):
            t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
            target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
            with torch.no_grad(), _autocast_context(device, args.mixed_precision):
                coarse = predict_stage1_batch(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale).to(device)
            brain_mask, wm_mask = build_training_masks(
                t1_img,
                target,
                args.brain_t1_threshold,
                args.brain_fa_threshold,
                args.wm_quantile,
                args.wm_min_threshold,
            )
            model_input = build_hp_refiner_input(t1_img, coarse)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                hp_pred = model(model_input)
                losses = build_hp_refiner_loss(
                    hp_pred,
                    coarse,
                    target,
                    brain_mask,
                    wm_mask,
                    args.hp_kernel_size,
                    args.lp_kernel_size,
                    args.hp_residual_weight,
                    args.hp_image_weight,
                    args.wm_hp_weight,
                    args.wm_final_l1_weight,
                    args.wm_guard_weight,
                    args.wm_guard_margin,
                    args.lowpass_weight,
                    args.final_l1_weight,
                    args.final_ssim_weight,
                    ssim_loss_fn,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key, value in losses.items():
                if key == "final":
                    continue
                train_totals[key] = train_totals.get(key, 0.0) + float(value.detach().cpu())
            if args.preview_every > 0 and global_step % args.preview_every == 0:
                save_preview(preview_dir, global_step, t1_img[: args.preview_batch_size], coarse[: args.preview_batch_size], losses["final"][: args.preview_batch_size], target[: args.preview_batch_size])

        model.eval()
        rows: list[Dict[str, float]] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"HPRefiner Val {epoch + 1}/{args.epochs}"):
                t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
                target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
                with _autocast_context(device, args.mixed_precision):
                    coarse = predict_stage1_batch(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale).to(device)
                    hp_pred = model(build_hp_refiner_input(t1_img, coarse))
                    refined = clamp_to_image_range(coarse + hp_pred)
                rows.append(_metric_dict(refined.float(), coarse.float(), target.float(), t1_img, ssim_loss_fn, args))

        metrics = _average_metrics(rows)
        train_batches = max(len(train_loader), 1)
        train_summary = {key: value / train_batches for key, value in train_totals.items()}
        score = (
            metrics["delta_sharp"]
            + 10.0 * max(metrics["delta_wm_l1"], 0.0)
            - 30.0 * max(-metrics["delta_wm_l1"], 0.0)
            - 0.05 * max(-metrics["delta_psnr"], 0.0)
        )
        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "score": score,
                    "metrics": metrics,
                    "stage1_channels": stage1_channels,
                },
                ckpt_dir / "best_hp_refiner.pt",
            )
            torch.save(model.state_dict(), root_dir / "best_hp_refiner.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "score": score,
                "metrics": metrics,
                "stage1_channels": stage1_channels,
            },
            ckpt_dir / "latest_hp_refiner.pt",
        )
        log_row = {"epoch": epoch + 1, "score": score, **{f"train_{k}": v for k, v in train_summary.items()}, **metrics}
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        print(
            f"[HPRefiner][Epoch {epoch + 1}] PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"L1={metrics['l1']:.6f} HP_L1={metrics['hp_l1']:.6f} WM_L1={metrics['wm_l1']:.6f} "
            f"SharpRatio={metrics['sharp_ratio']:.4f} CoarseSharpRatio={metrics['coarse_sharp_ratio']:.4f} "
            f"DeltaSharp={metrics['delta_sharp']:.4f} DeltaWM_L1={metrics['delta_wm_l1']:.6f} "
            f"DeltaPSNR={metrics['delta_psnr']:.4f} Score={score:.4f} Best={int(is_best)}"
        )


if __name__ == "__main__":
    main()
