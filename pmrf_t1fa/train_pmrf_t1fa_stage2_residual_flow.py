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

from pmrf_t1fa.models.pmrf_t1fa import (
    center_channel,
    prepare_stage1_input,
    reduce_rgb_to_single_channel,
    single_channel_laplacian,
)
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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PMRF-T1FA Stage 2 as an uncertainty-aware residual flow."
    )
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt")
    parser.add_argument("--stage1_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run_name", default="pmrf_t1fa_stage2_residual_flow_smoke")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=0.45)
    parser.add_argument("--source_noise_scale", type=float, default=1.0)
    parser.add_argument("--eval_steps", type=int, default=8)
    parser.add_argument("--eval_noise_scale", type=float, default=0.0)
    parser.add_argument("--eval_samples", type=int, default=1)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--velocity_weight", type=float, default=1.0)
    parser.add_argument("--residual_l1_weight", type=float, default=0.25)
    parser.add_argument("--image_l1_weight", type=float, default=0.20)
    parser.add_argument("--image_ssim_weight", type=float, default=0.04)
    parser.add_argument("--wm_residual_weight", type=float, default=0.50)
    parser.add_argument("--residual_energy_weight", type=float, default=0.10)
    parser.add_argument("--uncertainty_nll_weight", type=float, default=0.05)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--preview_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--best_metric", default="wm_paired", choices=["psnr", "wm_paired", "residual_calibrated"])
    return parser.parse_args()


def ensure_dirs(run_name: str) -> tuple[Path, Path, Path]:
    root = Path("outputs") / run_name
    ckpt = root / "checkpoints"
    preview = root / "previews"
    ckpt.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    return root, ckpt, preview


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


def build_residual_flow_condition(t1_stack: torch.Tensor, coarse: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    t1_center = center_channel(t1_stack)
    t1_edge = single_channel_laplacian(t1_center)
    coarse_edge = single_channel_laplacian(coarse)
    return torch.cat([t1_stack, coarse, sigma, t1_edge, coarse_edge], dim=1)


class ResidualFlowStage2(nn.Module):
    """Predict uncertainty and a conditional velocity field in residual space."""

    def __init__(
        self,
        stage1_channels: int,
        width: int = 48,
        num_blocks: int = 8,
        sigma_min: float = 0.01,
        sigma_max: float = 0.45,
    ):
        super().__init__()
        self.stage1_channels = int(stage1_channels)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        condition_channels = self.stage1_channels + 4
        self.sigma_net = nn.Sequential(
            nn.Conv2d(condition_channels - 1, width, 3, padding=1),
            NAFBlock(width),
            NAFBlock(width),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        velocity_channels = condition_channels + 2
        self.velocity_net = nn.Sequential(
            nn.Conv2d(velocity_channels, width, 3, padding=1),
            *[NAFBlock(width) for _ in range(num_blocks)],
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def predict_sigma(self, t1_stack: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        t1_center = center_channel(t1_stack)
        sigma_input = torch.cat(
            [
                t1_stack,
                coarse,
                single_channel_laplacian(t1_center),
                single_channel_laplacian(coarse),
            ],
            dim=1,
        )
        raw = self.sigma_net(sigma_input)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)

    def forward(
        self,
        residual: torch.Tensor,
        t: torch.Tensor,
        t1_stack: torch.Tensor,
        coarse: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() == 1:
            t_map = t.view(-1, 1, 1, 1).expand_as(residual)
        elif t.dim() == 4:
            t_map = t.expand_as(residual)
        else:
            raise ValueError(f"Unsupported t shape: {tuple(t.shape)}")
        condition = build_residual_flow_condition(t1_stack, coarse, sigma)
        return self.velocity_net(torch.cat([residual, t_map.to(residual.dtype), condition], dim=1))


def project_residual_endpoint(rt: torch.Tensor, velocity_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        remaining = (1.0 - t).view(-1, 1, 1, 1)
    elif t.dim() == 4:
        remaining = 1.0 - t
    else:
        raise ValueError(f"Unsupported t shape: {tuple(t.shape)}")
    return rt + remaining.to(dtype=rt.dtype, device=rt.device) * velocity_pred


def residual_energy_calibration_loss(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    wm_mask: torch.Tensor,
) -> torch.Tensor:
    pred_hp = laplacian_filter(pred_residual.float()).abs()
    target_hp = laplacian_filter(target_residual.float()).abs()
    wm_mask = wm_mask.to(dtype=pred_hp.dtype, device=pred_hp.device)
    denom = wm_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    pred_energy = (pred_hp * wm_mask).flatten(1).sum(dim=1) / denom
    target_energy = (target_hp * wm_mask).flatten(1).sum(dim=1) / denom
    return F.l1_loss(pred_energy, target_energy)


def uncertainty_nll_loss(
    target_residual: torch.Tensor,
    sigma: torch.Tensor,
    brain_mask: torch.Tensor,
) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-4)
    nll = torch.abs(target_residual.detach()) / sigma + torch.log(sigma)
    brain_mask = brain_mask.to(dtype=nll.dtype, device=nll.device)
    return (nll * brain_mask).sum() / brain_mask.sum().clamp_min(1.0)


def build_residual_flow_loss(
    velocity_pred: torch.Tensor,
    rt: torch.Tensor,
    r0: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    t: torch.Tensor,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    velocity_weight: float,
    residual_l1_weight: float,
    image_l1_weight: float,
    wm_residual_weight: float,
    residual_energy_weight: float,
    uncertainty_nll_weight: float,
    image_ssim_weight: float = 0.0,
    ssim_loss_fn: SSIMLoss | None = None,
) -> Dict[str, torch.Tensor]:
    target_residual = target - coarse
    velocity_target = target_residual - r0
    pred_residual = project_residual_endpoint(rt, velocity_pred, t)
    refined = clamp_to_image_range(coarse + pred_residual)
    velocity = F.mse_loss(velocity_pred, velocity_target)
    residual_l1 = F.l1_loss(pred_residual, target_residual)
    image_l1 = F.l1_loss(refined, target)
    wm_residual = masked_l1_loss(pred_residual, target_residual, wm_mask)
    residual_energy = residual_energy_calibration_loss(pred_residual, target_residual, wm_mask)
    uncertainty_nll = uncertainty_nll_loss(target_residual, sigma, brain_mask)
    image_ssim = refined.new_tensor(0.0) if ssim_loss_fn is None else ssim_loss_fn(refined, target)
    total = (
        velocity_weight * velocity
        + residual_l1_weight * residual_l1
        + image_l1_weight * image_l1
        + wm_residual_weight * wm_residual
        + residual_energy_weight * residual_energy
        + uncertainty_nll_weight * uncertainty_nll
        + image_ssim_weight * image_ssim
    )
    return {
        "total": total,
        "velocity": velocity,
        "residual_l1": residual_l1,
        "image_l1": image_l1,
        "wm_residual": wm_residual,
        "residual_energy": residual_energy,
        "uncertainty_nll": uncertainty_nll,
        "image_ssim": image_ssim,
        "pred_residual": pred_residual,
        "refined": refined,
        "velocity_target": velocity_target,
    }


@torch.no_grad()
def euler_sample_residual_flow(
    model: nn.Module,
    t1_stack: torch.Tensor,
    coarse: torch.Tensor,
    sigma: torch.Tensor,
    steps: int = 8,
    noise_scale: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if noise_scale == 0.0:
        residual = torch.zeros_like(coarse)
    else:
        residual = torch.randn(
            coarse.shape,
            generator=generator,
            device=coarse.device,
            dtype=coarse.dtype,
        ) * sigma * float(noise_scale)
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = torch.full((coarse.shape[0],), step / float(steps), device=coarse.device, dtype=coarse.dtype)
        velocity = model(residual, t, t1_stack, coarse, sigma)
        residual = residual + dt * velocity
    return residual


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def sample_time(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand((batch_size,), device=device, dtype=dtype)


def save_preview(
    preview_dir: Path,
    step: int,
    t1_img: torch.Tensor,
    coarse: torch.Tensor,
    refined: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> None:
    t1_vis = torch.clamp((center_channel(t1_img) + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    coarse_vis = torch.clamp((coarse + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    refined_vis = torch.clamp((refined + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    sigma_norm = sigma / sigma.flatten(1).amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    sigma_vis = torch.clamp(sigma_norm, 0.0, 1.0).repeat(1, 3, 1, 1)
    target_vis = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    panel = torch.cat([t1_vis, coarse_vis, refined_vis, sigma_vis, target_vis], dim=-1)
    save_image(panel, preview_dir / f"step_{step}.png", nrow=1)


def _average_metrics(rows: list[Dict[str, float]]) -> Dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}


def _metric_row(
    refined: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    t1_img: torch.Tensor,
    ssim_loss_fn: SSIMLoss,
    args: argparse.Namespace,
) -> Dict[str, float]:
    brain_mask, wm_mask = build_training_masks(
        t1_img,
        target,
        brain_t1_threshold=args.brain_t1_threshold,
        brain_fa_threshold=args.brain_fa_threshold,
        wm_quantile=args.wm_quantile,
        wm_min_threshold=args.wm_min_threshold,
    )
    target_residual = target - coarse
    pred_residual = refined - coarse
    target_sharp = laplacian_variance(target).item()
    sharp = laplacian_variance(refined).item()
    coarse_sharp = laplacian_variance(coarse).item()
    wm_l1 = masked_l1_loss(refined, target, wm_mask).item()
    coarse_wm_l1 = masked_l1_loss(coarse, target, wm_mask).item()
    energy = residual_energy_calibration_loss(pred_residual, target_residual, wm_mask).item()
    sigma_mae = masked_l1_loss(sigma, torch.abs(target_residual).detach(), brain_mask).item()
    return {
        "psnr": compute_psnr(refined, target).item(),
        "ssim": 1.0 - ssim_loss_fn(refined, target).item(),
        "mse": F.mse_loss(refined, target).item(),
        "l1": F.l1_loss(refined, target).item(),
        "wm_l1": wm_l1,
        "coarse_psnr": compute_psnr(coarse, target).item(),
        "coarse_ssim": 1.0 - ssim_loss_fn(coarse, target).item(),
        "coarse_wm_l1": coarse_wm_l1,
        "sharp_ratio": sharp / max(target_sharp, 1e-8),
        "coarse_sharp_ratio": coarse_sharp / max(target_sharp, 1e-8),
        "delta_psnr": compute_psnr(refined, target).item() - compute_psnr(coarse, target).item(),
        "delta_ssim": (1.0 - ssim_loss_fn(refined, target).item()) - (1.0 - ssim_loss_fn(coarse, target).item()),
        "delta_wm_l1": coarse_wm_l1 - wm_l1,
        "residual_energy_error": energy,
        "sigma_abs_residual_l1": sigma_mae,
        "sigma_mean": sigma.mean().item(),
    }


def selection_score(metrics: Dict[str, float], best_metric: str) -> float:
    if best_metric == "psnr":
        return metrics["psnr"]
    if best_metric == "residual_calibrated":
        return (
            metrics["psnr"]
            + 10.0 * metrics["ssim"]
            - 8.0 * metrics["wm_l1"]
            - 12.0 * metrics["residual_energy_error"]
        )
    return (
        metrics["psnr"]
        + 10.0 * metrics["ssim"]
        - 8.0 * metrics["wm_l1"]
        - 2.0 * metrics["residual_energy_error"]
    )


def main() -> None:
    args = parse_args()
    root_dir, ckpt_dir, preview_dir = ensure_dirs(args.run_name)
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

    model = ResidualFlowStage2(
        stage1_channels=stage1_channels,
        width=args.width,
        num_blocks=args.num_blocks,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    ssim_loss_fn = SSIMLoss().to(device)
    best_score = float("-inf")
    global_step = 0
    log_path = root_dir / "train.log"

    print(
        f"Residual flow Stage2 training on {len(train_dataset)} train slices / {len(val_dataset)} val slices | "
        f"stage1_channels={stage1_channels} | device={device} | width={args.width} blocks={args.num_blocks} | "
        f"sigma=[{args.sigma_min},{args.sigma_max}] noise_scale={args.source_noise_scale} "
        f"eval_steps={args.eval_steps} eval_noise={args.eval_noise_scale} eval_samples={args.eval_samples} | "
        f"weights velocity={args.velocity_weight} residual={args.residual_l1_weight} image={args.image_l1_weight}/{args.image_ssim_weight} "
        f"wm_residual={args.wm_residual_weight} energy={args.residual_energy_weight} nll={args.uncertainty_nll_weight}"
    )

    for epoch in range(args.epochs):
        model.train()
        train_totals: Dict[str, float] = {}
        for batch in tqdm(train_loader, desc=f"ResidualFlow Epoch {epoch + 1}/{args.epochs}"):
            t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
            target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
            with torch.no_grad(), _autocast_context(device, args.mixed_precision):
                coarse = predict_stage1_batch(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale).to(device)
            brain_mask, wm_mask = build_training_masks(
                t1_img,
                target,
                brain_t1_threshold=args.brain_t1_threshold,
                brain_fa_threshold=args.brain_fa_threshold,
                wm_quantile=args.wm_quantile,
                wm_min_threshold=args.wm_min_threshold,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                sigma = model.predict_sigma(t1_img, coarse)
                target_residual = target - coarse
                r0 = torch.randn_like(target_residual) * sigma * float(args.source_noise_scale)
                t = sample_time(target.shape[0], device, target.dtype)
                t_view = t.view(-1, 1, 1, 1)
                rt = (1.0 - t_view) * r0 + t_view * target_residual
                velocity_pred = model(rt, t, t1_img, coarse, sigma)
                losses = build_residual_flow_loss(
                    velocity_pred=velocity_pred,
                    rt=rt,
                    r0=r0,
                    coarse=coarse,
                    target=target,
                    sigma=sigma,
                    t=t,
                    brain_mask=brain_mask,
                    wm_mask=wm_mask,
                    velocity_weight=args.velocity_weight,
                    residual_l1_weight=args.residual_l1_weight,
                    image_l1_weight=args.image_l1_weight,
                    wm_residual_weight=args.wm_residual_weight,
                    residual_energy_weight=args.residual_energy_weight,
                    uncertainty_nll_weight=args.uncertainty_nll_weight,
                    image_ssim_weight=args.image_ssim_weight,
                    ssim_loss_fn=ssim_loss_fn,
                )
            if scaler.is_enabled():
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            for key, value in losses.items():
                if value.dim() == 0:
                    train_totals[key] = train_totals.get(key, 0.0) + float(value.detach().item())
            global_step += 1
            if args.preview_every > 0 and global_step % args.preview_every == 0:
                save_preview(preview_dir, global_step, t1_img[:1], coarse[:1], losses["refined"][:1], sigma[:1], target[:1])

        model.eval()
        val_rows: list[Dict[str, float]] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"ResidualFlow Val {epoch + 1}/{args.epochs}"):
                t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
                target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
                with _autocast_context(device, args.mixed_precision):
                    coarse = predict_stage1_batch(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale).to(device)
                    sigma = model.predict_sigma(t1_img, coarse)
                    residual_samples = []
                    sample_count = max(int(args.eval_samples), 1)
                    for _ in range(sample_count):
                        residual_samples.append(
                            euler_sample_residual_flow(
                                model,
                                t1_img,
                                coarse,
                                sigma,
                                steps=args.eval_steps,
                                noise_scale=args.eval_noise_scale,
                            )
                        )
                    pred_residual = torch.stack(residual_samples, dim=0).mean(dim=0)
                    refined = clamp_to_image_range(coarse + pred_residual)
                val_rows.append(_metric_row(refined, coarse, target, sigma, t1_img, ssim_loss_fn, args))

        metrics = _average_metrics(val_rows)
        score = selection_score(metrics, args.best_metric)
        is_best = score > best_score
        if is_best:
            best_score = score
            checkpoint = {
                "model": model.state_dict(),
                "args": vars(args),
                "stage1_channels": stage1_channels,
                "stage1_prediction_mode": stage1_prediction_mode,
                "stage1_detail_scale": stage1_detail_scale,
                "metrics": metrics,
                "score": score,
            }
            torch.save(checkpoint, ckpt_dir / "best_residual_flow.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "stage1_channels": stage1_channels,
                    "metrics": metrics,
                    "score": score,
                },
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
            )

        train_avg = {key: value / max(len(train_loader), 1) for key, value in train_totals.items()}
        line = (
            f"[ResidualFlow][Epoch {epoch + 1}] PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"coarse_PSNR={metrics['coarse_psnr']:.4f} coarse_SSIM={metrics['coarse_ssim']:.4f} "
            f"WM_L1={metrics['wm_l1']:.6f} DeltaPSNR={metrics['delta_psnr']:.4f} "
            f"DeltaSSIM={metrics['delta_ssim']:.4f} DeltaWM_L1={metrics['delta_wm_l1']:.6f} "
            f"SharpRatio={metrics['sharp_ratio']:.4f} CoarseSharpRatio={metrics['coarse_sharp_ratio']:.4f} "
            f"ResidualEnergyErr={metrics['residual_energy_error']:.6f} SigmaAbsResidualL1={metrics['sigma_abs_residual_l1']:.6f} "
            f"SigmaMean={metrics['sigma_mean']:.4f} TrainLoss={train_avg.get('total', 0.0):.5f} "
            f"Score={score:.4f} Best={int(is_best)}"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with open(root_dir / "last_metrics.json", "w", encoding="utf-8") as handle:
            json.dump({"metrics": metrics, "score": score, "train": train_avg}, handle, indent=2)


if __name__ == "__main__":
    main()
