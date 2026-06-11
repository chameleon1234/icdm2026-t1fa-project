import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd

from pmrf_t1fa.models.pmrf_t1fa import center_channel, prepare_stage1_input, reduce_rgb_to_single_channel
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
from src.eval.disease_sensitive_roi import roi_weight_tensor_from_frame, weighted_grid_roi_l1


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frequency-preserving Stage 2 medical fidelity corrector."
    )
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument(
        "--stage1_ckpt",
        default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt",
    )
    parser.add_argument("--stage1_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run_name", default="pmrf_t1fa_stage2_fidelity_flow_smoke")
    parser.add_argument("--corrector_mode", default="flow", choices=["flow", "direct"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--frequency_cutoff", type=float, default=0.12)
    parser.add_argument("--frequency_transition", type=float, default=0.04)
    parser.add_argument("--source_noise_scale", type=float, default=0.03)
    parser.add_argument("--eval_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--velocity_weight", type=float, default=1.0)
    parser.add_argument("--correction_l1_weight", type=float, default=1.0)
    parser.add_argument("--final_l1_weight", type=float, default=0.40)
    parser.add_argument("--final_mse_weight", type=float, default=0.30)
    parser.add_argument("--final_ssim_weight", type=float, default=0.10)
    parser.add_argument("--wm_l1_weight", type=float, default=1.0)
    parser.add_argument("--roi_weight", type=float, default=0.30)
    parser.add_argument("--disease_roi_csv", default="", help="CSV of disease-sensitive ROI weights built from train FA features.")
    parser.add_argument("--disease_roi_weight", type=float, default=0.0)
    parser.add_argument("--residual_magnitude_weight", type=float, default=0.05)
    parser.add_argument("--hf_preserve_weight", type=float, default=2.0)
    parser.add_argument("--background_weight", type=float, default=0.20)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--roi_rows", type=int, default=2)
    parser.add_argument("--roi_cols", type=int, default=3)
    parser.add_argument("--roi_min_pixels", type=int, default=16)
    parser.add_argument("--best_min_sharp_retention", type=float, default=0.97)
    parser.add_argument("--best_min_delta_wm_l1", type=float, default=0.0)
    parser.add_argument("--best_min_delta_roi", type=float, default=0.0)
    parser.add_argument("--best_min_delta_psnr", type=float, default=-0.03)
    parser.add_argument("--best_min_delta_ssim", type=float, default=-0.002)
    parser.add_argument("--preview_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", default="", help="Explicit latest/epoch checkpoint to resume.")
    return parser.parse_args()


def resume_required_keys() -> tuple[str, ...]:
    return (
        "stage1_ckpt",
        "corrector_mode",
        "width",
        "num_blocks",
        "frequency_cutoff",
        "frequency_transition",
        "source_noise_scale",
        "eval_steps",
        "lr",
        "weight_decay",
        "velocity_weight",
        "correction_l1_weight",
        "final_l1_weight",
        "final_mse_weight",
        "final_ssim_weight",
        "wm_l1_weight",
        "roi_weight",
        "disease_roi_csv",
        "disease_roi_weight",
        "residual_magnitude_weight",
        "hf_preserve_weight",
        "background_weight",
        "brain_t1_threshold",
        "brain_fa_threshold",
        "wm_quantile",
        "wm_min_threshold",
        "roi_rows",
        "roi_cols",
        "roi_min_pixels",
        "best_min_sharp_retention",
        "best_min_delta_wm_l1",
        "best_min_delta_roi",
        "best_min_delta_psnr",
        "best_min_delta_ssim",
    )


def validate_resume_configuration(
    current_args: Dict,
    checkpoint_args: Dict,
    required_keys: tuple[str, ...] | None = None,
) -> None:
    keys = required_keys or resume_required_keys()
    for key in keys:
        if key not in current_args:
            continue
        if key not in checkpoint_args:
            raise ValueError(f"Resume checkpoint missing required key: {key}")
        if checkpoint_args[key] != current_args[key]:
            raise ValueError(
                f"Resume configuration mismatch for {key}: "
                f"checkpoint={checkpoint_args[key]!r}, current={current_args[key]!r}"
            )


def ensure_dirs(run_name: str) -> tuple[Path, Path, Path]:
    root = Path("outputs") / run_name
    ckpt = root / "checkpoints"
    preview = root / "previews"
    ckpt.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    return root, ckpt, preview


def load_disease_roi_weights(path: str, roi_rows: int, roi_cols: int, device: torch.device) -> torch.Tensor | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    weights = roi_weight_tensor_from_frame(frame, roi_rows=roi_rows, roi_cols=roi_cols)
    return weights.to(device=device, dtype=torch.float32)


def frequency_lowpass_mask(
    height: int,
    width: int,
    cutoff: float,
    transition: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not 0.0 < cutoff < 0.5:
        raise ValueError(f"frequency cutoff must be in (0, 0.5), got {cutoff}")
    if transition < 0.0:
        raise ValueError(f"frequency transition must be non-negative, got {transition}")
    fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).view(1, width // 2 + 1)
    radius = torch.sqrt(fx.square() + fy.square())
    if transition == 0.0:
        mask = (radius <= float(cutoff)).float()
    else:
        phase = ((radius - float(cutoff)) / float(transition)).clamp(0.0, 1.0)
        mask = 0.5 * (1.0 + torch.cos(torch.pi * phase))
        mask = torch.where(radius <= float(cutoff), torch.ones_like(mask), mask)
        mask = torch.where(radius >= float(cutoff + transition), torch.zeros_like(mask), mask)
    return mask.to(dtype=dtype).view(1, 1, height, width // 2 + 1)


def frequency_lowpass(x: torch.Tensor, cutoff: float, transition: float) -> torch.Tensor:
    x_fp32 = x.float()
    mask = frequency_lowpass_mask(
        x.shape[-2],
        x.shape[-1],
        cutoff,
        transition,
        x.device,
        torch.float32,
    )
    filtered = torch.fft.irfft2(torch.fft.rfft2(x_fp32) * mask, s=x.shape[-2:])
    return filtered.to(dtype=x.dtype)


def frequency_highpass(x: torch.Tensor, cutoff: float, transition: float) -> torch.Tensor:
    return x - frequency_lowpass(x, cutoff, transition)


def frequency_preservation_error(
    refined: torch.Tensor,
    coarse: torch.Tensor,
    cutoff: float,
    transition: float,
) -> torch.Tensor:
    difference = refined.float() - coarse.float()
    low_mask = frequency_lowpass_mask(
        difference.shape[-2],
        difference.shape[-1],
        cutoff,
        transition,
        difference.device,
        torch.float32,
    )
    forbidden = (low_mask < 1e-6).to(dtype=torch.float32)
    forbidden_difference = torch.fft.irfft2(
        torch.fft.rfft2(difference) * forbidden,
        s=difference.shape[-2:],
    )
    return forbidden_difference.abs().mean()


def compose_frequency_preserving_output(
    coarse: torch.Tensor,
    raw_correction: torch.Tensor,
    cutoff: float,
    transition: float,
) -> torch.Tensor:
    """Allow Stage 2 to write only the configured low-frequency band."""

    return coarse + frequency_lowpass(raw_correction, cutoff, transition)


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
        y = self.dw(self.pw1(self.norm(x)))
        a, b = y.chunk(2, dim=1)
        return x + self.beta * self.pw2(a * torch.sigmoid(b))


def build_corrector_condition(
    t1_stack: torch.Tensor,
    coarse: torch.Tensor,
    cutoff: float,
    transition: float,
) -> torch.Tensor:
    coarse_low = frequency_lowpass(coarse, cutoff, transition)
    coarse_high = coarse - coarse_low
    return torch.cat([t1_stack, coarse, coarse_low, coarse_high], dim=1)


class FidelityCorrector(nn.Module):
    def __init__(self, stage1_channels: int, mode: str = "flow", width: int = 48, num_blocks: int = 8):
        super().__init__()
        if mode not in {"flow", "direct"}:
            raise ValueError(f"Unsupported corrector mode: {mode}")
        self.stage1_channels = int(stage1_channels)
        self.mode = mode
        condition_channels = self.stage1_channels + 3
        input_channels = condition_channels if mode == "direct" else condition_channels + 2
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, width, 3, padding=1),
            *[NAFBlock(width) for _ in range(num_blocks)],
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        t1_stack: torch.Tensor,
        coarse: torch.Tensor,
        state: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        cutoff: float = 0.12,
        transition: float = 0.04,
    ) -> torch.Tensor:
        condition = build_corrector_condition(t1_stack, coarse, cutoff, transition)
        if self.mode == "direct":
            return self.net(condition)
        if state is None or t is None:
            raise ValueError("Flow corrector requires state and t.")
        if t.dim() == 1:
            t_map = t.view(-1, 1, 1, 1).expand_as(state)
        else:
            t_map = t.expand_as(state)
        return self.net(torch.cat([state, t_map.to(state.dtype), condition], dim=1))


def project_flow_endpoint(state: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    remaining = (1.0 - t).view(-1, 1, 1, 1) if t.dim() == 1 else 1.0 - t
    return state + remaining.to(state) * velocity


@torch.no_grad()
def sample_fidelity_correction(
    model: FidelityCorrector,
    t1_stack: torch.Tensor,
    coarse: torch.Tensor,
    cutoff: float,
    transition: float,
    steps: int,
) -> torch.Tensor:
    if model.mode == "direct":
        return model(t1_stack, coarse, cutoff=cutoff, transition=transition)
    if steps <= 0:
        raise ValueError(f"eval steps must be positive, got {steps}")
    state = torch.zeros_like(coarse)
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = torch.full((coarse.shape[0],), step / float(steps), device=coarse.device, dtype=coarse.dtype)
        velocity = model(t1_stack, coarse, state=state, t=t, cutoff=cutoff, transition=transition)
        state = state + dt * velocity
    return state


def build_fidelity_corrector_loss(
    raw_correction: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    ssim_loss_fn: SSIMLoss,
    cutoff: float,
    transition: float,
    correction_l1_weight: float,
    final_l1_weight: float,
    final_mse_weight: float,
    final_ssim_weight: float,
    wm_l1_weight: float,
    roi_weight: float,
    residual_magnitude_weight: float,
    hf_preserve_weight: float,
    background_weight: float,
    roi_rows: int,
    roi_cols: int,
    roi_min_pixels: int,
    velocity_pred: torch.Tensor | None = None,
    velocity_target: torch.Tensor | None = None,
    velocity_weight: float = 0.0,
    disease_roi_weight: float = 0.0,
    disease_roi_weights: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    target_correction = frequency_lowpass(target - coarse, cutoff, transition)
    correction = frequency_lowpass(raw_correction, cutoff, transition)
    refined = coarse + correction
    correction_l1 = F.l1_loss(correction, target_correction)
    final_l1 = F.l1_loss(refined, target)
    final_mse = F.mse_loss(refined, target)
    final_ssim = ssim_loss_fn(refined, target)
    wm_l1 = masked_l1_loss(refined, target, wm_mask)
    roi = roi_consistency_loss(refined, target, wm_mask, roi_rows, roi_cols, roi_min_pixels)
    disease_roi = refined.new_tensor(0.0)
    if disease_roi_weight > 0.0 and disease_roi_weights is not None:
        disease_roi = weighted_grid_roi_l1(refined, target, wm_mask, disease_roi_weights, roi_min_pixels)
    residual_magnitude = correction.abs().mean()
    hf_preserve = F.l1_loss(
        frequency_highpass(refined, cutoff, transition),
        frequency_highpass(coarse, cutoff, transition),
    )
    background = ((1.0 - brain_mask.to(correction)) * correction.abs()).mean()
    velocity = refined.new_tensor(0.0)
    if velocity_pred is not None and velocity_target is not None:
        velocity = F.mse_loss(velocity_pred, velocity_target)
    total = (
        velocity_weight * velocity
        + correction_l1_weight * correction_l1
        + final_l1_weight * final_l1
        + final_mse_weight * final_mse
        + final_ssim_weight * final_ssim
        + wm_l1_weight * wm_l1
        + roi_weight * roi
        + disease_roi_weight * disease_roi
        + residual_magnitude_weight * residual_magnitude
        + hf_preserve_weight * hf_preserve
        + background_weight * background
    )
    return {
        "total": total,
        "velocity": velocity,
        "correction_l1": correction_l1,
        "final_l1": final_l1,
        "final_mse": final_mse,
        "final_ssim": final_ssim,
        "wm_l1": wm_l1,
        "roi": roi,
        "disease_roi": disease_roi,
        "residual_magnitude": residual_magnitude,
        "hf_preserve": hf_preserve,
        "background": background,
        "correction": correction,
        "target_correction": target_correction,
        "refined": refined,
    }


def passes_checkpoint_gate(
    metrics: Dict[str, float],
    min_sharp_retention: float,
    min_delta_wm_l1: float,
    min_delta_roi: float,
    min_delta_psnr: float,
    min_delta_ssim: float,
) -> bool:
    return (
        metrics["sharp_retention"] >= min_sharp_retention
        and metrics["delta_wm_l1"] >= min_delta_wm_l1
        and metrics["delta_roi"] >= min_delta_roi
        and metrics["delta_psnr"] >= min_delta_psnr
        and metrics["delta_ssim"] >= min_delta_ssim
    )


def selection_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["delta_psnr"]
        + 10.0 * metrics["delta_ssim"]
        + 20.0 * metrics["delta_wm_l1"]
        + 10.0 * metrics["delta_roi"]
        - 5.0 * metrics["hf_leak"]
    )


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _average_metrics(rows: list[Dict[str, float]]) -> Dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}


def _metric_row(
    refined: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    t1_img: torch.Tensor,
    ssim_loss_fn: SSIMLoss,
    args: argparse.Namespace,
) -> Dict[str, float]:
    _, wm_mask = build_training_masks(
        t1_img,
        target,
        args.brain_t1_threshold,
        args.brain_fa_threshold,
        args.wm_quantile,
        args.wm_min_threshold,
    )
    sharp = laplacian_variance(refined).item()
    coarse_sharp = laplacian_variance(coarse).item()
    wm_l1 = masked_l1_loss(refined, target, wm_mask).item()
    coarse_wm_l1 = masked_l1_loss(coarse, target, wm_mask).item()
    roi = roi_consistency_loss(refined, target, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels).item()
    coarse_roi = roi_consistency_loss(coarse, target, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels).item()
    psnr = compute_psnr(refined, target).item()
    coarse_psnr = compute_psnr(coarse, target).item()
    ssim = 1.0 - ssim_loss_fn(refined, target).item()
    coarse_ssim = 1.0 - ssim_loss_fn(coarse, target).item()
    hf_leak = frequency_preservation_error(
        refined,
        coarse,
        args.frequency_cutoff,
        args.frequency_transition,
    ).item()
    return {
        "psnr": psnr,
        "ssim": ssim,
        "wm_l1": wm_l1,
        "roi": roi,
        "sharp_ratio": sharp,
        "coarse_psnr": coarse_psnr,
        "coarse_ssim": coarse_ssim,
        "coarse_wm_l1": coarse_wm_l1,
        "coarse_roi": coarse_roi,
        "coarse_sharp_ratio": coarse_sharp,
        "sharp_retention": sharp / max(coarse_sharp, 1e-8),
        "delta_psnr": psnr - coarse_psnr,
        "delta_ssim": ssim - coarse_ssim,
        "delta_wm_l1": coarse_wm_l1 - wm_l1,
        "delta_roi": coarse_roi - roi,
        "hf_leak": hf_leak,
        "correction_l1": F.l1_loss(refined, coarse).item(),
    }


def save_preview(
    preview_dir: Path,
    step: int,
    t1_img: torch.Tensor,
    coarse: torch.Tensor,
    refined: torch.Tensor,
    target: torch.Tensor,
    cutoff: float,
    transition: float,
) -> None:
    images = [
        center_channel(t1_img),
        coarse,
        refined,
        target,
        4.0 * (refined - coarse),
        4.0 * frequency_highpass(refined - coarse, cutoff, transition),
    ]
    panel = torch.cat(
        [torch.clamp((image + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1) for image in images],
        dim=-1,
    )
    save_image(panel, preview_dir / f"step_{step}.png", nrow=1)


def _make_checkpoint(
    model: FidelityCorrector,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    stage1_channels: int,
    stage1_prediction_mode: str,
    stage1_detail_scale: float,
    epoch: int,
    score: float,
    metrics: Dict[str, float],
) -> Dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "stage1_channels": stage1_channels,
        "stage1_prediction_mode": stage1_prediction_mode,
        "stage1_detail_scale": stage1_detail_scale,
        "epoch": epoch,
        "score": score,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    root_dir, ckpt_dir, preview_dir = ensure_dirs(args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_device = device if args.stage1_device == "auto" else torch.device(args.stage1_device)
    stage1, prediction_mode, stage1_channels, detail_scale = load_stage1_model(args.stage1_ckpt, stage1_device)
    train_dataset = maybe_limit_dataset(make_slice_dataset(args.train_t1_dir, args.train_fa_dir, stage1_channels), args.train_limit)
    val_dataset = maybe_limit_dataset(make_slice_dataset(args.val_t1_dir, args.val_fa_dir, stage1_channels), args.val_limit)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    model = FidelityCorrector(stage1_channels, args.corrector_mode, args.width, args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    ssim_loss_fn = SSIMLoss().to(device)
    disease_roi_weights = load_disease_roi_weights(args.disease_roi_csv, args.roi_rows, args.roi_cols, device)
    start_epoch = 0
    best_score = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume_configuration(vars(args), checkpoint.get("args", {}))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", checkpoint.get("score", best_score)))

    log_path = root_dir / "train.log"
    global_step = 0
    print(
        f"Fidelity corrector training on {len(train_dataset)} train / {len(val_dataset)} val slices | "
        f"mode={args.corrector_mode} stage1_channels={stage1_channels} device={device} "
        f"width={args.width} blocks={args.num_blocks} frequency={args.frequency_cutoff}+{args.frequency_transition} "
        f"eval_steps={args.eval_steps} disease_roi_weight={args.disease_roi_weight} "
        f"disease_roi_csv={args.disease_roi_csv or 'none'}"
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_totals: Dict[str, float] = {}
        for batch in tqdm(train_loader, desc=f"FidelityCorrector Epoch {epoch + 1}/{args.epochs}"):
            t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
            target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
            with torch.no_grad(), _autocast_context(device, args.mixed_precision):
                coarse = predict_stage1_batch(stage1, t1_img, stage1_device, prediction_mode, detail_scale).to(device)
            brain_mask, wm_mask = build_training_masks(
                t1_img,
                target,
                args.brain_t1_threshold,
                args.brain_fa_threshold,
                args.wm_quantile,
                args.wm_min_threshold,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                velocity_pred = None
                velocity_target = None
                if args.corrector_mode == "flow":
                    target_correction = frequency_lowpass(
                        target - coarse,
                        args.frequency_cutoff,
                        args.frequency_transition,
                    )
                    source = frequency_lowpass(
                        torch.randn_like(target_correction) * args.source_noise_scale,
                        args.frequency_cutoff,
                        args.frequency_transition,
                    )
                    t = torch.rand((target.shape[0],), device=device, dtype=target.dtype)
                    t_view = t.view(-1, 1, 1, 1)
                    state = (1.0 - t_view) * source + t_view * target_correction
                    velocity_target = target_correction - source
                    velocity_pred = model(
                        t1_img,
                        coarse,
                        state=state,
                        t=t,
                        cutoff=args.frequency_cutoff,
                        transition=args.frequency_transition,
                    )
                    raw_correction = project_flow_endpoint(state, velocity_pred, t)
                else:
                    raw_correction = model(
                        t1_img,
                        coarse,
                        cutoff=args.frequency_cutoff,
                        transition=args.frequency_transition,
                    )
                losses = build_fidelity_corrector_loss(
                    raw_correction,
                    coarse,
                    target,
                    brain_mask,
                    wm_mask,
                    ssim_loss_fn,
                    args.frequency_cutoff,
                    args.frequency_transition,
                    args.correction_l1_weight,
                    args.final_l1_weight,
                    args.final_mse_weight,
                    args.final_ssim_weight,
                    args.wm_l1_weight,
                    args.roi_weight,
                    args.residual_magnitude_weight,
                    args.hf_preserve_weight,
                    args.background_weight,
                    args.roi_rows,
                    args.roi_cols,
                    args.roi_min_pixels,
                    velocity_pred,
                    velocity_target,
                    args.velocity_weight if args.corrector_mode == "flow" else 0.0,
                    args.disease_roi_weight,
                    disease_roi_weights,
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
                if torch.is_tensor(value) and value.dim() == 0:
                    train_totals[key] = train_totals.get(key, 0.0) + value.detach().item()
            global_step += 1
            if args.preview_every > 0 and global_step % args.preview_every == 0:
                save_preview(
                    preview_dir,
                    global_step,
                    t1_img[:1],
                    coarse[:1],
                    losses["refined"][:1],
                    target[:1],
                    args.frequency_cutoff,
                    args.frequency_transition,
                )

        model.eval()
        val_rows: list[Dict[str, float]] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"FidelityCorrector Val {epoch + 1}/{args.epochs}"):
                t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
                target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
                with _autocast_context(device, args.mixed_precision):
                    coarse = predict_stage1_batch(stage1, t1_img, stage1_device, prediction_mode, detail_scale).to(device)
                    raw_correction = sample_fidelity_correction(
                        model,
                        t1_img,
                        coarse,
                        args.frequency_cutoff,
                        args.frequency_transition,
                        args.eval_steps,
                    )
                    refined = compose_frequency_preserving_output(
                        coarse,
                        raw_correction,
                        args.frequency_cutoff,
                        args.frequency_transition,
                    )
                val_rows.append(_metric_row(refined, coarse, target, t1_img, ssim_loss_fn, args))
        metrics = _average_metrics(val_rows)
        score = selection_score(metrics)
        passed_gate = passes_checkpoint_gate(
            metrics,
            args.best_min_sharp_retention,
            args.best_min_delta_wm_l1,
            args.best_min_delta_roi,
            args.best_min_delta_psnr,
            args.best_min_delta_ssim,
        )
        is_best = passed_gate and score > best_score
        if is_best:
            best_score = score
            checkpoint = _make_checkpoint(
                model, optimizer, args, stage1_channels, prediction_mode, detail_scale, epoch, score, metrics
            )
            checkpoint["best_score"] = best_score
            torch.save(checkpoint, ckpt_dir / "best_fidelity_corrector.pt")
        latest = _make_checkpoint(
            model, optimizer, args, stage1_channels, prediction_mode, detail_scale, epoch, score, metrics
        )
        latest["best_score"] = best_score
        torch.save(latest, ckpt_dir / "latest_fidelity_corrector.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(latest, ckpt_dir / f"epoch_{epoch + 1:03d}.pt")

        train_avg = {key: value / max(len(train_loader), 1) for key, value in train_totals.items()}
        line = (
            f"[FidelityCorrector][Epoch {epoch + 1}] PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"WM_L1={metrics['wm_l1']:.6f} ROI={metrics['roi']:.6f} SharpRetention={metrics['sharp_retention']:.4f} "
            f"DeltaPSNR={metrics['delta_psnr']:.4f} DeltaSSIM={metrics['delta_ssim']:.4f} "
            f"DeltaWM_L1={metrics['delta_wm_l1']:.6f} DeltaROI={metrics['delta_roi']:.6f} "
            f"HFLeak={metrics['hf_leak']:.8f} CorrectionL1={metrics['correction_l1']:.6f} "
            f"TrainLoss={train_avg.get('total', 0.0):.6f} Score={score:.4f} Gate={int(passed_gate)} Best={int(is_best)}"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with open(root_dir / "last_metrics.json", "w", encoding="utf-8") as handle:
            json.dump({"metrics": metrics, "score": score, "gate": passed_gate, "train": train_avg}, handle, indent=2)


if __name__ == "__main__":
    main()
