import argparse
import os
from typing import Dict, List, Tuple

import lpips
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.utils import save_image
from tqdm import tqdm

from pmrf_t1fa.models.pmrf_t1fa import (
    DetailStage1Net,
    GradientLoss,
    SSIMLoss,
    Stage1Net,
    center_channel,
    compose_stage1_output,
    highpass_residual,
    prepare_stage1_input,
    predict_stage1_fa,
    reduce_rgb_to_single_channel,
    split_stage1_output,
)
from src.data.t1fa_stack_dataset import T1FAStackDataset
from src.datasets import T1FADataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class ZeroLPIPSLoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.new_zeros((pred.shape[0],))


def parse_args():
    parser = argparse.ArgumentParser(description="Train PMRF-T1FA Stage 1 coarse predictor")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--run_name", default="pmrf_t1fa_stage1")
    parser.add_argument("--stage1_model_variant", default="single", choices=["single", "detail"])
    parser.add_argument("--stage1_prediction_mode", default="residual", choices=["absolute", "residual"])
    parser.add_argument("--stage1_detail_scale", type=float, default=0.60)
    parser.add_argument("--context_slices", type=int, default=1, help="Odd number of adjacent T1 slices used as Stage 1 input channels.")
    parser.add_argument("--posterior_mean_preset", action="store_true", help="Use PSNR/SSIM-oriented Stage 1 weights with little/no perceptual pressure.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_limit", type=int, default=0, help="Use only the first N training slices for smoke tests.")
    parser.add_argument("--val_limit", type=int, default=0, help="Use only the first N validation slices for smoke tests.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--preview_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_true")
    parser.add_argument("--lpips_max_weight", type=float, default=0.03)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--mse_weight", type=float, default=0.50)
    parser.add_argument("--l1_start_weight", type=float, default=1.00)
    parser.add_argument("--l1_end_weight", type=float, default=0.60)
    parser.add_argument("--ssim_start_weight", type=float, default=1.00)
    parser.add_argument("--ssim_end_weight", type=float, default=0.50)
    parser.add_argument("--grad_weight", type=float, default=0.05)
    parser.add_argument("--hf_weight", type=float, default=0.02)
    parser.add_argument("--detail_hf_weight", type=float, default=0.0)
    parser.add_argument("--detail_lap_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--image_check_start_epoch", type=int, default=5)
    parser.add_argument("--image_check_patience", type=int, default=2)
    parser.add_argument("--disable_strict_image_check", action="store_false", dest="strict_image_check")
    parser.set_defaults(strict_image_check=True)
    parser.add_argument("--best_metric", default="paired", choices=["psnr", "mse", "mae", "fid", "sharpness", "balanced", "paired", "detail_paired"])
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--fid_eval_every", type=int, default=1)
    parser.add_argument("--kid_subset_size", type=int, default=100)
    parser.add_argument("--score_psnr_weight", type=float, default=1.0)
    parser.add_argument("--score_ssim_weight", type=float, default=10.0)
    parser.add_argument("--score_lpips_weight", type=float, default=2.0)
    parser.add_argument("--score_fid_weight", type=float, default=0.06)
    parser.add_argument("--score_sharp_weight", type=float, default=4.0)
    parser.add_argument("--paired_psnr_weight", type=float, default=1.0)
    parser.add_argument("--paired_ssim_weight", type=float, default=10.0)
    parser.add_argument("--paired_mse_weight", type=float, default=200.0)
    parser.add_argument("--paired_mae_weight", type=float, default=10.0)
    parser.add_argument("--paired_sharp_weight", type=float, default=6.0)
    parser.add_argument("--detail_target_sharp_ratio", type=float, default=0.90)
    parser.add_argument("--detail_max_sharp_ratio", type=float, default=1.20)
    parser.add_argument("--detail_oversharp_penalty_weight", type=float, default=12.0)
    parser.add_argument("--rollback_on_anomaly", action="store_true")
    parser.set_defaults(rollback_on_anomaly=True)
    parser.add_argument("--skip_nonfinite_batches", action="store_true")
    parser.set_defaults(skip_nonfinite_batches=True)
    args = parser.parse_args()
    if args.context_slices < 1 or args.context_slices % 2 == 0:
        raise ValueError(f"--context_slices must be a positive odd integer, got {args.context_slices}")
    if args.posterior_mean_preset:
        args.stage1_model_variant = "single"
        args.stage1_detail_scale = 0.0
        args.lpips_max_weight = 0.0
        args.mse_weight = 0.85
        args.l1_start_weight = 1.10
        args.l1_end_weight = 0.85
        args.ssim_start_weight = 0.70
        args.ssim_end_weight = 0.45
        args.grad_weight = 0.03
        args.hf_weight = 0.01
        args.best_metric = "paired"
    if args.stage1_model_variant == "single":
        args.stage1_detail_scale = 0.0
        args.detail_hf_weight = 0.0
        args.detail_lap_weight = 0.0
    if args.disable_lpips:
        args.lpips_max_weight = 0.0
    return args


def ensure_dirs(run_name: str):
    root = os.path.join("outputs", run_name)
    ckpt_dir = os.path.join(root, "checkpoints")
    preview_dir = os.path.join(root, "previews")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    return root, ckpt_dir, preview_dir


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
    target_01 = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
    mse = F.mse_loss(pred_01, target_01)
    return -10.0 * torch.log10(mse + 1e-8)


def clamp_to_image_range(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


def laplacian_filter(x: torch.Tensor) -> torch.Tensor:
    kernel = x.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def laplacian_variance(x: torch.Tensor) -> torch.Tensor:
    x_01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    lap = laplacian_filter(x_01)
    return lap.flatten(1).var(dim=1, unbiased=False).mean()


def compute_balanced_score(metrics: Dict[str, float], args) -> float:
    return (
        args.score_psnr_weight * metrics["psnr"]
        + args.score_ssim_weight * metrics["ssim"]
        - args.score_lpips_weight * metrics["lpips"]
        - args.score_fid_weight * metrics["fid"]
        + args.score_sharp_weight * metrics["sharp_ratio"]
    )


def compute_paired_score(metrics: Dict[str, float], args) -> float:
    return (
        args.paired_psnr_weight * metrics["psnr"]
        + args.paired_ssim_weight * metrics["ssim"]
        - args.paired_mse_weight * metrics["mse"]
        - args.paired_mae_weight * metrics["l1"]
    )


def compute_detail_paired_score(metrics: Dict[str, float], args) -> float:
    target_ratio = max(float(getattr(args, "detail_target_sharp_ratio", 0.90)), 1e-6)
    max_ratio = float(getattr(args, "detail_max_sharp_ratio", 1.20))
    sharp_ratio = float(metrics["sharp_ratio"])
    bounded_bonus = min(sharp_ratio, target_ratio) / target_ratio
    oversharp_penalty = max(sharp_ratio - max_ratio, 0.0)
    return (
        compute_paired_score(metrics, args)
        + args.paired_sharp_weight * bounded_bonus
        - getattr(args, "detail_oversharp_penalty_weight", 12.0) * oversharp_penalty
    )


def make_slice_dataset(t1_dir: str, fa_dir: str, context_slices: int):
    if context_slices > 1:
        return T1FAStackDataset(t1_dir, fa_dir, context_slices=context_slices, target_size=(224, 224))
    return T1FADataset(t1_dir, fa_dir, preload_ram=False)


def maybe_limit_dataset(dataset, limit: int):
    if limit <= 0:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def pick_fixed_preview_batch(val_dataset, device: torch.device, stage1_channels: int):
    total = len(val_dataset)
    candidate_ids = sorted({total // 8, total // 3, total // 2, (total * 3) // 4})
    t1_list = []
    fa_list = []
    for idx in candidate_ids[:4]:
        sample = val_dataset[idx]
        t1_list.append(prepare_stage1_input(sample["t1_slice"].unsqueeze(0).to(device), stage1_channels))
        fa_list.append(reduce_rgb_to_single_channel(sample["fa_slice"].unsqueeze(0).to(device)))
    return torch.cat(t1_list, dim=0), torch.cat(fa_list, dim=0)


def summarize_prediction_health(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
    target_01 = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
    pred_flat = pred_01.flatten()
    target_flat = target_01.flatten()

    q01 = torch.quantile(pred_flat, 0.01).item()
    q99 = torch.quantile(pred_flat, 0.99).item()
    stats = {
        "pred_mean": pred_flat.mean().item(),
        "target_mean": target_flat.mean().item(),
        "pred_std": pred_flat.std(unbiased=False).item(),
        "target_std": target_flat.std(unbiased=False).item(),
        "pred_low_sat": (pred_flat <= 0.02).float().mean().item(),
        "pred_high_sat": (pred_flat >= 0.98).float().mean().item(),
        "pred_dynamic_range": q99 - q01,
    }
    return stats


def detect_image_anomaly(pred: torch.Tensor, target: torch.Tensor) -> Tuple[bool, List[str], Dict[str, float]]:
    stats = summarize_prediction_health(pred, target)
    reasons: List[str] = []
    mean_gap = abs(stats["pred_mean"] - stats["target_mean"])
    std_ratio = stats["pred_std"] / max(stats["target_std"], 1e-6)

    if mean_gap > 0.18:
        reasons.append(f"mean_gap={mean_gap:.3f}")
    if stats["pred_dynamic_range"] < 0.08:
        reasons.append(f"low_dynamic_range={stats['pred_dynamic_range']:.3f}")
    if stats["pred_std"] < 0.04 or std_ratio < 0.45:
        reasons.append(f"low_std={stats['pred_std']:.3f},std_ratio={std_ratio:.3f}")
    if stats["pred_high_sat"] > 0.55:
        reasons.append(f"high_saturation={stats['pred_high_sat']:.3f}")
    if stats["pred_low_sat"] > 0.98:
        reasons.append(f"low_saturation={stats['pred_low_sat']:.3f}")
    if stats["pred_mean"] < max(0.5 * stats["target_mean"], 0.03):
        reasons.append(f"too_dark={stats['pred_mean']:.3f}")

    return len(reasons) > 0, reasons, stats


def is_better(metrics: Dict[str, float], best_metrics: Dict[str, float], best_metric: str) -> bool:
    if best_metric == "mse":
        return metrics["mse"] < best_metrics["mse"]
    if best_metric == "mae":
        return metrics["l1"] < best_metrics["l1"]
    if best_metric == "fid":
        return metrics["fid"] < best_metrics["fid"]
    if best_metric == "sharpness":
        return metrics["sharp_ratio"] > best_metrics["sharp_ratio"]
    if best_metric == "balanced":
        return metrics["balanced_score"] > best_metrics["balanced_score"]
    if best_metric == "paired":
        return metrics["paired_score"] > best_metrics["paired_score"]
    if best_metric == "detail_paired":
        return metrics["detail_paired_score"] > best_metrics["detail_paired_score"]
    if metrics["psnr"] > best_metrics["psnr"] + 1e-6:
        return True
    if abs(metrics["psnr"] - best_metrics["psnr"]) <= 1e-6 and metrics["ssim"] > best_metrics["ssim"]:
        return True
    return False


def build_stage1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    t1_img: torch.Tensor,
    detail_pred: torch.Tensor | None,
    ssim_loss_fn: SSIMLoss,
    grad_loss_fn: GradientLoss,
    lpips_loss_fn,
    epoch: int,
    total_epochs: int,
    mse_weight: float,
    l1_start_weight: float,
    l1_end_weight: float,
    ssim_start_weight: float,
    ssim_end_weight: float,
    grad_weight: float,
    hf_weight: float,
    detail_hf_weight: float,
    detail_lap_weight: float,
    stage1_prediction_mode: str,
    stage1_detail_scale: float,
    lpips_max_weight: float,
) -> Dict[str, torch.Tensor]:
    # Keep loss evaluation in fp32 even when the model runs with mixed precision.
    pred_fp32 = pred.float()
    target_fp32 = target.float()
    loss_mse = F.mse_loss(pred_fp32, target_fp32)
    pred_clamp = clamp_to_image_range(pred_fp32)
    loss_l1 = F.l1_loss(pred_clamp, target_fp32)
    loss_ssim = ssim_loss_fn(pred_clamp, target_fp32)
    loss_grad = grad_loss_fn(pred_clamp, target_fp32)
    loss_hf = F.l1_loss(laplacian_filter(pred_clamp), laplacian_filter(target_fp32))
    if detail_pred is None:
        loss_detail_hf = pred_fp32.new_tensor(0.0)
        loss_detail_lap = pred_fp32.new_tensor(0.0)
    else:
        if stage1_prediction_mode == "residual":
            detail_target = highpass_residual(target_fp32 - center_channel(t1_img).float())
        else:
            detail_target = highpass_residual(target_fp32)
        detail_contrib = float(stage1_detail_scale) * highpass_residual(detail_pred.float())
        loss_detail_hf = F.l1_loss(detail_contrib, detail_target)
        loss_detail_lap = F.l1_loss(laplacian_filter(detail_contrib), laplacian_filter(detail_target))
    with torch.autocast(device_type=pred.device.type, enabled=False):
        loss_lpips = lpips_loss_fn(
            pred_clamp.repeat(1, 3, 1, 1),
            target_fp32.repeat(1, 3, 1, 1),
        ).mean()

    progress_ratio = float(epoch) / max(total_epochs - 1, 1)
    w_l1 = l1_start_weight + (l1_end_weight - l1_start_weight) * progress_ratio
    w_ssim = ssim_start_weight + (ssim_end_weight - ssim_start_weight) * progress_ratio
    w_lpips = min(lpips_max_weight, 0.02 + max(lpips_max_weight - 0.02, 0.0) * progress_ratio)
    w_grad = grad_weight * (0.5 if epoch < 5 else 1.0)
    w_hf = hf_weight * (0.5 if epoch < 5 else 1.0)

    total = (
        mse_weight * loss_mse
        + w_l1 * loss_l1
        + w_ssim * loss_ssim
        + w_grad * loss_grad
        + w_hf * loss_hf
        + detail_hf_weight * loss_detail_hf
        + detail_lap_weight * loss_detail_lap
        + w_lpips * loss_lpips
    )
    return {
        "total": total,
        "mse": loss_mse,
        "l1": loss_l1,
        "ssim": loss_ssim,
        "grad": loss_grad,
        "hf": loss_hf,
        "detail_hf": loss_detail_hf,
        "detail_lap": loss_detail_lap,
        "lpips": loss_lpips,
        "pred_clamp": pred_clamp,
        "w_l1": torch.tensor(w_l1, device=pred.device, dtype=pred.dtype),
        "w_ssim": torch.tensor(w_ssim, device=pred.device, dtype=pred.dtype),
        "w_lpips": torch.tensor(w_lpips, device=pred.device, dtype=pred.dtype),
        "w_grad": torch.tensor(w_grad, device=pred.device, dtype=pred.dtype),
        "w_hf": torch.tensor(w_hf, device=pred.device, dtype=pred.dtype),
    }


def save_preview(preview_dir, step, t1_img, coarse_pred, fa_img, suffix: str = ""):
    t1_vis = torch.clamp((center_channel(t1_img) + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    coarse_vis = torch.clamp((clamp_to_image_range(coarse_pred) + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    fa_vis = torch.clamp((fa_img + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    panel = torch.cat([t1_vis, coarse_vis, fa_vis], dim=-1)
    save_image(panel, os.path.join(preview_dir, f"step_{step}{suffix}.png"), nrow=1)


def main():
    args = parse_args()
    root_dir, ckpt_dir, preview_dir = ensure_dirs(args.run_name)
    log_path = os.path.join(root_dir, "train.log")

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    stage1_channels = int(args.context_slices)
    train_dataset = make_slice_dataset(args.train_t1_dir, args.train_fa_dir, stage1_channels)
    val_dataset = make_slice_dataset(args.val_t1_dir, args.val_fa_dir, stage1_channels)
    train_dataset = maybe_limit_dataset(train_dataset, args.train_limit)
    val_dataset = maybe_limit_dataset(val_dataset, args.val_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if args.stage1_model_variant == "detail":
        model = DetailStage1Net(in_channels=stage1_channels, out_channels=1)
    else:
        model = Stage1Net(in_channels=stage1_channels, out_channels=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ssim_loss_fn = SSIMLoss()
    grad_loss_fn = GradientLoss()
    if args.disable_lpips or args.lpips_max_weight <= 0.0:
        args.lpips_max_weight = 0.0
        lpips_loss_fn = ZeroLPIPSLoss()
    else:
        lpips_loss_fn = lpips.LPIPS(net="vgg")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
    kid_metric = KernelInceptionDistance(subset_size=args.kid_subset_size, normalize=True)
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    model, optimizer, train_loader, val_loader, ssim_loss_fn, grad_loss_fn, lpips_loss_fn, fid_metric, kid_metric = accelerator.prepare(
        model, optimizer, train_loader, val_loader, ssim_loss_fn, grad_loss_fn, lpips_loss_fn, fid_metric, kid_metric
    )

    fixed_t1, fixed_fa = pick_fixed_preview_batch(val_dataset, accelerator.device, stage1_channels)

    best_metrics = {
        "psnr": float("-inf"),
        "ssim": float("-inf"),
        "mse": float("inf"),
        "l1": float("inf"),
        "lpips": float("inf"),
        "fid": float("inf"),
        "kid": float("inf"),
        "sharpness": float("-inf"),
        "sharp_ratio": float("-inf"),
        "balanced_score": float("-inf"),
        "paired_score": float("-inf"),
        "detail_paired_score": float("-inf"),
    }
    start_epoch = 0
    global_step = 0
    anomaly_epochs = 0
    no_improve_epochs = 0
    latest_path = os.path.join(ckpt_dir, "latest_stage1.pt")
    healthy_latest_path = os.path.join(ckpt_dir, "healthy_latest_stage1.pt")
    best_path = os.path.join(ckpt_dir, "best_stage1.pt")

    auto_resume = (not args.no_auto_resume) or args.resume
    resume_path = None
    if auto_resume:
        if os.path.exists(healthy_latest_path):
            resume_path = healthy_latest_path
        elif os.path.exists(latest_path):
            resume_path = latest_path

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_metrics["psnr"] = checkpoint.get("best_psnr", best_metrics["psnr"])
        best_metrics["ssim"] = checkpoint.get("best_ssim", best_metrics["ssim"])
        best_metrics["mse"] = checkpoint.get("best_mse", best_metrics["mse"])
        best_metrics["l1"] = checkpoint.get("best_l1", best_metrics["l1"])
        best_metrics["lpips"] = checkpoint.get("best_lpips", best_metrics["lpips"])
        best_metrics["fid"] = checkpoint.get("best_fid", best_metrics["fid"])
        best_metrics["kid"] = checkpoint.get("best_kid", best_metrics["kid"])
        best_metrics["sharpness"] = checkpoint.get("best_sharpness", best_metrics["sharpness"])
        best_metrics["sharp_ratio"] = checkpoint.get("best_sharp_ratio", best_metrics["sharp_ratio"])
        best_metrics["balanced_score"] = checkpoint.get("best_balanced_score", best_metrics["balanced_score"])
        best_metrics["paired_score"] = checkpoint.get("best_paired_score", best_metrics["paired_score"])
        best_metrics["detail_paired_score"] = checkpoint.get("best_detail_paired_score", best_metrics["detail_paired_score"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        anomaly_epochs = checkpoint.get("anomaly_epochs", 0)
        no_improve_epochs = checkpoint.get("no_improve_epochs", 0)
        accelerator.print(f"Resume Stage 1 from {resume_path}")

    accelerator.print(f"Stage 1 training on {len(train_dataset)} train slices / {len(val_dataset)} val slices")
    accelerator.print(
        f"Stage 1 input context_slices={stage1_channels} | posterior_mean_preset={args.posterior_mean_preset} | "
        f"stage1_model_variant={args.stage1_model_variant} | stage1_detail_scale={args.stage1_detail_scale}"
    )
    accelerator.print(f"Device={accelerator.device} | mixed_precision={args.mixed_precision}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {
            "total": 0.0,
            "mse": 0.0,
            "l1": 0.0,
            "ssim": 0.0,
            "grad": 0.0,
            "hf": 0.0,
            "detail_hf": 0.0,
            "detail_lap": 0.0,
            "lpips": 0.0,
        }
        skipped_nonfinite = 0

        progress = tqdm(
            train_loader,
            desc=f"Stage1 Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )
        for batch in progress:
            t1_img = prepare_stage1_input(batch["t1_slice"].to(accelerator.device), stage1_channels)
            fa_img = reduce_rgb_to_single_channel(batch["fa_slice"].to(accelerator.device))

            raw_output = model(t1_img)
            pred_delta = compose_stage1_output(raw_output, detail_scale=args.stage1_detail_scale)
            if args.stage1_prediction_mode == "residual":
                coarse_pred = center_channel(t1_img) + pred_delta
            else:
                coarse_pred = pred_delta
            _, detail_pred = split_stage1_output(raw_output)
            losses = build_stage1_loss(
                coarse_pred,
                fa_img,
                t1_img,
                detail_pred,
                ssim_loss_fn,
                grad_loss_fn,
                lpips_loss_fn,
                epoch,
                args.epochs,
                args.mse_weight,
                args.l1_start_weight,
                args.l1_end_weight,
                args.ssim_start_weight,
                args.ssim_end_weight,
                args.grad_weight,
                args.hf_weight,
                args.detail_hf_weight,
                args.detail_lap_weight,
                args.stage1_prediction_mode,
                args.stage1_detail_scale,
                args.lpips_max_weight,
            )

            finite_flags = {key: torch.isfinite(value).all().item() for key, value in losses.items() if torch.is_tensor(value)}
            if not all(finite_flags.values()):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if accelerator.is_main_process:
                    with open(log_path, "a", encoding="utf-8") as handle:
                        handle.write(
                            f"[NonFinite][Epoch {epoch + 1}][Step {global_step + 1}] "
                            + " ".join([f"{k}={finite_flags[k]}" for k in sorted(finite_flags.keys())])
                            + "\n"
                        )
                if args.skip_nonfinite_batches:
                    continue
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch + 1}, step {global_step + 1}")

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(losses["total"])
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            for key in running:
                running[key] += losses[key].item()

            if accelerator.is_main_process:
                progress.set_postfix(
                    loss=losses["total"].item(),
                    l1_w=losses["w_l1"].item(),
                    ssim_w=losses["w_ssim"].item(),
                    lpips_w=losses["w_lpips"].item(),
                    dhf=losses["detail_hf"].item(),
                )
                if global_step % args.preview_every == 0:
                    model.eval()
                    with torch.no_grad():
                        preview_pred = predict_stage1_fa(
                            model,
                            fixed_t1,
                            clamp=False,
                            prediction_mode=args.stage1_prediction_mode,
                            detail_scale=args.stage1_detail_scale,
                        )
                        save_preview(preview_dir, global_step, fixed_t1, preview_pred, fixed_fa)
                        is_bad, reasons, stats = detect_image_anomaly(preview_pred, fixed_fa)
                        if is_bad:
                            save_preview(preview_dir, global_step, fixed_t1, preview_pred, fixed_fa, suffix="_anomaly")
                            with open(log_path, "a", encoding="utf-8") as handle:
                                handle.write(
                                    f"[PreviewAnomaly][Step {global_step}] reasons={';'.join(reasons)} "
                                    f"pred_mean={stats['pred_mean']:.4f} target_mean={stats['target_mean']:.4f} "
                                    f"pred_std={stats['pred_std']:.4f} target_std={stats['target_std']:.4f}\n"
                                )
                    model.train()

        model.eval()
        fid_metric.reset()
        kid_metric.reset()
        val_total = {
            "mse": 0.0,
            "l1": 0.0,
            "ssim_loss": 0.0,
            "grad": 0.0,
            "hf": 0.0,
            "lpips": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "sharpness": 0.0,
            "target_sharpness": 0.0,
        }
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Stage1 Val {epoch + 1}/{args.epochs}",
                disable=not accelerator.is_local_main_process,
            ):
                t1_img = prepare_stage1_input(batch["t1_slice"].to(accelerator.device), stage1_channels)
                fa_img = reduce_rgb_to_single_channel(batch["fa_slice"].to(accelerator.device))
                coarse_pred = predict_stage1_fa(
                    model,
                    t1_img,
                    clamp=True,
                    prediction_mode=args.stage1_prediction_mode,
                    detail_scale=args.stage1_detail_scale,
                )

                val_total["mse"] += F.mse_loss(coarse_pred, fa_img).item()
                val_total["l1"] += F.l1_loss(coarse_pred, fa_img).item()
                ssim_loss = ssim_loss_fn(coarse_pred, fa_img)
                val_total["ssim_loss"] += ssim_loss.item()
                val_total["ssim"] += (1.0 - ssim_loss).item()
                val_total["grad"] += grad_loss_fn(coarse_pred, fa_img).item()
                val_total["hf"] += F.l1_loss(laplacian_filter(coarse_pred), laplacian_filter(fa_img)).item()
                val_total["lpips"] += lpips_loss_fn(
                    coarse_pred.repeat(1, 3, 1, 1), fa_img.repeat(1, 3, 1, 1)
                ).mean().item()
                val_total["psnr"] += compute_psnr(coarse_pred, fa_img).item()
                val_total["sharpness"] += laplacian_variance(coarse_pred).item()
                val_total["target_sharpness"] += laplacian_variance(fa_img).item()
                if (epoch + 1) % args.fid_eval_every == 0:
                    coarse_01_3c = torch.clamp((coarse_pred + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
                    fa_01_3c = torch.clamp((fa_img + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
                    fid_metric.update(fa_01_3c, real=True)
                    fid_metric.update(coarse_01_3c, real=False)
                    kid_metric.update(fa_01_3c, real=True)
                    kid_metric.update(coarse_01_3c, real=False)
                val_batches += 1

        metrics = {key: value / max(val_batches, 1) for key, value in val_total.items()}
        if (epoch + 1) % args.fid_eval_every == 0:
            kid_mean, _ = kid_metric.compute()
            metrics["fid"] = fid_metric.compute().item()
            metrics["kid"] = kid_mean.item()
        else:
            metrics["fid"] = best_metrics["fid"] if torch.isfinite(torch.tensor(best_metrics["fid"])) else float("inf")
            metrics["kid"] = best_metrics["kid"] if torch.isfinite(torch.tensor(best_metrics["kid"])) else float("inf")
        metrics["sharp_ratio"] = metrics["sharpness"] / max(metrics["target_sharpness"], 1e-8)
        metrics["balanced_score"] = compute_balanced_score(metrics, args)
        metrics["paired_score"] = compute_paired_score(metrics, args)
        metrics["detail_paired_score"] = compute_detail_paired_score(metrics, args)
        with torch.no_grad():
            fixed_preview_eval = predict_stage1_fa(
                model,
                fixed_t1,
                clamp=True,
                prediction_mode=args.stage1_prediction_mode,
                detail_scale=args.stage1_detail_scale,
            )
        is_bad_epoch = False
        anomaly_reasons: List[str] = []
        anomaly_stats: Dict[str, float] = {}
        if epoch + 1 >= args.image_check_start_epoch:
            is_bad_epoch, anomaly_reasons, anomaly_stats = detect_image_anomaly(fixed_preview_eval, fixed_fa)
            if metrics["psnr"] < 16.0:
                is_bad_epoch = True
                anomaly_reasons.append(f"low_val_psnr={metrics['psnr']:.3f}")

        anomaly_epochs = anomaly_epochs + 1 if is_bad_epoch else 0
        is_best = is_better(metrics, best_metrics, args.best_metric)
        no_improve_epochs = 0 if is_best else (no_improve_epochs + 1)

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_metrics["psnr"] if not is_best else metrics["psnr"],
            "best_ssim": best_metrics["ssim"] if not is_best else metrics["ssim"],
            "best_mse": best_metrics["mse"] if not is_best else metrics["mse"],
            "best_l1": best_metrics["l1"] if not is_best else metrics["l1"],
            "best_lpips": best_metrics["lpips"] if not is_best else metrics["lpips"],
            "best_fid": best_metrics["fid"] if not is_best else metrics["fid"],
            "best_kid": best_metrics["kid"] if not is_best else metrics["kid"],
            "best_sharpness": best_metrics["sharpness"] if not is_best else metrics["sharpness"],
            "best_sharp_ratio": best_metrics["sharp_ratio"] if not is_best else metrics["sharp_ratio"],
            "best_balanced_score": best_metrics["balanced_score"] if not is_best else metrics["balanced_score"],
            "best_paired_score": best_metrics["paired_score"] if not is_best else metrics["paired_score"],
            "best_detail_paired_score": best_metrics["detail_paired_score"] if not is_best else metrics["detail_paired_score"],
            "anomaly_epochs": anomaly_epochs,
            "no_improve_epochs": no_improve_epochs,
            "args": vars(args),
        }

        if accelerator.is_main_process:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"[Epoch {epoch + 1}] train_total={running['total'] / max(len(train_loader), 1):.6f} "
                    f"val_psnr={metrics['psnr']:.4f} val_ssim={metrics['ssim']:.4f} val_mse={metrics['mse']:.6f} "
                    f"val_l1={metrics['l1']:.6f} val_grad={metrics['grad']:.6f} val_hf={metrics['hf']:.6f} "
                    f"val_lpips={metrics['lpips']:.6f} val_fid={metrics['fid']:.4f} val_kid={metrics['kid']:.6f} "
                    f"sharp={metrics['sharpness']:.6f} sharp_ratio={metrics['sharp_ratio']:.4f} "
                    f"balanced={metrics['balanced_score']:.4f} paired={metrics['paired_score']:.4f} "
                    f"detail_paired={metrics['detail_paired_score']:.4f} anomaly_epochs={anomaly_epochs} "
                    f"skipped_nonfinite={skipped_nonfinite}\n"
                )
                if anomaly_reasons:
                    handle.write(
                        f"[ImageCheck][Epoch {epoch + 1}] reasons={';'.join(anomaly_reasons)} "
                        f"pred_mean={anomaly_stats.get('pred_mean', float('nan')):.4f} "
                        f"target_mean={anomaly_stats.get('target_mean', float('nan')):.4f} "
                        f"pred_std={anomaly_stats.get('pred_std', float('nan')):.4f} "
                        f"target_std={anomaly_stats.get('target_std', float('nan')):.4f}\n"
                    )
            torch.save(checkpoint, latest_path)
            if not is_bad_epoch:
                torch.save(checkpoint, healthy_latest_path)
            if (epoch + 1) % args.save_every == 0:
                torch.save(checkpoint, os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt"))

            if is_best and not is_bad_epoch:
                best_metrics["psnr"] = metrics["psnr"]
                best_metrics["ssim"] = metrics["ssim"]
                best_metrics["mse"] = metrics["mse"]
                best_metrics["l1"] = metrics["l1"]
                best_metrics["lpips"] = metrics["lpips"]
                best_metrics["fid"] = metrics["fid"]
                best_metrics["kid"] = metrics["kid"]
                best_metrics["sharpness"] = metrics["sharpness"]
                best_metrics["sharp_ratio"] = metrics["sharp_ratio"]
                best_metrics["balanced_score"] = metrics["balanced_score"]
                best_metrics["paired_score"] = metrics["paired_score"]
                best_metrics["detail_paired_score"] = metrics["detail_paired_score"]
                checkpoint["best_psnr"] = best_metrics["psnr"]
                checkpoint["best_ssim"] = best_metrics["ssim"]
                checkpoint["best_mse"] = best_metrics["mse"]
                checkpoint["best_l1"] = best_metrics["l1"]
                checkpoint["best_lpips"] = best_metrics["lpips"]
                checkpoint["best_fid"] = best_metrics["fid"]
                checkpoint["best_kid"] = best_metrics["kid"]
                checkpoint["best_sharpness"] = best_metrics["sharpness"]
                checkpoint["best_sharp_ratio"] = best_metrics["sharp_ratio"]
                checkpoint["best_balanced_score"] = best_metrics["balanced_score"]
                checkpoint["best_paired_score"] = best_metrics["paired_score"]
                checkpoint["best_detail_paired_score"] = best_metrics["detail_paired_score"]
                torch.save(checkpoint, best_path)
                accelerator.print(
                    f"New best Stage 1 checkpoint: PSNR={best_metrics['psnr']:.4f}, "
                    f"SSIM={best_metrics['ssim']:.4f}, MSE={best_metrics['mse']:.6f}, "
                    f"L1={best_metrics['l1']:.6f}, Detail_Paired={best_metrics['detail_paired_score']:.4f} -> {best_path}"
                )

        accelerator.print(
            f"[Stage 1][Epoch {epoch + 1}] "
            f"PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} MSE={metrics['mse']:.6f} "
            f"L1={metrics['l1']:.6f} Grad={metrics['grad']:.6f} HF={metrics['hf']:.6f} "
            f"LPIPS={metrics['lpips']:.6f} FID={metrics['fid']:.4f} SharpRatio={metrics['sharp_ratio']:.4f} "
            f"Score={metrics['balanced_score']:.4f} Paired={metrics['paired_score']:.4f} "
            f"Detail_Paired={metrics['detail_paired_score']:.4f} "
            f"AnomalyEpochs={anomaly_epochs} SkippedNonFinite={skipped_nonfinite}"
        )

        if args.strict_image_check and anomaly_epochs >= args.image_check_patience:
            if args.rollback_on_anomaly and os.path.exists(healthy_latest_path):
                healthy_checkpoint = torch.load(healthy_latest_path, map_location="cpu")
                accelerator.unwrap_model(model).load_state_dict(healthy_checkpoint["model"])
                optimizer.load_state_dict(healthy_checkpoint["optimizer"])
                accelerator.print(f"Rolled back to healthy checkpoint: {healthy_latest_path}")
            raise RuntimeError(
                "Stage 1 image check failed repeatedly. "
                f"Recent reasons: {', '.join(anomaly_reasons) if anomaly_reasons else 'unknown'}"
            )

        if no_improve_epochs >= args.early_stop_patience:
            accelerator.print(
                f"Early stopping triggered after {no_improve_epochs} epochs without improving {args.best_metric}."
            )
            break


if __name__ == "__main__":
    main()
