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

from pmrf_t1fa.checkpointing import select_resume_checkpoint, validate_resume_args
from pmrf_t1fa.models.pmrf_t1fa import (
    DetailRefinementFlowUNet,
    DetailStage1Net,
    GradientLoss,
    RefinementFlowUNet,
    SSIMLoss,
    Stage1Net,
    build_stage2_condition,
    build_xt,
    center_channel,
    compose_stage2_velocity,
    euler_refine,
    euler_refine_train,
    infer_stage1_detail_scale,
    infer_stage1_in_channels,
    infer_stage1_model_variant,
    infer_stage2_condition_mode,
    infer_stage1_prediction_mode,
    prepare_stage1_input,
    predict_stage1_fa,
    project_endpoint,
    reduce_rgb_to_single_channel,
    split_stage2_output,
    stage2_condition_channels,
)
from src.data.t1fa_stack_dataset import T1FAStackDataset
from src.datasets import T1FADataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_step_list(value: str) -> List[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not steps:
        raise ValueError("At least one rollout step count is required.")
    if any(step <= 0 for step in steps):
        raise ValueError(f"Rollout step counts must be positive, got {value!r}")
    return steps


def apply_stage2_training_preset(args):
    if args.stage2_training_preset != "detail_teacher":
        return args

    args.stage2_model_variant = "detail"
    args.condition_mode = "coarse_t1_edge"
    args.source_noise_std = max(float(args.source_noise_std), 0.05)
    if args.t_sampling == "endpoint":
        args.t_sampling = "uniform"
    if args.rollout_train_steps == [4, 8, 10]:
        args.rollout_train_steps = [4, 8, 10, 25]
    args.eval_steps = max(int(args.eval_steps), 10)
    args.velocity_weight = min(float(args.velocity_weight), 0.12)
    args.image_mse_weight = min(float(args.image_mse_weight), 0.35)
    args.l1_weight = min(float(args.l1_weight), 0.70)
    args.ssim_weight = min(float(args.ssim_weight), 0.45)
    args.grad_weight = max(float(args.grad_weight), 0.12)
    args.hf_weight = max(float(args.hf_weight), 0.12)
    args.residual_hf_weight = max(float(args.residual_hf_weight), 0.30)
    args.detail_weight = max(float(args.detail_weight), 0.25)
    args.detail_velocity_weight = max(float(args.detail_velocity_weight), 0.18)
    args.rollout_detail_weight = max(float(args.rollout_detail_weight), 0.35)
    args.rollout_hf_weight = max(float(args.rollout_hf_weight), 0.35)
    args.rollout_residual_hf_weight = max(float(args.rollout_residual_hf_weight), 0.45)
    args.rollout_wm_l1_weight = max(float(args.rollout_wm_l1_weight), 0.12)
    args.rollout_wm_grad_weight = max(float(args.rollout_wm_grad_weight), 0.08)
    args.wm_l1_weight = max(float(args.wm_l1_weight), 0.16)
    args.wm_grad_weight = max(float(args.wm_grad_weight), 0.08)
    args.roi_consistency_weight = max(float(args.roi_consistency_weight), 0.04)
    args.best_metric = "detail_paired"
    args.degrade_check_mode = "teacher"
    args.rollback_on_degrade = False
    args.dynamic_condition_rollout = True
    args.detail_refine_ratio_weight = max(float(args.detail_refine_ratio_weight), 4.0)
    args.detail_under_refine_penalty_weight = max(float(args.detail_under_refine_penalty_weight), 4.0)
    return args


class ZeroLPIPSLoss(torch.nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.new_zeros((pred.shape[0],))


def parse_args():
    parser = argparse.ArgumentParser(description="Train PMRF-T1FA Stage 2 refinement flow")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1/checkpoints/best_stage1.pt")
    parser.add_argument("--stage1_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run_name", default="pmrf_t1fa_stage2_b")
    parser.add_argument("--stage2_model_variant", default="single", choices=["single", "detail"])
    parser.add_argument("--stage2_training_preset", default="none", choices=["none", "detail_teacher"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0, help="Use only the first N training slices for smoke tests.")
    parser.add_argument("--val_limit", type=int, default=0, help="Use only the first N validation slices for smoke tests.")
    parser.add_argument("--preview_batch_size", type=int, default=1)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--source_noise_std", type=float, default=0.01)
    parser.add_argument("--detail_boost", type=float, default=0.90)
    parser.add_argument("--t_sampling", default="endpoint", choices=["endpoint", "uniform", "low_t"])
    parser.add_argument("--t_min", type=float, default=0.0)
    parser.add_argument("--t_max", type=float, default=0.25)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--condition_on_coarse", action="store_true")
    parser.add_argument("--disable_condition_on_coarse", action="store_false", dest="condition_on_coarse")
    parser.add_argument(
        "--condition_mode",
        default="coarse_t1",
        choices=["none", "coarse", "t1", "coarse_t1", "coarse_t1_edge"],
        help="Stage 2 conditioning. coarse_t1_edge adds T1/Stage1 edge and residual guidance for sharper multi-step refinement.",
    )
    parser.add_argument(
        "--rollout_train_steps",
        default="4,8,10",
        help="Comma-separated Stage 2 rollout lengths sampled during training, e.g. 4,8,10 or 10,25.",
    )
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--preview_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", default=None, help="Explicit Stage 2 checkpoint path to resume from.")
    parser.add_argument("--no_auto_resume", action="store_true")
    parser.add_argument("--lpips_warmup_epochs", type=int, default=20)
    parser.add_argument("--lpips_max_weight", type=float, default=0.02)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--velocity_weight", type=float, default=0.20)
    parser.add_argument("--image_mse_weight", type=float, default=0.65)
    parser.add_argument("--l1_weight", type=float, default=1.10)
    parser.add_argument("--ssim_weight", type=float, default=0.80)
    parser.add_argument("--grad_weight", type=float, default=0.06)
    parser.add_argument("--detail_weight", type=float, default=0.07)
    parser.add_argument("--hf_weight", type=float, default=0.04)
    parser.add_argument("--residual_hf_weight", type=float, default=0.12)
    parser.add_argument("--detail_velocity_weight", type=float, default=0.10)
    parser.add_argument("--rollout_detail_weight", type=float, default=0.0)
    parser.add_argument("--rollout_l1_weight", type=float, default=0.35)
    parser.add_argument("--rollout_ssim_weight", type=float, default=0.35)
    parser.add_argument("--rollout_hf_weight", type=float, default=0.18)
    parser.add_argument("--rollout_residual_hf_weight", type=float, default=0.18)
    parser.add_argument("--rollout_wm_l1_weight", type=float, default=0.08)
    parser.add_argument("--rollout_wm_grad_weight", type=float, default=0.04)
    parser.add_argument("--brain_l1_weight", type=float, default=0.08)
    parser.add_argument("--wm_l1_weight", type=float, default=0.12)
    parser.add_argument("--wm_grad_weight", type=float, default=0.04)
    parser.add_argument("--roi_consistency_weight", type=float, default=0.02)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--roi_rows", type=int, default=4)
    parser.add_argument("--roi_cols", type=int, default=4)
    parser.add_argument("--roi_min_pixels", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--best_metric", default="detail_paired", choices=["psnr", "mse", "mae", "fid", "sharpness", "balanced", "paired", "wm_paired", "detail_paired"])
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--degrade_patience", type=int, default=4)
    parser.add_argument("--degrade_margin_psnr", type=float, default=0.05)
    parser.add_argument("--degrade_margin_ssim", type=float, default=0.001)
    parser.add_argument("--degrade_check_mode", default="strict", choices=["strict", "teacher", "off"])
    parser.add_argument("--dynamic_condition_rollout", action="store_true")
    parser.add_argument("--disable_dynamic_condition_rollout", action="store_false", dest="dynamic_condition_rollout")
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
    parser.add_argument("--paired_brain_mae_weight", type=float, default=4.0)
    parser.add_argument("--paired_wm_mae_weight", type=float, default=8.0)
    parser.add_argument("--paired_grad_weight", type=float, default=0.8)
    parser.add_argument("--paired_roi_weight", type=float, default=4.0)
    parser.add_argument("--paired_sharp_weight", type=float, default=2.0)
    parser.add_argument("--detail_sharp_weight", type=float, default=8.0)
    parser.add_argument("--detail_coarse_penalty_weight", type=float, default=8.0)
    parser.add_argument("--detail_target_sharp_ratio", type=float, default=0.90)
    parser.add_argument("--detail_max_sharp_ratio", type=float, default=1.20)
    parser.add_argument("--detail_oversharp_penalty_weight", type=float, default=12.0)
    parser.add_argument("--detail_refine_ratio_weight", type=float, default=0.0)
    parser.add_argument("--detail_refine_target_ratio", type=float, default=0.50)
    parser.add_argument("--detail_under_refine_penalty_weight", type=float, default=0.0)
    parser.add_argument("--skip_nonfinite_batches", action="store_true")
    parser.add_argument("--rollback_on_degrade", action="store_true")
    parser.add_argument("--disable_rollback_on_degrade", action="store_false", dest="rollback_on_degrade")
    parser.set_defaults(
        condition_on_coarse=True,
        skip_nonfinite_batches=True,
        rollback_on_degrade=True,
        dynamic_condition_rollout=False,
    )
    args = parser.parse_args()
    args.rollout_train_steps = parse_step_list(args.rollout_train_steps)
    args = apply_stage2_training_preset(args)
    if not args.condition_on_coarse and args.condition_mode in {"coarse", "coarse_t1", "coarse_t1_edge"}:
        args.condition_mode = "none"
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
        - args.paired_mse_weight * metrics["mse_proxy"]
        - args.paired_mae_weight * metrics["l1"]
    )


def compute_bounded_sharp_bonus(sharp_ratio: float, weight: float, args) -> float:
    target_ratio = max(float(getattr(args, "detail_target_sharp_ratio", 0.90)), 1e-6)
    max_ratio = float(getattr(args, "detail_max_sharp_ratio", 1.20))
    bounded_bonus = min(float(sharp_ratio), target_ratio) / target_ratio
    oversharp_penalty = max(float(sharp_ratio) - max_ratio, 0.0)
    return (
        weight * bounded_bonus
        - getattr(args, "detail_oversharp_penalty_weight", 12.0) * oversharp_penalty
    )


def compute_wm_paired_score(metrics: Dict[str, float], args) -> float:
    return (
        compute_paired_score(metrics, args)
        - args.paired_brain_mae_weight * metrics["brain_l1"]
        - args.paired_wm_mae_weight * metrics["wm_l1"]
        - args.paired_grad_weight * metrics["grad"]
        - args.paired_roi_weight * metrics["roi"]
        + compute_bounded_sharp_bonus(metrics["sharp_ratio"], args.paired_sharp_weight, args)
    )


def compute_detail_paired_score(metrics: Dict[str, float], args) -> float:
    coarse_sharp_ratio = metrics.get("coarse_sharp_ratio", 0.0)
    target_ratio = max(float(getattr(args, "detail_target_sharp_ratio", 0.90)), 1e-6)
    sharp_gain = min(metrics["sharp_ratio"], target_ratio) - min(coarse_sharp_ratio, target_ratio)
    smooth_penalty = max(coarse_sharp_ratio - metrics["sharp_ratio"], 0.0)
    oversharp_penalty = max(metrics["sharp_ratio"] - getattr(args, "detail_max_sharp_ratio", 1.20), 0.0)
    refine_ratio = float(metrics.get("refine_ratio", 0.0))
    refine_target = max(float(getattr(args, "detail_refine_target_ratio", 0.50)), 1e-6)
    refine_bonus = float(getattr(args, "detail_refine_ratio_weight", 0.0)) * min(refine_ratio, refine_target) / refine_target
    under_refine_penalty = float(getattr(args, "detail_under_refine_penalty_weight", 0.0)) * max(
        refine_target - refine_ratio,
        0.0,
    )
    return (
        compute_wm_paired_score(metrics, args)
        + args.detail_sharp_weight * sharp_gain
        + refine_bonus
        - args.detail_coarse_penalty_weight * smooth_penalty
        - under_refine_penalty
        - getattr(args, "detail_oversharp_penalty_weight", 12.0) * oversharp_penalty
    )


def sample_stage2_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    if mode == "endpoint":
        return torch.zeros((batch_size,), device=device, dtype=dtype)
    if mode == "uniform":
        return torch.rand((batch_size,), device=device, dtype=dtype)
    if mode == "low_t":
        low = max(0.0, min(float(t_min), 1.0))
        high = max(low, min(float(t_max), 1.0))
        return low + (high - low) * torch.rand((batch_size,), device=device, dtype=dtype)
    raise ValueError(f"Unsupported t_sampling mode: {mode}")


def summarize_prediction_health(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
    target_01 = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
    pred_flat = pred_01.flatten()
    target_flat = target_01.flatten()
    q01 = torch.quantile(pred_flat, 0.01).item()
    q99 = torch.quantile(pred_flat, 0.99).item()
    return {
        "pred_mean": pred_flat.mean().item(),
        "target_mean": target_flat.mean().item(),
        "pred_std": pred_flat.std(unbiased=False).item(),
        "target_std": target_flat.std(unbiased=False).item(),
        "pred_dynamic_range": q99 - q01,
    }


def detect_stage2_anomaly(
    refined: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    psnr: float,
    ssim: float,
    coarse_psnr: float,
    coarse_ssim: float,
    margin_psnr: float,
    margin_ssim: float,
    mode: str,
) -> Tuple[bool, List[str], Dict[str, float]]:
    stats = summarize_prediction_health(refined, target)
    if mode == "off":
        return False, [], stats
    reasons: List[str] = []
    if mode == "strict":
        if psnr + margin_psnr < coarse_psnr:
            reasons.append(f"psnr_below_coarse={psnr:.4f}<{coarse_psnr:.4f}")
        if ssim + margin_ssim < coarse_ssim:
            reasons.append(f"ssim_below_coarse={ssim:.4f}<{coarse_ssim:.4f}")
        if stats["pred_std"] < 0.75 * max(stats["target_std"], 1e-6):
            reasons.append(f"low_std={stats['pred_std']:.4f}")
    elif mode == "teacher":
        if stats["pred_std"] < 0.35 * max(stats["target_std"], 1e-6):
            reasons.append(f"collapsed_std={stats['pred_std']:.4f}")
        if stats["pred_dynamic_range"] < 0.05:
            reasons.append(f"collapsed_dynamic_range={stats['pred_dynamic_range']:.4f}")
    else:
        raise ValueError(f"Unsupported degrade_check_mode: {mode}")
    residual_energy = F.l1_loss(refined, coarse).item()
    if mode == "strict" and residual_energy < 0.002:
        reasons.append(f"too_static={residual_energy:.5f}")
    return len(reasons) > 0, reasons, stats


def is_better(metrics: Dict[str, float], best_metrics: Dict[str, float], best_metric: str) -> bool:
    if best_metric == "mse":
        return metrics["mse_proxy"] < best_metrics["mse_proxy"]
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
    if best_metric == "wm_paired":
        return metrics["wm_paired_score"] > best_metrics["wm_paired_score"]
    if best_metric == "detail_paired":
        return metrics["detail_paired_score"] > best_metrics["detail_paired_score"]
    if metrics["psnr"] > best_metrics["psnr"] + 1e-6:
        return True
    if abs(metrics["psnr"] - best_metrics["psnr"]) <= 1e-6 and metrics["ssim"] > best_metrics["ssim"]:
        return True
    return False


def make_slice_dataset(t1_dir: str, fa_dir: str, stage1_channels: int):
    if stage1_channels > 1:
        return T1FAStackDataset(t1_dir, fa_dir, context_slices=stage1_channels, target_size=(224, 224))
    return T1FADataset(t1_dir, fa_dir, preload_ram=False)


def maybe_limit_dataset(dataset, limit: int):
    if limit <= 0:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_gradient_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    grad_loss_fn: GradientLoss,
) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    pred_x, pred_y = grad_loss_fn._gradients(pred)
    target_x, target_y = grad_loss_fn._gradients(target)
    denom = mask.sum().clamp_min(1.0)
    return ((pred_x - target_x).abs() * mask).sum() / denom + ((pred_y - target_y).abs() * mask).sum() / denom


def close_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
    return (eroded > 0.5).float()


def build_training_masks(
    t1_img: torch.Tensor,
    target: torch.Tensor,
    brain_t1_threshold: float,
    brain_fa_threshold: float,
    wm_quantile: float,
    wm_min_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t1_01 = torch.clamp((center_channel(t1_img).float() + 1.0) / 2.0, 0.0, 1.0)
    target_01 = torch.clamp((target.float() + 1.0) / 2.0, 0.0, 1.0)
    brain = close_mask(((t1_01 > brain_t1_threshold) | (target_01 > brain_fa_threshold)).float())
    wm_masks: List[torch.Tensor] = []
    for i in range(target_01.shape[0]):
        brain_i = brain[i : i + 1]
        values = target_01[i : i + 1][brain_i.bool()]
        if values.numel() == 0:
            wm_masks.append(torch.zeros_like(brain_i))
            continue
        threshold = max(float(torch.quantile(values.detach().float().cpu(), wm_quantile).item()), wm_min_threshold)
        wm_masks.append(close_mask((brain_i.bool() & (target_01[i : i + 1] >= threshold)).float()))
    return brain, torch.cat(wm_masks, dim=0)


def roi_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wm_mask: torch.Tensor,
    rows: int,
    cols: int,
    min_pixels: int,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    _, _, height, width = pred.shape
    for y in range(rows):
        y0 = y * height // rows
        y1 = (y + 1) * height // rows
        for x in range(cols):
            x0 = x * width // cols
            x1 = (x + 1) * width // cols
            region_mask = wm_mask[:, :, y0:y1, x0:x1].to(dtype=pred.dtype, device=pred.device)
            denom = region_mask.flatten(1).sum(dim=1)
            valid = denom >= float(min_pixels)
            if not bool(valid.any().item()):
                continue
            pred_mean = (pred[:, :, y0:y1, x0:x1] * region_mask).flatten(1).sum(dim=1) / denom.clamp_min(1.0)
            target_mean = (target[:, :, y0:y1, x0:x1] * region_mask).flatten(1).sum(dim=1) / denom.clamp_min(1.0)
            losses.append(torch.abs(pred_mean[valid] - target_mean[valid]).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def load_stage1_model(stage1_ckpt: str, device: torch.device) -> Tuple[torch.nn.Module, str, int, float]:
    checkpoint = torch.load(stage1_ckpt, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    stage1_channels = infer_stage1_in_channels(checkpoint)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if infer_stage1_model_variant(ckpt_args) == "detail":
        stage1 = DetailStage1Net(in_channels=stage1_channels, out_channels=1)
    else:
        stage1 = Stage1Net(in_channels=stage1_channels, out_channels=1)
    stage1.load_state_dict(state_dict)
    stage1.to(device)
    stage1.eval()
    for param in stage1.parameters():
        param.requires_grad = False
    prediction_mode = infer_stage1_prediction_mode(ckpt_args)
    return stage1, prediction_mode, stage1_channels, infer_stage1_detail_scale(ckpt_args)


def predict_stage1_batch(
    stage1: Stage1Net,
    t1_img: torch.Tensor,
    stage1_device: torch.device,
    prediction_mode: str,
    detail_scale: float,
) -> torch.Tensor:
    stage1_input = t1_img.to(stage1_device, non_blocking=True)
    coarse = predict_stage1_fa(
        stage1,
        stage1_input,
        clamp=True,
        prediction_mode=prediction_mode,
        detail_scale=detail_scale,
    )
    return coarse.to(t1_img.device, non_blocking=True)


def build_stage2_loss(
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    v_detail: torch.Tensor | None,
    v_target: torch.Tensor,
    coarse: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    rollout_pred: torch.Tensor,
    ssim_loss_fn: SSIMLoss,
    grad_loss_fn: GradientLoss,
    lpips_loss_fn,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    epoch: int,
    lpips_warmup_epochs: int,
    lpips_max_weight: float,
    velocity_weight: float,
    image_mse_weight: float,
    l1_weight: float,
    ssim_weight: float,
    grad_weight: float,
    detail_weight: float,
    hf_weight: float,
    brain_l1_weight: float,
    wm_l1_weight: float,
    wm_grad_weight: float,
    residual_hf_weight: float,
    detail_velocity_weight: float,
    rollout_detail_weight: float,
    rollout_l1_weight: float,
    rollout_ssim_weight: float,
    rollout_hf_weight: float,
    rollout_residual_hf_weight: float,
    rollout_wm_l1_weight: float,
    rollout_wm_grad_weight: float,
    roi_consistency_weight: float,
    roi_rows: int,
    roi_cols: int,
    roi_min_pixels: int,
) -> Dict[str, torch.Tensor]:
    x_hat = clamp_to_image_range(project_endpoint(x_t, v_pred, t).float())
    v_pred_fp32 = v_pred.float()
    v_target_fp32 = v_target.float()
    coarse_fp32 = coarse.float()
    target_fp32 = target.float()
    loss_vel = F.mse_loss(v_pred_fp32, v_target_fp32)
    loss_img_mse = F.mse_loss(x_hat, target_fp32)
    loss_l1 = F.l1_loss(x_hat, target_fp32)
    loss_ssim = ssim_loss_fn(x_hat, target_fp32)
    loss_grad = grad_loss_fn(x_hat, target_fp32)
    detail_pred = x_hat - coarse_fp32
    detail_target = target_fp32 - coarse_fp32
    loss_detail = F.l1_loss(detail_pred, detail_target)
    loss_hf = F.l1_loss(laplacian_filter(x_hat), laplacian_filter(target_fp32))
    loss_residual_hf = F.l1_loss(laplacian_filter(detail_pred), laplacian_filter(detail_target))
    if v_detail is None:
        loss_detail_velocity = x_t.new_tensor(0.0)
    else:
        loss_detail_velocity = F.l1_loss(v_detail.float(), laplacian_filter(target_fp32 - x_t.float()))
    loss_brain_l1 = masked_l1_loss(x_hat, target_fp32, brain_mask)
    loss_wm_l1 = masked_l1_loss(x_hat, target_fp32, wm_mask)
    loss_wm_grad = masked_gradient_l1_loss(x_hat, target_fp32, wm_mask, grad_loss_fn)
    rollout_fp32 = clamp_to_image_range(rollout_pred.float())
    rollout_detail_pred = rollout_fp32 - coarse_fp32
    loss_rollout_l1 = F.l1_loss(rollout_fp32, target_fp32)
    loss_rollout_detail = F.l1_loss(rollout_detail_pred, detail_target)
    loss_rollout_ssim = ssim_loss_fn(rollout_fp32, target_fp32)
    loss_rollout_hf = F.l1_loss(laplacian_filter(rollout_fp32), laplacian_filter(target_fp32))
    loss_rollout_residual_hf = F.l1_loss(laplacian_filter(rollout_detail_pred), laplacian_filter(detail_target))
    loss_rollout_wm_l1 = masked_l1_loss(rollout_fp32, target_fp32, wm_mask)
    loss_rollout_wm_grad = masked_gradient_l1_loss(rollout_fp32, target_fp32, wm_mask, grad_loss_fn)
    loss_roi = roi_consistency_loss(
        x_hat,
        target_fp32,
        wm_mask,
        rows=roi_rows,
        cols=roi_cols,
        min_pixels=roi_min_pixels,
    )
    with torch.autocast(device_type=x_t.device.type, enabled=False):
        loss_lpips = lpips_loss_fn(x_hat.repeat(1, 3, 1, 1), target_fp32.repeat(1, 3, 1, 1)).mean()

    warmup_ratio = min(float(epoch + 1) / max(lpips_warmup_epochs, 1), 1.0)
    w_lpips = lpips_max_weight * warmup_ratio
    total = (
        velocity_weight * loss_vel
        + image_mse_weight * loss_img_mse
        + l1_weight * loss_l1
        + ssim_weight * loss_ssim
        + grad_weight * loss_grad
        + detail_weight * loss_detail
        + hf_weight * loss_hf
        + brain_l1_weight * loss_brain_l1
        + wm_l1_weight * loss_wm_l1
        + wm_grad_weight * loss_wm_grad
        + residual_hf_weight * loss_residual_hf
        + detail_velocity_weight * loss_detail_velocity
        + rollout_detail_weight * loss_rollout_detail
        + rollout_l1_weight * loss_rollout_l1
        + rollout_ssim_weight * loss_rollout_ssim
        + rollout_hf_weight * loss_rollout_hf
        + rollout_residual_hf_weight * loss_rollout_residual_hf
        + rollout_wm_l1_weight * loss_rollout_wm_l1
        + rollout_wm_grad_weight * loss_rollout_wm_grad
        + roi_consistency_weight * loss_roi
        + w_lpips * loss_lpips
    )
    return {
        "total": total,
        "x_hat": x_hat,
        "vel": loss_vel,
        "img_mse": loss_img_mse,
        "l1": loss_l1,
        "ssim": loss_ssim,
        "grad": loss_grad,
        "lpips": loss_lpips,
        "detail": loss_detail,
        "hf": loss_hf,
        "residual_hf": loss_residual_hf,
        "detail_velocity": loss_detail_velocity,
        "rollout_detail": loss_rollout_detail,
        "rollout_l1": loss_rollout_l1,
        "rollout_ssim": loss_rollout_ssim,
        "rollout_hf": loss_rollout_hf,
        "rollout_residual_hf": loss_rollout_residual_hf,
        "rollout_wm_l1": loss_rollout_wm_l1,
        "rollout_wm_grad": loss_rollout_wm_grad,
        "brain_l1": loss_brain_l1,
        "wm_l1": loss_wm_l1,
        "wm_grad": loss_wm_grad,
        "roi": loss_roi,
        "w_lpips": torch.tensor(w_lpips, device=x_t.device, dtype=x_t.dtype),
    }


def save_preview(preview_dir, step, t1_img, coarse, refined, fa_img):
    t1_vis = torch.clamp((center_channel(t1_img) + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    refined_vis = torch.clamp((refined + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    fa_vis = torch.clamp((fa_img + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
    panel = torch.cat([t1_vis, refined_vis, fa_vis], dim=-1)
    save_image(panel, os.path.join(preview_dir, f"step_{step}.png"), nrow=1)


def main():
    args = parse_args()
    root_dir, ckpt_dir, preview_dir = ensure_dirs(args.run_name)
    log_path = os.path.join(root_dir, "train.log")

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.stage1_device == "cpu":
        stage1_device = torch.device("cpu")
    elif args.stage1_device == "cuda":
        stage1_device = torch.device("cuda")
    else:
        stage1_device = accelerator.device
    stage1, stage1_prediction_mode, stage1_channels, stage1_detail_scale = load_stage1_model(args.stage1_ckpt, stage1_device)
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

    args.condition_mode = infer_stage2_condition_mode(
        {"condition_mode": args.condition_mode, "condition_on_coarse": args.condition_on_coarse}
    )
    args.condition_on_coarse = args.condition_mode in {"coarse", "coarse_t1", "coarse_t1_edge"}
    condition_channels = stage2_condition_channels(args.condition_mode, stage1_channels)
    if args.stage2_model_variant == "detail":
        flow_model = DetailRefinementFlowUNet(input_channels=1, condition_channels=condition_channels)
    else:
        flow_model = RefinementFlowUNet(input_channels=1, condition_channels=condition_channels)
    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    flow_model, optimizer, train_loader, val_loader, ssim_loss_fn, grad_loss_fn, lpips_loss_fn, fid_metric, kid_metric = accelerator.prepare(
        flow_model, optimizer, train_loader, val_loader, ssim_loss_fn, grad_loss_fn, lpips_loss_fn, fid_metric, kid_metric
    )

    fixed_batch = next(iter(val_loader))
    preview_batch_size = max(1, int(args.preview_batch_size))
    fixed_t1 = prepare_stage1_input(fixed_batch["t1_slice"][:preview_batch_size].to(accelerator.device), stage1_channels)
    fixed_fa = reduce_rgb_to_single_channel(fixed_batch["fa_slice"][:preview_batch_size].to(accelerator.device))
    with torch.no_grad(), accelerator.autocast():
        fixed_coarse = predict_stage1_batch(stage1, fixed_t1, stage1_device, stage1_prediction_mode, stage1_detail_scale)

    best_metrics = {
        "psnr": float("-inf"),
        "ssim": float("-inf"),
        "mse_proxy": float("inf"),
        "l1": float("inf"),
        "lpips": float("inf"),
        "fid": float("inf"),
        "kid": float("inf"),
        "sharpness": float("-inf"),
        "sharp_ratio": float("-inf"),
        "balanced_score": float("-inf"),
        "paired_score": float("-inf"),
        "wm_paired_score": float("-inf"),
        "detail_paired_score": float("-inf"),
        "refine_l1": float("-inf"),
        "coarse_target_l1": float("inf"),
        "refine_ratio": float("-inf"),
    }
    start_epoch = 0
    global_step = 0
    no_improve_epochs = 0
    degrade_epochs = 0
    latest_path = os.path.join(ckpt_dir, "latest_stage2.pt")
    healthy_latest_path = os.path.join(ckpt_dir, "healthy_latest_stage2.pt")
    best_path = os.path.join(ckpt_dir, "best_stage2.pt")

    auto_resume = ((not args.no_auto_resume) or args.resume) and args.resume_from is None
    resume_path = select_resume_checkpoint(
        ckpt_dir,
        "stage2",
        explicit_path=args.resume_from,
        auto_resume=auto_resume,
        prefer_healthy=True,
    )

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        validate_resume_args(
            checkpoint,
            args,
            required_keys=[
                "stage1_ckpt",
                "stage2_model_variant",
                "condition_mode",
                "dynamic_condition_rollout",
                "detail_weight",
                "rollout_detail_weight",
                "rollout_residual_hf_weight",
                "detail_refine_ratio_weight",
                "detail_under_refine_penalty_weight",
            ],
            checkpoint_path=resume_path,
        )
        accelerator.unwrap_model(flow_model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_metrics["psnr"] = checkpoint.get("best_psnr", best_metrics["psnr"])
        best_metrics["ssim"] = checkpoint.get("best_ssim", best_metrics["ssim"])
        best_metrics["mse_proxy"] = checkpoint.get("best_mse_proxy", best_metrics["mse_proxy"])
        best_metrics["l1"] = checkpoint.get("best_l1", best_metrics["l1"])
        best_metrics["lpips"] = checkpoint.get("best_lpips", best_metrics["lpips"])
        best_metrics["fid"] = checkpoint.get("best_fid", best_metrics["fid"])
        best_metrics["kid"] = checkpoint.get("best_kid", best_metrics["kid"])
        best_metrics["sharpness"] = checkpoint.get("best_sharpness", best_metrics["sharpness"])
        best_metrics["sharp_ratio"] = checkpoint.get("best_sharp_ratio", best_metrics["sharp_ratio"])
        best_metrics["balanced_score"] = checkpoint.get("best_balanced_score", best_metrics["balanced_score"])
        best_metrics["paired_score"] = checkpoint.get("best_paired_score", best_metrics["paired_score"])
        best_metrics["wm_paired_score"] = checkpoint.get("best_wm_paired_score", best_metrics["wm_paired_score"])
        best_metrics["detail_paired_score"] = checkpoint.get("best_detail_paired_score", best_metrics["detail_paired_score"])
        best_metrics["refine_l1"] = checkpoint.get("best_refine_l1", best_metrics["refine_l1"])
        best_metrics["coarse_target_l1"] = checkpoint.get("best_coarse_target_l1", best_metrics["coarse_target_l1"])
        best_metrics["refine_ratio"] = checkpoint.get("best_refine_ratio", best_metrics["refine_ratio"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        no_improve_epochs = checkpoint.get("no_improve_epochs", 0)
        degrade_epochs = checkpoint.get("degrade_epochs", 0)
        accelerator.print(f"Resume Stage 2 from {resume_path}")

    accelerator.print(
        f"Stage 2 training on {len(train_dataset)} train slices / {len(val_dataset)} val slices | "
        f"condition_mode={args.condition_mode} condition_channels={condition_channels} | "
        f"condition_on_coarse={args.condition_on_coarse} | Device={accelerator.device} | "
        f"mixed_precision={args.mixed_precision} | stage1_channels={stage1_channels} | "
        f"source_noise_std={args.source_noise_std} | rollout_train_steps={args.rollout_train_steps} | "
        f"eval_steps={args.eval_steps} | stage1_device={stage1_device} | "
        f"stage2_model_variant={args.stage2_model_variant} | detail_boost={args.detail_boost} | "
        f"dynamic_condition_rollout={args.dynamic_condition_rollout} | "
        f"stage1_detail_scale={stage1_detail_scale}"
    )

    for epoch in range(start_epoch, args.epochs):
        flow_model.train()
        running = {
            "total": 0.0,
            "vel": 0.0,
            "img_mse": 0.0,
            "l1": 0.0,
            "ssim": 0.0,
            "grad": 0.0,
            "lpips": 0.0,
            "detail": 0.0,
            "hf": 0.0,
            "residual_hf": 0.0,
            "detail_velocity": 0.0,
            "rollout_detail": 0.0,
            "rollout_l1": 0.0,
            "rollout_ssim": 0.0,
            "rollout_hf": 0.0,
            "rollout_residual_hf": 0.0,
            "rollout_wm_l1": 0.0,
            "rollout_wm_grad": 0.0,
            "brain_l1": 0.0,
            "wm_l1": 0.0,
            "wm_grad": 0.0,
            "roi": 0.0,
        }
        skipped_nonfinite = 0

        progress = tqdm(
            train_loader,
            desc=f"Stage2 Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )
        for batch in progress:
            t1_img = prepare_stage1_input(batch["t1_slice"].to(accelerator.device), stage1_channels)
            fa_img = reduce_rgb_to_single_channel(batch["fa_slice"].to(accelerator.device))

            with torch.no_grad(), accelerator.autocast():
                coarse = predict_stage1_batch(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale)

            with accelerator.autocast():
                t = sample_stage2_time(
                    batch_size=t1_img.shape[0],
                    device=accelerator.device,
                    dtype=t1_img.dtype,
                    mode=args.t_sampling,
                    t_min=args.t_min,
                    t_max=args.t_max,
                )
                x_t, _, v_target = build_xt(
                    target=fa_img,
                    coarse=coarse,
                    t=t,
                    source_noise_std=args.source_noise_std,
                    deterministic_source=args.t_sampling == "endpoint",
                )
                condition = build_stage2_condition(coarse, t1_img, args.condition_mode)
                raw_output = flow_model(x_t, t, condition=condition)
                v_pred = compose_stage2_velocity(raw_output, t, detail_boost=args.detail_boost)
                _, v_detail = split_stage2_output(raw_output)
                rollout_index = int(
                    torch.randint(
                        low=0,
                        high=len(args.rollout_train_steps),
                        size=(1,),
                        device=accelerator.device,
                    ).item()
                )
                rollout_steps = int(args.rollout_train_steps[rollout_index])
                rollout_pred = euler_refine_train(
                    flow_model,
                    coarse,
                    num_steps=rollout_steps,
                    condition=condition,
                    clamp=True,
                    detail_boost=args.detail_boost,
                    dynamic_condition=args.dynamic_condition_rollout,
                    condition_mode=args.condition_mode,
                    t1_img=t1_img,
                )
                brain_mask, wm_mask = build_training_masks(
                    t1_img,
                    fa_img,
                    brain_t1_threshold=args.brain_t1_threshold,
                    brain_fa_threshold=args.brain_fa_threshold,
                    wm_quantile=args.wm_quantile,
                    wm_min_threshold=args.wm_min_threshold,
                )

                losses = build_stage2_loss(
                    x_t=x_t,
                    v_pred=v_pred,
                    v_detail=v_detail,
                    v_target=v_target,
                    coarse=coarse,
                    target=fa_img,
                    t=t,
                    rollout_pred=rollout_pred,
                    ssim_loss_fn=ssim_loss_fn,
                    grad_loss_fn=grad_loss_fn,
                    lpips_loss_fn=lpips_loss_fn,
                    brain_mask=brain_mask,
                    wm_mask=wm_mask,
                    epoch=epoch,
                    lpips_warmup_epochs=args.lpips_warmup_epochs,
                    lpips_max_weight=args.lpips_max_weight,
                    velocity_weight=args.velocity_weight,
                    image_mse_weight=args.image_mse_weight,
                    l1_weight=args.l1_weight,
                    ssim_weight=args.ssim_weight,
                    grad_weight=args.grad_weight,
                    detail_weight=args.detail_weight,
                    hf_weight=args.hf_weight,
                    brain_l1_weight=args.brain_l1_weight,
                    wm_l1_weight=args.wm_l1_weight,
                    wm_grad_weight=args.wm_grad_weight,
                    residual_hf_weight=args.residual_hf_weight,
                    detail_velocity_weight=args.detail_velocity_weight,
                    rollout_detail_weight=args.rollout_detail_weight,
                    rollout_l1_weight=args.rollout_l1_weight,
                    rollout_ssim_weight=args.rollout_ssim_weight,
                    rollout_hf_weight=args.rollout_hf_weight,
                    rollout_residual_hf_weight=args.rollout_residual_hf_weight,
                    rollout_wm_l1_weight=args.rollout_wm_l1_weight,
                    rollout_wm_grad_weight=args.rollout_wm_grad_weight,
                    roi_consistency_weight=args.roi_consistency_weight,
                    roi_rows=args.roi_rows,
                    roi_cols=args.roi_cols,
                    roi_min_pixels=args.roi_min_pixels,
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
                raise RuntimeError(f"Non-finite Stage 2 loss at epoch {epoch + 1}, step {global_step + 1}")

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(losses["total"])
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(flow_model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            for key in running:
                running[key] += losses[key].item()

            if accelerator.is_main_process:
                progress.set_postfix(
                    loss=losses["total"].item(),
                    detail=losses["detail"].item(),
                    hf=losses["hf"].item(),
                    rhf=losses["residual_hf"].item(),
                    dv=losses["detail_velocity"].item(),
                    roll_hf=losses["rollout_hf"].item(),
                    wm=losses["wm_l1"].item(),
                    lpips_w=losses["w_lpips"].item(),
                )
                if global_step % args.preview_every == 0:
                    flow_model.eval()
                    with torch.no_grad(), accelerator.autocast():
                        condition = build_stage2_condition(fixed_coarse, fixed_t1, args.condition_mode)
                        preview_refined = euler_refine(
                            flow_model,
                            fixed_coarse,
                            num_steps=args.eval_steps,
                            condition=condition,
                            detail_boost=args.detail_boost,
                            dynamic_condition=args.dynamic_condition_rollout,
                            condition_mode=args.condition_mode,
                            t1_img=fixed_t1,
                        )
                        save_preview(preview_dir, global_step, fixed_t1, fixed_coarse, preview_refined, fixed_fa)
                    flow_model.train()

        flow_model.eval()
        fid_metric.reset()
        kid_metric.reset()
        val_total = {
            "psnr": 0.0,
            "ssim": 0.0,
            "mse_proxy": 0.0,
            "l1": 0.0,
            "grad": 0.0,
            "hf": 0.0,
            "lpips": 0.0,
            "vel": 0.0,
            "coarse_psnr": 0.0,
            "coarse_ssim": 0.0,
            "coarse_sharpness": 0.0,
            "coarse_target_sharpness": 0.0,
            "sharpness": 0.0,
            "target_sharpness": 0.0,
            "refine_l1": 0.0,
            "coarse_target_l1": 0.0,
            "brain_l1": 0.0,
            "wm_l1": 0.0,
            "wm_grad": 0.0,
            "roi": 0.0,
        }
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Stage2 Val {epoch + 1}/{args.epochs}",
                disable=not accelerator.is_local_main_process,
            ):
                t1_img = prepare_stage1_input(batch["t1_slice"].to(accelerator.device), stage1_channels)
                fa_img = reduce_rgb_to_single_channel(batch["fa_slice"].to(accelerator.device))
                with accelerator.autocast():
                    coarse = predict_stage1_batch(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale)

                    t_eval = sample_stage2_time(
                        batch_size=t1_img.shape[0],
                        device=accelerator.device,
                        dtype=t1_img.dtype,
                        mode=args.t_sampling,
                        t_min=args.t_min,
                        t_max=args.t_max,
                    )
                    x_t, _, v_target = build_xt(
                        target=fa_img,
                        coarse=coarse,
                        t=t_eval,
                        source_noise_std=args.source_noise_std,
                        deterministic_source=True,
                    )
                    condition = build_stage2_condition(coarse, t1_img, args.condition_mode)
                    raw_output = flow_model(x_t, t_eval, condition=condition)
                    v_pred = compose_stage2_velocity(raw_output, t_eval, detail_boost=args.detail_boost)
                    refined = euler_refine(
                        flow_model,
                        coarse,
                        num_steps=args.eval_steps,
                        condition=condition,
                        detail_boost=args.detail_boost,
                        dynamic_condition=args.dynamic_condition_rollout,
                        condition_mode=args.condition_mode,
                        t1_img=t1_img,
                    )
                val_total["vel"] += F.mse_loss(v_pred.float(), v_target.float()).item()
                refined = clamp_to_image_range(refined.float())
                fa_fp32 = fa_img.float()
                coarse_fp32 = coarse.float()
                ssim_loss = ssim_loss_fn(refined, fa_img)
                coarse_ssim_loss = ssim_loss_fn(coarse_fp32, fa_fp32)
                brain_mask, wm_mask = build_training_masks(
                    t1_img,
                    fa_img,
                    brain_t1_threshold=args.brain_t1_threshold,
                    brain_fa_threshold=args.brain_fa_threshold,
                    wm_quantile=args.wm_quantile,
                    wm_min_threshold=args.wm_min_threshold,
                )

                val_total["psnr"] += compute_psnr(refined, fa_fp32).item()
                val_total["ssim"] += (1.0 - ssim_loss).item()
                val_total["mse_proxy"] += F.mse_loss(refined, fa_fp32).item()
                val_total["l1"] += F.l1_loss(refined, fa_fp32).item()
                val_total["refine_l1"] += F.l1_loss(refined, coarse_fp32).item()
                val_total["coarse_target_l1"] += F.l1_loss(coarse_fp32, fa_fp32).item()
                val_total["grad"] += grad_loss_fn(refined, fa_fp32).item()
                val_total["hf"] += F.l1_loss(laplacian_filter(refined), laplacian_filter(fa_fp32)).item()
                val_total["brain_l1"] += masked_l1_loss(refined, fa_fp32, brain_mask).item()
                val_total["wm_l1"] += masked_l1_loss(refined, fa_fp32, wm_mask).item()
                val_total["wm_grad"] += masked_gradient_l1_loss(refined, fa_fp32, wm_mask, grad_loss_fn).item()
                val_total["roi"] += roi_consistency_loss(
                    refined,
                    fa_fp32,
                    wm_mask,
                    rows=args.roi_rows,
                    cols=args.roi_cols,
                    min_pixels=args.roi_min_pixels,
                ).item()
                with torch.autocast(device_type=accelerator.device.type, enabled=False):
                    val_total["lpips"] += lpips_loss_fn(
                        refined.repeat(1, 3, 1, 1), fa_fp32.repeat(1, 3, 1, 1)
                    ).mean().item()
                val_total["sharpness"] += laplacian_variance(refined).item()
                val_total["target_sharpness"] += laplacian_variance(fa_fp32).item()
                val_total["coarse_sharpness"] += laplacian_variance(coarse_fp32).item()
                val_total["coarse_target_sharpness"] += laplacian_variance(fa_fp32).item()
                if (epoch + 1) % args.fid_eval_every == 0:
                    refined_01_3c = torch.clamp((refined + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
                    fa_01_3c = torch.clamp((fa_fp32 + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)
                    fid_metric.update(fa_01_3c, real=True)
                    fid_metric.update(refined_01_3c, real=False)
                    kid_metric.update(fa_01_3c, real=True)
                    kid_metric.update(refined_01_3c, real=False)
                val_total["coarse_psnr"] += compute_psnr(coarse_fp32, fa_fp32).item()
                val_total["coarse_ssim"] += (1.0 - coarse_ssim_loss).item()
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
        metrics["coarse_sharp_ratio"] = metrics["coarse_sharpness"] / max(metrics["coarse_target_sharpness"], 1e-8)
        metrics["refine_ratio"] = metrics["refine_l1"] / max(metrics["coarse_target_l1"], 1e-8)
        metrics["balanced_score"] = compute_balanced_score(metrics, args)
        metrics["paired_score"] = compute_paired_score(metrics, args)
        metrics["wm_paired_score"] = compute_wm_paired_score(metrics, args)
        metrics["detail_paired_score"] = compute_detail_paired_score(metrics, args)
        with torch.no_grad():
            fixed_condition = build_stage2_condition(fixed_coarse, fixed_t1, args.condition_mode)
            fixed_preview = clamp_to_image_range(
                euler_refine(
                    flow_model,
                    fixed_coarse,
                    num_steps=args.eval_steps,
                    condition=fixed_condition,
                    detail_boost=args.detail_boost,
                    dynamic_condition=args.dynamic_condition_rollout,
                    condition_mode=args.condition_mode,
                    t1_img=fixed_t1,
                ).float()
            )
        is_degraded, degrade_reasons, degrade_stats = detect_stage2_anomaly(
            refined=fixed_preview,
            coarse=fixed_coarse.float(),
            target=fixed_fa.float(),
            psnr=metrics["psnr"],
            ssim=metrics["ssim"],
            coarse_psnr=metrics["coarse_psnr"],
            coarse_ssim=metrics["coarse_ssim"],
            margin_psnr=args.degrade_margin_psnr,
            margin_ssim=args.degrade_margin_ssim,
            mode=args.degrade_check_mode,
        )
        degrade_epochs = degrade_epochs + 1 if is_degraded else 0
        is_best = is_better(metrics, best_metrics, args.best_metric)
        no_improve_epochs = 0 if is_best else (no_improve_epochs + 1)

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model": accelerator.unwrap_model(flow_model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_metrics["psnr"] if not is_best else metrics["psnr"],
            "best_ssim": best_metrics["ssim"] if not is_best else metrics["ssim"],
            "best_mse_proxy": best_metrics["mse_proxy"] if not is_best else metrics["mse_proxy"],
            "best_l1": best_metrics["l1"] if not is_best else metrics["l1"],
            "best_lpips": best_metrics["lpips"] if not is_best else metrics["lpips"],
            "best_fid": best_metrics["fid"] if not is_best else metrics["fid"],
            "best_kid": best_metrics["kid"] if not is_best else metrics["kid"],
            "best_sharpness": best_metrics["sharpness"] if not is_best else metrics["sharpness"],
            "best_sharp_ratio": best_metrics["sharp_ratio"] if not is_best else metrics["sharp_ratio"],
            "best_balanced_score": best_metrics["balanced_score"] if not is_best else metrics["balanced_score"],
            "best_paired_score": best_metrics["paired_score"] if not is_best else metrics["paired_score"],
            "best_wm_paired_score": best_metrics["wm_paired_score"] if not is_best else metrics["wm_paired_score"],
            "best_detail_paired_score": best_metrics["detail_paired_score"] if not is_best else metrics["detail_paired_score"],
            "best_refine_l1": best_metrics["refine_l1"] if not is_best else metrics["refine_l1"],
            "best_coarse_target_l1": best_metrics["coarse_target_l1"] if not is_best else metrics["coarse_target_l1"],
            "best_refine_ratio": best_metrics["refine_ratio"] if not is_best else metrics["refine_ratio"],
            "no_improve_epochs": no_improve_epochs,
            "degrade_epochs": degrade_epochs,
            "args": vars(args),
        }

        if accelerator.is_main_process:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"[Epoch {epoch + 1}] train_total={running['total'] / max(len(train_loader), 1):.6f} "
                    f"train_wm_l1={running['wm_l1'] / max(len(train_loader), 1):.6f} "
                    f"train_rollout_detail={running['rollout_detail'] / max(len(train_loader), 1):.6f} "
                    f"train_rollout_l1={running['rollout_l1'] / max(len(train_loader), 1):.6f} "
                    f"train_rollout_hf={running['rollout_hf'] / max(len(train_loader), 1):.6f} "
                    f"train_rollout_residual_hf={running['rollout_residual_hf'] / max(len(train_loader), 1):.6f} "
                    f"train_roi={running['roi'] / max(len(train_loader), 1):.6f} "
                    f"val_psnr={metrics['psnr']:.4f} val_ssim={metrics['ssim']:.4f} val_mse={metrics['mse_proxy']:.6f} "
                    f"val_l1={metrics['l1']:.6f} val_grad={metrics['grad']:.6f} val_hf={metrics['hf']:.6f} "
                    f"val_brain_l1={metrics['brain_l1']:.6f} val_wm_l1={metrics['wm_l1']:.6f} "
                    f"val_wm_grad={metrics['wm_grad']:.6f} val_roi={metrics['roi']:.6f} "
                    f"val_lpips={metrics['lpips']:.6f} val_fid={metrics['fid']:.4f} val_kid={metrics['kid']:.6f} "
                    f"sharp={metrics['sharpness']:.6f} sharp_ratio={metrics['sharp_ratio']:.4f} "
                    f"coarse_sharp_ratio={metrics['coarse_sharp_ratio']:.4f} "
                    f"refine_l1={metrics['refine_l1']:.6f} refine_ratio={metrics['refine_ratio']:.4f} "
                    f"balanced={metrics['balanced_score']:.4f} paired={metrics['paired_score']:.4f} "
                    f"wm_paired={metrics['wm_paired_score']:.4f} detail_paired={metrics['detail_paired_score']:.4f} "
                    f"coarse_psnr={metrics['coarse_psnr']:.4f} "
                    f"coarse_ssim={metrics['coarse_ssim']:.4f} "
                    f"no_improve_epochs={no_improve_epochs} degrade_epochs={degrade_epochs} "
                    f"skipped_nonfinite={skipped_nonfinite}\n"
                )
                if degrade_reasons:
                    handle.write(
                        f"[DegradeCheck][Epoch {epoch + 1}] reasons={';'.join(degrade_reasons)} "
                        f"pred_mean={degrade_stats['pred_mean']:.4f} target_mean={degrade_stats['target_mean']:.4f} "
                        f"pred_std={degrade_stats['pred_std']:.4f} target_std={degrade_stats['target_std']:.4f}\n"
                    )
            torch.save(checkpoint, latest_path)
            if not is_degraded:
                torch.save(checkpoint, healthy_latest_path)
            if (epoch + 1) % args.save_every == 0:
                torch.save(checkpoint, os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt"))

            if is_best:
                best_metrics["psnr"] = metrics["psnr"]
                best_metrics["ssim"] = metrics["ssim"]
                best_metrics["mse_proxy"] = metrics["mse_proxy"]
                best_metrics["l1"] = metrics["l1"]
                best_metrics["lpips"] = metrics["lpips"]
                best_metrics["fid"] = metrics["fid"]
                best_metrics["kid"] = metrics["kid"]
                best_metrics["sharpness"] = metrics["sharpness"]
                best_metrics["sharp_ratio"] = metrics["sharp_ratio"]
                best_metrics["balanced_score"] = metrics["balanced_score"]
                best_metrics["paired_score"] = metrics["paired_score"]
                best_metrics["wm_paired_score"] = metrics["wm_paired_score"]
                best_metrics["detail_paired_score"] = metrics["detail_paired_score"]
                best_metrics["refine_l1"] = metrics["refine_l1"]
                best_metrics["coarse_target_l1"] = metrics["coarse_target_l1"]
                best_metrics["refine_ratio"] = metrics["refine_ratio"]
                checkpoint["best_psnr"] = best_metrics["psnr"]
                checkpoint["best_ssim"] = best_metrics["ssim"]
                checkpoint["best_mse_proxy"] = best_metrics["mse_proxy"]
                checkpoint["best_l1"] = best_metrics["l1"]
                checkpoint["best_lpips"] = best_metrics["lpips"]
                checkpoint["best_fid"] = best_metrics["fid"]
                checkpoint["best_kid"] = best_metrics["kid"]
                checkpoint["best_sharpness"] = best_metrics["sharpness"]
                checkpoint["best_sharp_ratio"] = best_metrics["sharp_ratio"]
                checkpoint["best_balanced_score"] = best_metrics["balanced_score"]
                checkpoint["best_paired_score"] = best_metrics["paired_score"]
                checkpoint["best_wm_paired_score"] = best_metrics["wm_paired_score"]
                checkpoint["best_detail_paired_score"] = best_metrics["detail_paired_score"]
                checkpoint["best_refine_l1"] = best_metrics["refine_l1"]
                checkpoint["best_coarse_target_l1"] = best_metrics["coarse_target_l1"]
                checkpoint["best_refine_ratio"] = best_metrics["refine_ratio"]
                torch.save(checkpoint, best_path)
                accelerator.print(
                    f"New best Stage 2 checkpoint: PSNR={best_metrics['psnr']:.4f}, "
                    f"SSIM={best_metrics['ssim']:.4f}, MSE={best_metrics['mse_proxy']:.6f}, "
                    f"L1={best_metrics['l1']:.6f}, Detail_Paired={best_metrics['detail_paired_score']:.4f} -> {best_path}"
                )

        accelerator.print(
            f"[Stage 2][Epoch {epoch + 1}] refined_PSNR={metrics['psnr']:.4f} "
            f"refined_SSIM={metrics['ssim']:.4f} refined_MSE={metrics['mse_proxy']:.6f} "
            f"coarse_PSNR={metrics['coarse_psnr']:.4f} coarse_SSIM={metrics['coarse_ssim']:.4f} L1={metrics['l1']:.6f} "
            f"Grad={metrics['grad']:.6f} WM_L1={metrics['wm_l1']:.6f} ROI={metrics['roi']:.6f} "
            f"HF={metrics['hf']:.6f} LPIPS={metrics['lpips']:.6f} "
            f"FID={metrics['fid']:.4f} SharpRatio={metrics['sharp_ratio']:.4f} "
            f"CoarseSharpRatio={metrics['coarse_sharp_ratio']:.4f} "
            f"RefineL1={metrics['refine_l1']:.6f} RefineRatio={metrics['refine_ratio']:.4f} "
            f"Score={metrics['balanced_score']:.4f} Paired={metrics['paired_score']:.4f} "
            f"WM_Paired={metrics['wm_paired_score']:.4f} Detail_Paired={metrics['detail_paired_score']:.4f} "
            f"NoImprove={no_improve_epochs} Degrade={degrade_epochs} SkippedNonFinite={skipped_nonfinite}"
        )

        if degrade_epochs >= args.degrade_patience:
            if args.rollback_on_degrade and os.path.exists(healthy_latest_path):
                healthy_checkpoint = torch.load(healthy_latest_path, map_location="cpu")
                accelerator.unwrap_model(flow_model).load_state_dict(healthy_checkpoint["model"])
                optimizer.load_state_dict(healthy_checkpoint["optimizer"])
                accelerator.print(f"Rolled back to healthy checkpoint: {healthy_latest_path}")
            accelerator.print("Stage 2 stopped because refinement kept degrading relative to coarse.")
            break

        if no_improve_epochs >= args.early_stop_patience:
            accelerator.print(
                f"Early stopping triggered after {no_improve_epochs} epochs without improving {args.best_metric}."
            )
            break


if __name__ == "__main__":
    main()
