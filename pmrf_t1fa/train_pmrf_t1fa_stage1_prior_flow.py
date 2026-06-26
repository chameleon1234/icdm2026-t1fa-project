import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from pmrf_t1fa.models.pmrf_t1fa import (
    SSIMLoss,
    TemplatePriorFlowNet,
    build_prior_flow_state,
    center_channel,
    euler_sample_prior_flow,
    prepare_stage1_input,
    reduce_rgb_to_single_channel,
)
from pmrf_t1fa.train_pmrf_t1fa_stage1 import (
    _batch_slice_ids,
    build_fa_slice_template,
    build_t1_anatomical_wm_prob_map,
    build_training_masks,
    compute_psnr,
    ensure_dirs,
    gather_template_batch_from_slice_ids,
    laplacian_filter,
    laplacian_variance,
    make_slice_dataset,
    masked_l1_loss,
    roi_consistency_loss,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train template-guided prior-flow Stage1 for T1-to-FA generation.")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--run_name", default="pmrf_t1fa_stage1_prior_flow")
    parser.add_argument("--context_slices", type=int, default=1)
    parser.add_argument(
        "--flow_source",
        default="template",
        choices=["template", "pred_folder", "t1"],
        help="Initial source for x0. Use t1 to start the flow from the center T1 slice.",
    )
    parser.add_argument("--flow_source_pred_dir", default="", help="PNG folder for pred_folder mode.")
    parser.add_argument("--val_flow_source_pred_dir", default="", help="Validation PNG folder for pred_folder mode; defaults to flow_source_pred_dir.")
    parser.add_argument("--flow_source_name", default="", help="Short name for logging when using pred_folder mode.")
    parser.add_argument("--flow_steps", type=int, default=4, help="Euler steps used during validation.")
    parser.add_argument("--sample_clamp_each_step", action="store_true", help="Clamp Euler samples after each step instead of only at the end.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable CUDA pinned host memory for DataLoader. Disabled by default to avoid RAM pressure on 32GB machines.",
    )
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--end_weight", type=float, default=0.5)
    parser.add_argument("--wm_flow_weight", type=float, default=0.2)
    parser.add_argument("--roi_flow_weight", type=float, default=0.1)
    parser.add_argument("--freq_flow_weight", type=float, default=0.05)
    parser.add_argument("--ssim_weight", type=float, default=0.05)
    parser.add_argument("--preserve_lowfreq_weight", type=float, default=0.0)
    parser.add_argument("--preserve_lowfreq_kernel", type=int, default=13)
    parser.add_argument("--source_preserve_weight", type=float, default=0.0)
    parser.add_argument("--non_wm_overbright_weight", type=float, default=0.0, help="Penalize bright hallucinations outside the WM proxy mask.")
    parser.add_argument("--non_wm_overbright_margin", type=float, default=0.03, help="Allowed pred-target margin before non-WM overbright penalty is applied.")
    parser.add_argument("--source_artifact_weight", type=float, default=0.0, help="Extra correction on locations where the source is already non-WM overbright.")
    parser.add_argument("--brain_overbright_weight", type=float, default=0.0, help="Penalize any brain-region positive pred-target overshoot.")
    parser.add_argument("--brain_overbright_margin", type=float, default=0.06)
    parser.add_argument("--local_spike_weight", type=float, default=0.0, help="Penalize local bright spikes not present in the paired FA target.")
    parser.add_argument("--local_spike_margin", type=float, default=0.035)
    parser.add_argument("--local_spike_kernel", type=int, default=7)
    parser.add_argument("--source_spike_weight", type=float, default=0.0, help="Extra suppression where source already has unsupported local spikes.")
    parser.add_argument("--anatomy_unsupported_spike_weight", type=float, default=0.0, help="Penalize local spikes in low-WM, low-T1-edge regions.")
    parser.add_argument("--source_anatomy_spike_weight", type=float, default=0.0, help="Extra correction where source has anatomy-unsupported spikes.")
    parser.add_argument("--start_anchor_weight", type=float, default=0.0)
    parser.add_argument("--start_anchor_wm_weight", type=float, default=0.2)
    parser.add_argument("--start_anchor_roi_weight", type=float, default=0.1)
    parser.add_argument("--start_anchor_freq_weight", type=float, default=0.03)
    parser.add_argument("--t_sampling", default="uniform", choices=["uniform", "low", "mixed"])
    parser.add_argument("--t_gamma", type=float, default=2.0)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.72)
    parser.add_argument("--wm_min_threshold", type=float, default=0.10)
    parser.add_argument("--roi_rows", type=int, default=4)
    parser.add_argument("--roi_cols", type=int, default=4)
    parser.add_argument("--roi_min_pixels", type=int, default=20)
    parser.add_argument("--sharp_score_weight", type=float, default=5.0)
    parser.add_argument("--wm_score_weight", type=float, default=20.0)
    parser.add_argument("--best_sharp_target", type=float, default=1.05)
    parser.add_argument("--best_min_psnr", type=float, default=27.0)
    parser.add_argument("--best_min_sharp", type=float, default=0.85)
    parser.add_argument("--best_max_wm_l1", type=float, default=0.070)
    parser.add_argument("--best_max_sharp", type=float, default=99.0)
    parser.add_argument("--best_min_tenengrad", type=float, default=0.0)
    parser.add_argument("--best_min_edge_grad", type=float, default=0.0)
    parser.add_argument("--best_max_blur_deficit", type=float, default=1.0)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--preview_count", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def maybe_limit_dataset(dataset, limit: int):
    if int(limit) <= 0:
        return dataset
    return Subset(dataset, range(min(int(limit), len(dataset))))


def autocast_context(device: torch.device, mode: str):
    if device.type != "cuda" or mode == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def normalize_z_map(slice_ids: torch.Tensor, like_tensor: torch.Tensor, z_min: float, z_max: float) -> torch.Tensor:
    if z_max <= z_min:
        z = torch.zeros_like(slice_ids, dtype=like_tensor.dtype, device=like_tensor.device)
    else:
        z = (slice_ids.to(dtype=like_tensor.dtype) - float(z_min)) / (float(z_max) - float(z_min))
        z = z * 2.0 - 1.0
    return z.view(-1, 1, 1, 1).expand(-1, 1, like_tensor.shape[-2], like_tensor.shape[-1])


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Small differentiability-free Sobel magnitude helper for validation clarity metrics."""
    if x.shape[1] != 1:
        x = center_channel(x)
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)


def batch_clarity_metrics(pred: torch.Tensor, target: torch.Tensor, brain_mask: torch.Tensor) -> Dict[str, float]:
    pred_mag = sobel_magnitude(pred)
    target_mag = sobel_magnitude(target)
    mask = brain_mask.to(dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    pred_ten = (pred_mag.square() * mask).sum() / denom
    target_ten = (target_mag.square() * mask).sum() / denom
    tenengrad_ratio = pred_ten / target_ten.clamp_min(1e-8)
    flat = (target_mag * mask).flatten(start_dim=1)
    quantiles = torch.quantile(flat, 0.85, dim=1).view(-1, 1, 1, 1)
    edge_mask = ((target_mag >= quantiles) & (mask > 0)).to(dtype=pred.dtype)
    edge_denom = edge_mask.sum().clamp_min(1.0)
    edge_grad_ratio = (pred_mag * edge_mask).sum() / (target_mag * edge_mask).sum().clamp_min(1e-8)
    blur_deficit = torch.relu(target_ten - pred_ten) / target_ten.clamp_min(1e-8)
    pred_hp = laplacian_filter(pred).flatten(start_dim=1)
    target_hp = laplacian_filter(target).flatten(start_dim=1)
    pred_hp = pred_hp - pred_hp.mean(dim=1, keepdim=True)
    target_hp = target_hp - target_hp.mean(dim=1, keepdim=True)
    hf_corr = (pred_hp * target_hp).sum(dim=1) / (
        torch.sqrt((pred_hp.square()).sum(dim=1) * (target_hp.square()).sum(dim=1)).clamp_min(1e-8)
    )
    return {
        "tenengrad_ratio": float(tenengrad_ratio.item()),
        "edge_grad_ratio": float(edge_grad_ratio.item()),
        "blur_deficit": float(blur_deficit.item()),
        "hf_corr": float(hf_corr.mean().item()),
    }


def lowpass_filter(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = max(3, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)


def local_positive_spike(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    local_mean = lowpass_filter(x, kernel_size)
    return F.relu(x - local_mean)


def anatomy_unsupported_gate(t1_img: torch.Tensor, wm_prob: torch.Tensor, brain_mask: torch.Tensor) -> torch.Tensor:
    t1_edge = sobel_magnitude(center_channel(t1_img)).detach()
    flat = (t1_edge * brain_mask).flatten(start_dim=1)
    q = torch.quantile(flat, 0.85, dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    edge_support = torch.clamp(t1_edge / q, 0.0, 1.0)
    return brain_mask * (1.0 - wm_prob.clamp(0.0, 1.0)) * (1.0 - edge_support)


def sample_flow_time(batch_size: int, device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> torch.Tensor:
    u = torch.rand((batch_size,), device=device, dtype=dtype)
    if args.t_sampling == "uniform":
        return u
    low_t = u ** float(args.t_gamma)
    if args.t_sampling == "low":
        return low_t
    if args.t_sampling == "mixed":
        choose_low = (torch.rand_like(u) < 0.5).to(dtype=dtype)
        return choose_low * low_t + (1.0 - choose_low) * u
    raise ValueError(f"Unsupported t_sampling: {args.t_sampling}")


def prepare_batch(
    batch: Dict[str, Any],
    device: torch.device,
    context_slices: int,
    template_by_slice: dict[int, torch.Tensor],
    default_template: torch.Tensor,
    z_min: float,
    z_max: float,
    flow_source: str = "template",
    pred_dir: str = "",
):
    import cv2, numpy as np
    t1_img = prepare_stage1_input(batch["t1_slice"].to(device), context_slices).float()
    target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device)).float()
    slice_ids = _batch_slice_ids(batch, device)
    if flow_source == "pred_folder":
        x0_list = []
        for fname in batch["fname"]:
            p = os.path.join(pred_dir, str(fname))
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                if os.path.exists(p):
                    try:
                        data = np.fromfile(p, dtype=np.uint8)
                        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
                    except Exception:
                        img = None
            if img is None:
                slice_id = int(str(fname).split("_z")[-1].replace(".png", ""))
                fallback = template_by_slice.get(slice_id, default_template)
                img = (fallback.squeeze().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
            img = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
            x0_list.append(img)
        x0 = torch.cat(x0_list, dim=0).to(device=device, dtype=target.dtype)
    elif flow_source == "t1":
        x0 = center_channel(t1_img).to(device=device, dtype=target.dtype)
    else:
        x0 = gather_template_batch_from_slice_ids(template_by_slice, slice_ids, target, default_template)
    wm_prob = build_t1_anatomical_wm_prob_map(t1_img).to(device=device, dtype=target.dtype)
    z_map = normalize_z_map(slice_ids, target, z_min, z_max)
    return t1_img, target, x0, wm_prob, z_map


def prior_flow_loss(
    model: TemplatePriorFlowNet,
    t1_img: torch.Tensor,
    target: torch.Tensor,
    x0: torch.Tensor,
    wm_prob: torch.Tensor,
    z_map: torch.Tensor,
    args: argparse.Namespace,
    ssim_loss: SSIMLoss,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    t = sample_flow_time(target.shape[0], target.device, target.dtype, args)
    x_t, v_target = build_prior_flow_state(x0, target, t)
    v_pred = model(x_t, x0, t1_img, wm_prob, z_map, t)
    t_map = t.view(-1, 1, 1, 1)
    x_hat = torch.clamp(x_t + (1.0 - t_map) * v_pred, -1.0, 1.0)
    brain_mask, wm_mask = build_training_masks(
        t1_img,
        target,
        args.brain_t1_threshold,
        args.brain_fa_threshold,
        args.wm_quantile,
        args.wm_min_threshold,
    )
    flow_mse = (((v_pred - v_target) ** 2) * brain_mask).sum() / brain_mask.sum().clamp_min(1.0)
    end_l1 = masked_l1_loss(x_hat, target, brain_mask)
    wm_l1 = masked_l1_loss(x_hat, target, wm_mask)
    roi = roi_consistency_loss(x_hat, target, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels)
    freq = F.l1_loss(laplacian_filter(x_hat), laplacian_filter(target))
    ssim = ssim_loss(x_hat, target)
    lowfreq_preserve = F.l1_loss(lowpass_filter(x_hat, args.preserve_lowfreq_kernel), lowpass_filter(x0, args.preserve_lowfreq_kernel))
    source_preserve = masked_l1_loss(x_hat, x0, brain_mask)
    non_wm = brain_mask * (1.0 - wm_prob.clamp(0.0, 1.0))
    non_wm_denom = non_wm.sum().clamp_min(1.0)
    overbright = F.relu(x_hat - target - float(args.non_wm_overbright_margin))
    non_wm_overbright = (overbright * non_wm).sum() / non_wm_denom
    source_artifact_mask = (F.relu(x0 - target - float(args.non_wm_overbright_margin)) * non_wm).detach()
    source_artifact_denom = source_artifact_mask.sum().clamp_min(1.0)
    source_artifact = (overbright * source_artifact_mask).sum() / source_artifact_denom
    brain_overbright = (F.relu(x_hat - target - float(args.brain_overbright_margin)) * brain_mask).sum() / brain_mask.sum().clamp_min(1.0)
    pred_spike = local_positive_spike(x_hat, args.local_spike_kernel)
    target_spike = local_positive_spike(target, args.local_spike_kernel)
    source_spike = local_positive_spike(x0, args.local_spike_kernel)
    unsupported_spike = F.relu(pred_spike - target_spike - float(args.local_spike_margin))
    local_spike = (unsupported_spike * brain_mask).sum() / brain_mask.sum().clamp_min(1.0)
    source_spike_mask = F.relu(source_spike - target_spike - float(args.local_spike_margin)).detach() * brain_mask
    source_spike_artifact = (unsupported_spike * source_spike_mask).sum() / source_spike_mask.sum().clamp_min(1.0)
    anatomy_gate = anatomy_unsupported_gate(t1_img, wm_prob, brain_mask)
    anatomy_spike = (unsupported_spike * anatomy_gate).sum() / anatomy_gate.sum().clamp_min(1.0)
    source_anatomy_spike_mask = (
        F.relu(source_spike - target_spike - float(args.local_spike_margin)).detach() * anatomy_gate
    )
    source_anatomy_spike = (unsupported_spike * source_anatomy_spike_mask).sum() / source_anatomy_spike_mask.sum().clamp_min(1.0)
    total = (
        args.flow_weight * flow_mse
        + args.end_weight * end_l1
        + args.wm_flow_weight * wm_l1
        + args.roi_flow_weight * roi
        + args.freq_flow_weight * freq
        + args.ssim_weight * ssim
        + args.preserve_lowfreq_weight * lowfreq_preserve
        + args.source_preserve_weight * source_preserve
        + args.non_wm_overbright_weight * non_wm_overbright
        + args.source_artifact_weight * source_artifact
        + args.brain_overbright_weight * brain_overbright
        + args.local_spike_weight * local_spike
        + args.source_spike_weight * source_spike_artifact
        + args.anatomy_unsupported_spike_weight * anatomy_spike
        + args.source_anatomy_spike_weight * source_anatomy_spike
    )
    anchor_l1 = target.new_tensor(0.0)
    anchor_wm = target.new_tensor(0.0)
    anchor_roi = target.new_tensor(0.0)
    anchor_freq = target.new_tensor(0.0)
    if float(args.start_anchor_weight) > 0.0:
        t0 = torch.zeros((target.shape[0],), device=target.device, dtype=target.dtype)
        v0 = model(x0, x0, t1_img, wm_prob, z_map, t0)
        x0_hat = torch.clamp(x0 + v0, -1.0, 1.0)
        anchor_l1 = masked_l1_loss(x0_hat, target, brain_mask)
        anchor_wm = masked_l1_loss(x0_hat, target, wm_mask)
        anchor_roi = roi_consistency_loss(x0_hat, target, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels)
        anchor_freq = F.l1_loss(laplacian_filter(x0_hat), laplacian_filter(target))
        total = total + float(args.start_anchor_weight) * (
            anchor_l1
            + float(args.start_anchor_wm_weight) * anchor_wm
            + float(args.start_anchor_roi_weight) * anchor_roi
            + float(args.start_anchor_freq_weight) * anchor_freq
        )
    return total, {
        "flow": flow_mse.detach(),
        "end_l1": end_l1.detach(),
        "wm_l1": wm_l1.detach(),
        "roi": roi.detach(),
        "freq": freq.detach(),
        "ssim_loss": ssim.detach(),
        "lowfreq_preserve": lowfreq_preserve.detach(),
        "source_preserve": source_preserve.detach(),
        "non_wm_overbright": non_wm_overbright.detach(),
        "source_artifact": source_artifact.detach(),
        "brain_overbright": brain_overbright.detach(),
        "local_spike": local_spike.detach(),
        "source_spike_artifact": source_spike_artifact.detach(),
        "anatomy_spike": anatomy_spike.detach(),
        "source_anatomy_spike": source_anatomy_spike.detach(),
        "anchor_l1": anchor_l1.detach(),
        "anchor_wm": anchor_wm.detach(),
        "anchor_roi": anchor_roi.detach(),
        "anchor_freq": anchor_freq.detach(),
    }


@torch.no_grad()
def validate(
    model: TemplatePriorFlowNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    template_by_slice: dict[int, torch.Tensor],
    default_template: torch.Tensor,
    z_min: float,
    z_max: float,
    preview_dir: Path,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "psnr": 0.0,
        "ssim": 0.0,
        "mse": 0.0,
        "l1": 0.0,
        "wm_l1": 0.0,
        "roi": 0.0,
        "sharp": 0.0,
        "source_sharp": 0.0,
        "source_residual_l1": 0.0,
        "tenengrad_ratio": 0.0,
        "edge_grad_ratio": 0.0,
        "blur_deficit": 0.0,
        "hf_corr": 0.0,
        "non_wm_overbright": 0.0,
        "source_non_wm_overbright": 0.0,
        "brain_overbright": 0.0,
        "local_spike": 0.0,
        "source_local_spike": 0.0,
        "anatomy_spike": 0.0,
        "source_anatomy_spike": 0.0,
    }
    count = 0
    ssim_loss = SSIMLoss().to(device)
    preview_saved = 0
    for batch in tqdm(loader, desc=f"PriorFlow Val {epoch}/{args.epochs}"):
        pred_dir = (args.val_flow_source_pred_dir or args.flow_source_pred_dir) if args.flow_source == "pred_folder" else ""
        t1_img, target, x0, wm_prob, z_map = prepare_batch(
            batch,
            device,
            args.context_slices,
            template_by_slice,
            default_template,
            z_min,
            z_max,
            flow_source=args.flow_source,
            pred_dir=pred_dir,
        )
        pred = euler_sample_prior_flow(model, x0, t1_img, wm_prob, z_map, steps=args.flow_steps, clamp=bool(args.sample_clamp_each_step))
        pred = torch.clamp(pred, -1.0, 1.0)
        brain_mask, wm_mask = build_training_masks(
            t1_img,
            target,
            args.brain_t1_threshold,
            args.brain_fa_threshold,
            args.wm_quantile,
            args.wm_min_threshold,
        )
        batch_size = pred.shape[0]
        totals["psnr"] += float(compute_psnr(pred, target).item()) * batch_size
        totals["ssim"] += float((1.0 - ssim_loss(pred, target)).item()) * batch_size
        pred_01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
        target_01 = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
        totals["mse"] += float(F.mse_loss(pred_01, target_01).item()) * batch_size
        totals["l1"] += float(F.l1_loss(pred_01, target_01).item()) * batch_size
        totals["wm_l1"] += float(masked_l1_loss(pred_01, target_01, wm_mask).item()) * batch_size
        totals["roi"] += float(roi_consistency_loss(pred_01, target_01, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels).item()) * batch_size
        target_sharp = laplacian_variance(target).clamp_min(1e-8)
        totals["sharp"] += float((laplacian_variance(pred) / target_sharp).item()) * batch_size
        totals["source_sharp"] += float((laplacian_variance(x0) / target_sharp).item()) * batch_size
        active = (target - x0).abs() > torch.quantile((target - x0).abs().flatten(), 0.70)
        totals["source_residual_l1"] += float((pred - target).abs()[active].mean().item()) * batch_size
        clarity = batch_clarity_metrics(pred, target, brain_mask)
        for key, value in clarity.items():
            totals[key] += value * batch_size
        non_wm = brain_mask * (1.0 - wm_prob.clamp(0.0, 1.0))
        non_wm_denom = non_wm.sum().clamp_min(1.0)
        margin = float(args.non_wm_overbright_margin)
        totals["non_wm_overbright"] += float((F.relu(pred - target - margin) * non_wm).sum().item() / float(non_wm_denom.item())) * batch_size
        totals["source_non_wm_overbright"] += float((F.relu(x0 - target - margin) * non_wm).sum().item() / float(non_wm_denom.item())) * batch_size
        brain_denom = brain_mask.sum().clamp_min(1.0)
        totals["brain_overbright"] += float((F.relu(pred - target - float(args.brain_overbright_margin)) * brain_mask).sum().item() / float(brain_denom.item())) * batch_size
        pred_spike = local_positive_spike(pred, args.local_spike_kernel)
        target_spike = local_positive_spike(target, args.local_spike_kernel)
        source_spike = local_positive_spike(x0, args.local_spike_kernel)
        spike_margin = float(args.local_spike_margin)
        totals["local_spike"] += float((F.relu(pred_spike - target_spike - spike_margin) * brain_mask).sum().item() / float(brain_denom.item())) * batch_size
        totals["source_local_spike"] += float((F.relu(source_spike - target_spike - spike_margin) * brain_mask).sum().item() / float(brain_denom.item())) * batch_size
        anatomy_gate = anatomy_unsupported_gate(t1_img, wm_prob, brain_mask)
        anatomy_denom = anatomy_gate.sum().clamp_min(1.0)
        totals["anatomy_spike"] += float((F.relu(pred_spike - target_spike - spike_margin) * anatomy_gate).sum().item() / float(anatomy_denom.item())) * batch_size
        totals["source_anatomy_spike"] += float((F.relu(source_spike - target_spike - spike_margin) * anatomy_gate).sum().item() / float(anatomy_denom.item())) * batch_size
        count += batch_size
        if preview_saved < args.preview_count:
            names = batch.get("fname", [f"sample_{preview_saved}.png"])
            for i in range(min(pred.shape[0], args.preview_count - preview_saved)):
                panel = torch.cat([center_channel(t1_img[i : i + 1]), x0[i : i + 1], pred[i : i + 1], target[i : i + 1]], dim=3)
                save_image((panel + 1.0) / 2.0, preview_dir / f"epoch{epoch:03d}_{Path(str(names[i])).stem}_t1_source_pred_gt.png")
                preview_saved += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def old_score_metrics(metrics: Dict[str, float], args: argparse.Namespace) -> float:
    sharp_bonus = min(metrics["sharp"], 1.15)
    return metrics["psnr"] + 10.0 * metrics["ssim"] + args.sharp_score_weight * sharp_bonus - args.wm_score_weight * metrics["wm_l1"]


def score_metrics(metrics: Dict[str, float], args: argparse.Namespace) -> float:
    sharp_ratio = float(metrics["sharp"])
    sharp_target = float(args.best_sharp_target)
    sharp_penalty = abs(sharp_ratio - sharp_target)
    score = (
        float(metrics["psnr"])
        + 12.0 * float(metrics["ssim"])
        - 80.0 * float(metrics["wm_l1"])
        - 40.0 * float(metrics["roi"])
        - 2.0 * sharp_penalty
    )
    if sharp_ratio < 0.90:
        score -= 2.0 * (0.90 - sharp_ratio)
    if sharp_ratio > 1.20:
        score -= 3.0 * (sharp_ratio - 1.20)
    return score


def passes_best_gate(metrics: Dict[str, float], args: argparse.Namespace) -> bool:
    """Hard gate: only save checkpoint if all clinical-quality thresholds pass."""
    if metrics["psnr"] < float(getattr(args, "best_min_psnr", 27.0)):
        return False
    if metrics["sharp"] < float(getattr(args, "best_min_sharp", 0.85)):
        return False
    if metrics["sharp"] > float(getattr(args, "best_max_sharp", 99.0)):
        return False
    if metrics["wm_l1"] > float(getattr(args, "best_max_wm_l1", 0.070)):
        return False
    if metrics.get("tenengrad_ratio", 0.0) < float(getattr(args, "best_min_tenengrad", 0.0)):
        return False
    if metrics.get("edge_grad_ratio", 0.0) < float(getattr(args, "best_min_edge_grad", 0.0)):
        return False
    if metrics.get("blur_deficit", 1.0) > float(getattr(args, "best_max_blur_deficit", 1.0)):
        return False
    return True


def build_checkpoint(
    model: TemplatePriorFlowNet,
    optimizer: torch.optim.Optimizer,
    args_dict: dict[str, Any],
    epoch: int,
    metrics: Dict[str, float],
    template_by_slice: dict[int, torch.Tensor],
    default_template: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": args_dict,
        "epoch": epoch,
        "metrics": metrics,
        "stage1_template_by_slice": {int(k): v.cpu() for k, v in template_by_slice.items()},
        "stage1_template_default": default_template.cpu(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    root, ckpt_dir_str, preview_dir_str = ensure_dirs(args.run_name)
    ckpt_dir = Path(ckpt_dir_str)
    preview_dir = Path(preview_dir_str)
    train_dataset = maybe_limit_dataset(make_slice_dataset(args.train_t1_dir, args.train_fa_dir, args.context_slices), args.train_limit)
    val_dataset = maybe_limit_dataset(make_slice_dataset(args.val_t1_dir, args.val_fa_dir, args.context_slices), args.val_limit)
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    template_by_slice, default_template = build_fa_slice_template(args.train_fa_dir)
    z_min = float(min(template_by_slice.keys()))
    z_max = float(max(template_by_slice.keys()))
    model = TemplatePriorFlowNet(t1_channels=args.context_slices, width=args.width, num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    ssim_loss = SSIMLoss().to(device)
    best_score = -float("inf")
    args_dict = vars(args).copy()
    print(
        f"Prior-flow Stage1 training on {len(train_dataset)} train / {len(val_dataset)} val slices | "
        f"context={args.context_slices} source={args.flow_source} steps={args.flow_steps} "
        f"source_name={args.flow_source_name or 'template'} width={args.width} blocks={args.num_blocks} "
        f"device={device} num_workers={args.num_workers} pin_memory={pin_memory}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"PriorFlow Epoch {epoch}/{args.epochs}")
        pred_dir = args.flow_source_pred_dir if args.flow_source == "pred_folder" else ""
        for batch in progress:
            t1_img, target, x0, wm_prob, z_map = prepare_batch(
                batch,
                device,
                args.context_slices,
                template_by_slice,
                default_template,
                z_min,
                z_max,
                flow_source=args.flow_source,
                pred_dir=pred_dir,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.mixed_precision):
                loss, parts = prior_flow_loss(model, t1_img, target, x0, wm_prob, z_map, args, ssim_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().item())
            progress.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                flow=f"{float(parts['flow'].item()):.4f}",
                wm=f"{float(parts['wm_l1'].item()):.4f}",
            )
        metrics = validate(model, val_loader, device, args, template_by_slice, default_template, z_min, z_max, preview_dir, epoch)
        metrics["loss"] = running / max(len(train_loader), 1)
        metrics["old_score"] = old_score_metrics(metrics, args)
        metrics["score"] = score_metrics(metrics, args)
        gate_ok = passes_best_gate(metrics, args)
        is_best = gate_ok and (metrics["score"] > best_score)
        checkpoint = build_checkpoint(model, optimizer, args_dict, epoch, metrics, template_by_slice, default_template)
        if is_best:
            best_score = metrics["score"]
            torch.save(checkpoint, ckpt_dir / "best_prior_flow_stage1.pt")
            torch.save(checkpoint, ckpt_dir / "best_stage1.pt")
        torch.save(checkpoint, ckpt_dir / "latest_prior_flow_stage1.pt")
        torch.save(checkpoint, ckpt_dir / "latest_stage1.pt")
        print(
            f"[PriorFlow][Epoch {epoch}] PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"MSE={metrics['mse']:.6f} MAE={metrics['l1']:.6f} WM_L1={metrics['wm_l1']:.6f} "
            f"ROI={metrics['roi']:.6f} SharpRatio={metrics['sharp']:.4f} "
            f"SourceSharp={metrics['source_sharp']:.4f} SourceResidualL1={metrics['source_residual_l1']:.6f} "
            f"Tenengrad={metrics['tenengrad_ratio']:.4f} EdgeGrad={metrics['edge_grad_ratio']:.4f} "
            f"BlurDeficit={metrics['blur_deficit']:.4f} HFCorr={metrics['hf_corr']:.4f} "
            f"NonWMOver={metrics['non_wm_overbright']:.6f} SourceNonWMOver={metrics['source_non_wm_overbright']:.6f} "
            f"BrainOver={metrics['brain_overbright']:.6f} LocalSpike={metrics['local_spike']:.6f} SourceLocalSpike={metrics['source_local_spike']:.6f} "
            f"AnatSpike={metrics['anatomy_spike']:.6f} SourceAnatSpike={metrics['source_anatomy_spike']:.6f} "
            f"OldScore={metrics['old_score']:.4f} NewScore={metrics['score']:.4f} Gate={int(gate_ok)} Best={int(is_best)}"
        )
        with open(Path(root) / "train_metrics.json", "w", encoding="utf-8") as handle:
            json.dump({"last": metrics, "best_score": best_score, "gate_ok": gate_ok}, handle, indent=2)
        if args.save_every_epoch:
            torch.save(checkpoint, ckpt_dir / f"epoch_{epoch:03d}_stage1.pt")


if __name__ == "__main__":
    main()
