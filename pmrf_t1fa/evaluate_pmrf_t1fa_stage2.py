import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from pmrf_t1fa.models.pmrf_t1fa import (
    DetailRefinementFlowUNet,
    DetailStage1Net,
    RefinementFlowUNet,
    Stage1Net,
    build_stage2_condition,
    center_channel,
    euler_refine,
    infer_stage1_detail_scale,
    infer_stage1_in_channels,
    infer_stage1_model_variant,
    infer_stage1_prediction_mode,
    infer_stage2_condition_mode,
    prepare_stage1_input,
    predict_stage1_fa,
    reduce_rgb_to_single_channel,
    stage2_condition_channels,
)
from src.data.t1fa_stack_dataset import T1FAStackDataset
from src.datasets import T1FADataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PMRF-T1FA Stage 2 checkpoint")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/processed/test/fa_slices")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1/checkpoints/best_stage1.pt")
    parser.add_argument("--stage2_ckpt", default="outputs/pmrf_t1fa_stage2_b/checkpoints/best_stage2.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps", type=int, default=-1, help="Use checkpoint args when set to -1")
    parser.add_argument("--dynamic_condition_rollout", action="store_true", help="Override checkpoint and rebuild condition from the current rollout state.")
    parser.add_argument("--disable_dynamic_condition_rollout", action="store_true", help="Override checkpoint and use the static initial condition.")
    parser.add_argument(
        "--condition_on_coarse",
        action="store_true",
        help="Override checkpoint setting and enable coarse conditioning",
    )
    parser.add_argument(
        "--auto_condition_from_ckpt",
        action="store_true",
        help="Use condition settings from checkpoint args when available",
    )
    parser.add_argument(
        "--condition_mode",
        default="auto",
        choices=["auto", "none", "coarse", "t1", "coarse_t1", "coarse_t1_edge"],
        help="Stage 2 conditioning. auto reads the checkpoint and keeps old coarse-only checkpoints compatible.",
    )
    parser.add_argument("--kid_subset_size", type=int, default=100)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--prediction_limit", type=int, default=128)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--roi_rows", type=int, default=2)
    parser.add_argument("--roi_cols", type=int, default=3)
    parser.add_argument("--roi_min_pixels", type=int, default=32)
    parser.add_argument("--hist_bins", type=int, default=64)
    return parser.parse_args()


def clamp_to_image_range(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def checkpoint_args(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("args", {})
    return {}


def load_stage1(stage1_ckpt: str, device: torch.device):
    checkpoint = torch.load(stage1_ckpt, map_location="cpu")
    stage1_channels = infer_stage1_in_channels(checkpoint)
    ckpt_args = checkpoint_args(checkpoint)
    if infer_stage1_model_variant(ckpt_args) == "detail":
        model = DetailStage1Net(in_channels=stage1_channels, out_channels=1)
    else:
        model = Stage1Net(in_channels=stage1_channels, out_channels=1)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    prediction_mode = infer_stage1_prediction_mode(ckpt_args)
    return model, prediction_mode, stage1_channels, infer_stage1_detail_scale(ckpt_args)


def make_slice_dataset(t1_dir: str, fa_dir: str, stage1_channels: int):
    if stage1_channels > 1:
        return T1FAStackDataset(t1_dir, fa_dir, context_slices=stage1_channels, target_size=(224, 224))
    return T1FADataset(t1_dir, fa_dir, preload_ram=False)


def load_stage2(
    stage2_ckpt: str,
    device: torch.device,
    condition_mode: str,
    stage1_channels: int,
) -> RefinementFlowUNet:
    checkpoint = torch.load(stage2_ckpt, map_location="cpu")
    state_dict = checkpoint_state_dict(checkpoint)
    inferred_condition_channels = int(state_dict["inc.weight"].shape[1]) - 1 if "inc.weight" in state_dict else None
    expected_with_delta = stage2_condition_channels(condition_mode, stage1_channels, include_delta_from_initial=True)
    expected_without_delta = stage2_condition_channels(condition_mode, stage1_channels, include_delta_from_initial=False)
    if inferred_condition_channels in {expected_with_delta, expected_without_delta}:
        condition_channels = inferred_condition_channels
    else:
        condition_channels = expected_with_delta
    include_delta_from_initial = condition_channels == expected_with_delta
    stage2_args = checkpoint_args(checkpoint)
    if stage2_args.get("stage2_model_variant", "single") == "detail":
        model = DetailRefinementFlowUNet(input_channels=1, condition_channels=condition_channels)
    else:
        model = RefinementFlowUNet(input_channels=1, condition_channels=condition_channels)
    model.load_state_dict(state_dict)
    model.include_delta_from_initial = include_delta_from_initial
    model.to(device)
    model.eval()
    return model


def ensure_output_dir(output_dir: str, stage2_ckpt: str) -> str:
    if output_dir:
        root = output_dir
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(stage2_ckpt))
        root = os.path.join(os.path.dirname(ckpt_dir), "eval")
    os.makedirs(root, exist_ok=True)
    return root


def save_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_csv(path: str, payload: dict):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def save_prediction_panel(path: str, t1: torch.Tensor, coarse: torch.Tensor, refined: torch.Tensor, gt: torch.Tensor):
    from torchvision.utils import save_image

    def _vis(t: torch.Tensor) -> torch.Tensor:
        return torch.clamp((t + 1.0) / 2.0, 0.0, 1.0).repeat(1, 3, 1, 1)

    panel = torch.cat([_vis(t) for t in (center_channel(t1), coarse, refined, gt)], dim=-1)
    save_image(panel, path, nrow=1)


def sobel_gradients(x: torch.Tensor):
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return grad_x, grad_y


def laplacian_response(x: torch.Tensor):
    kernel = x.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1)


def morphological_close(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=kernel_size, stride=1, padding=pad)
    return eroded > 0.5


def build_brain_mask(t1_01: torch.Tensor, fa_01: torch.Tensor, t1_threshold: float, fa_threshold: float) -> torch.Tensor:
    mask = (t1_01 > t1_threshold) | (fa_01 > fa_threshold)
    return morphological_close(mask.float()).bool()


def build_wm_mask(fa_01: torch.Tensor, brain_mask: torch.Tensor, quantile: float, min_threshold: float) -> torch.Tensor:
    masks = []
    for i in range(fa_01.shape[0]):
        fa_vals = fa_01[i][brain_mask[i]]
        if fa_vals.numel() == 0:
            masks.append(torch.zeros_like(brain_mask[i], dtype=torch.bool))
            continue
        thr = max(min_threshold, torch.quantile(fa_vals, quantile).item())
        wm = brain_mask[i] & (fa_01[i] >= thr)
        masks.append(morphological_close(wm.float()).bool())
    return torch.stack(masks, dim=0)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    vals = torch.abs(pred - target)[mask]
    return float("nan") if vals.numel() == 0 else vals.mean().item()


def masked_gradient_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred_gx, pred_gy = sobel_gradients(pred)
    target_gx, target_gy = sobel_gradients(target)
    vals_x = torch.abs(pred_gx - target_gx)[mask]
    vals_y = torch.abs(pred_gy - target_gy)[mask]
    if vals_x.numel() == 0 or vals_y.numel() == 0:
        return float("nan")
    return 0.5 * (vals_x.mean().item() + vals_y.mean().item())


def masked_laplacian_variance(x: torch.Tensor, mask: torch.Tensor) -> float:
    vals = laplacian_response(x)[mask]
    return float("nan") if vals.numel() == 0 else vals.var(unbiased=False).item()


def histogram_wasserstein_distance(pred_values: torch.Tensor, target_values: torch.Tensor, bins: int = 64) -> float:
    if pred_values.numel() == 0 or target_values.numel() == 0:
        return float("nan")
    pred_hist = torch.histc(pred_values.float(), bins=bins, min=0.0, max=1.0)
    target_hist = torch.histc(target_values.float(), bins=bins, min=0.0, max=1.0)
    pred_hist = pred_hist / pred_hist.sum().clamp_min(1e-8)
    target_hist = target_hist / target_hist.sum().clamp_min(1e-8)
    pred_cdf = torch.cumsum(pred_hist, dim=0)
    target_cdf = torch.cumsum(target_hist, dim=0)
    return (torch.abs(pred_cdf - target_cdf).sum() * (1.0 / bins)).item()


def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mean_x) * (y - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float("nan") if denom <= 1e-12 else (2.0 * cov_xy) / denom


def collect_spatial_roi_means(
    pred_01: torch.Tensor,
    target_01: torch.Tensor,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    roi_rows: int,
    roi_cols: int,
    min_pixels: int,
):
    pred_means = []
    target_means = []
    for i in range(pred_01.shape[0]):
        coords = torch.nonzero(brain_mask[i, 0], as_tuple=False)
        if coords.numel() == 0:
            continue
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item()) + 1
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item()) + 1
        y_edges = torch.linspace(y_min, y_max, steps=roi_rows + 1, device=pred_01.device).round().long()
        x_edges = torch.linspace(x_min, x_max, steps=roi_cols + 1, device=pred_01.device).round().long()
        for r in range(roi_rows):
            for c in range(roi_cols):
                y0, y1 = int(y_edges[r].item()), int(y_edges[r + 1].item())
                x0, x1 = int(x_edges[c].item()), int(x_edges[c + 1].item())
                if y1 <= y0 or x1 <= x0:
                    continue
                region_mask = wm_mask[i, 0, y0:y1, x0:x1]
                if int(region_mask.sum().item()) < min_pixels:
                    continue
                pred_region = pred_01[i, 0, y0:y1, x0:x1][region_mask]
                target_region = target_01[i, 0, y0:y1, x0:x1][region_mask]
                pred_means.append(float(pred_region.mean().item()))
                target_means.append(float(target_region.mean().item()))
    return pred_means, target_means


def nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def nanstd(values):
    arr = np.asarray(values, dtype=np.float64)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanstd(arr))


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    stage2_checkpoint = torch.load(args.stage2_ckpt, map_location="cpu")
    stage2_args = checkpoint_args(stage2_checkpoint)

    condition_on_coarse = args.condition_on_coarse
    if args.auto_condition_from_ckpt or not args.condition_on_coarse or args.condition_mode == "auto":
        condition_on_coarse = bool(stage2_args.get("condition_on_coarse", condition_on_coarse))
    if args.condition_mode == "auto":
        condition_mode = infer_stage2_condition_mode(stage2_args)
    else:
        condition_mode = infer_stage2_condition_mode(
            {"condition_mode": args.condition_mode, "condition_on_coarse": condition_on_coarse}
        )
    condition_on_coarse = condition_mode in {"coarse", "coarse_t1", "coarse_t1_edge"}

    eval_steps = args.eval_steps if args.eval_steps > 0 else int(stage2_args.get("eval_steps", 1))
    detail_boost = float(stage2_args.get("detail_boost", 0.0))
    dynamic_condition = bool(stage2_args.get("dynamic_condition_rollout", False))
    if args.dynamic_condition_rollout:
        dynamic_condition = True
    if args.disable_dynamic_condition_rollout:
        dynamic_condition = False
    velocity_schedule = str(stage2_args.get("velocity_schedule", "constant"))

    stage1, stage1_prediction_mode, stage1_channels, stage1_detail_scale = load_stage1(args.stage1_ckpt, device)
    test_dataset = make_slice_dataset(args.test_t1_dir, args.test_fa_dir, stage1_channels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    stage2 = load_stage2(
        args.stage2_ckpt,
        device,
        condition_mode=condition_mode,
        stage1_channels=stage1_channels,
    )
    include_delta = bool(getattr(stage2, "include_delta_from_initial", True))

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    per_image = {
        "PSNR": [],
        "SSIM": [],
        "MS_SSIM": [],
        "MSE": [],
        "MAE": [],
        "Brain_Masked_MAE": [],
        "WM_Masked_MAE": [],
        "Gradient_Error": [],
        "Sharpness": [],
        "Target_Sharpness": [],
        "Sharpness_Ratio": [],
        "WM_Hist_Wasserstein": [],
    }
    coarse_per_image = {
        "PSNR": [],
        "SSIM": [],
        "MS_SSIM": [],
        "MSE": [],
        "MAE": [],
        "Brain_Masked_MAE": [],
        "WM_Masked_MAE": [],
        "Gradient_Error": [],
        "Sharpness": [],
        "Target_Sharpness": [],
        "Sharpness_Ratio": [],
        "WM_Hist_Wasserstein": [],
    }
    refined_roi_pred_means = []
    refined_roi_target_means = []
    coarse_roi_pred_means = []
    coarse_roi_target_means = []

    output_dir = ensure_output_dir(args.output_dir, args.stage2_ckpt)
    prediction_dir = os.path.join(output_dir, "predictions")
    if args.save_predictions:
        os.makedirs(prediction_dir, exist_ok=True)

    saved_predictions = 0

    print(
        f"Evaluating Stage 2 checkpoint: {args.stage2_ckpt}\n"
        f"Stage 1 checkpoint: {args.stage1_ckpt}\n"
        f"Device: {device}\n"
        f"Condition mode: {condition_mode}\n"
        f"Condition on coarse: {condition_on_coarse}\n"
        f"Eval steps: {eval_steps}\n"
        f"Detail boost: {detail_boost}\n"
        f"Dynamic condition rollout: {dynamic_condition}\n"
        f"Delta-from-initial condition: {include_delta}\n"
        f"Velocity schedule: {velocity_schedule}\n"
        f"Stage 1 input channels: {stage1_channels}\n"
        f"Test slices: {len(test_dataset)}"
    )

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating Stage 2", leave=False)):
        t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
        fa_real = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))

        coarse = predict_stage1_fa(
            stage1,
            t1_img,
            clamp=True,
            prediction_mode=stage1_prediction_mode,
            detail_scale=stage1_detail_scale,
        )
        condition = build_stage2_condition(
            coarse,
            t1_img,
            condition_mode,
            initial_coarse=coarse,
            include_delta_from_initial=include_delta,
        )
        refined = clamp_to_image_range(
            euler_refine(
                stage2,
                coarse,
                num_steps=eval_steps,
                condition=condition,
                detail_boost=detail_boost,
                dynamic_condition=dynamic_condition,
                condition_mode=condition_mode,
                t1_img=t1_img,
                initial_coarse=coarse,
                include_delta_from_initial=include_delta,
            ).float()
        )

        coarse_01 = torch.clamp((coarse + 1.0) / 2.0, 0.0, 1.0)
        refined_01 = torch.clamp((refined + 1.0) / 2.0, 0.0, 1.0)
        real_01 = torch.clamp((fa_real + 1.0) / 2.0, 0.0, 1.0)
        brain_mask = build_brain_mask(
            t1_01=torch.clamp((center_channel(t1_img) + 1.0) / 2.0, 0.0, 1.0),
            fa_01=real_01,
            t1_threshold=args.brain_t1_threshold,
            fa_threshold=args.brain_fa_threshold,
        )
        wm_mask = build_wm_mask(
            fa_01=real_01,
            brain_mask=brain_mask,
            quantile=args.wm_quantile,
            min_threshold=args.wm_min_threshold,
        )

        batch_refined_roi_pred, batch_refined_roi_target = collect_spatial_roi_means(
            refined_01, real_01, brain_mask, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels
        )
        refined_roi_pred_means.extend(batch_refined_roi_pred)
        refined_roi_target_means.extend(batch_refined_roi_target)
        batch_coarse_roi_pred, batch_coarse_roi_target = collect_spatial_roi_means(
            coarse_01, real_01, brain_mask, wm_mask, args.roi_rows, args.roi_cols, args.roi_min_pixels
        )
        coarse_roi_pred_means.extend(batch_coarse_roi_pred)
        coarse_roi_target_means.extend(batch_coarse_roi_target)

        batch_size = refined.shape[0]
        for i in range(batch_size):
            pred_img = refined_01[i : i + 1]
            coarse_img = coarse_01[i : i + 1]
            real_img = real_01[i : i + 1]

            per_image["PSNR"].append(psnr_metric(pred_img, real_img).item())
            per_image["SSIM"].append(ssim_metric(pred_img, real_img).item())
            per_image["MS_SSIM"].append(ms_ssim_metric(pred_img, real_img).item())
            per_image["MSE"].append(((pred_img - real_img) ** 2).mean().item())
            per_image["MAE"].append((pred_img - real_img).abs().mean().item())
            brain_mask_i = brain_mask[i : i + 1]
            wm_mask_i = wm_mask[i : i + 1]
            per_image["Brain_Masked_MAE"].append(masked_mae(pred_img, real_img, brain_mask_i))
            per_image["WM_Masked_MAE"].append(masked_mae(pred_img, real_img, wm_mask_i))
            per_image["Gradient_Error"].append(masked_gradient_error(pred_img, real_img, brain_mask_i))
            pred_sharp = masked_laplacian_variance(pred_img, brain_mask_i)
            target_sharp = masked_laplacian_variance(real_img, brain_mask_i)
            per_image["Sharpness"].append(pred_sharp)
            per_image["Target_Sharpness"].append(target_sharp)
            per_image["Sharpness_Ratio"].append(pred_sharp / max(target_sharp, 1e-8) if np.isfinite(target_sharp) else float("nan"))
            per_image["WM_Hist_Wasserstein"].append(
                histogram_wasserstein_distance(pred_img[wm_mask_i], real_img[wm_mask_i], bins=args.hist_bins)
            )

            coarse_per_image["PSNR"].append(psnr_metric(coarse_img, real_img).item())
            coarse_per_image["SSIM"].append(ssim_metric(coarse_img, real_img).item())
            coarse_per_image["MS_SSIM"].append(ms_ssim_metric(coarse_img, real_img).item())
            coarse_per_image["MSE"].append(((coarse_img - real_img) ** 2).mean().item())
            coarse_per_image["MAE"].append((coarse_img - real_img).abs().mean().item())
            coarse_per_image["Brain_Masked_MAE"].append(masked_mae(coarse_img, real_img, brain_mask_i))
            coarse_per_image["WM_Masked_MAE"].append(masked_mae(coarse_img, real_img, wm_mask_i))
            coarse_per_image["Gradient_Error"].append(masked_gradient_error(coarse_img, real_img, brain_mask_i))
            coarse_sharp = masked_laplacian_variance(coarse_img, brain_mask_i)
            coarse_per_image["Sharpness"].append(coarse_sharp)
            coarse_per_image["Target_Sharpness"].append(target_sharp)
            coarse_per_image["Sharpness_Ratio"].append(coarse_sharp / max(target_sharp, 1e-8) if np.isfinite(target_sharp) else float("nan"))
            coarse_per_image["WM_Hist_Wasserstein"].append(
                histogram_wasserstein_distance(coarse_img[wm_mask_i], real_img[wm_mask_i], bins=args.hist_bins)
            )

            if args.save_predictions and saved_predictions < args.prediction_limit:
                sample_id = batch_idx * args.batch_size + i
                save_prediction_panel(
                    os.path.join(prediction_dir, f"{sample_id:04d}.png"),
                    t1_img[i : i + 1].cpu(),
                    coarse[i : i + 1].cpu(),
                    refined[i : i + 1].cpu(),
                    fa_real[i : i + 1].cpu(),
                )
                saved_predictions += 1

    summary = {
        "stage2_ckpt": os.path.abspath(args.stage2_ckpt),
        "stage1_ckpt": os.path.abspath(args.stage1_ckpt),
        "stage1_channels": stage1_channels,
        "condition_mode": condition_mode,
        "condition_on_coarse": condition_on_coarse,
        "eval_steps": eval_steps,
        "detail_boost": detail_boost,
        "dynamic_condition_rollout": dynamic_condition,
        "include_delta_from_initial": include_delta,
        "velocity_schedule": velocity_schedule,
        "stage1_detail_scale": stage1_detail_scale,
        "test_slices": len(test_dataset),
        "PSNR_mean": float(np.mean(per_image["PSNR"])),
        "PSNR_std": float(np.std(per_image["PSNR"])),
        "SSIM_mean": float(np.mean(per_image["SSIM"])),
        "SSIM_std": float(np.std(per_image["SSIM"])),
        "MS_SSIM_mean": nanmean(per_image["MS_SSIM"]),
        "MS_SSIM_std": nanstd(per_image["MS_SSIM"]),
        "MSE_mean": float(np.mean(per_image["MSE"])),
        "MSE_std": float(np.std(per_image["MSE"])),
        "MAE_mean": float(np.mean(per_image["MAE"])),
        "MAE_std": float(np.std(per_image["MAE"])),
        "Brain_Masked_MAE_mean": nanmean(per_image["Brain_Masked_MAE"]),
        "WM_Masked_MAE_mean": nanmean(per_image["WM_Masked_MAE"]),
        "Gradient_Error_mean": nanmean(per_image["Gradient_Error"]),
        "Sharpness_mean": nanmean(per_image["Sharpness"]),
        "Target_Sharpness_mean": nanmean(per_image["Target_Sharpness"]),
        "Sharpness_Ratio_mean": nanmean(per_image["Sharpness_Ratio"]),
        "WM_Hist_Wasserstein_mean": nanmean(per_image["WM_Hist_Wasserstein"]),
        "ROI_CCC": concordance_correlation_coefficient(
            np.asarray(refined_roi_pred_means, dtype=np.float64),
            np.asarray(refined_roi_target_means, dtype=np.float64),
        ),
        "Coarse_PSNR_mean": float(np.mean(coarse_per_image["PSNR"])),
        "Coarse_SSIM_mean": float(np.mean(coarse_per_image["SSIM"])),
        "Coarse_MS_SSIM_mean": nanmean(coarse_per_image["MS_SSIM"]),
        "Coarse_MSE_mean": float(np.mean(coarse_per_image["MSE"])),
        "Coarse_MAE_mean": float(np.mean(coarse_per_image["MAE"])),
        "Coarse_Brain_Masked_MAE_mean": nanmean(coarse_per_image["Brain_Masked_MAE"]),
        "Coarse_WM_Masked_MAE_mean": nanmean(coarse_per_image["WM_Masked_MAE"]),
        "Coarse_Gradient_Error_mean": nanmean(coarse_per_image["Gradient_Error"]),
        "Coarse_Sharpness_mean": nanmean(coarse_per_image["Sharpness"]),
        "Coarse_Target_Sharpness_mean": nanmean(coarse_per_image["Target_Sharpness"]),
        "Coarse_Sharpness_Ratio_mean": nanmean(coarse_per_image["Sharpness_Ratio"]),
        "Coarse_WM_Hist_Wasserstein_mean": nanmean(coarse_per_image["WM_Hist_Wasserstein"]),
        "Coarse_ROI_CCC": concordance_correlation_coefficient(
            np.asarray(coarse_roi_pred_means, dtype=np.float64),
            np.asarray(coarse_roi_target_means, dtype=np.float64),
        ),
    }

    json_path = os.path.join(output_dir, "stage2_metrics.json")
    csv_path = os.path.join(output_dir, "stage2_metrics.csv")
    save_json(json_path, summary)
    save_csv(csv_path, summary)

    print(
        f"PSNR: {summary['PSNR_mean']:.3f} | SSIM: {summary['SSIM_mean']:.4f} | "
        f"MS-SSIM: {summary['MS_SSIM_mean']:.4f} | "
        f"MSE: {summary['MSE_mean']:.6f} | MAE: {summary['MAE_mean']:.6f}"
    )
    print(
        f"Brain-MAE: {summary['Brain_Masked_MAE_mean']:.4f} | WM-MAE: {summary['WM_Masked_MAE_mean']:.4f} | "
        f"GradErr: {summary['Gradient_Error_mean']:.4f} | SharpRatio: {summary['Sharpness_Ratio_mean']:.4f} | "
        f"ROI-CCC: {summary['ROI_CCC']:.4f} | WM-Wass: {summary['WM_Hist_Wasserstein_mean']:.4f}"
    )
    print(
        f"Coarse -> PSNR: {summary['Coarse_PSNR_mean']:.3f} | SSIM: {summary['Coarse_SSIM_mean']:.4f} | "
        f"MS-SSIM: {summary['Coarse_MS_SSIM_mean']:.4f} | "
        f"MSE: {summary['Coarse_MSE_mean']:.6f} | MAE: {summary['Coarse_MAE_mean']:.6f} | "
        f"Brain-MAE: {summary['Coarse_Brain_Masked_MAE_mean']:.4f} | WM-MAE: {summary['Coarse_WM_Masked_MAE_mean']:.4f} | "
        f"SharpRatio: {summary['Coarse_Sharpness_Ratio_mean']:.4f} | ROI-CCC: {summary['Coarse_ROI_CCC']:.4f}"
    )
    print(f"Saved metrics to: {json_path}")
    print(f"Saved metrics table to: {csv_path}")


if __name__ == "__main__":
    evaluate(parse_args())
