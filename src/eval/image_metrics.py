import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _as_bchw(x: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(x).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D image tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _single_channel(x: torch.Tensor) -> torch.Tensor:
    x = _as_bchw(x)
    if x.shape[1] == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def compute_mse(pred_01: torch.Tensor, target_01: torch.Tensor) -> float:
    pred = _as_bchw(pred_01)
    target = _as_bchw(target_01).to(pred.device)
    return float(torch.mean((pred - target) ** 2).item())


def compute_mae(pred_01: torch.Tensor, target_01: torch.Tensor) -> float:
    pred = _as_bchw(pred_01)
    target = _as_bchw(target_01).to(pred.device)
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_psnr(pred_01: torch.Tensor, target_01: torch.Tensor, data_range: float = 1.0) -> float:
    mse = compute_mse(pred_01, target_01)
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range**2) / mse))


def masked_mse(pred_01: torch.Tensor, target_01: torch.Tensor, mask: torch.Tensor) -> float:
    pred = _single_channel(pred_01)
    target = _single_channel(target_01).to(pred.device)
    mask_tensor = _single_channel(mask).bool().to(pred.device)
    vals = (pred - target)[mask_tensor]
    return float("nan") if vals.numel() == 0 else float(torch.mean(vals * vals).item())


def masked_psnr(
    pred_01: torch.Tensor,
    target_01: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    mse = masked_mse(pred_01, target_01, mask)
    if math.isnan(mse):
        return float("nan")
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10((data_range**2) / mse))


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def compute_ssim(
    pred_01: torch.Tensor,
    target_01: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    pred = _as_bchw(pred_01)
    target = _as_bchw(target_01).to(pred.device)
    if torch.equal(pred, target):
        return 1.0

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_cross

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    return float(torch.clamp((numerator / denominator.clamp_min(1e-12)).mean(), min=-1.0, max=1.0).item())


def sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    image = _single_channel(x)
    kernel_x = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = image.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    return F.conv2d(image, kernel_x, padding=1), F.conv2d(image, kernel_y, padding=1)


def laplacian_response(x: torch.Tensor) -> torch.Tensor:
    image = _single_channel(x)
    kernel = image.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
    return F.conv2d(image, kernel, padding=1)


def morphological_close(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    mask = _single_channel(mask).float()
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - dilated, kernel_size=kernel_size, stride=1, padding=pad)
    return eroded > 0.5


def build_brain_mask(
    t1_01: torch.Tensor,
    fa_01: torch.Tensor,
    t1_threshold: float = 0.05,
    fa_threshold: float = 0.02,
) -> torch.Tensor:
    t1 = _single_channel(t1_01)
    fa = _single_channel(fa_01).to(t1.device)
    mask = (t1 > t1_threshold) | (fa > fa_threshold)
    return morphological_close(mask.float()).bool()


def build_wm_mask(
    fa_01: torch.Tensor,
    brain_mask: torch.Tensor,
    quantile: float = 0.65,
    min_threshold: float = 0.20,
) -> torch.Tensor:
    fa = _single_channel(fa_01)
    brain = _single_channel(brain_mask).bool().to(fa.device)
    masks = []
    for i in range(fa.shape[0]):
        fa_vals = fa[i][brain[i]]
        if fa_vals.numel() == 0:
            masks.append(torch.zeros_like(brain[i], dtype=torch.bool))
            continue
        threshold = max(min_threshold, float(torch.quantile(fa_vals, quantile).item()))
        wm = brain[i] & (fa[i] >= threshold)
        masks.append(morphological_close(wm.float().unsqueeze(0)).squeeze(0).bool())
    return torch.stack(masks, dim=0)


def masked_mae(pred_01: torch.Tensor, target_01: torch.Tensor, mask: torch.Tensor) -> float:
    pred = _single_channel(pred_01)
    target = _single_channel(target_01).to(pred.device)
    mask_tensor = _single_channel(mask).bool().to(pred.device)
    vals = torch.abs(pred - target)[mask_tensor]
    return float("nan") if vals.numel() == 0 else float(vals.mean().item())


def gradient_error(pred_01: torch.Tensor, target_01: torch.Tensor, mask: torch.Tensor) -> float:
    pred = _single_channel(pred_01)
    target = _single_channel(target_01).to(pred.device)
    mask_tensor = _single_channel(mask).bool().to(pred.device)
    pred_gx, pred_gy = sobel_gradients(pred)
    target_gx, target_gy = sobel_gradients(target)
    vals_x = torch.abs(pred_gx - target_gx)[mask_tensor]
    vals_y = torch.abs(pred_gy - target_gy)[mask_tensor]
    if vals_x.numel() == 0 or vals_y.numel() == 0:
        return float("nan")
    return float(0.5 * (vals_x.mean().item() + vals_y.mean().item()))


def masked_laplacian_variance(x_01: torch.Tensor, mask: torch.Tensor) -> float:
    response = laplacian_response(x_01)
    mask_tensor = _single_channel(mask).bool().to(response.device)
    vals = response[mask_tensor]
    return float("nan") if vals.numel() == 0 else float(vals.var(unbiased=False).item())


def histogram_wasserstein_distance(pred_values: torch.Tensor, target_values: torch.Tensor, bins: int = 64) -> float:
    pred = torch.as_tensor(pred_values).float()
    target = torch.as_tensor(target_values).float().to(pred.device)
    if pred.numel() == 0 or target.numel() == 0:
        return float("nan")
    pred_hist = torch.histc(pred, bins=bins, min=0.0, max=1.0)
    target_hist = torch.histc(target, bins=bins, min=0.0, max=1.0)
    pred_hist = pred_hist / pred_hist.sum().clamp_min(1e-8)
    target_hist = target_hist / target_hist.sum().clamp_min(1e-8)
    pred_cdf = torch.cumsum(pred_hist, dim=0)
    target_cdf = torch.cumsum(target_hist, dim=0)
    return float((torch.abs(pred_cdf - target_cdf).sum() * (1.0 / bins)).item())


def roi_ccc(pred_roi_values: Iterable[float], target_roi_values: Iterable[float]) -> float:
    pred = np.asarray(list(pred_roi_values), dtype=np.float64)
    target = np.asarray(list(target_roi_values), dtype=np.float64)
    if pred.size < 2 or target.size < 2:
        return float("nan")
    if np.allclose(pred, target, equal_nan=False):
        return 1.0
    mean_pred = float(np.mean(pred))
    mean_target = float(np.mean(target))
    var_pred = float(np.var(pred))
    var_target = float(np.var(target))
    cov = float(np.mean((pred - mean_pred) * (target - mean_target)))
    denom = var_pred + var_target + (mean_pred - mean_target) ** 2
    return float("nan") if denom <= 1e-12 else float((2.0 * cov) / denom)


def collect_spatial_roi_means(
    pred_01: torch.Tensor,
    target_01: torch.Tensor,
    brain_mask: torch.Tensor,
    wm_mask: torch.Tensor,
    roi_rows: int = 2,
    roi_cols: int = 3,
    min_pixels: int = 32,
) -> tuple[list[float], list[float]]:
    pred = _single_channel(pred_01)
    target = _single_channel(target_01).to(pred.device)
    brain = _single_channel(brain_mask).bool().to(pred.device)
    wm = _single_channel(wm_mask).bool().to(pred.device)
    pred_means: list[float] = []
    target_means: list[float] = []

    for i in range(pred.shape[0]):
        coords = torch.nonzero(brain[i, 0], as_tuple=False)
        if coords.numel() == 0:
            continue
        y_min = int(coords[:, 0].min().item())
        y_max = int(coords[:, 0].max().item()) + 1
        x_min = int(coords[:, 1].min().item())
        x_max = int(coords[:, 1].max().item()) + 1
        y_edges = torch.linspace(y_min, y_max, steps=roi_rows + 1, device=pred.device).round().long()
        x_edges = torch.linspace(x_min, x_max, steps=roi_cols + 1, device=pred.device).round().long()
        for row_idx in range(roi_rows):
            for col_idx in range(roi_cols):
                y0, y1 = int(y_edges[row_idx].item()), int(y_edges[row_idx + 1].item())
                x0, x1 = int(x_edges[col_idx].item()), int(x_edges[col_idx + 1].item())
                if y1 <= y0 or x1 <= x0:
                    continue
                region_mask = wm[i, 0, y0:y1, x0:x1]
                if int(region_mask.sum().item()) < min_pixels:
                    continue
                pred_region = pred[i, 0, y0:y1, x0:x1][region_mask]
                target_region = target[i, 0, y0:y1, x0:x1][region_mask]
                pred_means.append(float(pred_region.mean().item()))
                target_means.append(float(target_region.mean().item()))
    return pred_means, target_means
