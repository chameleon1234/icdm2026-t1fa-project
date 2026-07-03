from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _odd_kernel(kernel_size: int) -> int:
    k = int(kernel_size)
    return k if k % 2 == 1 else k + 1


def lowpass(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    k = _odd_kernel(kernel_size)
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def highpass(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    return x - lowpass(x, kernel_size=kernel_size)


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx.abs(), (0, 1, 0, 0))
    dy = F.pad(dy.abs(), (0, 0, 0, 1))
    return dx + dy


def center_channel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return x[:, x.shape[1] // 2 : x.shape[1] // 2 + 1]


def make_brain_mask(t1: torch.Tensor) -> torch.Tensor:
    t1_01 = ((center_channel(t1).float() + 1.0) * 0.5).clamp(0.0, 1.0)
    return (t1_01 > 0.04).float()


def make_wm_proxy(t1: torch.Tensor) -> torch.Tensor:
    t1_01 = ((center_channel(t1).float() + 1.0) * 0.5).clamp(0.0, 1.0)
    brain = make_brain_mask(t1)
    flat = t1_01.flatten(1)
    thr = torch.quantile(flat, 0.62, dim=1).view(-1, 1, 1, 1)
    wm = torch.sigmoid((t1_01 - thr) * 14.0) * brain
    return wm.clamp(0.0, 1.0)


def grid_roi_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    rows: int = 2,
    cols: int = 3,
    min_pixels: int = 8,
) -> torch.Tensor:
    losses = []
    h, w = pred.shape[-2:]
    for r in range(rows):
        y0, y1 = int(round(r * h / rows)), int(round((r + 1) * h / rows))
        for c in range(cols):
            x0, x1 = int(round(c * w / cols)), int(round((c + 1) * w / cols))
            m = mask[:, :, y0:y1, x0:x1]
            valid = m.sum(dim=(1, 2, 3)) >= float(min_pixels)
            if not bool(valid.any()):
                continue
            p = pred[:, :, y0:y1, x0:x1]
            t = target[:, :, y0:y1, x0:x1]
            denom = m.sum(dim=(1, 2, 3)).clamp_min(1.0)
            p_mean = (p * m).sum(dim=(1, 2, 3)) / denom
            t_mean = (t * m).sum(dim=(1, 2, 3)) / denom
            losses.append((p_mean[valid] - t_mean[valid]).abs().mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).abs() * mask).sum() / denom


def stripe_penalty(pred: torch.Tensor, target: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    x = pred if target is None else pred - target
    if mask is not None:
        x = x * mask
    row_bias = x.mean(dim=3, keepdim=True)
    col_bias = x.mean(dim=2, keepdim=True)
    return row_bias.abs().mean() + col_bias.abs().mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 7) -> None:
        super().__init__()
        self.window_size = _odd_kernel(window_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = self.window_size
        c1 = 0.01**2
        c2 = 0.03**2
        pred01 = ((pred.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        target01 = ((target.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        mu_x = F.avg_pool2d(pred01, k, stride=1, padding=k // 2)
        mu_y = F.avg_pool2d(target01, k, stride=1, padding=k // 2)
        sigma_x = F.avg_pool2d(pred01 * pred01, k, stride=1, padding=k // 2) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target01 * target01, k, stride=1, padding=k // 2) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred01 * target01, k, stride=1, padding=k // 2) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        return 1.0 - ssim.clamp(0.0, 1.0).mean()


class NAFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(self.pw1(self.norm(x)))
        a, b = y.chunk(2, dim=1)
        y = self.pw2(a * torch.sigmoid(b))
        return x + self.beta * y


class SingleBranchNet(nn.Module):
    def __init__(self, width: int = 32, num_blocks: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            *[NAFBlock(width) for _ in range(num_blocks)],
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, t1: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(center_channel(t1)))


class E2EDiTStage1(nn.Module):
    def __init__(self, width: int = 32, num_blocks: int = 6, detail_scale: float = 0.25) -> None:
        super().__init__()
        self.detail_scale = float(detail_scale)
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.base_head = nn.Conv2d(width, 1, 3, padding=1)
        self.detail_head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, t1: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.body(self.stem(center_channel(t1)))
        base_raw = self.base_head(x)
        detail_raw = self.detail_head(x)
        base = torch.tanh(lowpass(base_raw, 9))
        detail = self.detail_scale * torch.tanh(highpass(detail_raw, 5))
        prior = torch.clamp(base + detail, -1.0, 1.0)
        return {"base": base, "detail": detail, "prior": prior}


class E2EDiTCorrector(nn.Module):
    def __init__(self, width: int = 32, num_blocks: int = 6, correction_scale: float = 0.08) -> None:
        super().__init__()
        self.correction_scale = float(correction_scale)
        in_channels = 8
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            *[NAFBlock(width) for _ in range(num_blocks)],
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _condition(self, t1: torch.Tensor, prior: torch.Tensor, wm_proxy: torch.Tensor) -> torch.Tensor:
        t1c = center_channel(t1)
        lp = lowpass(prior, 13)
        hp = highpass(prior, 5)
        edge = gradient_magnitude(t1c)
        mismatch = (edge - hp.abs()).abs()
        b, _, h, w = prior.shape
        yy = torch.linspace(-1.0, 1.0, h, device=prior.device, dtype=prior.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=prior.device, dtype=prior.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([t1c, prior, lp, hp, edge, wm_proxy, mismatch, xx + yy], dim=1)

    def forward(self, t1: torch.Tensor, prior: torch.Tensor, wm_proxy: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if wm_proxy is None:
            wm_proxy = make_wm_proxy(t1)
        brain = make_brain_mask(t1)
        gate = brain * (0.35 + 0.65 * wm_proxy.clamp(0.0, 1.0))
        raw = self.net(self._condition(t1, prior, wm_proxy))
        correction = self.correction_scale * torch.tanh(raw) * gate
        final = torch.clamp(prior + correction, -1.0, 1.0)
        return {"correction": correction, "final": final, "gate": gate}


class E2EDiTFull(nn.Module):
    def __init__(self, width: int = 32, num_blocks: int = 6, correction_scale: float = 0.08) -> None:
        super().__init__()
        self.stage1 = E2EDiTStage1(width=width, num_blocks=num_blocks)
        self.corrector = E2EDiTCorrector(width=width, num_blocks=num_blocks, correction_scale=correction_scale)

    def forward(self, t1: torch.Tensor) -> Dict[str, torch.Tensor]:
        s1 = self.stage1(t1)
        corr = self.corrector(t1, s1["prior"])
        return {**s1, **corr}


def e2edit_m0_loss(pred: torch.Tensor, target: torch.Tensor, wm_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ssim = SSIMLoss()(pred, target)
    l1 = F.l1_loss(pred, target)
    wm = masked_l1(pred, target, wm_mask)
    total = l1 + 0.05 * ssim + 0.2 * wm
    return total, {"recon": l1, "ssim": ssim, "wm": wm}


def e2edit_m1_loss(base: torch.Tensor, detail: torch.Tensor, target: torch.Tensor, wm_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    prior = torch.clamp(base + detail, -1.0, 1.0)
    ssim = SSIMLoss()(prior, target)
    recon = F.l1_loss(prior, target)
    wm = masked_l1(prior, target, wm_mask)
    total = recon + 0.05 * ssim + 0.2 * wm
    return total, {"recon": recon, "ssim": ssim, "wm": wm}


def e2edit_m2_loss(base: torch.Tensor, detail: torch.Tensor, target: torch.Tensor, wm_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    prior = torch.clamp(base + detail, -1.0, 1.0)
    ssim = SSIMLoss()(prior, target)
    recon = F.l1_loss(prior, target)
    base_low = F.l1_loss(lowpass(base, 9), lowpass(target, 9))
    base_hf_suppress = highpass(base, 9).abs().mean()
    detail_hp = F.l1_loss(highpass(prior, 5), highpass(target, 5))
    wm = masked_l1(prior, target, wm_mask)
    roi = grid_roi_loss(prior, target, wm_mask)
    artifact = stripe_penalty(prior, target, make_brain_mask(target))
    total = (
        recon
        + 0.05 * ssim
        + 0.5 * base_low
        + 0.2 * base_hf_suppress
        + 0.3 * detail_hp
        + 0.2 * wm
        + 0.1 * roi
        + 0.1 * artifact
    )
    return total, {
        "recon": recon,
        "ssim": ssim,
        "base_low": base_low,
        "base_hf_suppress": base_hf_suppress,
        "detail_hp": detail_hp,
        "wm": wm,
        "roi": roi,
        "artifact": artifact,
    }


def e2edit_m3_loss(
    final: torch.Tensor,
    prior: torch.Tensor,
    correction: torch.Tensor,
    target: torch.Tensor,
    wm_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ssim = SSIMLoss()(final, target)
    final_l1 = F.l1_loss(final, target)
    wm = masked_l1(final, target, wm_mask)
    roi = grid_roi_loss(final, target, wm_mask)
    artifact = stripe_penalty(final, target, make_brain_mask(target))
    correction_mag = correction.abs().mean()
    preserve_hp = F.l1_loss(highpass(final, 5), highpass(prior.detach(), 5))
    total = (
        final_l1
        + 0.05 * ssim
        + 0.3 * wm
        + 0.15 * roi
        + 0.1 * artifact
        + 0.05 * correction_mag
        + 0.1 * preserve_hp
    )
    return total, {
        "final_l1": final_l1,
        "ssim": ssim,
        "wm": wm,
        "roi": roi,
        "artifact": artifact,
        "correction_mag": correction_mag,
        "preserve_hp": preserve_hp,
    }


def _pearson(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        x = x[mask > 0.5]
        y = y[mask > 0.5]
    else:
        x = x.flatten()
        y = y.flatten()
    if x.numel() < 4:
        return torch.tensor(0.0, device=y.device)
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.sqrt((x * x).mean() * (y * y).mean()).clamp_min(1e-8)
    return ((x * y).mean() / denom).clamp(-1.0, 1.0)


def _lap_var(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype)
    k = k.view(1, 1, 3, 3)
    y = F.conv2d(x.float(), k.float(), padding=1)
    return y.var(dim=(1, 2, 3), unbiased=False)


def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor, wm_mask: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(-1.0, 1.0)
    target = target.detach().float().clamp(-1.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    psnr = 80.0 if mse <= 1e-12 else 20.0 * math.log10(2.0) - 10.0 * math.log10(mse)
    mae = F.l1_loss(pred, target).item()
    wm_mae = float(masked_l1(pred, target, wm_mask).item())
    ssim = float(1.0 - SSIMLoss()(pred, target).item())
    target_sharp = _lap_var(target).mean().clamp_min(1e-8)
    sharp = _lap_var(pred).mean()
    ten_target = gradient_magnitude(target).mean().clamp_min(1e-8)
    ten_pred = gradient_magnitude(pred).mean()
    hp_pred = highpass(pred, 5)
    hp_target = highpass(target, 5)
    return {
        "psnr": float(psnr),
        "ssim": ssim,
        "mae": mae,
        "wm_mae": wm_mae,
        "roi_ccc_proxy": float(1.0 / (1.0 + grid_roi_loss(pred, target, wm_mask).item())),
        "sharp_ratio": float((sharp / target_sharp).item()),
        "tenengrad_ratio": float((ten_pred / ten_target).item()),
        "hf_corr": float(_pearson(hp_pred, hp_target, wm_mask).item()),
        "overbright": float(F.relu(pred - target - 0.10).mean().item()),
    }


def branch_diagnostics(
    base: torch.Tensor,
    detail: torch.Tensor,
    prior: torch.Tensor,
    target: torch.Tensor,
    correction: torch.Tensor | None = None,
) -> Dict[str, float]:
    hp_base = highpass(base.detach().float(), 5)
    hp_detail = highpass(detail.detach().float(), 5)
    hp_prior = highpass(prior.detach().float(), 5)
    hp_target = highpass(target.detach().float(), 5)
    out = {
        "base_hp_abs": float(hp_base.abs().mean().item()),
        "detail_hp_abs": float(hp_detail.abs().mean().item()),
        "highpass_corr": float(_pearson(hp_prior, hp_target).item()),
        "base_low_corr": float(_pearson(lowpass(base.detach().float(), 9), lowpass(target.detach().float(), 9)).item()),
        "detail_hp_corr": float(_pearson(hp_detail, hp_target).item()),
    }
    if correction is not None:
        out["correction_magnitude"] = float(correction.detach().float().abs().mean().item())
        out["correction_hp_change"] = float(highpass(correction.detach().float(), 5).abs().mean().item())
    return out
