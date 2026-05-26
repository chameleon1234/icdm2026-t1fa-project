import math

import torch


def test_image_metrics_are_exact_for_identical_images():
    from src.eval.image_metrics import compute_mae, compute_mse, compute_psnr, compute_ssim

    target = torch.linspace(0.0, 1.0, steps=64 * 64).reshape(1, 1, 64, 64)
    pred = target.clone()

    assert math.isinf(compute_psnr(pred, target))
    assert compute_ssim(pred, target) == 1.0
    assert compute_mse(pred, target) == 0.0
    assert compute_mae(pred, target) == 0.0


def test_image_metrics_and_medical_masks_for_simple_difference():
    from src.eval.image_metrics import (
        build_brain_mask,
        build_wm_mask,
        collect_spatial_roi_means,
        compute_mae,
        compute_mse,
        compute_psnr,
        gradient_error,
        histogram_wasserstein_distance,
        masked_laplacian_variance,
        masked_mae,
        roi_ccc,
    )

    target = torch.zeros(1, 1, 32, 32)
    pred = torch.full_like(target, 0.5)
    t1 = torch.zeros_like(target)
    fa = torch.zeros_like(target)
    t1[:, :, 8:24, 8:24] = 0.4
    fa[:, :, 10:22, 10:22] = 0.8

    brain_mask = build_brain_mask(t1, fa)
    wm_mask = build_wm_mask(fa, brain_mask)

    assert compute_mse(pred, target) == 0.25
    assert compute_mae(pred, target) == 0.5
    assert compute_psnr(pred, target) == 10.0 * math.log10(1.0 / 0.25)
    assert brain_mask.dtype == torch.bool
    assert brain_mask.sum().item() > 0
    assert wm_mask.sum().item() > 0
    assert masked_mae(pred, target, brain_mask) == 0.5
    assert gradient_error(target, target, brain_mask) == 0.0
    assert masked_laplacian_variance(target, brain_mask) == 0.0
    assert histogram_wasserstein_distance(fa[wm_mask], fa[wm_mask]) == 0.0

    roi_pred, roi_target = collect_spatial_roi_means(fa, fa, brain_mask, wm_mask, roi_rows=2, roi_cols=2, min_pixels=1)
    assert roi_pred
    assert roi_ccc(roi_pred, roi_target) == 1.0

