from argparse import Namespace

import torch
import torch.nn as nn

from pmrf_t1fa.train_pmrf_t1fa_stage2 import SSIMLoss
from pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector import (
    SingleSliceCorrector,
    build_loss,
    checkpoint_payload,
    passes_gate,
    selection_score,
)


def test_relaxed_stripe_gate_accepts_high_roi_checkpoint():
    metrics = {
        "delta_psnr": 0.17,
        "delta_ssim": 0.005,
        "delta_wm_l1": 0.010,
        "delta_roi": 0.020,
        "delta_disease_roi": 0.020,
        "delta_stripe": -0.0035,
        "sharp_retention": 1.19,
    }
    strict_args = Namespace(
        best_min_sharp_retention=0.95,
        best_min_delta_disease_roi=0.0,
        best_max_delta_stripe=0.002,
    )
    relaxed_args = Namespace(
        best_min_sharp_retention=0.95,
        best_min_delta_disease_roi=0.0,
        best_max_delta_stripe=0.005,
    )

    assert selection_score(metrics) > 0.0
    assert not passes_gate(metrics, strict_args)
    assert passes_gate(metrics, relaxed_args)


def test_checkpoint_payload_keeps_stage1_export_metadata():
    model = nn.Conv2d(1, 1, kernel_size=1)
    args = Namespace(stage1_ckpt="outputs/stage1/checkpoints/best_stage1.pt", variant="hybrid")
    metrics = {"psnr": 27.6, "ssim": 0.89}

    payload = checkpoint_payload(
        model=model,
        args=args,
        stage1_channels=1,
        prediction_mode="residual",
        detail_scale=0.45,
        epoch=3,
        score=1.23,
        metrics=metrics,
    )

    assert "model" in payload
    assert payload["args"]["variant"] == "hybrid"
    assert payload["stage1_ckpt"] == args.stage1_ckpt
    assert payload["stage1_channels"] == 1
    assert payload["stage1_prediction_mode"] == "residual"
    assert payload["stage1_detail_scale"] == 0.45
    assert payload["epoch"] == 3
    assert torch.equal(payload["model"]["weight"], model.state_dict()["weight"])


def test_multihead_corrector_composes_single_channel_refined_image():
    model = SingleSliceCorrector(in_channels=9, width=8, num_blocks=1, variant="multihead")
    condition = torch.zeros(2, 9, 16, 16)
    raw, log_sigma = model(condition)
    coarse = torch.zeros(2, 1, 16, 16)
    target = torch.zeros(2, 1, 16, 16)
    brain_mask = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    wm_mask = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    roi_map = torch.zeros(2, 1, 16, 16)
    args = Namespace(
        variant="multihead",
        correction_scale=0.2,
        lowpass_kernel=5,
        hp_kernel=3,
        final_l1_weight=0.25,
        final_mse_weight=0.20,
        final_ssim_weight=0.15,
        wm_l1_weight=1.0,
        roi_weight=0.5,
        disease_roi_weight=0.0,
        correction_l1_weight=0.35,
        bounded_weight=0.02,
        sharp_retention_weight=0.5,
        stripe_weight=2.0,
        hf_preserve_weight=1.0,
        uncertainty_weight=0.25,
        atlas_smooth_weight=0.15,
        roi_rows=2,
        roi_cols=2,
        roi_min_pixels=4,
    )

    _, losses, refined = build_loss(
        raw,
        log_sigma,
        coarse,
        target,
        brain_mask,
        wm_mask,
        roi_map,
        None,
        SSIMLoss(),
        args,
    )

    assert raw.shape == (2, 3, 16, 16)
    assert log_sigma is not None and log_sigma.shape == (2, 1, 16, 16)
    assert refined.shape == target.shape
    assert "low_correction_l1" in losses
    assert "high_correction_l1" in losses
    assert "stripe_correction_l1" in losses
