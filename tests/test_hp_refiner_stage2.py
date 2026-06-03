import torch

from pmrf_t1fa.train_pmrf_t1fa_stage2_hp_refiner import (
    HighPassRefinerNet,
    apply_hp_residual_cap,
    build_residual_gate,
    build_hp_refiner_input,
    build_hp_refiner_loss,
    hp_refiner_best_key,
    highpass,
    multiscale_highpass_loss,
    sharpness_floor_loss,
    lowpass,
)


def test_highpass_and_lowpass_recompose_input_shape():
    x = torch.rand((2, 1, 16, 16))

    hp = highpass(x, kernel_size=5)
    lp = lowpass(x, kernel_size=5)

    assert hp.shape == x.shape
    assert lp.shape == x.shape
    assert torch.allclose(hp + lp, x)


def test_hp_refiner_input_uses_t1_stack_coarse_and_edges():
    t1_stack = torch.rand((2, 5, 16, 16))
    coarse = torch.rand((2, 1, 16, 16))

    model_input = build_hp_refiner_input(t1_stack, coarse)

    assert model_input.shape == (2, 8, 16, 16)


def test_highpass_refiner_net_preserves_spatial_shape():
    model = HighPassRefinerNet(in_channels=8, width=8, num_blocks=2)
    x = torch.rand((2, 8, 16, 16))

    y = model(x)

    assert y.shape == (2, 1, 16, 16)


def test_hp_residual_cap_bounds_prediction_amplitude():
    raw = torch.tensor([[[[-10.0, -0.2, 0.2, 10.0]]]])

    capped = apply_hp_residual_cap(raw, residual_scale=0.15)
    unchanged = apply_hp_residual_cap(raw, residual_scale=0.0)

    assert capped.min() >= -0.15
    assert capped.max() <= 0.15
    assert torch.allclose(unchanged, raw)


def test_hp_residual_cap_applies_spatial_gate():
    raw = torch.full((1, 1, 4, 4), 10.0)
    gate = torch.full_like(raw, 0.25)

    capped = apply_hp_residual_cap(raw, residual_scale=0.4, residual_gate=gate)

    assert capped.max() <= 0.1


def test_edge_residual_gate_has_valid_shape_and_range():
    t1_stack = torch.zeros((1, 5, 8, 8))
    coarse = torch.zeros((1, 1, 8, 8))
    t1_stack[:, 2, :, 4:] = 1.0
    coarse[:, :, 4:, :] = 1.0

    gate = build_residual_gate(t1_stack, coarse, mode="edge", gate_min=0.2)
    none_gate = build_residual_gate(t1_stack, coarse, mode="none", gate_min=0.2)

    assert gate.shape == coarse.shape
    assert gate.min() >= 0.2
    assert gate.max() <= 1.0
    assert gate[:, :, :, 3:5].mean() > gate[:, :, :, :2].mean()
    assert none_gate is None


def test_hp_refiner_residual_loss_zero_when_prediction_matches_target_residual():
    coarse = torch.zeros((1, 1, 16, 16))
    target = torch.zeros((1, 1, 16, 16))
    target[:, :, 4:12, 8:] = 0.5
    hp_target = highpass(target, kernel_size=5) - highpass(coarse, kernel_size=5)
    mask = torch.ones_like(coarse)

    losses = build_hp_refiner_loss(
        hp_pred=hp_target,
        coarse=coarse,
        target=target,
        brain_mask=mask,
        wm_mask=mask,
        hp_kernel_size=5,
        lp_kernel_size=9,
        hp_residual_weight=1.0,
        hp_image_weight=1.0,
        wm_hp_weight=1.0,
        multiscale_hp_weight=0.0,
        multiscale_wm_hp_weight=0.0,
        multiscale_hp_kernels=(3, 5, 9),
        wm_final_l1_weight=0.0,
        wm_guard_weight=0.0,
        wm_guard_margin=0.0,
        lowpass_weight=1.0,
        sharpness_floor_weight=0.0,
        sharpness_floor_margin=0.0,
        final_l1_weight=0.0,
        final_ssim_weight=0.0,
        ssim_loss_fn=None,
    )

    assert torch.allclose(losses["hp_residual"], torch.zeros_like(losses["hp_residual"]), atol=1e-6)


def test_hp_refiner_image_losses_zero_when_final_matches_target():
    coarse = torch.zeros((1, 1, 16, 16))
    target = torch.zeros((1, 1, 16, 16))
    target[:, :, 4:12, 8:] = 0.5
    hp_pred = target - coarse
    mask = torch.ones_like(coarse)

    losses = build_hp_refiner_loss(
        hp_pred=hp_pred,
        coarse=coarse,
        target=target,
        brain_mask=mask,
        wm_mask=mask,
        hp_kernel_size=5,
        lp_kernel_size=9,
        hp_residual_weight=0.0,
        hp_image_weight=1.0,
        wm_hp_weight=1.0,
        multiscale_hp_weight=0.0,
        multiscale_wm_hp_weight=0.0,
        multiscale_hp_kernels=(3, 5, 9),
        wm_final_l1_weight=1.0,
        wm_guard_weight=0.0,
        wm_guard_margin=0.0,
        lowpass_weight=0.0,
        sharpness_floor_weight=0.0,
        sharpness_floor_margin=0.0,
        final_l1_weight=1.0,
        final_ssim_weight=0.0,
        ssim_loss_fn=None,
    )

    assert torch.allclose(losses["hp_image"], torch.zeros_like(losses["hp_image"]), atol=1e-6)
    assert torch.allclose(losses["wm_hp"], torch.zeros_like(losses["wm_hp"]), atol=1e-6)
    assert torch.allclose(losses["wm_final_l1"], torch.zeros_like(losses["wm_final_l1"]), atol=1e-6)
    assert torch.allclose(losses["final_l1"], torch.zeros_like(losses["final_l1"]), atol=1e-6)


def test_hp_refiner_wm_guard_penalizes_predictions_worse_than_coarse():
    coarse = torch.zeros((1, 1, 8, 8))
    target = torch.zeros((1, 1, 8, 8))
    bad_hp_pred = torch.full_like(coarse, 0.25)
    good_hp_pred = torch.zeros_like(coarse)
    mask = torch.ones_like(coarse)

    common = dict(
        coarse=coarse,
        target=target,
        brain_mask=mask,
        wm_mask=mask,
        hp_kernel_size=5,
        lp_kernel_size=9,
        hp_residual_weight=0.0,
        hp_image_weight=0.0,
        wm_hp_weight=0.0,
        multiscale_hp_weight=0.0,
        multiscale_wm_hp_weight=0.0,
        multiscale_hp_kernels=(3, 5, 9),
        wm_final_l1_weight=0.0,
        wm_guard_weight=1.0,
        wm_guard_margin=0.0,
        lowpass_weight=0.0,
        sharpness_floor_weight=0.0,
        sharpness_floor_margin=0.0,
        final_l1_weight=0.0,
        final_ssim_weight=0.0,
        ssim_loss_fn=None,
    )
    bad_losses = build_hp_refiner_loss(hp_pred=bad_hp_pred, **common)
    good_losses = build_hp_refiner_loss(hp_pred=good_hp_pred, **common)

    assert bad_losses["wm_guard"] > 0.0
    assert torch.allclose(good_losses["wm_guard"], torch.zeros_like(good_losses["wm_guard"]))


def test_sharpness_floor_penalizes_refined_blurrier_than_coarse():
    coarse = torch.zeros((1, 1, 16, 16))
    coarse[:, :, :, 8:] = 1.0
    target = coarse.clone()
    blurry = lowpass(coarse, kernel_size=7)
    sharper = coarse.clone()

    blurry_loss = sharpness_floor_loss(blurry, coarse, target, margin=0.0)
    sharper_loss = sharpness_floor_loss(sharper, coarse, target, margin=0.0)

    assert blurry_loss > 0.0
    assert torch.allclose(sharper_loss, torch.zeros_like(sharper_loss), atol=1e-6)


def test_multiscale_highpass_loss_zero_when_refined_matches_target():
    target = torch.zeros((1, 1, 16, 16))
    target[:, :, 4:12, 8:] = 0.5
    mask = torch.ones_like(target)

    losses = multiscale_highpass_loss(target, target, mask, kernels=(3, 5, 9))

    assert torch.allclose(losses["multiscale_hp"], torch.zeros_like(losses["multiscale_hp"]), atol=1e-6)
    assert torch.allclose(losses["multiscale_wm_hp"], torch.zeros_like(losses["multiscale_wm_hp"]), atol=1e-6)


def test_multiscale_highpass_loss_penalizes_blurry_refined_image():
    target = torch.zeros((1, 1, 16, 16))
    target[:, :, 4:12, 8:] = 0.5
    blurry = lowpass(target, kernel_size=7)
    mask = torch.ones_like(target)

    losses = multiscale_highpass_loss(blurry, target, mask, kernels=(3, 5, 9))

    assert losses["multiscale_hp"] > 0.0
    assert losses["multiscale_wm_hp"] > 0.0


def test_hp_refiner_best_key_rejects_blurry_or_wm_worse_epoch():
    blurry_metrics = {"delta_sharp": -0.01, "delta_wm_l1": 0.001, "psnr": 30.0}
    wm_worse_metrics = {"delta_sharp": 0.05, "delta_wm_l1": -0.001, "psnr": 30.0}

    assert hp_refiner_best_key(blurry_metrics, min_delta_sharp=0.03, min_delta_wm_l1=0.0) is None
    assert hp_refiner_best_key(wm_worse_metrics, min_delta_sharp=0.03, min_delta_wm_l1=0.0) is None


def test_hp_refiner_best_key_prioritizes_sharpness_then_wm_then_psnr():
    sharper = {"delta_sharp": 0.08, "delta_wm_l1": 0.0001, "psnr": 28.0}
    higher_psnr_but_less_sharp = {"delta_sharp": 0.04, "delta_wm_l1": 0.01, "psnr": 32.0}
    same_sharp_better_wm = {"delta_sharp": 0.08, "delta_wm_l1": 0.001, "psnr": 27.0}

    sharper_key = hp_refiner_best_key(sharper, min_delta_sharp=0.03, min_delta_wm_l1=0.0)
    psnr_key = hp_refiner_best_key(higher_psnr_but_less_sharp, min_delta_sharp=0.03, min_delta_wm_l1=0.0)
    wm_key = hp_refiner_best_key(same_sharp_better_wm, min_delta_sharp=0.03, min_delta_wm_l1=0.0)

    assert sharper_key is not None
    assert psnr_key is not None
    assert wm_key is not None
    assert sharper_key > psnr_key
    assert wm_key > sharper_key
