import torch

from pmrf_t1fa.train_pmrf_t1fa_stage2_hp_refiner import (
    HighPassRefinerNet,
    build_hp_refiner_input,
    build_hp_refiner_loss,
    highpass,
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
        wm_final_l1_weight=0.0,
        wm_guard_weight=0.0,
        wm_guard_margin=0.0,
        lowpass_weight=1.0,
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
        wm_final_l1_weight=1.0,
        wm_guard_weight=0.0,
        wm_guard_margin=0.0,
        lowpass_weight=0.0,
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
        wm_final_l1_weight=0.0,
        wm_guard_weight=1.0,
        wm_guard_margin=0.0,
        lowpass_weight=0.0,
        final_l1_weight=0.0,
        final_ssim_weight=0.0,
        ssim_loss_fn=None,
    )
    bad_losses = build_hp_refiner_loss(hp_pred=bad_hp_pred, **common)
    good_losses = build_hp_refiner_loss(hp_pred=good_hp_pred, **common)

    assert bad_losses["wm_guard"] > 0.0
    assert torch.allclose(good_losses["wm_guard"], torch.zeros_like(good_losses["wm_guard"]))
