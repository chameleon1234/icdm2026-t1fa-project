import torch


def test_frequency_preserving_composition_only_changes_allowed_low_band():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import (
        compose_frequency_preserving_output,
        frequency_lowpass_mask,
    )

    coarse = torch.randn((2, 1, 32, 32))
    raw_correction = torch.randn_like(coarse)
    refined = compose_frequency_preserving_output(
        coarse,
        raw_correction,
        cutoff=0.12,
        transition=0.04,
    )

    correction_fft = torch.fft.rfft2(refined - coarse)
    low_mask = frequency_lowpass_mask(32, 32, 0.12, 0.04, coarse.device, coarse.dtype)
    forbidden_energy = (correction_fft * (low_mask < 1e-6)).abs().amax()

    assert forbidden_energy < 1e-5


def test_frequency_preserving_composition_is_identity_for_zero_correction():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import compose_frequency_preserving_output

    coarse = torch.randn((1, 1, 24, 24))
    refined = compose_frequency_preserving_output(coarse, torch.zeros_like(coarse), cutoff=0.12, transition=0.04)

    assert torch.allclose(refined, coarse, atol=1e-6)


def test_flow_endpoint_projection_recovers_exact_low_frequency_residual():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import project_flow_endpoint

    source = torch.zeros((1, 1, 8, 8))
    target = torch.full_like(source, 0.2)
    t = torch.tensor([0.4])
    t_view = t.view(-1, 1, 1, 1)
    state = (1.0 - t_view) * source + t_view * target
    velocity = target - source

    endpoint = project_flow_endpoint(state, velocity, t)

    assert torch.allclose(endpoint, target, atol=1e-6)


def test_direct_and_flow_correctors_return_single_channel_corrections():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import FidelityCorrector

    t1 = torch.randn((2, 5, 32, 32))
    coarse = torch.randn((2, 1, 32, 32))
    state = torch.zeros_like(coarse)
    t = torch.tensor([0.2, 0.7])

    direct = FidelityCorrector(stage1_channels=5, mode="direct", width=8, num_blocks=2)
    flow = FidelityCorrector(stage1_channels=5, mode="flow", width=8, num_blocks=2)

    direct_output = direct(t1, coarse)
    flow_output = flow(t1, coarse, state=state, t=t)

    assert direct_output.shape == coarse.shape
    assert flow_output.shape == coarse.shape
    assert torch.allclose(direct_output, torch.zeros_like(coarse), atol=1e-6)
    assert torch.allclose(flow_output, torch.zeros_like(coarse), atol=1e-6)


def test_checkpoint_gate_requires_sharpness_wm_and_roi_constraints():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import passes_checkpoint_gate

    good = {
        "sharp_retention": 0.99,
        "delta_wm_l1": 0.001,
        "delta_roi": 0.002,
        "delta_psnr": 0.01,
        "delta_ssim": 0.0,
    }

    assert passes_checkpoint_gate(good, 0.97, 0.0, 0.0, 0.0, -0.002)
    assert not passes_checkpoint_gate({**good, "sharp_retention": 0.90}, 0.97, 0.0, 0.0, 0.0, -0.002)
    assert not passes_checkpoint_gate({**good, "delta_wm_l1": -0.001}, 0.97, 0.0, 0.0, 0.0, -0.002)
    assert not passes_checkpoint_gate({**good, "delta_roi": -0.001}, 0.97, 0.0, 0.0, 0.0, -0.002)
    assert passes_checkpoint_gate({**good, "delta_psnr": -0.02}, 0.97, 0.0, 0.0, -0.03, -0.002)
    assert not passes_checkpoint_gate({**good, "delta_psnr": -0.04}, 0.97, 0.0, 0.0, -0.03, -0.002)


def test_training_loss_uses_same_single_frequency_projection_as_export():
    from pmrf_t1fa.train_pmrf_t1fa_stage2 import SSIMLoss
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import (
        build_fidelity_corrector_loss,
        compose_frequency_preserving_output,
    )

    coarse = torch.randn((1, 1, 16, 16))
    raw_correction = torch.randn_like(coarse)
    target = torch.randn_like(coarse)
    mask = torch.ones_like(coarse)
    expected = compose_frequency_preserving_output(coarse, raw_correction, 0.12, 0.04)

    losses = build_fidelity_corrector_loss(
        raw_correction=raw_correction,
        coarse=coarse,
        target=target,
        brain_mask=mask,
        wm_mask=mask,
        ssim_loss_fn=SSIMLoss(),
        cutoff=0.12,
        transition=0.04,
        correction_l1_weight=1.0,
        final_l1_weight=1.0,
        final_mse_weight=1.0,
        final_ssim_weight=0.0,
        wm_l1_weight=0.0,
        roi_weight=0.0,
        residual_magnitude_weight=0.0,
        hf_preserve_weight=0.0,
        background_weight=0.0,
        roi_rows=2,
        roi_cols=2,
        roi_min_pixels=1,
    )

    assert torch.allclose(losses["refined"], expected, atol=1e-6)


def test_frequency_preservation_error_ignores_allowed_transition_and_detects_forbidden_change():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import (
        compose_frequency_preserving_output,
        frequency_preservation_error,
    )

    coarse = torch.randn((1, 1, 32, 32))
    allowed = compose_frequency_preserving_output(coarse, torch.randn_like(coarse), 0.12, 0.04)
    checkerboard = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(16, 16).view(1, 1, 32, 32)

    assert frequency_preservation_error(allowed, coarse, 0.12, 0.04) < 1e-5
    assert frequency_preservation_error(coarse + checkerboard, coarse, 0.12, 0.04) > 0.1


def test_resume_validation_rejects_missing_or_changed_required_configuration():
    import pytest

    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import validate_resume_configuration

    current = {
        "stage1_ckpt": "sharp.pt",
        "corrector_mode": "flow",
        "width": 48,
        "frequency_cutoff": 0.12,
    }

    with pytest.raises(ValueError, match="missing required key"):
        validate_resume_configuration(current, {"stage1_ckpt": "sharp.pt"}, tuple(current))
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_resume_configuration(current, {**current, "frequency_cutoff": 0.2}, tuple(current))


def test_metric_restore_preset_raises_fidelity_weights_and_keeps_sharpness_gate():
    import argparse

    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import apply_training_preset

    args = argparse.Namespace(
        training_preset="metric_restore",
        corrector_mode="flow",
        frequency_cutoff=0.12,
        frequency_transition=0.04,
        correction_l1_weight=1.0,
        final_l1_weight=0.4,
        final_mse_weight=0.3,
        final_ssim_weight=0.1,
        wm_l1_weight=1.0,
        roi_weight=0.3,
        residual_magnitude_weight=0.05,
        hf_preserve_weight=2.0,
        best_min_sharp_retention=0.90,
        best_min_delta_psnr=-0.03,
        best_min_delta_ssim=-0.002,
        eval_steps=4,
    )

    updated = apply_training_preset(args)

    assert updated.frequency_cutoff >= 0.16
    assert updated.frequency_transition >= 0.06
    assert updated.correction_l1_weight >= 1.2
    assert updated.final_l1_weight >= 0.8
    assert updated.final_mse_weight >= 0.8
    assert updated.final_ssim_weight >= 0.5
    assert updated.residual_magnitude_weight <= 0.03
    assert updated.hf_preserve_weight >= 2.5
    assert updated.best_min_sharp_retention >= 0.97
    assert updated.best_min_delta_psnr >= 0.0
    assert updated.best_min_delta_ssim >= 0.0
    assert updated.eval_steps >= 6


def test_init_checkpoint_loads_model_weights_without_optimizer_or_arg_validation(tmp_path):
    import torch

    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import (
        FidelityCorrector,
        load_init_checkpoint,
    )

    source = FidelityCorrector(stage1_channels=5, mode="flow", width=8, num_blocks=1)
    target = FidelityCorrector(stage1_channels=5, mode="flow", width=8, num_blocks=1)
    for parameter in source.parameters():
        torch.nn.init.constant_(parameter, 0.123)
    ckpt_path = tmp_path / "init.pt"
    torch.save({"model": source.state_dict(), "optimizer": {"ignored": True}, "args": {"old": 1}}, ckpt_path)

    load_init_checkpoint(target, ckpt_path)

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(source_param, target_param)
