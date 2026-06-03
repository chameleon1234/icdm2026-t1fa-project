import torch


def test_residual_flow_condition_uses_uncertainty_without_target_leakage():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_residual_flow import (
        ResidualFlowStage2,
        build_residual_flow_condition,
    )

    t1_stack = torch.rand((2, 5, 16, 16))
    coarse = torch.rand((2, 1, 16, 16))
    model = ResidualFlowStage2(stage1_channels=5, width=8, num_blocks=2)

    sigma = model.predict_sigma(t1_stack, coarse)
    condition = build_residual_flow_condition(t1_stack, coarse, sigma)

    assert sigma.shape == coarse.shape
    assert sigma.min() >= model.sigma_min
    assert sigma.max() <= model.sigma_max
    assert condition.shape == (2, 9, 16, 16)

    changed_target = torch.rand_like(coarse)
    condition_again = build_residual_flow_condition(t1_stack, coarse, sigma)

    assert torch.allclose(condition, condition_again)
    assert changed_target.shape == coarse.shape


def test_project_residual_endpoint_matches_target_when_velocity_is_exact():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_residual_flow import project_residual_endpoint

    r0 = torch.zeros((1, 1, 8, 8))
    target_residual = torch.ones_like(r0) * 0.4
    t = torch.full((1,), 0.25)
    rt = (1.0 - t.view(1, 1, 1, 1)) * r0 + t.view(1, 1, 1, 1) * target_residual
    exact_velocity = target_residual - r0

    endpoint = project_residual_endpoint(rt, exact_velocity, t)

    assert torch.allclose(endpoint, target_residual, atol=1e-6)


def test_residual_flow_loss_zero_for_exact_velocity_and_endpoint():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_residual_flow import build_residual_flow_loss

    coarse = torch.zeros((1, 1, 8, 8))
    target = torch.ones_like(coarse) * 0.25
    r0 = torch.zeros_like(coarse)
    t = torch.full((1,), 0.5)
    rt = t.view(1, 1, 1, 1) * (target - coarse)
    exact_velocity = target - coarse
    mask = torch.ones_like(coarse)

    losses = build_residual_flow_loss(
        velocity_pred=exact_velocity,
        rt=rt,
        r0=r0,
        coarse=coarse,
        target=target,
        sigma=torch.ones_like(coarse) * 0.25,
        t=t,
        brain_mask=mask,
        wm_mask=mask,
        velocity_weight=1.0,
        residual_l1_weight=1.0,
        image_l1_weight=1.0,
        wm_residual_weight=1.0,
        residual_energy_weight=0.0,
        uncertainty_nll_weight=0.0,
    )

    assert torch.allclose(losses["velocity"], torch.zeros_like(losses["velocity"]), atol=1e-6)
    assert torch.allclose(losses["residual_l1"], torch.zeros_like(losses["residual_l1"]), atol=1e-6)
    assert torch.allclose(losses["image_l1"], torch.zeros_like(losses["image_l1"]), atol=1e-6)


def test_residual_flow_euler_sampler_integrates_constant_velocity():
    from pmrf_t1fa.train_pmrf_t1fa_stage2_residual_flow import euler_sample_residual_flow

    class ConstantVelocity(torch.nn.Module):
        sigma_min = 0.01
        sigma_max = 0.50

        def forward(self, residual, t, t1_stack, coarse, sigma):
            return torch.ones_like(residual) * 0.2

    t1_stack = torch.zeros((1, 5, 8, 8))
    coarse = torch.zeros((1, 1, 8, 8))
    sigma = torch.ones_like(coarse) * 0.1

    pred = euler_sample_residual_flow(
        model=ConstantVelocity(),
        t1_stack=t1_stack,
        coarse=coarse,
        sigma=sigma,
        steps=4,
        noise_scale=0.0,
    )

    assert torch.allclose(pred, torch.ones_like(coarse) * 0.2, atol=1e-6)
