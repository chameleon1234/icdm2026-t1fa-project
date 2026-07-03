import torch


def test_stage1_returns_decoupled_bounded_outputs():
    from pmrf_t1fa.e2edit import E2EDiTStage1

    model = E2EDiTStage1(width=8, num_blocks=2)
    t1 = torch.randn(2, 1, 32, 32)

    out = model(t1)

    assert out["base"].shape == (2, 1, 32, 32)
    assert out["detail"].shape == (2, 1, 32, 32)
    assert out["prior"].shape == (2, 1, 32, 32)
    assert out["prior"].min() >= -1.0
    assert out["prior"].max() <= 1.0


def test_corrector_returns_small_bounded_correction():
    from pmrf_t1fa.e2edit import E2EDiTCorrector, make_wm_proxy

    model = E2EDiTCorrector(width=8, num_blocks=2, correction_scale=0.08)
    t1 = torch.randn(2, 1, 32, 32)
    prior = torch.tanh(torch.randn(2, 1, 32, 32))
    wm = make_wm_proxy(t1)

    out = model(t1, prior, wm)

    assert out["correction"].shape == (2, 1, 32, 32)
    assert out["final"].shape == (2, 1, 32, 32)
    assert float(out["correction"].abs().max()) <= 0.0801
    assert out["final"].min() >= -1.0
    assert out["final"].max() <= 1.0


def test_frequency_losses_penalize_base_high_frequency():
    from pmrf_t1fa.e2edit import highpass, e2edit_m2_loss

    target = torch.zeros(1, 1, 32, 32)
    noisy_base = torch.randn_like(target) * 0.2
    clean_base = torch.zeros_like(target)
    detail = torch.zeros_like(target)
    wm = torch.ones_like(target)

    noisy_loss, noisy_parts = e2edit_m2_loss(noisy_base, detail, target, wm)
    clean_loss, clean_parts = e2edit_m2_loss(clean_base, detail, target, wm)

    assert noisy_parts["base_hf_suppress"] > clean_parts["base_hf_suppress"]
    assert noisy_loss > clean_loss
    assert highpass(noisy_base).abs().mean() > highpass(clean_base).abs().mean()


def test_metrics_report_sharpness_and_hf_correlation():
    from pmrf_t1fa.e2edit import compute_batch_metrics

    target = torch.linspace(-1, 1, 32 * 32).view(1, 1, 32, 32)
    pred = target.clone()
    wm = torch.ones_like(target)

    metrics = compute_batch_metrics(pred, target, wm)

    assert metrics["psnr"] > 70
    assert metrics["mae"] == 0.0
    assert 0.99 <= metrics["sharp_ratio"] <= 1.01
    assert metrics["hf_corr"] > 0.99
