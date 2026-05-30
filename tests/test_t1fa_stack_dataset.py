from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _write_gray_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 6), value, dtype=np.uint8), mode="L").save(path)


def _to_norm(value: int) -> float:
    return value / 127.5 - 1.0


def test_stack_dataset_returns_center_target_with_clamped_context(tmp_path):
    from src.data.t1fa_stack_dataset import T1FAStackDataset

    t1_dir = tmp_path / "t1"
    fa_dir = tmp_path / "fa"
    for z, value in enumerate([10, 20, 30]):
        name = f"sub-001_z{z:03d}.png"
        _write_gray_png(t1_dir / name, value)
        _write_gray_png(fa_dir / name, 100 + z)

    dataset = T1FAStackDataset(t1_dir, fa_dir, context_slices=3, target_size=(4, 4))

    edge = dataset[0]
    middle = dataset[1]

    assert edge["fname"] == "sub-001_z000.png"
    assert edge["t1_slice"].shape == torch.Size([3, 4, 4])
    assert edge["fa_slice"].shape == torch.Size([1, 4, 4])
    assert [round(float(edge["t1_slice"][i, 0, 0]), 4) for i in range(3)] == [
        round(_to_norm(10), 4),
        round(_to_norm(10), 4),
        round(_to_norm(20), 4),
    ]
    assert [round(float(middle["t1_slice"][i, 0, 0]), 4) for i in range(3)] == [
        round(_to_norm(10), 4),
        round(_to_norm(20), 4),
        round(_to_norm(30), 4),
    ]


def test_select_resume_checkpoint_prefers_healthy_latest(tmp_path):
    from pmrf_t1fa.checkpointing import select_resume_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    latest = ckpt_dir / "latest_stage1.pt"
    healthy = ckpt_dir / "healthy_latest_stage1.pt"
    latest.write_bytes(b"latest")
    healthy.write_bytes(b"healthy")

    assert select_resume_checkpoint(ckpt_dir, "stage1") == str(healthy)
    assert select_resume_checkpoint(ckpt_dir, "stage1", prefer_healthy=False) == str(latest)


def test_select_resume_checkpoint_uses_explicit_path_and_errors_if_missing(tmp_path):
    import pytest

    from pmrf_t1fa.checkpointing import select_resume_checkpoint

    explicit = tmp_path / "epoch_010.pt"
    explicit.write_bytes(b"checkpoint")

    assert select_resume_checkpoint(tmp_path, "stage2", explicit_path=explicit, auto_resume=False) == str(explicit)
    with pytest.raises(FileNotFoundError):
        select_resume_checkpoint(tmp_path, "stage2", explicit_path=tmp_path / "missing.pt")


def test_validate_resume_args_rejects_incompatible_stage1_checkpoint():
    import pytest
    from types import SimpleNamespace

    from pmrf_t1fa.checkpointing import validate_resume_args

    checkpoint = {"args": {"context_slices": 3, "stage1_model_variant": "detail"}}
    args = SimpleNamespace(context_slices=5, stage1_model_variant="detail")

    with pytest.raises(ValueError, match="context_slices"):
        validate_resume_args(checkpoint, args, ["context_slices", "stage1_model_variant"], "bad.pt")


def test_validate_resume_args_accepts_equivalent_checkpoint_paths():
    from types import SimpleNamespace

    from pmrf_t1fa.checkpointing import validate_resume_args

    checkpoint = {"args": {"stage1_ckpt": "outputs/foo/checkpoints/best_stage1.pt"}}
    args = SimpleNamespace(stage1_ckpt="outputs\\foo\\checkpoints\\best_stage1.pt")

    validate_resume_args(checkpoint, args, ["stage1_ckpt"], "resume.pt")


def test_predict_stage1_residual_uses_center_slice_for_stacked_input():
    from pmrf_t1fa.models.pmrf_t1fa import predict_stage1_fa

    class ZeroResidual(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

    t1_stack = torch.tensor([[[[-0.5]], [[0.25]], [[0.75]]]])

    coarse = predict_stage1_fa(ZeroResidual(), t1_stack, prediction_mode="residual")

    assert coarse.shape == torch.Size([1, 1, 1, 1])
    assert coarse.item() == 0.25


def test_predict_stage1_detail_head_adds_highpass_detail_to_residual():
    from pmrf_t1fa.models.pmrf_t1fa import predict_stage1_fa

    class FixedDetail(torch.nn.Module):
        def forward(self, x):
            base = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            detail = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
            detail[:, :, 2, 2] = 1.0
            return base, detail

    t1_stack = torch.zeros((1, 3, 5, 5))

    no_detail = predict_stage1_fa(FixedDetail(), t1_stack, prediction_mode="residual", detail_scale=0.0)
    with_detail = predict_stage1_fa(FixedDetail(), t1_stack, prediction_mode="residual", detail_scale=1.0)

    assert torch.allclose(no_detail, torch.zeros_like(no_detail))
    assert with_detail[:, :, 2, 2].item() > 0.0
    assert abs(with_detail.sum().item()) < 1e-5


def test_detail_stage1_net_returns_base_and_detail_heads():
    from pmrf_t1fa.models.pmrf_t1fa import DetailStage1Net

    model = DetailStage1Net(in_channels=3, out_channels=1, dim=8)
    x = torch.zeros((1, 3, 16, 16))

    base, detail = model(x)

    assert base.shape == torch.Size([1, 1, 16, 16])
    assert detail.shape == torch.Size([1, 1, 16, 16])


def test_stage1_checkpoint_channel_inference_supports_export_and_stage2_eval(tmp_path):
    from pmrf_t1fa.models.pmrf_t1fa import Stage1Net, infer_stage1_in_channels
    from scripts.export_pm_dirf_predictions import load_stage1 as load_export_stage1
    from pmrf_t1fa.evaluate_pmrf_t1fa_stage2 import load_stage1 as load_eval_stage1

    checkpoint_path = tmp_path / "stage1_3slice.pt"
    torch.save(
        {
            "model": Stage1Net(in_channels=3, out_channels=1).state_dict(),
            "args": {"context_slices": 3, "stage1_prediction_mode": "residual"},
        },
        checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert infer_stage1_in_channels(checkpoint) == 3

    export_model, export_mode, export_channels, export_detail_scale = load_export_stage1(checkpoint_path, torch.device("cpu"))
    eval_model, eval_mode, eval_channels, eval_detail_scale = load_eval_stage1(checkpoint_path, torch.device("cpu"))

    assert export_model.patch_embed.weight.shape[1] == 3
    assert eval_model.patch_embed.weight.shape[1] == 3
    assert export_mode == eval_mode == "residual"
    assert export_channels == eval_channels == 3
    assert export_detail_scale == 0.0
    assert eval_detail_scale == 0.0


def test_stage1_detail_checkpoint_loader_preserves_variant_and_detail_scale(tmp_path):
    from pmrf_t1fa.models.pmrf_t1fa import DetailStage1Net
    from scripts.export_pm_dirf_predictions import load_stage1 as load_export_stage1

    checkpoint_path = tmp_path / "stage1_detail_3slice.pt"
    torch.save(
        {
            "model": DetailStage1Net(in_channels=3, out_channels=1).state_dict(),
            "args": {
                "context_slices": 3,
                "stage1_prediction_mode": "residual",
                "stage1_model_variant": "detail",
                "stage1_detail_scale": 0.75,
            },
        },
        checkpoint_path,
    )

    model, mode, channels, detail_scale = load_export_stage1(checkpoint_path, torch.device("cpu"))

    assert hasattr(model, "output_detail")
    assert mode == "residual"
    assert channels == 3
    assert detail_scale == 0.75


def test_wm_paired_score_prefers_white_matter_and_roi_fidelity():
    from types import SimpleNamespace

    from pmrf_t1fa.train_pmrf_t1fa_stage2 import compute_wm_paired_score

    args = SimpleNamespace(
        paired_psnr_weight=1.0,
        paired_ssim_weight=10.0,
        paired_mse_weight=200.0,
        paired_mae_weight=10.0,
        paired_brain_mae_weight=4.0,
        paired_wm_mae_weight=8.0,
        paired_grad_weight=0.8,
        paired_roi_weight=4.0,
        paired_sharp_weight=2.0,
    )
    base = {
        "psnr": 28.0,
        "ssim": 0.90,
        "mse_proxy": 0.0017,
        "l1": 0.018,
        "brain_l1": 0.052,
        "wm_l1": 0.060,
        "grad": 0.112,
        "roi": 0.020,
        "sharp_ratio": 0.70,
    }
    worse_wm = dict(base, wm_l1=0.090, roi=0.050, sharp_ratio=0.50)

    assert compute_wm_paired_score(base, args) > compute_wm_paired_score(worse_wm, args)


def test_stage2_coarse_t1_condition_uses_coarse_and_full_t1_stack():
    from pmrf_t1fa.models.pmrf_t1fa import (
        build_stage2_condition,
        stage2_condition_channels,
    )

    coarse = torch.full((2, 1, 4, 4), 0.25)
    t1_stack = torch.stack(
        [
            torch.full((4, 4), -0.5),
            torch.full((4, 4), 0.0),
            torch.full((4, 4), 0.5),
        ],
        dim=0,
    ).unsqueeze(0).repeat(2, 1, 1, 1)

    condition = build_stage2_condition(coarse, t1_stack, "coarse_t1")

    assert stage2_condition_channels("coarse_t1", stage1_channels=3) == 4
    assert condition.shape == torch.Size([2, 4, 4, 4])
    assert torch.allclose(condition[:, :1], coarse)
    assert torch.allclose(condition[:, 1:], t1_stack)


def test_stage2_condition_mode_defaults_keep_old_coarse_checkpoints_compatible():
    from pmrf_t1fa.models.pmrf_t1fa import infer_stage2_condition_mode

    assert infer_stage2_condition_mode({"condition_on_coarse": True}) == "coarse"
    assert infer_stage2_condition_mode({"condition_on_coarse": False}) == "none"
    assert infer_stage2_condition_mode({"condition_mode": "coarse_t1", "condition_on_coarse": True}) == "coarse_t1"


def test_stage2_coarse_t1_edge_condition_keeps_structure_and_detail_guides():
    from pmrf_t1fa.models.pmrf_t1fa import (
        build_stage2_condition,
        stage2_condition_channels,
    )

    coarse = torch.zeros((1, 1, 4, 4))
    coarse[:, :, 1:3, 1:3] = 0.5
    t1_stack = torch.zeros((1, 3, 4, 4))
    t1_stack[:, 0] = -0.25
    t1_stack[:, 1, 1:3, 1:3] = 0.25
    t1_stack[:, 2, :, 2:] = 0.75

    condition = build_stage2_condition(coarse, t1_stack, "coarse_t1_edge")

    assert stage2_condition_channels("coarse_t1_edge", stage1_channels=3) == 10
    assert condition.shape == torch.Size([1, 10, 4, 4])
    assert torch.allclose(condition[:, :1], coarse)
    assert torch.allclose(condition[:, 1:4], t1_stack)
    assert torch.any(condition[:, 4:5].abs() > 0.0)  # T1 center edge
    assert torch.any(condition[:, 7:8].abs() > 0.0)  # coarse edge
    assert torch.allclose(condition[:, 8:9], coarse - t1_stack[:, 1:2])
    assert torch.any(condition[:, 9:10].abs() > 0.0)  # residual edge


def test_euler_refine_train_backpropagates_through_multistep_rollout():
    from pmrf_t1fa.models.pmrf_t1fa import euler_refine_train

    class LinearVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, x_t, t, condition=None):
            return self.weight * x_t

    model = LinearVelocity()
    coarse = torch.ones((1, 1, 3, 3), requires_grad=True)

    refined = euler_refine_train(model, coarse, num_steps=4, condition=None, clamp=False)
    loss = refined.mean()
    loss.backward()

    assert model.weight.grad is not None
    assert model.weight.grad.abs().item() > 0.0
    assert coarse.grad is not None
    assert coarse.grad.abs().mean().item() > 0.0


def test_detail_refinement_velocity_boosts_detail_head_at_early_steps():
    from pmrf_t1fa.models.pmrf_t1fa import compose_stage2_velocity

    avg = torch.full((1, 1, 2, 2), 0.2)
    detail = torch.full((1, 1, 2, 2), 0.5)
    t0 = torch.zeros(1)
    t1 = torch.ones(1)

    early = compose_stage2_velocity((avg, detail), t0, detail_boost=1.0)
    late = compose_stage2_velocity((avg, detail), t1, detail_boost=1.0)

    assert torch.allclose(early, torch.full((1, 1, 2, 2), 0.7))
    assert torch.allclose(late, torch.full((1, 1, 2, 2), 0.525))


def test_detail_refinement_unet_returns_avg_and_detail_heads():
    from pmrf_t1fa.models.pmrf_t1fa import DetailRefinementFlowUNet

    model = DetailRefinementFlowUNet(input_channels=1, condition_channels=2, base_channels=8)
    x = torch.zeros((1, 1, 16, 16))
    condition = torch.zeros((1, 2, 16, 16))
    t = torch.zeros(1)

    avg, detail = model(x, t, condition=condition)

    assert avg.shape == torch.Size([1, 1, 16, 16])
    assert detail.shape == torch.Size([1, 1, 16, 16])


def test_stage2_endpoint_time_sampler_matches_one_step_inference():
    from pmrf_t1fa.train_pmrf_t1fa_stage2 import sample_stage2_time

    t = sample_stage2_time(
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        mode="endpoint",
        t_min=0.0,
        t_max=1.0,
    )

    assert torch.equal(t, torch.zeros(4))


def test_stage2_rollout_step_parser_supports_multistep_training():
    from pmrf_t1fa.train_pmrf_t1fa_stage2 import parse_step_list

    assert parse_step_list("4, 8,10") == [4, 8, 10]


def test_zero_lpips_loss_keeps_stage2_smoke_tests_lightweight():
    from pmrf_t1fa.train_pmrf_t1fa_stage2 import ZeroLPIPSLoss

    loss_fn = ZeroLPIPSLoss()
    pred = torch.ones((2, 3, 4, 4))
    target = torch.zeros((2, 3, 4, 4))

    loss = loss_fn(pred, target)

    assert loss.shape == torch.Size([2])
    assert torch.equal(loss, torch.zeros(2))


def test_detail_paired_score_rewards_sharpness_over_smoothing():
    from types import SimpleNamespace

    from pmrf_t1fa.train_pmrf_t1fa_stage2 import compute_detail_paired_score

    args = SimpleNamespace(
        paired_psnr_weight=1.0,
        paired_ssim_weight=10.0,
        paired_mse_weight=200.0,
        paired_mae_weight=10.0,
        paired_brain_mae_weight=4.0,
        paired_wm_mae_weight=8.0,
        paired_grad_weight=0.8,
        paired_roi_weight=4.0,
        paired_sharp_weight=2.0,
        detail_sharp_weight=8.0,
        detail_coarse_penalty_weight=8.0,
        detail_target_sharp_ratio=0.90,
        detail_max_sharp_ratio=1.20,
        detail_oversharp_penalty_weight=12.0,
    )
    base = {
        "psnr": 28.0,
        "ssim": 0.90,
        "mse_proxy": 0.0017,
        "l1": 0.018,
        "brain_l1": 0.052,
        "wm_l1": 0.060,
        "grad": 0.112,
        "roi": 0.020,
        "sharp_ratio": 0.72,
        "coarse_sharp_ratio": 0.55,
    }
    smoother = dict(base, psnr=28.2, sharp_ratio=0.45)

    assert compute_detail_paired_score(base, args) > compute_detail_paired_score(smoother, args)

    noisy = dict(base, psnr=28.3, sharp_ratio=2.20)
    assert compute_detail_paired_score(base, args) > compute_detail_paired_score(noisy, args)


def test_stage1_detail_paired_score_rewards_sharper_valid_predictions():
    from types import SimpleNamespace

    from pmrf_t1fa.train_pmrf_t1fa_stage1 import compute_detail_paired_score

    args = SimpleNamespace(
        paired_psnr_weight=1.0,
        paired_ssim_weight=10.0,
        paired_mse_weight=200.0,
        paired_mae_weight=10.0,
        paired_sharp_weight=6.0,
        detail_target_sharp_ratio=0.90,
        detail_max_sharp_ratio=1.20,
        detail_oversharp_penalty_weight=12.0,
    )
    sharp = {
        "psnr": 27.8,
        "ssim": 0.90,
        "mse": 0.0018,
        "l1": 0.018,
        "sharp_ratio": 0.80,
    }
    smooth = dict(sharp, psnr=28.0, sharp_ratio=0.45)

    assert compute_detail_paired_score(sharp, args) > compute_detail_paired_score(smooth, args)

    noisy = dict(sharp, psnr=28.1, sharp_ratio=2.50)
    assert compute_detail_paired_score(sharp, args) > compute_detail_paired_score(noisy, args)
