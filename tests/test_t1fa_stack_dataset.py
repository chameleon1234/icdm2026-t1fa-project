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


def test_predict_stage1_residual_uses_center_slice_for_stacked_input():
    from pmrf_t1fa.models.pmrf_t1fa import predict_stage1_fa

    class ZeroResidual(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

    t1_stack = torch.tensor([[[[-0.5]], [[0.25]], [[0.75]]]])

    coarse = predict_stage1_fa(ZeroResidual(), t1_stack, prediction_mode="residual")

    assert coarse.shape == torch.Size([1, 1, 1, 1])
    assert coarse.item() == 0.25


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

    export_model, export_mode, export_channels = load_export_stage1(checkpoint_path, torch.device("cpu"))
    eval_model, eval_mode, eval_channels = load_eval_stage1(checkpoint_path, torch.device("cpu"))

    assert export_model.patch_embed.weight.shape[1] == 3
    assert eval_model.patch_embed.weight.shape[1] == 3
    assert export_mode == eval_mode == "residual"
    assert export_channels == eval_channels == 3


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
