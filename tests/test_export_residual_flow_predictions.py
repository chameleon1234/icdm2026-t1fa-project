from pathlib import Path

import torch


def test_residual_flow_default_output_dir():
    from scripts.export_residual_flow_predictions import resolve_output_dir

    assert resolve_output_dir("outputs/icdm2026/predictions", "") == Path(
        "outputs/icdm2026/predictions/PM_DIRF_RESIDUAL_FLOW"
    )
    assert resolve_output_dir("outputs/icdm2026/predictions", "custom") == Path("custom")


def test_load_residual_flow_uses_checkpoint_args(tmp_path):
    from pmrf_t1fa.train_pmrf_t1fa_stage2_residual_flow import ResidualFlowStage2
    from scripts.export_residual_flow_predictions import load_residual_flow

    ckpt_path = tmp_path / "best_residual_flow.pt"
    model = ResidualFlowStage2(stage1_channels=5, width=8, num_blocks=2, sigma_min=0.02, sigma_max=0.35)
    torch.save(
        {
            "model": model.state_dict(),
            "args": {
                "width": 8,
                "num_blocks": 2,
                "sigma_min": 0.02,
                "sigma_max": 0.35,
                "eval_steps": 6,
                "eval_noise_scale": 0.1,
            },
            "stage1_channels": 5,
        },
        ckpt_path,
    )

    loaded, flow_args = load_residual_flow(ckpt_path, torch.device("cpu"), stage1_channels=5)

    assert loaded.stage1_channels == 5
    assert loaded.sigma_min == 0.02
    assert loaded.sigma_max == 0.35
    assert flow_args["eval_steps"] == 6
    assert flow_args["eval_noise_scale"] == 0.1
