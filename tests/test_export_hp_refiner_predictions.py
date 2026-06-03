from pathlib import Path

import torch


def test_hp_refiner_default_output_dir():
    from scripts.export_hp_refiner_predictions import resolve_output_dir

    assert resolve_output_dir("outputs/icdm2026/predictions", "") == Path(
        "outputs/icdm2026/predictions/PM_DIRF_HP_REFINER"
    )
    assert resolve_output_dir("outputs/icdm2026/predictions", "custom") == Path("custom")


def test_load_hp_refiner_uses_checkpoint_model_args(tmp_path):
    from pmrf_t1fa.train_pmrf_t1fa_stage2_hp_refiner import HighPassRefinerNet
    from scripts.export_hp_refiner_predictions import load_hp_refiner

    ckpt_path = tmp_path / "best_hp_refiner.pt"
    model = HighPassRefinerNet(in_channels=8, width=8, num_blocks=2)
    torch.save(
        {
            "model": model.state_dict(),
            "args": {
                "width": 8,
                "num_blocks": 2,
                "residual_scale": 0.35,
                "residual_gate_mode": "edge",
                "residual_gate_min": 0.75,
            },
            "stage1_channels": 5,
        },
        ckpt_path,
    )

    loaded, hp_args = load_hp_refiner(ckpt_path, torch.device("cpu"), stage1_channels=5)

    assert loaded.in_proj.weight.shape[1] == 8
    assert hp_args["residual_scale"] == 0.35
    assert hp_args["residual_gate_mode"] == "edge"
    assert hp_args["residual_gate_min"] == 0.75
