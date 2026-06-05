from pathlib import Path

import torch


def test_fidelity_corrector_default_output_dir():
    from scripts.export_fidelity_corrector_predictions import resolve_output_dir

    assert resolve_output_dir("outputs/icdm2026/predictions", "") == Path(
        "outputs/icdm2026/predictions/PM_DIRF_FIDELITY_CORRECTOR"
    )


def test_load_fidelity_corrector_uses_checkpoint_architecture(tmp_path):
    from pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector import FidelityCorrector
    from scripts.export_fidelity_corrector_predictions import load_fidelity_corrector

    model = FidelityCorrector(stage1_channels=5, mode="flow", width=8, num_blocks=2)
    ckpt = tmp_path / "best_fidelity_corrector.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "args": {
                "corrector_mode": "flow",
                "width": 8,
                "num_blocks": 2,
                "frequency_cutoff": 0.10,
                "frequency_transition": 0.03,
                "eval_steps": 4,
            },
            "stage1_channels": 5,
        },
        ckpt,
    )

    loaded, settings = load_fidelity_corrector(ckpt, torch.device("cpu"), stage1_channels=5)

    assert loaded.mode == "flow"
    assert settings["frequency_cutoff"] == 0.10
    assert settings["frequency_transition"] == 0.03
    assert settings["eval_steps"] == 4
