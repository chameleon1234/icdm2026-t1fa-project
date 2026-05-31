from pathlib import Path

import cv2
import numpy as np
import torch


def test_export_stage_defaults_resolve_common_prediction_folders():
    from scripts.export_pm_dirf_predictions import resolve_output_dir, stage_to_method

    assert stage_to_method("stage1") == "PM_STAGE1"
    assert stage_to_method("stage2") == "PM_DIRF"
    assert resolve_output_dir("stage1", "outputs/icdm2026/predictions", "") == Path(
        "outputs/icdm2026/predictions/PM_STAGE1"
    )
    assert resolve_output_dir("stage2", "outputs/icdm2026/predictions", "") == Path(
        "outputs/icdm2026/predictions/PM_DIRF"
    )
    assert resolve_output_dir("stage1", "outputs/icdm2026/predictions", "custom") == Path("custom")


def test_save_prediction_png_writes_grayscale_8bit_file(tmp_path):
    from scripts.export_pm_dirf_predictions import save_prediction_png

    prediction = torch.linspace(-1.0, 1.0, steps=16).reshape(1, 1, 4, 4)
    output_path = tmp_path / "sub-001_z020.png"

    save_prediction_png(prediction, output_path)

    image = cv2.imdecode(np.fromfile(str(output_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert output_path.exists()
    assert image.shape == (4, 4)
    assert image.dtype == np.uint8
    assert image.min() == 0
    assert image.max() == 255


def test_load_stage2_uses_checkpoint_condition_mode_for_t1_aware_model(tmp_path):
    from pmrf_t1fa.models.pmrf_t1fa import RefinementFlowUNet
    from scripts.export_pm_dirf_predictions import load_stage2, resolve_stage2_settings

    ckpt_path = tmp_path / "stage2_coarse_t1.pt"
    model = RefinementFlowUNet(input_channels=1, condition_channels=4)
    torch.save(
        {
            "model": model.state_dict(),
            "args": {"condition_mode": "coarse_t1", "condition_on_coarse": True, "eval_steps": 1},
        },
        ckpt_path,
    )

    args = type(
        "Args",
        (),
        {
            "stage": "stage2",
            "stage2_ckpt": str(ckpt_path),
            "condition_on_coarse": True,
            "auto_condition_from_ckpt": True,
            "condition_mode": "auto",
            "eval_steps": -1,
            "detail_boost_override": -1.0,
            "dynamic_condition_rollout": False,
            "force_static_condition_rollout": False,
        },
    )()

    condition_mode, condition_on_coarse, eval_steps, detail_boost, dynamic_condition = resolve_stage2_settings(args)
    loaded = load_stage2(ckpt_path, torch.device("cpu"), condition_mode=condition_mode, stage1_channels=3)

    assert condition_mode == "coarse_t1"
    assert condition_on_coarse is True
    assert eval_steps == 1
    assert detail_boost == 0.0
    assert dynamic_condition is False
    assert loaded.inc.weight.shape[1] == 5


def test_load_stage2_supports_stage1_guided_edge_condition_mode(tmp_path):
    from pmrf_t1fa.models.pmrf_t1fa import DetailRefinementFlowUNet
    from scripts.export_pm_dirf_predictions import load_stage2, resolve_stage2_settings

    ckpt_path = tmp_path / "stage2_coarse_t1_edge.pt"
    model = DetailRefinementFlowUNet(input_channels=1, condition_channels=10)
    torch.save(
        {
            "model": model.state_dict(),
            "args": {
                "condition_mode": "coarse_t1_edge",
                "condition_on_coarse": True,
                "eval_steps": 10,
                "stage2_model_variant": "detail",
                "detail_boost": 0.9,
                "dynamic_condition_rollout": True,
            },
        },
        ckpt_path,
    )

    args = type(
        "Args",
        (),
        {
            "stage": "stage2",
            "stage2_ckpt": str(ckpt_path),
            "condition_on_coarse": True,
            "auto_condition_from_ckpt": True,
            "condition_mode": "auto",
            "eval_steps": -1,
            "detail_boost_override": -1.0,
            "dynamic_condition_rollout": False,
            "force_static_condition_rollout": False,
        },
    )()

    condition_mode, condition_on_coarse, eval_steps, detail_boost, dynamic_condition = resolve_stage2_settings(args)
    loaded = load_stage2(ckpt_path, torch.device("cpu"), condition_mode=condition_mode, stage1_channels=3)

    assert condition_mode == "coarse_t1_edge"
    assert condition_on_coarse is True
    assert eval_steps == 10
    assert detail_boost == 0.9
    assert dynamic_condition is True
    assert loaded.inc.weight.shape[1] == 11
    assert hasattr(loaded, "out_detail")
