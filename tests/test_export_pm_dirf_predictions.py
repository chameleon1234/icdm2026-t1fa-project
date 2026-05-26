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
