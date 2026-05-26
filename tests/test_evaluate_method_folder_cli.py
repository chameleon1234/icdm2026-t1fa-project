import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_evaluate_method_folder_cli_writes_slice_subject_and_summary_outputs(tmp_path):
    gt_dir = Path("data/processed/test/fa_slices")
    pred_dir = tmp_path / "predictions"
    metrics_root = tmp_path / "metrics"
    pred_dir.mkdir()

    selected_files = sorted(gt_dir.glob("*.png"))[:3]
    for path in selected_files:
        shutil.copy2(path, pred_dir / path.name)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_method_folder.py",
            "--pred_dir",
            str(pred_dir),
            "--method",
            "UNIT_GT",
            "--config",
            "configs/icdm2026.yaml",
            "--metrics_root",
            str(metrics_root),
            "--limit",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "UNIT_GT" in result.stdout
    assert "n_slices=3" in result.stdout

    slice_csv = metrics_root / "UNIT_GT_slice_metrics.csv"
    subject_csv = metrics_root / "UNIT_GT_subject_metrics.csv"
    summary_json = metrics_root / "UNIT_GT_summary.json"

    assert slice_csv.exists()
    assert subject_csv.exists()
    assert summary_json.exists()

    slice_df = pd.read_csv(slice_csv)
    subject_df = pd.read_csv(subject_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    assert len(slice_df) == 3
    assert set(["method", "fname", "subject_id", "group_name", "PSNR", "SSIM", "MSE", "MAE"]).issubset(slice_df.columns)
    assert len(subject_df) == 1
    assert subject_df.loc[0, "subject_id"].startswith("sub-")
    assert summary["method"] == "UNIT_GT"
    assert summary["n_slices"] == 3
    assert math.isinf(summary["PSNR_mean"]) or summary["PSNR_mean"] > 90.0
    assert summary["SSIM_mean"] > 0.999
    assert summary["MSE_mean"] == 0.0
    assert summary["MAE_mean"] == 0.0

