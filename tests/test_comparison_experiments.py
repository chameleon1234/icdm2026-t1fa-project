import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.comparison_experiments import (
    build_downstream_command,
    build_image_eval_command,
    build_table_rows,
    load_comparison_manifest,
    write_markdown_table,
)


def test_load_comparison_manifest_keeps_table_groups_and_fusion_specs(tmp_path):
    manifest_path = tmp_path / "methods.yaml"
    manifest_path.write_text(
        """
methods:
  - name: Stage1_Baseline
    metric_name: PM_STAGE1_WMROI_DETAIL_5SLICE
    downstream_name: Stage1_Baseline
    prediction_dir: outputs/icdm2026/predictions/PM_STAGE1_WMROI_DETAIL_5SLICE
    table_group: main
    display_name: Baseline
  - name: Fidelity_Flow
    prediction_dir: outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL
    table_group: main
    is_ours: true
baselines:
  include_t1: true
  include_fa_gt: true
downstream:
  tasks: cn_scd_vs_mci_ad
  repeat_seeds: 0,1,2
  max_features: 12
  fusions:
    - T1_PLUS_FLOW=T1_ONLY+Fidelity_Flow
""",
        encoding="utf-8",
    )

    manifest = load_comparison_manifest(manifest_path)

    assert [method.name for method in manifest.methods] == ["Stage1_Baseline", "Fidelity_Flow"]
    assert manifest.methods[0].display_name == "Baseline"
    assert manifest.methods[0].metric_name == "PM_STAGE1_WMROI_DETAIL_5SLICE"
    assert manifest.methods[0].downstream_name == "Stage1_Baseline"
    assert manifest.methods[1].is_ours is True
    assert manifest.baselines.include_t1 is True
    assert manifest.baselines.include_fa_gt is True
    assert manifest.downstream.fusions == ["T1_PLUS_FLOW=T1_ONLY+Fidelity_Flow"]


def test_load_comparison_manifest_rejects_duplicate_method_names(tmp_path):
    manifest_path = tmp_path / "methods.yaml"
    manifest_path.write_text(
        """
methods:
  - name: DUP
    prediction_dir: a
  - name: DUP
    prediction_dir: b
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate method name"):
        load_comparison_manifest(manifest_path)


def test_build_commands_use_existing_project_scripts(tmp_path):
    method = load_comparison_manifest(
        tmp_path / "methods.yaml",
        raw={
            "methods": [
                {
                    "name": "Fidelity_Flow",
                    "prediction_dir": "outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL",
                }
            ],
            "baselines": {"include_t1": True, "include_fa_gt": True},
            "downstream": {"tasks": "cn_scd_vs_mci_ad", "repeat_seeds": "0,1"},
        },
    ).methods[0]

    image_cmd = build_image_eval_command(method, visualize_count=8)
    downstream_cmd = build_downstream_command(
        load_comparison_manifest(
            tmp_path / "methods.yaml",
            raw={
                "methods": [{"name": "Fidelity_Flow", "prediction_dir": "preds/flow"}],
                "baselines": {"include_t1": True, "include_fa_gt": True},
                "downstream": {"tasks": "cn_scd_vs_mci_ad", "repeat_seeds": "0,1", "max_features": 12},
            },
        ),
        output_root=Path("outputs/icdm2026/downstream_comparison"),
    )

    assert image_cmd[:3] == ["python", "scripts/evaluate_method_folder.py", "--pred_dir"]
    assert "--method" in image_cmd
    assert "Fidelity_Flow" in image_cmd
    assert downstream_cmd[:2] == ["python", "scripts/evaluate_downstream_classification.py"]
    assert "--include_t1" in downstream_cmd
    assert "--include_fa_gt" in downstream_cmd
    assert "Fidelity_Flow=preds/flow" in downstream_cmd
    assert "--repeat_seeds" in downstream_cmd


def test_build_table_rows_merges_image_metrics_and_repeated_downstream(tmp_path):
    metrics_root = tmp_path / "metrics"
    downstream_root = tmp_path / "downstream"
    metrics_root.mkdir()
    downstream_root.mkdir()
    (metrics_root / "PM_DIRF_FIDELITY_FLOW_FULL_summary.json").write_text(
        json.dumps(
            {
                "method": "Fidelity_Flow",
                "PSNR_mean": 27.58,
                "SSIM_mean": 0.8925,
                "Sharpness_Ratio_mean": 0.8692,
                "WM_Masked_MAE_mean": 0.0592,
                "ROI_CCC": 0.8665,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "method": "Fidelity_Flow",
                "task": "cn_scd_vs_mci_ad",
                "macro_f1_mean": 0.519,
                "macro_f1_std": 0.045,
            }
        ]
    ).to_csv(downstream_root / "classification_repeated_summary.csv", index=False)

    manifest = load_comparison_manifest(
        tmp_path / "methods.yaml",
        raw={
            "methods": [
                {
                    "name": "Fidelity_Flow",
                    "metric_name": "PM_DIRF_FIDELITY_FLOW_FULL",
                    "downstream_name": "Fidelity_Flow",
                    "prediction_dir": "preds/flow",
                    "display_name": "Fidelity Flow",
                    "is_ours": True,
                }
            ]
        },
    )
    rows = build_table_rows(
        manifest,
        metrics_root=metrics_root,
        downstream_root=downstream_root,
        downstream_task="cn_scd_vs_mci_ad",
    )

    assert rows == [
        {
            "method": "Fidelity_Flow",
            "display_name": "Fidelity Flow",
            "table_group": "main",
            "table_groups": ["main"],
            "is_ours": True,
            "PSNR": 27.58,
            "SSIM": 0.8925,
            "SharpRatio": 0.8692,
            "WM_MAE": 0.0592,
            "ROI_CCC": 0.8665,
            "Macro_F1": 0.519,
            "Macro_F1_Std": 0.045,
        }
    ]


def test_write_markdown_table_has_cn_variant_and_ascii_safe_arrows(tmp_path):
    rows = [
        {
            "display_name": "Fidelity Flow",
            "PSNR": 27.58,
            "SSIM": 0.8925,
            "SharpRatio": 0.8692,
            "WM_MAE": 0.0592,
            "ROI_CCC": 0.8665,
            "Macro_F1": 0.519,
            "Macro_F1_Std": 0.045,
        }
    ]

    path = tmp_path / "main_results_table.md"
    write_markdown_table(rows, path, title="Main Results", title_cn="主结果表")

    text = path.read_text(encoding="utf-8")
    cn_text = (tmp_path / "main_results_table_cn.md").read_text(encoding="utf-8")
    assert "higher is better" in text
    assert "越高越好" in cn_text
    assert "Fidelity Flow" in text
    assert "0.519 +/- 0.045" in text
