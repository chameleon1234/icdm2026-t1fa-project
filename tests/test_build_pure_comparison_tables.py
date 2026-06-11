import json
from pathlib import Path

import pandas as pd


def test_build_pure_comparison_tables_excludes_legacy_and_ours_methods(tmp_path):
    from scripts.build_pure_comparison_tables import build_downstream_table, build_image_table

    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    summaries = {
        "UNet_CurrentSplit_E100": {"PSNR_mean": 28.2, "SSIM_mean": 0.89},
        "Pix2Pix_CurrentSplit_E100": {"PSNR_mean": 28.1, "SSIM_mean": 0.90},
        "UNet_E99": {"PSNR_mean": 37.0, "SSIM_mean": 0.97},
        "PM_DIRF_FIDELITY_FLOW_FULL": {"PSNR_mean": 28.0, "SSIM_mean": 0.90},
    }
    for method, summary in summaries.items():
        (metrics_root / f"{method}_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    downstream = tmp_path / "classification_summary.csv"
    pd.DataFrame(
        [
            {"method": "UNet_CurrentSplit_E100", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.53, "balanced_accuracy": 0.54, "macro_auc_ovr": 0.56},
            {"method": "Pix2Pix_CurrentSplit_E100", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.57, "balanced_accuracy": 0.57, "macro_auc_ovr": 0.56},
            {"method": "UNet_E99", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.90, "balanced_accuracy": 0.90, "macro_auc_ovr": 0.95},
            {"method": "Fidelity_Flow", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.58, "balanced_accuracy": 0.58, "macro_auc_ovr": 0.61},
        ]
    ).to_csv(downstream, index=False)

    image = build_image_table(metrics_root)
    utility = build_downstream_table([downstream])

    assert "UNet_E99" not in set(image["method"])
    assert "PM_DIRF_FIDELITY_FLOW_FULL" not in set(image["method"])
    assert "UNet_E99" not in set(utility["method"])
    assert "Fidelity_Flow" not in set(utility["method"])
    assert {"UNet_CurrentSplit_E100", "Pix2Pix_CurrentSplit_E100", "CycleGAN_CurrentSplit_E100", "DIRF_V5_3SLICE_K6"}.issubset(set(image["method"]))
    assert utility.loc[utility["method"].eq("UNet_CurrentSplit_E100"), "auc"].item() == 0.56


def test_build_pure_comparison_tables_can_append_ours_separately(tmp_path):
    from scripts.build_pure_comparison_tables import build_downstream_table, build_image_table

    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    (metrics_root / "PM_DIRF_FIDELITY_FLOW_FULL_summary.json").write_text(
        json.dumps({"PSNR_mean": 28.09, "SSIM_mean": 0.905}),
        encoding="utf-8",
    )
    downstream = tmp_path / "classification_summary.csv"
    pd.DataFrame(
        [
            {"method": "Fidelity_Flow", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.58, "balanced_accuracy": 0.58, "macro_auc_ovr": 0.61},
        ]
    ).to_csv(downstream, index=False)

    image = build_image_table(metrics_root, include_ours=True)
    utility = build_downstream_table([downstream], include_ours=True)

    ours_image = image[image["method"].eq("PM_DIRF_FIDELITY_FLOW_FULL")].iloc[0]
    ours_utility = utility[utility["method"].eq("PM_DIRF_FIDELITY_FLOW_FULL")].iloc[0]
    assert ours_image["role"] == "final_two_stage"
    assert ours_image["PSNR"] == 28.09
    assert ours_utility["role"] == "final_two_stage"
    assert ours_utility["auc"] == 0.61
