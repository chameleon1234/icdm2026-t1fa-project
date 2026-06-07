import json
from pathlib import Path

import pandas as pd


def test_build_pure_comparison_tables_excludes_legacy_and_fusion_methods(tmp_path):
    from scripts.build_pure_comparison_tables import build_downstream_table, build_image_table

    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    summaries = {
        "UNet_CurrentSplit_E100": {"PSNR_mean": 28.2, "SSIM_mean": 0.89},
        "Pix2Pix_CurrentSplit_E100": {"PSNR_mean": 28.1, "SSIM_mean": 0.90},
        "UNet_E99": {"PSNR_mean": 37.0, "SSIM_mean": 0.97},
        "FREQ_FLOWBASE_PMLOW_B035": {"PSNR_mean": 28.3, "SSIM_mean": 0.91},
    }
    for method, summary in summaries.items():
        (metrics_root / f"{method}_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    downstream = tmp_path / "classification_summary.csv"
    pd.DataFrame(
        [
            {"method": "UNet_CurrentSplit_E100", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.53, "balanced_accuracy": 0.54, "macro_auc_ovr": 0.56},
            {"method": "Pix2Pix_CurrentSplit_E100", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.57, "balanced_accuracy": 0.57, "macro_auc_ovr": 0.56},
            {"method": "UNet_E99", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.90, "balanced_accuracy": 0.90, "macro_auc_ovr": 0.95},
            {"method": "FREQ_FLOWBASE_B035", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.60, "balanced_accuracy": 0.60, "macro_auc_ovr": 0.57},
        ]
    ).to_csv(downstream, index=False)

    image = build_image_table(metrics_root)
    utility = build_downstream_table([downstream])

    assert "UNet_E99" not in set(image["method"])
    assert "FREQ_FLOWBASE_PMLOW_B035" not in set(image["method"])
    assert "UNet_E99" not in set(utility["method"])
    assert "FREQ_FLOWBASE_B035" not in set(utility["method"])
    assert {"UNet_CurrentSplit_E100", "Pix2Pix_CurrentSplit_E100", "CycleGAN_CurrentSplit_E100", "DIRF_V5_3SLICE_K6"}.issubset(set(image["method"]))
    assert utility.loc[utility["method"].eq("UNet_CurrentSplit_E100"), "auc"].item() == 0.56


def test_build_pure_comparison_tables_can_append_ours_separately(tmp_path):
    from scripts.build_pure_comparison_tables import build_downstream_table, build_image_table

    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    (metrics_root / "FREQ_FLOWBASE_PMLOW_B035_summary.json").write_text(
        json.dumps({"PSNR_mean": 28.18, "SSIM_mean": 0.90}),
        encoding="utf-8",
    )
    downstream = tmp_path / "classification_summary.csv"
    pd.DataFrame(
        [
            {"method": "FREQ_FLOWBASE_B035", "task": "cn_scd_vs_mci_ad", "n_subjects": 38, "macro_f1": 0.60, "balanced_accuracy": 0.60, "macro_auc_ovr": 0.57},
        ]
    ).to_csv(downstream, index=False)

    image = build_image_table(metrics_root, include_ours=True)
    utility = build_downstream_table([downstream], include_ours=True)

    ours_image = image[image["method"].eq("FREQ_FLOWBASE_B035")].iloc[0]
    ours_utility = utility[utility["method"].eq("FREQ_FLOWBASE_B035")].iloc[0]
    assert ours_image["role"] == "main_candidate"
    assert ours_image["PSNR"] == 28.18
    assert ours_utility["role"] == "main_candidate"
    assert ours_utility["auc"] == 0.57
