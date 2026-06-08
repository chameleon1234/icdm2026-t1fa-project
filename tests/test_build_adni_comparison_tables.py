import json
from pathlib import Path

import pandas as pd


def test_build_adni_comparison_tables_keeps_adni_methods_and_auc(tmp_path):
    from scripts.build_adni_comparison_tables import build_downstream_table, build_image_table

    metrics_root = tmp_path / "adni_metrics"
    metrics_root.mkdir()
    summaries = {
        "ADNI_UNet_E50": {"PSNR_mean": 23.4, "SSIM_mean": 0.81, "Sharpness_Ratio_mean": 0.72, "ROI_CCC": 0.42},
        "ADNI_T1_ONLY": {"PSNR_mean": 14.0, "SSIM_mean": 0.74, "Sharpness_Ratio_mean": 3.6},
        "UNet_CurrentSplit_E100": {"PSNR_mean": 28.2, "SSIM_mean": 0.89},
    }
    for method, summary in summaries.items():
        (metrics_root / f"{method}_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    downstream = tmp_path / "adni_downstream.csv"
    pd.DataFrame(
        [
            {
                "method": "ADNI_UNet_E50",
                "task": "cn_vs_mci_spectrum_ad",
                "n_subjects": 78,
                "macro_f1": 0.58,
                "balanced_accuracy": 0.59,
                "macro_auc_ovr": 0.62,
            },
            {
                "method": "ADNI_T1_ONLY",
                "task": "cn_vs_mci_spectrum_ad",
                "n_subjects": 78,
                "macro_f1": 0.56,
                "balanced_accuracy": 0.56,
                "macro_auc_ovr": 0.55,
            },
            {
                "method": "UNet_CurrentSplit_E100",
                "task": "cn_vs_mci_spectrum_ad",
                "n_subjects": 78,
                "macro_f1": 0.99,
                "balanced_accuracy": 0.99,
                "macro_auc_ovr": 0.99,
            },
        ]
    ).to_csv(downstream, index=False)

    image = build_image_table(metrics_root)
    utility = build_downstream_table([downstream])

    assert "ADNI_UNet_E50" in set(image["method"])
    assert "ADNI_T1_ONLY" in set(image["method"])
    assert "UNet_CurrentSplit_E100" not in set(image["method"])
    assert "UNet_CurrentSplit_E100" not in set(utility["method"])

    unet = image[image["method"].eq("ADNI_UNet_E50")].iloc[0]
    t1 = utility[utility["method"].eq("ADNI_T1_ONLY")].iloc[0]
    assert unet["family"] == "CNN"
    assert unet["role"] == "pure_comparison"
    assert unet["PSNR"] == 23.4
    assert unet["ROI_CCC"] == 0.42
    assert t1["role"] == "reference"
    assert t1["auc"] == 0.55
