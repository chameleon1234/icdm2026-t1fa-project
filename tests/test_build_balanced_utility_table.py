import json
from pathlib import Path

import pandas as pd


def test_balanced_utility_rewards_sharp_roi_auc_and_low_wm_error(tmp_path):
    from scripts.build_balanced_utility_table import build_balanced_table

    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    summaries = {
        "OursSharp": {
            "PSNR_mean": 28.0,
            "SSIM_mean": 0.90,
            "Sharpness_Ratio_mean": 0.86,
            "WM_Masked_MAE_mean": 0.056,
            "ROI_CCC": 0.86,
        },
        "SmoothBaseline": {
            "PSNR_mean": 28.2,
            "SSIM_mean": 0.91,
            "Sharpness_Ratio_mean": 0.45,
            "WM_Masked_MAE_mean": 0.060,
            "ROI_CCC": 0.80,
        },
    }
    for name, summary in summaries.items():
        (metrics_root / f"{name}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    downstream = tmp_path / "downstream.csv"
    pd.DataFrame(
        [
            {"method": "T1_PLUS_OursSharp", "task": "cn_vs_ad", "accuracy_mean": 0.79, "macro_auc_ovr_mean": 0.74},
            {"method": "T1_PLUS_OursSharp", "task": "cn_vs_mci", "accuracy_mean": 0.61, "macro_auc_ovr_mean": 0.62},
            {"method": "T1_PLUS_SmoothBaseline", "task": "cn_vs_ad", "accuracy_mean": 0.70, "macro_auc_ovr_mean": 0.55},
            {"method": "T1_PLUS_SmoothBaseline", "task": "cn_vs_mci", "accuracy_mean": 0.60, "macro_auc_ovr_mean": 0.57},
        ]
    ).to_csv(downstream, index=False)

    table = build_balanced_table(
        method_specs=["OursSharp=OursSharp=T1_PLUS_OursSharp", "SmoothBaseline=SmoothBaseline=T1_PLUS_SmoothBaseline"],
        metrics_root=metrics_root,
        downstream_csv=downstream,
        tasks=["cn_vs_ad", "cn_vs_mci"],
    )

    ours = table[table["method"].eq("OursSharp")].iloc[0]
    smooth = table[table["method"].eq("SmoothBaseline")].iloc[0]
    assert ours["mean_auc"] > smooth["mean_auc"]
    assert ours["balanced_score"] > smooth["balanced_score"]
    assert ours["Sharpness_Ratio"] > smooth["Sharpness_Ratio"]
