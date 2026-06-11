from pathlib import Path

import pandas as pd


def test_final_package_filters_frequency_and_scores_methods(tmp_path):
    from scripts.build_final_experiment_package import build_final_package_tables

    table_root = tmp_path / "tables"
    table_root.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "private",
                "method": "PM_DIRF_FIDELITY_FLOW_FULL",
                "display": "Ours Fidelity Flow",
                "family": "Ours",
                "role": "final_two_stage",
                "available": True,
                "PSNR": 28.0,
                "SSIM": 0.91,
                "MAE": 0.017,
                "SharpRatio": 0.87,
                "WM_MAE": 0.055,
                "ROI_CCC": 0.88,
            },
            {
                "dataset": "private",
                "method": "FREQ_FLOWBASE_B035",
                "display": "Frequency Fusion",
                "family": "Ours",
                "role": "discarded",
                "available": True,
                "PSNR": 28.2,
                "SSIM": 0.91,
                "MAE": 0.016,
                "SharpRatio": 0.85,
                "WM_MAE": 0.055,
                "ROI_CCC": 0.88,
            },
            {
                "dataset": "private",
                "method": "UNet_CurrentSplit_E100",
                "display": "U-Net",
                "family": "CNN",
                "role": "comparison",
                "available": True,
                "PSNR": 27.0,
                "SSIM": 0.88,
                "MAE": 0.02,
                "SharpRatio": 0.50,
                "WM_MAE": 0.070,
                "ROI_CCC": 0.70,
            },
        ]
    ).to_csv(table_root / "two_dataset_image_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "private",
                "task": "cn_vs_ad",
                "display": "Ours Fidelity Flow",
                "method": "PM_DIRF_FIDELITY_FLOW_FULL",
                "available": True,
                "macro_f1_mean": 0.60,
                "macro_auc_ovr_mean": 0.62,
            },
            {
                "dataset": "private",
                "task": "cn_vs_ad",
                "display": "Frequency Fusion",
                "method": "FREQ_FLOWBASE_B035",
                "available": True,
                "macro_f1_mean": 0.99,
                "macro_auc_ovr_mean": 0.99,
            },
            {
                "dataset": "private",
                "task": "cn_vs_ad",
                "display": "U-Net",
                "method": "UNet_CurrentSplit_E100",
                "available": True,
                "macro_f1_mean": 0.50,
                "macro_auc_ovr_mean": 0.52,
            },
        ]
    ).to_csv(table_root / "two_dataset_downstream_macro_f1.csv", index=False)

    summary, task_table = build_final_package_tables(table_root)

    assert "FREQ_FLOWBASE_B035" not in set(summary["method"])
    assert "FREQ_FLOWBASE_B035" not in set(task_table["method"])
    ours = summary[summary["method"].eq("PM_DIRF_FIDELITY_FLOW_FULL")].iloc[0]
    unet = summary[summary["method"].eq("UNet_CurrentSplit_E100")].iloc[0]
    assert ours["quality_score"] > unet["quality_score"]
    assert ours["utility_score"] > unet["utility_score"]
    assert ours["overall_score"] > unet["overall_score"]


def test_final_package_marks_dataset_best_per_metric(tmp_path):
    from scripts.build_final_experiment_package import build_final_package_tables

    table_root = tmp_path / "tables"
    table_root.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "adni",
                "method": "A",
                "display": "A",
                "family": "CNN",
                "role": "comparison",
                "available": True,
                "PSNR": 28.0,
                "SSIM": 0.90,
                "MAE": 0.018,
                "SharpRatio": 0.70,
                "WM_MAE": 0.06,
                "ROI_CCC": 0.80,
            },
            {
                "dataset": "adni",
                "method": "B",
                "display": "B",
                "family": "Ours",
                "role": "final_two_stage",
                "available": True,
                "PSNR": 29.0,
                "SSIM": 0.92,
                "MAE": 0.015,
                "SharpRatio": 0.90,
                "WM_MAE": 0.05,
                "ROI_CCC": 0.90,
            },
        ]
    ).to_csv(table_root / "two_dataset_image_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"dataset": "adni", "task": "cn_vs_ad", "method": "A", "display": "A", "available": True, "macro_f1_mean": 0.50},
            {"dataset": "adni", "task": "cn_vs_ad", "method": "B", "display": "B", "available": True, "macro_f1_mean": 0.55},
        ]
    ).to_csv(table_root / "two_dataset_downstream_macro_f1.csv", index=False)

    summary, _ = build_final_package_tables(table_root)
    best = summary[summary["method"].eq("B")].iloc[0]

    assert best["is_best_quality_score"]
    assert best["is_best_overall_score"]


def test_final_package_visual_gate_excludes_blurry_high_psnr_method(tmp_path):
    from scripts.build_final_experiment_package import build_final_package_tables

    table_root = tmp_path / "tables"
    table_root.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "private",
                "method": "DIRF_V5_3SLICE_K6",
                "display": "DIRF V5",
                "family": "Flow",
                "role": "comparison",
                "available": True,
                "PSNR": 28.4,
                "SSIM": 0.91,
                "MAE": 0.016,
                "SharpRatio": 0.45,
                "WM_MAE": 0.055,
                "ROI_CCC": 0.88,
            },
            {
                "dataset": "private",
                "method": "PM_DIRF_FIDELITY_FLOW_FULL",
                "display": "Ours Fidelity Flow",
                "family": "Ours",
                "role": "final_two_stage",
                "available": True,
                "PSNR": 27.6,
                "SSIM": 0.89,
                "MAE": 0.019,
                "SharpRatio": 0.87,
                "WM_MAE": 0.059,
                "ROI_CCC": 0.86,
            },
        ]
    ).to_csv(table_root / "two_dataset_image_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"dataset": "private", "task": "cn_vs_ad", "method": "DIRF_V5_3SLICE_K6", "display": "DIRF V5", "available": True, "macro_f1_mean": 0.65},
            {"dataset": "private", "task": "cn_vs_ad", "method": "PM_DIRF_FIDELITY_FLOW_FULL", "display": "Ours Fidelity Flow", "available": True, "macro_f1_mean": 0.56},
        ]
    ).to_csv(table_root / "two_dataset_downstream_macro_f1.csv", index=False)

    summary, _ = build_final_package_tables(table_root)
    dirf = summary[summary["method"].eq("DIRF_V5_3SLICE_K6")].iloc[0]
    ours = summary[summary["method"].eq("PM_DIRF_FIDELITY_FLOW_FULL")].iloc[0]

    assert not dirf["passes_visual_gate"]
    assert pd.isna(dirf["eligible_overall_score"])
    assert ours["passes_visual_gate"]
    assert ours["eligible_overall_score_rank"] == 1
