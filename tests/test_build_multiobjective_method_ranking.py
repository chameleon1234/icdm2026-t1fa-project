import pandas as pd

from scripts.build_multiobjective_method_ranking import (
    canonical_method_name,
    compute_pareto_front,
    normalize_series,
    score_methods,
)


def test_normalize_series_respects_metric_direction():
    higher = normalize_series(pd.Series([2.0, 4.0, 6.0]), higher_is_better=True)
    lower = normalize_series(pd.Series([2.0, 4.0, 6.0]), higher_is_better=False)

    assert higher.tolist() == [0.0, 0.5, 1.0]
    assert lower.tolist() == [1.0, 0.5, 0.0]


def test_normalize_series_handles_infinite_upper_bound_without_flattening_finite_values():
    normalized = normalize_series(pd.Series([28.0, 30.0, float("inf")]), higher_is_better=True)

    assert normalized.tolist() == [0.0, 1.0, 1.0]


def test_compute_pareto_front_excludes_dominated_rows():
    df = pd.DataFrame(
        [
            {"method": "balanced", "PSNR_mean": 28.0, "SSIM_mean": 0.90, "MAE_mean": 0.018},
            {"method": "dominated", "PSNR_mean": 27.0, "SSIM_mean": 0.89, "MAE_mean": 0.020},
            {"method": "low_mae_tradeoff", "PSNR_mean": 27.5, "SSIM_mean": 0.88, "MAE_mean": 0.016},
        ]
    )

    front = compute_pareto_front(
        df,
        {
            "PSNR_mean": "max",
            "SSIM_mean": "max",
            "MAE_mean": "min",
        },
    )

    assert set(front["method"]) == {"balanced", "low_mae_tradeoff"}


def test_score_methods_prefers_balanced_image_and_downstream_quality():
    df = pd.DataFrame(
        [
            {
                "dataset": "private",
                "method": "sharp_but_weak",
                "PSNR_mean": 27.0,
                "SSIM_mean": 0.88,
                "MSE_mean": 0.0022,
                "MAE_mean": 0.020,
                "Sharpness_Ratio_mean": 0.95,
                "WM_Masked_MAE_mean": 0.065,
                "ROI_CCC": 0.82,
                "mean_accuracy": 0.62,
                "mean_auc": 0.55,
                "mean_macro_f1": 0.52,
            },
            {
                "dataset": "private",
                "method": "smooth_but_useful",
                "PSNR_mean": 28.2,
                "SSIM_mean": 0.91,
                "MSE_mean": 0.0015,
                "MAE_mean": 0.016,
                "Sharpness_Ratio_mean": 0.50,
                "WM_Masked_MAE_mean": 0.055,
                "ROI_CCC": 0.88,
                "mean_accuracy": 0.70,
                "mean_auc": 0.73,
                "mean_macro_f1": 0.68,
            },
            {
                "dataset": "private",
                "method": "balanced_ours",
                "PSNR_mean": 27.9,
                "SSIM_mean": 0.90,
                "MSE_mean": 0.0017,
                "MAE_mean": 0.017,
                "Sharpness_Ratio_mean": 0.82,
                "WM_Masked_MAE_mean": 0.056,
                "ROI_CCC": 0.87,
                "mean_accuracy": 0.69,
                "mean_auc": 0.72,
                "mean_macro_f1": 0.66,
            },
        ]
    )

    scored = score_methods(df)
    top = scored.sort_values("overall_score", ascending=False).iloc[0]

    assert top["method"] == "balanced_ours"


def test_canonical_method_name_maps_adni_prefix_and_full_suffix():
    assert canonical_method_name("ADNI_PM_DIRF_FIDELITY_FLOW_FULL") == "PM_DIRF_FIDELITY_FLOW"
    assert canonical_method_name("PM_DIRF_FIDELITY_FLOW_FULL") == "PM_DIRF_FIDELITY_FLOW"
    assert canonical_method_name("Fidelity_Flow") == "PM_DIRF_FIDELITY_FLOW"
    assert canonical_method_name("Stage1_LPIPS_GAN") == "PM_STAGE1_LPIPS_GAN"
    assert canonical_method_name("UNet_CurrentSplit_E100") == "UNet"
    assert canonical_method_name("ADNI_UNet_E50") == "UNet"
