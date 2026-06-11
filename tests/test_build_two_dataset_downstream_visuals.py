import json
from pathlib import Path

import pandas as pd


def _write_summary(metrics_root: Path, method: str, *, psnr: float, sharp: float) -> None:
    (metrics_root / f"{method}_summary.json").write_text(
        json.dumps(
            {
                "PSNR_mean": psnr,
                "SSIM_mean": 0.9,
                "MSE_mean": 0.001,
                "MAE_mean": 0.02,
                "Sharpness_Ratio_mean": sharp,
                "WM_Masked_MAE_mean": 0.06,
                "ROI_CCC_mean": 0.86,
            }
        ),
        encoding="utf-8",
    )


def test_two_dataset_builder_filters_frequency_and_keeps_final_method(tmp_path):
    from scripts.build_two_dataset_downstream_visuals import build_downstream_table, build_image_table

    metrics_root = tmp_path / "metrics"
    metrics_root.mkdir()
    _write_summary(metrics_root, "PM_DIRF_FIDELITY_FLOW_FULL", psnr=27.58, sharp=0.87)
    _write_summary(metrics_root, "FREQ_FLOWBASE_B035", psnr=28.18, sharp=0.84)
    _write_summary(metrics_root, "ADNI_PM_DIRF_FIDELITY_FLOW_FULL", psnr=28.09, sharp=0.89)

    downstream = tmp_path / "classification_repeated_summary.csv"
    pd.DataFrame(
        [
            {
                "method": "Fidelity_Flow",
                "task": "cn_scd_vs_mci_ad",
                "n_subjects": 38,
                "macro_f1_mean": 0.52,
                "macro_f1_std": 0.04,
                "balanced_accuracy_mean": 0.53,
                "macro_auc_ovr_mean": 0.55,
            },
            {
                "method": "FREQ_FLOWBASE_B035",
                "task": "cn_scd_vs_mci_ad",
                "n_subjects": 38,
                "macro_f1_mean": 0.99,
                "macro_f1_std": 0.01,
                "balanced_accuracy_mean": 0.99,
                "macro_auc_ovr_mean": 0.99,
            },
            {
                "method": "ADNI_PM_DIRF_FIDELITY_FLOW_FULL",
                "task": "cn_vs_mci_spectrum",
                "n_subjects": 73,
                "macro_f1_mean": 0.57,
                "macro_f1_std": 0.03,
                "balanced_accuracy_mean": 0.56,
                "macro_auc_ovr_mean": 0.58,
            },
        ]
    ).to_csv(downstream, index=False)

    image = build_image_table(metrics_root, dataset="private")
    utility = build_downstream_table([downstream], dataset="private")

    assert "FREQ_FLOWBASE_B035" not in set(image["method"])
    assert "FREQ_FLOWBASE_B035" not in set(utility["method"])
    assert image.loc[image["method"].eq("PM_DIRF_FIDELITY_FLOW_FULL"), "display"].item() == "Ours Fidelity Flow"
    ours_main = utility[utility["method"].eq("PM_DIRF_FIDELITY_FLOW_FULL") & utility["task"].eq("cn_scd_vs_mci_ad")]
    assert ours_main["macro_f1_mean"].item() == 0.52


def test_two_dataset_builder_prefers_repeated_columns_and_accepts_single_run(tmp_path):
    from scripts.build_two_dataset_downstream_visuals import build_downstream_table

    repeated = tmp_path / "repeated.csv"
    single = tmp_path / "single.csv"
    pd.DataFrame(
        [
            {
                "method": "T1_ONLY",
                "task": "cn_vs_ad",
                "n_subjects": 21,
                "macro_f1_mean": 0.49,
                "macro_f1_std": 0.12,
                "balanced_accuracy_mean": 0.50,
                "macro_auc_ovr_mean": 0.48,
            }
        ]
    ).to_csv(repeated, index=False)
    pd.DataFrame(
        [
            {
                "method": "FA_GT",
                "task": "cn_vs_ad",
                "n_subjects": 21,
                "macro_f1": 0.61,
                "balanced_accuracy": 0.62,
                "macro_auc_ovr": 0.64,
            }
        ]
    ).to_csv(single, index=False)

    table = build_downstream_table([repeated, single], dataset="private")
    t1 = table[(table["method"].eq("T1_ONLY")) & (table["task"].eq("cn_vs_ad"))].iloc[0]
    gt = table[(table["method"].eq("FA_GT")) & (table["task"].eq("cn_vs_ad"))].iloc[0]

    assert t1["macro_f1_mean"] == 0.49
    assert t1["macro_f1_std"] == 0.12
    assert gt["macro_f1_mean"] == 0.61
    assert pd.isna(gt["macro_f1_std"])
