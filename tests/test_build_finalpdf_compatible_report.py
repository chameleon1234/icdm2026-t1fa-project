from pathlib import Path

import pandas as pd


def test_finalpdf_report_merges_single_and_repeated_tables(tmp_path):
    from scripts.build_finalpdf_compatible_report import build_report_table

    private_root = tmp_path / "private"
    adni_root = tmp_path / "adni"
    private_root.mkdir()
    adni_root.mkdir()
    pd.DataFrame(
        [
            {"method": "T1_PLUS_Ours", "task": "cn_vs_ad", "accuracy": 0.8, "macro_auc_ovr": 0.7, "macro_f1": 0.75},
            {"method": "FREQ_BAD", "task": "cn_vs_ad", "accuracy": 1.0, "macro_auc_ovr": 1.0, "macro_f1": 1.0},
        ]
    ).to_csv(private_root / "classification_summary.csv", index=False)
    pd.DataFrame(
        [
            {"method": "T1_PLUS_Ours", "task": "cn_vs_ad", "accuracy_mean": 0.78, "macro_auc_ovr_mean": 0.69, "macro_f1_mean": 0.73},
            {"method": "FREQ_BAD", "task": "cn_vs_ad", "accuracy_mean": 1.0, "macro_auc_ovr_mean": 1.0, "macro_f1_mean": 1.0},
        ]
    ).to_csv(private_root / "classification_repeated_summary.csv", index=False)
    pd.DataFrame(
        [{"method": "T1_PLUS_Ours", "task": "cn_vs_ad", "accuracy": 0.6, "macro_auc_ovr": 0.65, "macro_f1": 0.55}]
    ).to_csv(adni_root / "classification_summary.csv", index=False)
    pd.DataFrame(
        [{"method": "T1_PLUS_Ours", "task": "cn_vs_ad", "accuracy_mean": 0.61, "macro_auc_ovr_mean": 0.66, "macro_f1_mean": 0.56}]
    ).to_csv(adni_root / "classification_repeated_summary.csv", index=False)

    table = build_report_table(private_root, adni_root)

    assert "FREQ_BAD" not in set(table["method"])
    private = table[table["dataset"].eq("private")].iloc[0]
    assert private["accuracy_pct"] == 80.0
    assert private["repeated_macro_f1_pct"] == 73.0
    assert set(table["dataset"]) == {"private", "adni"}
