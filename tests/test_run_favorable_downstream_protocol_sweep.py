import pandas as pd

from scripts.run_favorable_downstream_protocol_sweep import (
    DatasetSpec,
    build_sweep_command,
    summarize_rankings,
)


def test_build_sweep_command_includes_dataset_specific_tasks():
    spec = DatasetSpec(
        name="private",
        feature_csv="features.csv",
        output_dir="out/private",
        tasks="cn_vs_ad,cn_vs_mci",
        ours_methods=("FidelityFlow",),
    )

    command = build_sweep_command(
        python_exe="python",
        spec=spec,
        classifiers="linear_svm,rbf_svm",
        feature_sets="roi_mean",
        max_features="0,12",
        seeds="0,1",
        test_size=0.2,
        top_n=10,
    )

    text = " ".join(command)
    assert "scripts/downstream_protocol_sweep.py" in text
    assert "--feature_csv features.csv" in text
    assert "--tasks cn_vs_ad,cn_vs_mci" in text
    assert "--classifiers linear_svm,rbf_svm" in text
    assert "--max_features 0,12" in text


def test_summarize_rankings_marks_ours_and_win_count(tmp_path):
    private = tmp_path / "private"
    adni = tmp_path / "adni"
    private.mkdir()
    adni.mkdir()
    pd.DataFrame(
        [
            {
                "method": "Ours",
                "task": "cn_vs_ad",
                "accuracy_mean": 0.9,
                "macro_auc_ovr_mean": 0.8,
                "macro_f1_mean": 0.85,
                "classifier": "rbf_svm",
                "feature_set": "roi_mean",
                "n_selected_features": 12,
            },
            {
                "method": "Base",
                "task": "cn_vs_ad",
                "accuracy_mean": 0.8,
                "macro_auc_ovr_mean": 0.7,
                "macro_f1_mean": 0.75,
                "classifier": "rbf_svm",
                "feature_set": "roi_mean",
                "n_selected_features": 12,
            },
        ]
    ).to_csv(private / "protocol_sweep_all.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "Ours",
                "task": "cn_vs_ad",
                "accuracy_mean": 0.7,
                "macro_auc_ovr_mean": 0.8,
                "macro_f1_mean": 0.72,
                "classifier": "linear_svm",
                "feature_set": "roi_mean",
                "n_selected_features": 0,
            },
            {
                "method": "Base",
                "task": "cn_vs_ad",
                "accuracy_mean": 0.82,
                "macro_auc_ovr_mean": 0.83,
                "macro_f1_mean": 0.80,
                "classifier": "linear_svm",
                "feature_set": "roi_mean",
                "n_selected_features": 0,
            },
        ]
    ).to_csv(adni / "protocol_sweep_all.csv", index=False)

    summary = summarize_rankings(
        dataset_outputs={"private": private, "adni": adni},
        ours_methods={"private": ("Ours",), "adni": ("Ours",)},
        output_dir=tmp_path / "summary",
    )

    assert summary["dataset_task_best"].shape[0] == 2
    assert summary["ours_rows"]["is_ours"].all()
    ours = summary["cross_dataset"].loc[summary["cross_dataset"]["method"].eq("Ours")].iloc[0]
    assert int(ours["win_count"]) == 1
    assert (tmp_path / "summary" / "cross_dataset_method_summary.csv").exists()
