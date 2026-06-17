import pandas as pd

from scripts.ensemble_mil_seed_scores import ensemble_subject_scores


def test_ensemble_subject_scores_averages_scores_before_metrics():
    seed_a = pd.DataFrame(
        [
            {"method": "Ours", "task": "cn_vs_ad", "subject_id": "sub-001", "group_name": "CN", "y_true": 0, "subject_score": -0.2},
            {"method": "Ours", "task": "cn_vs_ad", "subject_id": "sub-002", "group_name": "AD", "y_true": 1, "subject_score": 0.1},
        ]
    )
    seed_b = pd.DataFrame(
        [
            {"method": "Ours", "task": "cn_vs_ad", "subject_id": "sub-001", "group_name": "CN", "y_true": 0, "subject_score": -0.4},
            {"method": "Ours", "task": "cn_vs_ad", "subject_id": "sub-002", "group_name": "AD", "y_true": 1, "subject_score": 0.9},
        ]
    )

    summary, pred = ensemble_subject_scores([seed_a, seed_b], run_name="allroi")

    assert pred["subject_score"].round(3).tolist() == [-0.3, 0.5]
    assert summary.loc[0, "accuracy"] == 1.0
    assert summary.loc[0, "macro_auc_ovr"] == 1.0
    assert summary.loc[0, "n_seeds"] == 2
