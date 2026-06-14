import numpy as np
import pandas as pd

from scripts.evaluate_slice_mil_roi import (
    aggregate_slice_scores_by_subject,
    build_subject_bags,
    select_shared_disease_roi_columns,
)


def test_aggregate_slice_scores_reports_subject_level_metrics():
    rows = pd.DataFrame(
        [
            {"subject_id": "sub-001", "group_name": "CN", "y_true": 0, "decision_score": -1.0},
            {"subject_id": "sub-001", "group_name": "CN", "y_true": 0, "decision_score": -0.5},
            {"subject_id": "sub-002", "group_name": "AD", "y_true": 1, "decision_score": 0.2},
            {"subject_id": "sub-002", "group_name": "AD", "y_true": 1, "decision_score": 1.2},
        ]
    )

    summary, subject_predictions = aggregate_slice_scores_by_subject(rows, method="Ours", task="cn_vs_ad")

    assert summary["accuracy"] == 1.0
    assert summary["macro_auc_ovr"] == 1.0
    assert summary["n_test_subjects"] == 2
    assert summary["n_test_slices"] == 4
    assert subject_predictions["subject_score"].round(3).tolist() == [-0.75, 0.7]


def test_build_subject_bags_keeps_slice_rows_grouped_by_subject():
    features = pd.DataFrame(
        [
            {"subject_id": "sub-001", "slice_idx": 21, "group_name": "CN", "roi_label_1_mean": 0.2},
            {"subject_id": "sub-001", "slice_idx": 20, "group_name": "CN", "roi_label_1_mean": 0.1},
            {"subject_id": "sub-002", "slice_idx": 20, "group_name": "AD", "roi_label_1_mean": 0.9},
        ]
    )
    columns = ["roi_label_1_mean"]
    labels = {"CN": 0, "AD": 1}

    bags = build_subject_bags(features, columns, labels)

    assert [bag.subject_id for bag in bags] == ["sub-001", "sub-002"]
    assert bags[0].x.shape == (2, 1)
    np.testing.assert_allclose(np.asarray(bags[0].x).reshape(-1), [0.1, 0.2], atol=1e-6)
    assert bags[1].y == 1


def test_shared_disease_roi_selection_averages_across_tasks():
    rows = []
    for idx, group in enumerate(["CN", "SCD", "MCI", "AD"]):
        for rep in range(4):
            rows.append(
                {
                    "method": "Ours",
                    "subject_id": f"sub-{idx}-{rep}",
                    "sample_id": f"sub-{idx}-{rep}_z020",
                    "slice_idx": 20,
                    "group_name": group,
                    "roi_label_1_mean": float(idx),  # separates all disease progression tasks
                    "roi_label_2_mean": float(rep % 2),  # nuisance
                    "roi_label_3_mean": float(idx == 3),  # AD-specific only
                }
            )
    features = pd.DataFrame(rows)

    selected = select_shared_disease_roi_columns(
        features,
        feature_set="roi_mean",
        tasks=["cn_vs_mci", "cn_vs_ad", "mci_vs_ad"],
        top_k=2,
    )

    assert "roi_label_1_mean" in selected
    assert len(selected) == 2
