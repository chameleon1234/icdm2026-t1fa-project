import numpy as np
import pandas as pd

from scripts.evaluate_slice_mil_roi import aggregate_slice_scores_by_subject, build_subject_bags


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
