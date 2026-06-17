import numpy as np
import pandas as pd

from scripts.evaluate_slice_mil_roi import (
    _select_and_scale,
    aggregate_slice_scores_by_subject,
    build_subject_bags,
    build_multitask_subject_bags,
    select_shared_disease_roi_columns,
    split_late_fusion_columns,
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


def test_split_late_fusion_columns_uses_two_modal_prefixes():
    columns = [
        "T1_ONLY__roi_label_1_mean",
        "T1_ONLY__roi_label_2_mean",
        "PM_DIRF_FIDELITY_FLOW__roi_label_1_mean",
        "PM_DIRF_FIDELITY_FLOW__roi_label_2_mean",
    ]

    left, right = split_late_fusion_columns(columns)

    assert left == columns[:2]
    assert right == columns[2:]


def test_select_and_scale_can_preserve_late_fusion_prefixes():
    train = pd.DataFrame(
        [
            {"method": "Fused", "sample_id": "a_z020", "subject_id": "a", "slice_idx": 20, "group_name": "CN", "y_label": 0, "T1__roi_label_1_mean": 0.1, "FA__roi_label_1_mean": 0.2},
            {"method": "Fused", "sample_id": "b_z020", "subject_id": "b", "slice_idx": 20, "group_name": "AD", "y_label": 1, "T1__roi_label_1_mean": 0.9, "FA__roi_label_1_mean": 0.8},
        ]
    )
    test = train.copy()

    _, _, columns = _select_and_scale(train, test, "roi_mean", 0, preserve_column_names=True)

    assert columns == ["T1__roi_label_1_mean", "FA__roi_label_1_mean"]


def test_build_multitask_subject_bags_uses_task_specific_labels():
    features = pd.DataFrame(
        [
            {"subject_id": "sub-cn", "slice_idx": 20, "group_name": "CN", "feature_000": 0.1},
            {"subject_id": "sub-mci", "slice_idx": 20, "group_name": "MCI", "feature_000": 0.5},
            {"subject_id": "sub-ad", "slice_idx": 20, "group_name": "AD", "feature_000": 0.9},
        ]
    )

    task_bags = build_multitask_subject_bags(features, ["feature_000"], ["cn_vs_mci", "mci_vs_ad"])

    labels = {(task, bag.subject_id): bag.y for task, bag in task_bags}
    assert labels[("cn_vs_mci", "sub-cn")] == 0
    assert labels[("cn_vs_mci", "sub-mci")] == 1
    assert ("cn_vs_mci", "sub-ad") not in labels
    assert labels[("mci_vs_ad", "sub-mci")] == 0
    assert labels[("mci_vs_ad", "sub-ad")] == 1
