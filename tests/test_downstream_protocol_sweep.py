import pandas as pd

from scripts.downstream_protocol_sweep import build_estimator, evaluate_repeated_holdout


def _row(method: str, subject: str, group: str, x0: float, x1: float) -> dict:
    return {
        "method": method,
        "subject_id": subject,
        "group_name": group,
        "group_id": {"CN": 1, "MCI": 3}[group],
        "split": "test",
        "roi_mean_r0_c0_mean": x0,
        "roi_mean_r0_c1_mean": x1,
        "noise": 1.0 - x0,
    }


def test_build_estimator_keeps_feature_selection_inside_pipeline():
    estimator = build_estimator("rbf_svm", n_features=10, max_features=3, random_state=7)

    step_names = [name for name, _ in estimator.steps]

    assert step_names[0] == "select"
    assert step_names[1] == "scale"
    assert step_names[-1] == "clf"


def test_repeated_holdout_scores_separable_subject_features():
    rows = []
    for idx in range(10):
        rows.append(_row("Ours", f"sub-cn-{idx}", "CN", 0.05 + idx * 0.005, 0.1))
        rows.append(_row("Ours", f"sub-mci-{idx}", "MCI", 0.85 - idx * 0.005, 0.8))
    features = pd.DataFrame(rows)

    result = evaluate_repeated_holdout(
        features,
        method="Ours",
        task="cn_vs_mci",
        classifier="linear_svm",
        feature_set="roi_mean",
        max_features=1,
        test_size=0.2,
        seeds=[0, 1, 2],
    )

    assert result["accuracy_mean"] > 0.95
    assert result["macro_auc_ovr_mean"] > 0.95
    assert result["n_selected_features"] == 1
