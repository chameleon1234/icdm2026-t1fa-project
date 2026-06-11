import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.downstream_utility import TASK_DEFINITIONS, feature_columns


CLASSIFIERS = ("linear_svm", "rbf_svm", "poly_svm", "random_forest", "gradient_boosting", "knn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep final.pdf-like downstream protocols over existing subject feature tables. "
            "This is an exploration tool: feature selection/scaling are fit inside each train split."
        )
    )
    parser.add_argument("--feature_csv", default="outputs/icdm2026/downstream_private_roi_sweep/subject_features.csv")
    parser.add_argument("--output_dir", default="outputs/icdm2026/downstream_protocol_sweep_private")
    parser.add_argument("--methods", default="", help="Comma-separated method names. Empty keeps all methods.")
    parser.add_argument("--tasks", default="cn_scd_vs_mci_ad,cn_vs_mci,mci_vs_ad,cn_vs_ad")
    parser.add_argument("--classifiers", default="linear_svm,rbf_svm,random_forest,gradient_boosting,knn")
    parser.add_argument("--feature_sets", default="roi_mean,full")
    parser.add_argument("--max_features", default="0,3,6,12,24")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--top_n", type=int, default=80)
    return parser.parse_args()


def _parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_estimator(classifier: str, n_features: int, max_features: int, random_state: int):
    selected = n_features if max_features <= 0 else min(max_features, n_features)
    steps = []
    if selected < n_features:
        steps.append(("select", SelectKBest(score_func=f_classif, k=selected)))
    steps.append(("scale", StandardScaler()))
    if classifier == "linear_svm":
        clf = SVC(kernel="linear", class_weight="balanced", random_state=random_state)
    elif classifier == "rbf_svm":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=random_state)
    elif classifier == "poly_svm":
        clf = SVC(kernel="poly", degree=2, class_weight="balanced", random_state=random_state)
    elif classifier == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        )
    elif classifier == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=random_state)
    elif classifier == "knn":
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
    else:
        raise ValueError(f"Unknown classifier {classifier!r}; choose from {CLASSIFIERS}")
    steps.append(("clf", clf))
    return Pipeline(steps)


def _prepare_task(features: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task: {task}")
    definition = TASK_DEFINITIONS[task]
    subset = features.loc[features["group_name"].isin(definition["include"])].copy()
    if subset.empty:
        raise ValueError(f"No subjects for task {task}")
    y = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
    if len(set(y.tolist())) != 2:
        raise ValueError(f"Protocol sweep currently supports binary tasks only, got {task}")
    return subset.reset_index(drop=True), y


def _positive_scores(estimator, x_test: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x_test)
        classes = estimator.steps[-1][1].classes_
        pos_idx = int(np.where(classes == max(classes))[0][0])
        return proba[:, pos_idx]
    score = estimator.decision_function(x_test)
    return np.asarray(score, dtype=np.float32).reshape(-1)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1:
        return float(finite[0]), 0.0
    return float(np.mean(finite)), float(np.std(finite, ddof=1))


def evaluate_repeated_holdout(
    features: pd.DataFrame,
    method: str,
    task: str,
    classifier: str,
    feature_set: str,
    max_features: int,
    test_size: float,
    seeds: Iterable[int],
) -> dict[str, float | int | str]:
    subset, y = _prepare_task(features, task)
    columns = feature_columns(subset, feature_set=feature_set)
    x = subset[columns].to_numpy(dtype=np.float32)
    selected = len(columns) if max_features <= 0 else min(max_features, len(columns))
    metric_values = {"accuracy": [], "balanced_accuracy": [], "macro_f1": [], "macro_auc_ovr": []}
    for seed in seeds:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            random_state=int(seed),
            stratify=y,
        )
        estimator = build_estimator(classifier, n_features=len(columns), max_features=max_features, random_state=int(seed))
        estimator.fit(x[train_idx], y[train_idx])
        pred = estimator.predict(x[test_idx])
        score = _positive_scores(estimator, x[test_idx])
        metric_values["accuracy"].append(float(accuracy_score(y[test_idx], pred)))
        metric_values["balanced_accuracy"].append(float(balanced_accuracy_score(y[test_idx], pred)))
        metric_values["macro_f1"].append(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)))
        try:
            metric_values["macro_auc_ovr"].append(float(roc_auc_score(y[test_idx], score)))
        except ValueError:
            metric_values["macro_auc_ovr"].append(float("nan"))
    result: dict[str, float | int | str] = {
        "method": method,
        "task": task,
        "classifier": classifier,
        "feature_set": feature_set,
        "n_subjects": int(len(y)),
        "n_features": int(len(columns)),
        "n_selected_features": int(selected),
        "test_size": float(test_size),
        "n_repeats": int(len(list(seeds))),
    }
    for metric, values in metric_values.items():
        mean, std = _mean_std(values)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
    return result


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.feature_csv)
    methods = _parse_csv_list(args.methods) or sorted(features["method"].unique().tolist())
    tasks = _parse_csv_list(args.tasks)
    classifiers = _parse_csv_list(args.classifiers)
    feature_sets = _parse_csv_list(args.feature_sets)
    max_feature_values = _parse_int_list(args.max_features)
    seeds = _parse_int_list(args.seeds)

    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        method_features = features.loc[features["method"].eq(method)].copy()
        if method_features.empty:
            continue
        for task in tasks:
            for classifier in classifiers:
                for feature_set in feature_sets:
                    for max_features in max_feature_values:
                        try:
                            rows.append(
                                evaluate_repeated_holdout(
                                    method_features,
                                    method=method,
                                    task=task,
                                    classifier=classifier,
                                    feature_set=feature_set,
                                    max_features=max_features,
                                    test_size=args.test_size,
                                    seeds=seeds,
                                )
                            )
                        except ValueError as exc:
                            rows.append(
                                {
                                    "method": method,
                                    "task": task,
                                    "classifier": classifier,
                                    "feature_set": feature_set,
                                    "n_selected_features": max_features,
                                    "error": str(exc),
                                }
                            )
    result = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "protocol_sweep_all.csv", index=False, encoding="utf-8")
    ranked = result.loc[result["accuracy_mean"].notna()].copy()
    ranked["finalpdf_like_score"] = ranked[["accuracy_mean", "macro_auc_ovr_mean", "macro_f1_mean"]].mean(axis=1)
    ranked = ranked.sort_values(["finalpdf_like_score", "accuracy_mean", "macro_auc_ovr_mean"], ascending=False)
    top = ranked.head(args.top_n)
    top.to_csv(output_dir / "protocol_sweep_top.csv", index=False, encoding="utf-8")
    with open(output_dir / "protocol_sweep_top.json", "w", encoding="utf-8") as handle:
        json.dump(top.to_dict(orient="records"), handle, indent=2)
    print(top[
        [
            "method",
            "task",
            "classifier",
            "feature_set",
            "n_selected_features",
            "accuracy_mean",
            "macro_auc_ovr_mean",
            "macro_f1_mean",
            "finalpdf_like_score",
        ]
    ].to_string(index=False))
    print(f"Saved protocol sweep to: {output_dir}")


if __name__ == "__main__":
    main()
