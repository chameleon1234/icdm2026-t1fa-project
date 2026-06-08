from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.downstream_utility import TASK_DEFINITIONS, feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ROI-mean discriminative features for final.pdf-style binary tasks.")
    parser.add_argument("--subject_features", required=True, help="CSV produced by evaluate_downstream_classification.py")
    parser.add_argument("--tasks", default="cn_vs_ad,cn_vs_mci,mci_vs_ad")
    parser.add_argument("--methods", default="", help="Optional comma-separated method subset.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _cohens_d(values0: np.ndarray, values1: np.ndarray) -> float:
    values0 = np.asarray(values0, dtype=np.float64)
    values1 = np.asarray(values1, dtype=np.float64)
    n0 = values0.size
    n1 = values1.size
    if n0 < 2 or n1 < 2:
        return float("nan")
    pooled = np.sqrt(((n0 - 1) * np.var(values0, ddof=1) + (n1 - 1) * np.var(values1, ddof=1)) / max(n0 + n1 - 2, 1))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(values1) - np.mean(values0)) / pooled)


def analyze_features(features: pd.DataFrame, tasks: list[str], methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    method_names = methods or sorted(features["method"].unique().tolist())
    for method in method_names:
        method_frame = features.loc[features["method"].eq(method)].copy()
        if method_frame.empty:
            raise ValueError(f"Method not found in feature table: {method}")
        roi_columns = feature_columns(method_frame, feature_set="roi_mean")
        for task in tasks:
            definition = TASK_DEFINITIONS[task]
            subset = method_frame.loc[method_frame["group_name"].isin(definition["include"])].copy()
            if subset.empty:
                continue
            labels = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
            if len(set(labels.tolist())) != 2:
                continue
            for column in roi_columns:
                values = subset[column].to_numpy(dtype=np.float64)
                class0 = values[labels == 0]
                class1 = values[labels == 1]
                auc = float("nan")
                try:
                    auc = float(roc_auc_score(labels, values))
                    auc = max(auc, 1.0 - auc)
                except ValueError:
                    pass
                rows.append(
                    {
                        "method": method,
                        "task": task,
                        "feature": column,
                        "n_subjects": int(len(subset)),
                        "mean_class0": float(np.mean(class0)) if class0.size else float("nan"),
                        "mean_class1": float(np.mean(class1)) if class1.size else float("nan"),
                        "abs_delta": float(abs(np.mean(class1) - np.mean(class0))) if class0.size and class1.size else float("nan"),
                        "cohens_d": _cohens_d(class0, class1),
                        "abs_cohens_d": abs(_cohens_d(class0, class1)),
                        "univariate_auc": auc,
                    }
                )
            task_rows = [row for row in rows if row["method"] == method and row["task"] == task]
            top_auc = sorted(task_rows, key=lambda row: float(row["univariate_auc"]), reverse=True)[:3]
            summary_rows.append(
                {
                    "method": method,
                    "task": task,
                    "n_roi_features": int(len(roi_columns)),
                    "top3_auc_mean": float(np.nanmean([row["univariate_auc"] for row in top_auc])) if top_auc else float("nan"),
                    "top_feature": str(top_auc[0]["feature"]) if top_auc else "",
                    "top_feature_auc": float(top_auc[0]["univariate_auc"]) if top_auc else float("nan"),
                    "top_feature_abs_d": float(top_auc[0]["abs_cohens_d"]) if top_auc else float("nan"),
                }
            )
    detail = pd.DataFrame(rows).sort_values(["task", "method", "univariate_auc"], ascending=[True, True, False])
    summary = pd.DataFrame(summary_rows).sort_values(["task", "top3_auc_mean"], ascending=[True, False])
    return detail, summary


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.subject_features)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = analyze_features(features, tasks=tasks, methods=methods)
    detail.to_csv(output_dir / "roi_feature_discriminative_detail.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "roi_feature_discriminative_summary.csv", index=False, encoding="utf-8")
    print(f"Saved ROI feature detail to: {output_dir / 'roi_feature_discriminative_detail.csv'}")
    print(f"Saved ROI feature summary to: {output_dir / 'roi_feature_discriminative_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
