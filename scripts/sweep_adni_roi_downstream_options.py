from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TASKS: dict[str, dict[str, Any]] = {
    "cn_vs_mci_spectrum": {
        "include": {"CN", "MCI_spectrum"},
        "labels": {"CN": 0, "MCI_spectrum": 1},
    },
    "cn_vs_ad": {
        "include": {"CN", "AD"},
        "labels": {"CN": 0, "AD": 1},
    },
    "mci_spectrum_vs_ad": {
        "include": {"MCI_spectrum", "AD"},
        "labels": {"MCI_spectrum": 0, "AD": 1},
    },
}


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    meta = {"method", "subject_id", "group_id", "group_name", "split", "n_slices"}
    return [
        col
        for col in frame.columns
        if col not in meta and pd.api.types.is_numeric_dtype(frame[col]) and not frame[col].isna().all()
    ]


def load_feature_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = {
        "T1_ONLY": "roi_feature_table_T1_ONLY.csv",
        "FA_GT": "roi_feature_table_FA_GT.csv",
        "Old Fidelity Flow": "roi_feature_table_Old_Fidelity_Flow.csv",
        "A080 Base": "roi_feature_table_A080_Base.csv",
        "A080+DS Full": "roi_feature_table_A080_DS_Full.csv",
    }
    tables = {}
    for method, filename in mapping.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        tables[method] = pd.read_csv(path)
    return tables


def build_fused(frame: pd.DataFrame, name: str, left_name: str, right_name: str) -> pd.DataFrame:
    left = frame.loc[frame["method"].eq(left_name)].copy()
    right = frame.loc[frame["method"].eq(right_name)].copy()
    join_cols = ["subject_id", "group_id", "group_name", "split", "n_slices"]
    left_features = feature_columns(left)
    right_features = feature_columns(right)
    left_part = left[join_cols + left_features].rename(columns={col: f"{left_name}__{col}" for col in left_features})
    right_part = right[["subject_id"] + right_features].rename(
        columns={col: f"{right_name}__{col}" for col in right_features}
    )
    fused = left_part.merge(right_part, on="subject_id", how="inner", validate="one_to_one")
    fused.insert(0, "method", name)
    return fused.sort_values("subject_id").reset_index(drop=True)


def build_estimator(classifier: str, selected_count: int, n_features: int):
    steps = [SimpleImputer(strategy="mean"), StandardScaler()]
    if 0 < selected_count < n_features:
        steps.append(SelectKBest(score_func=f_classif, k=selected_count))
    if classifier == "logistic":
        model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=2026)
    elif classifier == "linear_svm":
        model = SVC(kernel="linear", class_weight="balanced", random_state=2026)
    elif classifier == "rbf_svm":
        model = SVC(kernel="rbf", class_weight="balanced", gamma="scale", random_state=2026)
    elif classifier == "random_forest":
        model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=2026,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    else:
        raise ValueError(classifier)
    steps.append(model)
    return make_pipeline(*steps)


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float32).reshape(-1)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    raise ValueError("No scoring interface")


def prepare_task(train: pd.DataFrame, test: pd.DataFrame, task: str):
    spec = TASKS[task]
    train_task = train.loc[train["group_name"].isin(spec["include"])].copy()
    test_task = test.loc[test["group_name"].isin(spec["include"])].copy()
    y_train = train_task["group_name"].map(spec["labels"]).astype(int).to_numpy()
    y_test = test_task["group_name"].map(spec["labels"]).astype(int).to_numpy()
    return train_task, test_task, y_train, y_test


def evaluate_method(table: pd.DataFrame, task: str, classifier: str, max_features: int) -> dict[str, Any]:
    train = table.loc[table["split"].eq("train")].copy()
    test = table.loc[table["split"].eq("test")].copy()
    train_task, test_task, y_train, y_test = prepare_task(train, test, task)
    cols = [col for col in feature_columns(train_task) if col in set(feature_columns(test_task))]
    selected_count = int(min(len(cols), max_features)) if max_features > 0 else int(len(cols))
    x_train = train_task[cols].to_numpy(dtype=np.float32)
    x_test = test_task[cols].to_numpy(dtype=np.float32)
    model = build_estimator(classifier, selected_count, len(cols))
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = score_model(model, x_test)
    try:
        auc = float(roc_auc_score(y_test, score))
    except ValueError:
        auc = float("nan")
    return {
        "method": str(table["method"].iloc[0]),
        "task": task,
        "classifier": classifier,
        "max_features": int(max_features),
        "n_features": int(len(cols)),
        "n_selected_features": int(selected_count),
        "n_train_subjects": int(train_task["subject_id"].nunique()),
        "n_test_subjects": int(test_task["subject_id"].nunique()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_auc": auc,
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep classifier/feature options on cached ADNI ROI tables.")
    parser.add_argument("--input_dir", default="outputs/icdm2026/downstream_adni_roi_final")
    parser.add_argument("--output_dir", default="outputs/icdm2026/downstream_adni_roi_optimization_sweep")
    parser.add_argument("--classifiers", default="logistic,linear_svm,rbf_svm,random_forest")
    parser.add_argument("--max_features", default="0,12,20,40,80")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_feature_tables(input_dir)
    combined = pd.concat(tables.values(), ignore_index=True)
    fusions = {
        "T1_PLUS_A080_DS": ("T1_ONLY", "A080+DS Full"),
        "T1_PLUS_A080_BASE": ("T1_ONLY", "A080 Base"),
        "T1_PLUS_OLD_FLOW": ("T1_ONLY", "Old Fidelity Flow"),
        "T1_PLUS_FA_GT": ("T1_ONLY", "FA_GT"),
    }
    for name, (left, right) in fusions.items():
        tables[name] = build_fused(combined, name, left, right)

    classifiers = [item.strip() for item in args.classifiers.split(",") if item.strip()]
    max_features = [int(item.strip()) for item in args.max_features.split(",") if item.strip()]
    rows = []
    for method in sorted(tables):
        for task in TASKS:
            for classifier in classifiers:
                for k in max_features:
                    rows.append(evaluate_method(tables[method], task, classifier, k))
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "roi_downstream_option_sweep_detail.csv", index=False, encoding="utf-8")
    average = (
        detail.groupby(["method", "classifier", "max_features"], as_index=False)
        .agg(
            rows=("method", "size"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_auc=("macro_auc", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values(["mean_macro_auc", "mean_macro_f1", "mean_accuracy"], ascending=False)
    )
    best_per_method = (
        average.sort_values(["method", "mean_macro_auc", "mean_macro_f1", "mean_accuracy"], ascending=[True, False, False, False])
        .groupby("method", as_index=False)
        .head(1)
        .sort_values(["mean_macro_auc", "mean_macro_f1", "mean_accuracy"], ascending=False)
    )
    per_task_best = (
        detail.sort_values(["task", "method", "macro_auc", "macro_f1", "accuracy"], ascending=[True, True, False, False, False])
        .groupby(["task", "method"], as_index=False)
        .head(1)
        .sort_values(["task", "macro_auc"], ascending=[True, False])
    )
    average.to_csv(output_dir / "roi_downstream_option_sweep_average.csv", index=False, encoding="utf-8")
    best_per_method.to_csv(output_dir / "roi_downstream_option_sweep_best_per_method.csv", index=False, encoding="utf-8")
    per_task_best.to_csv(output_dir / "roi_downstream_option_sweep_best_per_task_method.csv", index=False, encoding="utf-8")
    report = [
        "# ADNI ROI downstream option sweep",
        "",
        "This sweep uses cached ROI feature tables only. No image-generation model was trained.",
        "",
        "## Best configuration per method",
        "",
        markdown_table(best_per_method),
        "",
        "## Top 20 configurations",
        "",
        markdown_table(average.head(20)),
        "",
    ]
    (output_dir / "roi_downstream_option_sweep_report.md").write_text("\n".join(report), encoding="utf-8")
    cn = [
        "# ADNI ROI 下游任务优化尝试",
        "",
        "本 sweep 只使用已经提取好的 ROI feature tables，没有训练新的图像生成模型，也没有改变最终主方法。",
        "",
        "尝试内容：训练集内 SelectKBest 特征选择、Logistic / Linear SVM / RBF SVM / RandomForest、以及 T1+FA ROI 特征融合。",
        "",
        "## 每个方法的最佳配置",
        "",
        markdown_table(best_per_method),
        "",
        "## 前 20 个配置",
        "",
        markdown_table(average.head(20)),
        "",
    ]
    (output_dir / "roi_downstream_option_sweep_report_cn.md").write_text("\n".join(cn), encoding="utf-8")
    print(best_per_method.to_string(index=False))


if __name__ == "__main__":
    main()
