from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def ensemble_subject_scores(seed_predictions: list[pd.DataFrame], run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not seed_predictions:
        raise ValueError("At least one seed prediction table is required")
    normalized = []
    for seed_idx, table in enumerate(seed_predictions):
        required = {"method", "task", "subject_id", "group_name", "y_true", "subject_score"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Seed table {seed_idx} is missing columns: {sorted(missing)}")
        keep = table[list(required)].copy()
        keep["seed_idx"] = int(seed_idx)
        normalized.append(keep)
    stacked = pd.concat(normalized, ignore_index=True)
    grouped = stacked.groupby(["method", "task", "subject_id"], sort=True)
    rows: list[dict[str, Any]] = []
    for (method, task, subject_id), group in grouped:
        y_values = group["y_true"].astype(int).unique().tolist()
        group_names = group["group_name"].astype(str).unique().tolist()
        if len(y_values) != 1 or len(group_names) != 1:
            raise ValueError(f"Inconsistent labels for {method}/{task}/{subject_id}")
        rows.append(
            {
                "run_name": run_name,
                "method": method,
                "task": task,
                "subject_id": subject_id,
                "group_name": group_names[0],
                "y_true": int(y_values[0]),
                "subject_score": float(group["subject_score"].mean()),
                "n_seeds": int(group["seed_idx"].nunique()),
            }
        )
    predictions = pd.DataFrame(rows).sort_values(["task", "method", "subject_id"]).reset_index(drop=True)
    predictions["y_pred"] = (predictions["subject_score"] >= 0.0).astype(int)

    summaries: list[dict[str, Any]] = []
    for (method, task), group in predictions.groupby(["method", "task"], sort=True):
        y_true = group["y_true"].to_numpy(dtype=np.int64)
        y_pred = group["y_pred"].to_numpy(dtype=np.int64)
        scores = group["subject_score"].to_numpy(dtype=np.float32)
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = float("nan")
        summaries.append(
            {
                "run_name": run_name,
                "method": method,
                "task": task,
                "protocol": "seed_score_ensemble",
                "n_seeds": int(group["n_seeds"].max()),
                "n_test_subjects": int(group["subject_id"].nunique()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_auc_ovr": auc,
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["task", "accuracy", "macro_auc_ovr"], ascending=[True, False, False])
    return summary, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average subject-level MIL scores across random seeds.")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--prediction_csv", action="append", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tables = [pd.read_csv(path) for path in args.prediction_csv]
    summary, predictions = ensemble_subject_scores(tables, run_name=args.run_name)
    summary.to_csv(output_root / "classification_subject_ensemble_summary.csv", index=False, encoding="utf-8")
    predictions.to_csv(output_root / "classification_subject_ensemble_predictions.csv", index=False, encoding="utf-8")
    with open(output_root / "classification_subject_ensemble_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary.to_dict(orient="records")), handle, indent=2, ensure_ascii=False)
    print(f"Saved ensemble summary to: {output_root / 'classification_subject_ensemble_summary.csv'}")
    print(summary[["run_name", "method", "task", "n_seeds", "n_test_subjects", "accuracy", "macro_auc_ovr", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
