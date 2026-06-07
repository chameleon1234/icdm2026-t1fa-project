import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import load_subject_index, normalize_subject_id
from src.eval.downstream_utility import (
    build_fused_feature_table,
    confusion_matrix_frame,
    extract_subject_features_from_folder,
    parse_method_specs,
    run_classification_cv,
    run_repeated_classification_cv,
    run_regression_cv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate subject-level downstream disease utility from image folders.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--subject_index_csv", default="", help="Optional prebuilt subject index CSV for tests or custom splits.")
    parser.add_argument("--adni_slice_manifest", default="", help="Optional ADNI slice manifest CSV produced by preprocess_adni_slices.py.")
    parser.add_argument("--split", default="test", help="Subject split to evaluate with stratified CV.")
    parser.add_argument("--method", action="append", default=[], help="Method spec in NAME=DIR format. Can be repeated.")
    parser.add_argument("--fusion", action="append", default=[], help="Fusion spec in NAME=LEFT+RIGHT format using selected method names.")
    parser.add_argument("--include_t1", action="store_true", help="Include the split T1 folder as a T1-only baseline.")
    parser.add_argument("--include_fa_gt", action="store_true", help="Include the split FA folder as an upper-bound reference.")
    parser.add_argument("--tasks", default="", help="Comma-separated task list. Defaults to config evaluation.classification_tasks.")
    parser.add_argument("--regression_targets", default="", help="Comma-separated subject-level regression targets, e.g. MMSE.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--repeat_seeds", default="", help="Comma-separated seeds for repeated classification CV robustness output.")
    parser.add_argument("--max_features", type=int, default=0, help="Fold-internal SelectKBest feature cap. 0 keeps all image features.")
    parser.add_argument("--regression_clip_min", type=float, default=0.0, help="Minimum clipped regression prediction.")
    parser.add_argument("--regression_clip_max", type=float, default=30.0, help="Maximum clipped regression prediction.")
    parser.add_argument("--output_root", default="", help="Override downstream output root.")
    parser.add_argument("--brain_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--feature_view", choices=["full", "lowpass", "highpass"], default="full")
    parser.add_argument("--frequency_sigma", type=float, default=1.5)
    return parser.parse_args()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def _load_subject_index(args: argparse.Namespace, config: dict[str, Any]) -> pd.DataFrame:
    if args.subject_index_csv:
        return pd.read_csv(args.subject_index_csv)
    if args.adni_slice_manifest:
        return _load_adni_subject_index(args.adni_slice_manifest)
    data_config = config["data"]
    return load_subject_index(
        excel_path=data_config["excel_path"],
        split_json=data_config["split_json"],
        sheet_name=data_config.get("excel_sheet", "re_order"),
    )


def _load_adni_subject_index(slice_manifest_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(slice_manifest_csv)
    required = {"subject", "split", "normalized_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ADNI slice manifest is missing columns: {sorted(missing)}")
    subjects = frame[["subject", "split", "normalized_group"]].drop_duplicates("subject").copy()
    subjects = subjects.rename(columns={"normalized_group": "group_name"})
    group_ids = {"UNLABELED": 0, "CN": 1, "MCI_spectrum": 2, "AD": 3, "EXCLUDE": 99}
    subjects["subject_id"] = subjects["subject"].map(normalize_subject_id)
    subjects["group_id"] = subjects["group_name"].map(group_ids).fillna(99).astype(int)
    for column in ["gender", "age", "edu", "MMSE"]:
        subjects[column] = np.nan
    columns = ["subject_id", "group_id", "group_name", "gender", "age", "edu", "MMSE", "split"]
    return subjects[columns].sort_values("subject_id").reset_index(drop=True)


def _default_split_dir(config: dict[str, Any], split: str, modality: str) -> Path:
    processed_root = Path(config["data"]["processed_root"])
    return processed_root / split / f"{modality}_slices"


def _parse_fusion_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec or "+" not in spec:
        raise ValueError(f"Fusion spec must use NAME=LEFT+RIGHT format, got {spec!r}")
    name, pair = spec.split("=", 1)
    left, right = pair.split("+", 1)
    name = name.strip()
    left = left.strip()
    right = right.strip()
    if not name or not left or not right:
        raise ValueError(f"Invalid fusion spec: {spec!r}")
    return name, left, right


def _parse_seed_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _write_confusion_png(path: Path, matrix_df: pd.DataFrame, title: str) -> None:
    cell = 74
    top = 52
    left = 92
    height = top + cell * (len(matrix_df.index) + 1)
    width = left + cell * (len(matrix_df.columns) + 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    values = matrix_df.to_numpy(dtype=np.float32)
    max_value = float(values.max()) if values.size and values.max() > 0 else 1.0
    cv2.putText(canvas, title[:48], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    for col_idx, label in enumerate(matrix_df.columns):
        x = left + col_idx * cell
        cv2.putText(canvas, label.replace("pred_", "P"), (x + 8, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for row_idx, label in enumerate(matrix_df.index):
        y = top + row_idx * cell
        cv2.putText(canvas, label.replace("true_", "T"), (10, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        for col_idx, value in enumerate(values[row_idx]):
            x = left + col_idx * cell
            intensity = int(255 - 170 * (float(value) / max_value))
            color = (255, intensity, intensity)
            cv2.rectangle(canvas, (x, y), (x + cell - 2, y + cell - 2), color, thickness=-1)
            cv2.rectangle(canvas, (x, y), (x + cell - 2, y + cell - 2), (180, 180, 180), thickness=1)
            cv2.putText(canvas, str(int(value)), (x + 26, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError(f"Failed to encode confusion matrix: {path}")
    encoded.tofile(str(path))


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)
    output_root = Path(args.output_root or Path(config["outputs"]["root"]) / "downstream_classification")
    output_root.mkdir(parents=True, exist_ok=True)

    method_specs = list(parse_method_specs(args.method)) if args.method else []
    if args.include_t1:
        method_specs.insert(0, parse_method_specs([f"T1_ONLY={_default_split_dir(config, args.split, 't1')}"])[0])
    if args.include_fa_gt:
        method_specs.append(parse_method_specs([f"FA_GT={_default_split_dir(config, args.split, 'fa')}"])[0])
    if not method_specs:
        raise ValueError("No methods selected. Use --method NAME=DIR and/or --include_t1/--include_fa_gt.")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        tasks = list(config.get("evaluation", {}).get("classification_tasks", ["four_class"]))
    regression_targets = [target.strip() for target in args.regression_targets.split(",") if target.strip()]
    repeat_seeds = _parse_seed_list(args.repeat_seeds)

    subject_index = _load_subject_index(args, config)
    feature_tables: dict[str, pd.DataFrame] = {}
    all_summaries = []
    all_repeated_summaries = []
    all_predictions = []
    all_regression_summaries = []
    all_regression_predictions = []
    confusion_root = output_root / "confusion_matrices"
    confusion_root.mkdir(parents=True, exist_ok=True)

    for spec in method_specs:
        features = extract_subject_features_from_folder(
            image_dir=spec.image_dir,
            method=spec.name,
            subject_index=subject_index,
            split=args.split,
            brain_threshold=args.brain_threshold,
            wm_quantile=args.wm_quantile,
            feature_view=args.feature_view,
            frequency_sigma=args.frequency_sigma,
        )
        feature_tables[spec.name] = features

    if args.fusion:
        combined_base = pd.concat(feature_tables.values(), ignore_index=True)
        for raw_spec in args.fusion:
            fused_name, left_name, right_name = _parse_fusion_spec(raw_spec)
            if fused_name in feature_tables:
                raise ValueError(f"Fusion name duplicates an existing method: {fused_name}")
            feature_tables[fused_name] = build_fused_feature_table(
                combined_base,
                fused_name=fused_name,
                left_method=left_name,
                right_method=right_name,
            )

    for method_name, features in feature_tables.items():
        for task in tasks:
            summary, predictions = run_classification_cv(
                features,
                method=method_name,
                task=task,
                n_splits=args.n_splits,
                random_state=args.random_state,
                max_features=args.max_features,
            )
            all_summaries.append(summary)
            all_predictions.append(predictions)
            if repeat_seeds:
                all_repeated_summaries.append(
                    run_repeated_classification_cv(
                        features,
                        method=method_name,
                        task=task,
                        n_splits=args.n_splits,
                        seeds=repeat_seeds,
                        max_features=args.max_features,
                    )
                )
            matrix_df = confusion_matrix_frame(predictions)
            matrix_csv = confusion_root / f"{method_name}_{task}_confusion.csv"
            matrix_df.to_csv(matrix_csv, encoding="utf-8")
            _write_confusion_png(confusion_root / f"{method_name}_{task}_confusion.png", matrix_df, f"{method_name} {task}")
        for target in regression_targets:
            summary, predictions = run_regression_cv(
                features,
                method=method_name,
                target=target,
                n_splits=args.n_splits,
                random_state=args.random_state,
                max_features=args.max_features,
                clip_range=(args.regression_clip_min, args.regression_clip_max),
            )
            all_regression_summaries.append(summary)
            all_regression_predictions.append(predictions)

    feature_df = pd.concat(feature_tables.values(), ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    prediction_df = pd.concat(all_predictions, ignore_index=True)

    feature_path = output_root / "subject_features.csv"
    summary_path = output_root / "classification_summary.csv"
    prediction_path = output_root / "classification_predictions.csv"
    json_path = output_root / "classification_summary.json"
    feature_df.to_csv(feature_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(all_summaries), handle, indent=2, ensure_ascii=False)

    if all_regression_summaries:
        regression_summary_df = pd.DataFrame(all_regression_summaries)
        regression_prediction_df = pd.concat(all_regression_predictions, ignore_index=True)
        regression_summary_path = output_root / "regression_summary.csv"
        regression_prediction_path = output_root / "regression_predictions.csv"
        regression_json_path = output_root / "regression_summary.json"
        regression_summary_df.to_csv(regression_summary_path, index=False, encoding="utf-8")
        regression_prediction_df.to_csv(regression_prediction_path, index=False, encoding="utf-8")
        with open(regression_json_path, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(all_regression_summaries), handle, indent=2, ensure_ascii=False)
    if all_repeated_summaries:
        repeated_summary_df = pd.DataFrame(all_repeated_summaries)
        repeated_summary_path = output_root / "classification_repeated_summary.csv"
        repeated_json_path = output_root / "classification_repeated_summary.json"
        repeated_summary_df.to_csv(repeated_summary_path, index=False, encoding="utf-8")
        with open(repeated_json_path, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(all_repeated_summaries), handle, indent=2, ensure_ascii=False)

    print(f"Saved subject features to: {feature_path}")
    print(f"Saved classification summary to: {summary_path}")
    print(f"Saved predictions to: {prediction_path}")
    print(f"Saved confusion matrices to: {confusion_root}")
    if all_regression_summaries:
        print(f"Saved regression summary to: {output_root / 'regression_summary.csv'}")
        print(f"Saved regression predictions to: {output_root / 'regression_predictions.csv'}")
    if all_repeated_summaries:
        print(f"Saved repeated classification summary to: {output_root / 'classification_repeated_summary.csv'}")
    if not summary_df.empty:
        display_cols = ["method", "task", "n_subjects", "macro_f1", "balanced_accuracy", "macro_auc_ovr"]
        print(summary_df[display_cols].to_string(index=False))
    if all_regression_summaries:
        print(pd.DataFrame(all_regression_summaries)[["method", "target", "n_subjects", "mae", "rmse", "pearson_r", "spearman_r"]].to_string(index=False))
    if all_repeated_summaries:
        print(pd.DataFrame(all_repeated_summaries)[["method", "task", "n_repeats", "macro_f1_mean", "macro_f1_std", "macro_f1_ci95_low", "macro_f1_ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
