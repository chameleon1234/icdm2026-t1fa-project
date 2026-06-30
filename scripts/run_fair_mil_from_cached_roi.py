from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_finalpdf_train_test_roi import (  # noqa: E402
    MethodPair,
    _default_split_dir,
    _json_ready,
    _load_adni_subject_index,
    _load_subject_index,
    _read_yaml,
    extract_slice_features_with_atlas,
    parse_method_pair,
)
from scripts.evaluate_slice_mil_roi import (  # noqa: E402
    evaluate_attention_mil,
    evaluate_multi_task_mil,
    evaluate_slice_svm_vote,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recoverable fair train/test downstream evaluation with cached ROI slice features."
    )
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--subject_index_csv", default="")
    parser.add_argument("--adni_slice_manifest", default="")
    parser.add_argument("--processed_root", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--include_t1", action="store_true")
    parser.add_argument("--include_fa_gt", action="store_true")
    parser.add_argument("--method", action="append", default=[], help="NAME=TRAIN_DIR|TEST_DIR")
    parser.add_argument("--atlas_dir", required=True)
    parser.add_argument("--tasks", default="cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad")
    parser.add_argument("--protocols", default="slice_svm_vote,attention_mil,multi_task_mil")
    parser.add_argument("--classifier", choices=["linear_svm", "rbf_svm"], default="linear_svm")
    parser.add_argument("--feature_set", choices=["roi_mean", "full"], default="roi_mean")
    parser.add_argument("--max_features", type=int, default=0)
    parser.add_argument("--mil_epochs", type=int, default=80)
    parser.add_argument("--mil_lr", type=float, default=1e-3)
    parser.add_argument("--mil_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mil_width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--force_reextract", action="store_true")
    return parser.parse_args()


def _load_subject_index_for_args(args: argparse.Namespace) -> pd.DataFrame:
    config = _read_yaml(args.config)
    if args.adni_slice_manifest:
        return _load_adni_subject_index(args.adni_slice_manifest)
    return _load_subject_index(config, args.subject_index_csv)


def _build_pairs(args: argparse.Namespace) -> list[MethodPair]:
    config = _read_yaml(args.config)
    pairs: list[MethodPair] = []
    if args.include_t1:
        pairs.append(
            MethodPair(
                "T1_ONLY",
                _default_split_dir(config, args.train_split, "t1", args.processed_root),
                _default_split_dir(config, args.test_split, "t1", args.processed_root),
            )
        )
    if args.include_fa_gt:
        pairs.append(
            MethodPair(
                "FA_GT",
                _default_split_dir(config, args.train_split, "fa", args.processed_root),
                _default_split_dir(config, args.test_split, "fa", args.processed_root),
            )
        )
    pairs.extend(parse_method_pair(item) for item in args.method)
    if not pairs:
        raise ValueError("No methods selected")
    return pairs


def _cache_path(output_root: Path, method: str, split: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in method)
    return output_root / "feature_cache" / f"{safe}_{split}_roi_slice_features.csv"


def _extract_or_load(
    output_root: Path,
    pair: MethodPair,
    split_name: str,
    image_dir: str,
    atlas_dir: str,
    subject_index: pd.DataFrame,
    force: bool,
) -> pd.DataFrame:
    path = _cache_path(output_root, pair.name, split_name)
    if path.exists() and not force:
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = extract_slice_features_with_atlas(
        image_dir=image_dir,
        atlas_dir=atlas_dir,
        method=pair.name,
        subject_index=subject_index,
        split=split_name,
    )
    frame.to_csv(path, index=False, encoding="utf-8")
    return frame


def _compute_per_class_f1(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required = {"method", "protocol", "task", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    for (method, protocol, task), group in predictions.groupby(["method", "protocol", "task"], sort=True):
        labels = sorted(set(group["y_true"].astype(int)).union(set(group["y_pred"].astype(int))))
        scores = f1_score(
            group["y_true"].astype(int).to_numpy(),
            group["y_pred"].astype(int).to_numpy(),
            labels=labels,
            average=None,
            zero_division=0,
        )
        for label, score in zip(labels, scores):
            rows.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "task": task,
                    "class_label": int(label),
                    "f1": float(score),
                    "n_subject_rows": int(group.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _write_reports(output_root: Path, dataset_name: str, summary: pd.DataFrame, eligible_methods: list[str]) -> None:
    avg = (
        summary.groupby("method", as_index=False)
        .agg(
            rows=("method", "size"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_auc=("macro_auc_ovr", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_n_train_subjects=("n_train_subjects", "mean"),
            mean_n_test_subjects=("n_test_subjects", "mean"),
        )
        .sort_values(["mean_macro_auc", "mean_macro_f1", "mean_accuracy"], ascending=False)
    )
    task = (
        summary.groupby(["method", "task"], as_index=False)
        .agg(
            rows=("method", "size"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_auc=("macro_auc_ovr", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values(["task", "mean_macro_auc"], ascending=[True, False])
    )
    avg.to_csv(output_root / "method_average_summary.csv", index=False, encoding="utf-8")
    task.to_csv(output_root / "per_task_summary.csv", index=False, encoding="utf-8")

    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "(empty)"
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        headers = [str(column) for column in display.columns]
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

    cn = [
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run/validate",
        "- Origin Date: 2026-06-30",
        "- Verification Status: ANALYZED",
        "- Version Label: fair_downstream_cached_roi_v1",
        "",
        f"## {dataset_name} 公平 train/test MIL 下游报告",
        "",
        f"- eligible methods: {', '.join(eligible_methods)}",
        "- 协议: slice_svm_vote, attention_mil, multi_task_mil",
        "- 训练和测试严格使用不同 split；没有把 test prediction 当作 train prediction。",
        "- per-class F1 由 subject-level predictions 中的 y_true/y_pred 重新计算。",
        "",
        "## Method Average Summary",
        "",
        md_table(avg),
        "",
        "## Per Task Summary",
        "",
        md_table(task),
        "",
    ]
    en = [
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run/validate",
        "- Origin Date: 2026-06-30",
        "- Verification Status: ANALYZED",
        "- Version Label: fair_downstream_cached_roi_v1",
        "",
        f"## {dataset_name} Fair Train/Test MIL Downstream Report",
        "",
        f"- eligible methods: {', '.join(eligible_methods)}",
        "- protocols: slice_svm_vote, attention_mil, multi_task_mil",
        "- Train and test splits are separate; test predictions were not used as training predictions.",
        "- Per-class F1 was recomputed from subject-level y_true/y_pred predictions.",
        "",
        "## Method Average Summary",
        "",
        md_table(avg),
        "",
        "## Per Task Summary",
        "",
        md_table(task),
        "",
    ]
    stem = "downstream_adni_fair_full_all_methods_report" if dataset_name.upper() == "ADNI" else "downstream_private_fair_full_all_methods_report"
    (output_root / f"{stem}_cn.md").write_text("\n".join(cn), encoding="utf-8")
    (output_root / f"{stem}.md").write_text("\n".join(en), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subject_index = _load_subject_index_for_args(args)
    pairs = _build_pairs(args)
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]

    train_tables: dict[str, pd.DataFrame] = {}
    test_tables: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        print(f"[cache] {pair.name} train")
        train_tables[pair.name] = _extract_or_load(
            output_root, pair, args.train_split, pair.train_dir, args.atlas_dir, subject_index, args.force_reextract
        )
        print(f"[cache] {pair.name} test")
        test_tables[pair.name] = _extract_or_load(
            output_root, pair, args.test_split, pair.test_dir, args.atlas_dir, subject_index, args.force_reextract
        )

    summaries: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for method in sorted(train_tables):
        print(f"[eval] {method}")
        if "multi_task_mil" in protocols:
            multi_summaries, multi_pred = evaluate_multi_task_mil(
                train_tables[method],
                test_tables[method],
                method,
                tasks=tasks,
                feature_set=args.feature_set,
                max_features=args.max_features,
                epochs=args.mil_epochs,
                lr=args.mil_lr,
                weight_decay=args.mil_weight_decay,
                width=args.mil_width,
                seed=args.seed,
                device=args.device,
            )
            summaries.extend(multi_summaries)
            predictions.append(multi_pred.assign(protocol="multi_task_mil"))
        for task in tasks:
            if "slice_svm_vote" in protocols:
                summary, pred = evaluate_slice_svm_vote(
                    train_tables[method],
                    test_tables[method],
                    method,
                    task,
                    classifier=args.classifier,
                    feature_set=args.feature_set,
                    max_features=args.max_features,
                )
                summaries.append(summary)
                predictions.append(pred.assign(protocol="slice_svm_vote"))
            if "attention_mil" in protocols:
                summary, pred = evaluate_attention_mil(
                    train_tables[method],
                    test_tables[method],
                    method,
                    task,
                    feature_set=args.feature_set,
                    max_features=args.max_features,
                    epochs=args.mil_epochs,
                    lr=args.mil_lr,
                    weight_decay=args.mil_weight_decay,
                    width=args.mil_width,
                    seed=args.seed,
                    device=args.device,
                )
                summaries.append(summary)
                predictions.append(pred.assign(protocol="attention_mil"))

    summary_df = pd.DataFrame(summaries).sort_values(["task", "protocol", "method"]).reset_index(drop=True)
    pred_df = pd.concat(predictions, ignore_index=True)
    summary_df.to_csv(output_root / "classification_subject_summary.csv", index=False, encoding="utf-8")
    pred_df.to_csv(output_root / "classification_subject_predictions.csv", index=False, encoding="utf-8")
    with open(output_root / "classification_subject_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summaries), handle, indent=2, ensure_ascii=False)
    per_class = _compute_per_class_f1(pred_df)
    if not per_class.empty:
        per_class.to_csv(output_root / "per_class_f1_summary.csv", index=False, encoding="utf-8")
    _write_reports(
        output_root,
        dataset_name="ADNI" if args.adni_slice_manifest else "Private",
        summary=summary_df,
        eligible_methods=[pair.name for pair in pairs],
    )
    print(f"Saved summary to {output_root / 'classification_subject_summary.csv'}")
    print(f"Saved predictions to {output_root / 'classification_subject_predictions.csv'}")


if __name__ == "__main__":
    main()
