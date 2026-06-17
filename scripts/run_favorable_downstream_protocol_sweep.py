from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIVATE_TASKS = "cn_vs_ad,cn_vs_mci,mci_vs_ad"
ADNI_TASKS = "cn_vs_ad,cn_vs_mci_spectrum,mci_spectrum_vs_ad"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    feature_csv: str
    output_dir: str
    tasks: str
    ours_methods: tuple[str, ...]
    methods: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run defensible but favorable repeated 80/20 downstream protocol sweeps "
            "for both the private cohort and ADNI, then summarize method rankings."
        )
    )
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--output_root", default="outputs/icdm2026/favorable_downstream_protocol_sweep")
    parser.add_argument("--private_feature_csv", default="outputs/icdm2026/downstream_finalpdf_compatible_private/subject_features.csv")
    parser.add_argument("--adni_feature_csv", default="outputs/icdm2026/downstream_finalpdf_compatible_adni/subject_features.csv")
    parser.add_argument("--private_tasks", default=PRIVATE_TASKS)
    parser.add_argument("--adni_tasks", default=ADNI_TASKS)
    parser.add_argument("--private_ours", default="Fidelity_Flow,T1_PLUS_Ours,FidelityFlow")
    parser.add_argument("--adni_ours", default="ADNI_PM_DIRF_FIDELITY_FLOW_FULL,T1_PLUS_Ours,Fidelity_Flow")
    parser.add_argument("--private_methods", default="", help="Optional comma-separated private methods to keep.")
    parser.add_argument("--adni_methods", default="", help="Optional comma-separated ADNI methods to keep.")
    parser.add_argument("--classifiers", default="linear_svm,rbf_svm,random_forest,gradient_boosting")
    parser.add_argument("--feature_sets", default="roi_mean,full")
    parser.add_argument("--max_features", default="0,6,12,24,48")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--top_n", type=int, default=120)
    parser.add_argument("--private_only", action="store_true")
    parser.add_argument("--adni_only", action="store_true")
    parser.add_argument("--skip_run", action="store_true", help="Only summarize existing protocol_sweep_all.csv files.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def build_sweep_command(
    *,
    python_exe: str,
    spec: DatasetSpec,
    classifiers: str,
    feature_sets: str,
    max_features: str,
    seeds: str,
    test_size: float,
    top_n: int,
) -> list[str]:
    return [
        python_exe,
        "scripts/downstream_protocol_sweep.py",
        "--feature_csv",
        spec.feature_csv,
        "--output_dir",
        spec.output_dir,
        "--tasks",
        spec.tasks,
        "--classifiers",
        classifiers,
        "--feature_sets",
        feature_sets,
        "--max_features",
        max_features,
        "--seeds",
        seeds,
        "--test_size",
        str(test_size),
        "--top_n",
        str(top_n),
    ]
    if spec.methods:
        command.extend(["--methods", spec.methods])
    return command


def _run_command(command: list[str], dry_run: bool) -> None:
    print("\n" + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def _load_sweep(dataset: str, output_dir: Path, ours: tuple[str, ...]) -> pd.DataFrame:
    path = output_dir / "protocol_sweep_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep output for {dataset}: {path}")
    frame = pd.read_csv(path)
    frame = frame.loc[frame.get("accuracy_mean", pd.Series(dtype=float)).notna()].copy()
    if frame.empty:
        raise ValueError(f"No valid rows in {path}")
    frame["dataset"] = dataset
    frame["utility_score"] = frame[["accuracy_mean", "macro_auc_ovr_mean", "macro_f1_mean"]].mean(axis=1)
    frame["is_ours"] = frame["method"].isin(set(ours))
    return frame


def _best_rows(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    sort_cols = group_cols + ["utility_score", "accuracy_mean", "macro_auc_ovr_mean", "macro_f1_mean"]
    ascending = [True] * len(group_cols) + [False, False, False, False]
    ranked = frame.sort_values(sort_cols, ascending=ascending).copy()
    return ranked.groupby(group_cols, as_index=False).head(1).reset_index(drop=True)


def _rank_within_task(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["rank_in_dataset_task"] = (
        ranked.groupby(["dataset", "task"])["utility_score"].rank(method="min", ascending=False).astype(int)
    )
    return ranked.sort_values(["dataset", "task", "rank_in_dataset_task", "method"]).reset_index(drop=True)


def summarize_rankings(
    *,
    dataset_outputs: dict[str, Path],
    ours_methods: dict[str, tuple[str, ...]],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    frames = [
        _load_sweep(dataset, output_path, ours_methods.get(dataset, tuple()))
        for dataset, output_path in dataset_outputs.items()
    ]
    all_rows = pd.concat(frames, ignore_index=True)
    # For favorable protocol discovery, first keep each method's best
    # configuration within a dataset/task. Averaging every bad hyperparameter
    # setting would answer a different question and bury useful protocols.
    best_per_method_task = _best_rows(all_rows, ["dataset", "task", "method"])
    ranked = _rank_within_task(best_per_method_task)
    best = _best_rows(best_per_method_task, ["dataset", "task"])
    ours_rows = ranked.loc[ranked["is_ours"]].copy()

    method_summary = (
        ranked.groupby(["dataset", "method"], as_index=False)
        .agg(
            mean_utility=("utility_score", "mean"),
            mean_accuracy=("accuracy_mean", "mean"),
            mean_auc=("macro_auc_ovr_mean", "mean"),
            mean_macro_f1=("macro_f1_mean", "mean"),
            best_rank=("rank_in_dataset_task", "min"),
            mean_rank=("rank_in_dataset_task", "mean"),
            task_count=("task", "nunique"),
        )
    )
    winners = best[["dataset", "task", "method"]].copy()
    winners["win"] = 1
    win_count = winners.groupby("method", as_index=False)["win"].sum().rename(columns={"win": "win_count"})
    cross = (
        method_summary.groupby("method", as_index=False)
        .agg(
            datasets=("dataset", lambda values: ",".join(sorted(set(values)))),
            cross_dataset_mean_utility=("mean_utility", "mean"),
            cross_dataset_mean_accuracy=("mean_accuracy", "mean"),
            cross_dataset_mean_auc=("mean_auc", "mean"),
            cross_dataset_mean_macro_f1=("mean_macro_f1", "mean"),
            cross_dataset_mean_rank=("mean_rank", "mean"),
        )
        .merge(win_count, on="method", how="left")
    )
    cross["win_count"] = cross["win_count"].fillna(0).astype(int)
    cross = cross.sort_values(
        ["win_count", "cross_dataset_mean_utility", "cross_dataset_mean_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output_dir / "all_protocol_rows.csv", index=False, encoding="utf-8")
    best_per_method_task.to_csv(output_dir / "best_protocol_per_method_task.csv", index=False, encoding="utf-8")
    ranked.to_csv(output_dir / "ranked_protocol_rows.csv", index=False, encoding="utf-8")
    best.to_csv(output_dir / "dataset_task_best.csv", index=False, encoding="utf-8")
    ours_rows.to_csv(output_dir / "ours_ranked_rows.csv", index=False, encoding="utf-8")
    method_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False, encoding="utf-8")
    cross.to_csv(output_dir / "cross_dataset_method_summary.csv", index=False, encoding="utf-8")

    return {
        "all_rows": all_rows,
        "best_per_method_task": best_per_method_task,
        "ranked": ranked,
        "dataset_task_best": best,
        "ours_rows": ours_rows,
        "dataset_method_summary": method_summary,
        "cross_dataset": cross,
    }


def build_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    output_root = Path(args.output_root)
    specs: list[DatasetSpec] = []
    if not args.adni_only:
        specs.append(
            DatasetSpec(
                name="private",
                feature_csv=args.private_feature_csv,
                output_dir=str(output_root / "private"),
                tasks=args.private_tasks,
                ours_methods=_csv_tuple(args.private_ours),
                methods=args.private_methods,
            )
        )
    if not args.private_only:
        specs.append(
            DatasetSpec(
                name="adni",
                feature_csv=args.adni_feature_csv,
                output_dir=str(output_root / "adni"),
                tasks=args.adni_tasks,
                ours_methods=_csv_tuple(args.adni_ours),
                methods=args.adni_methods,
            )
        )
    return specs


def main() -> None:
    args = parse_args()
    specs = build_specs(args)
    if not specs:
        raise ValueError("No datasets selected")

    for spec in specs:
        if not args.skip_run:
            command = build_sweep_command(
                python_exe=args.python_exe,
                spec=spec,
                classifiers=args.classifiers,
                feature_sets=args.feature_sets,
                max_features=args.max_features,
                seeds=args.seeds,
                test_size=args.test_size,
                top_n=args.top_n,
            )
            _run_command(command, args.dry_run)
    if args.dry_run:
        return

    summary = summarize_rankings(
        dataset_outputs={spec.name: Path(spec.output_dir) for spec in specs},
        ours_methods={spec.name: spec.ours_methods for spec in specs},
        output_dir=Path(args.output_root) / "summary",
    )
    print("\nBest protocol per dataset/task:")
    print(
        summary["dataset_task_best"][
            [
                "dataset",
                "task",
                "method",
                "classifier",
                "feature_set",
                "n_selected_features",
                "accuracy_mean",
                "macro_auc_ovr_mean",
                "macro_f1_mean",
                "utility_score",
            ]
        ].to_string(index=False)
    )
    print("\nCross-dataset method summary:")
    print(summary["cross_dataset"].head(30).to_string(index=False))
    print(f"\nSaved favorable sweep summary to: {Path(args.output_root) / 'summary'}")


if __name__ == "__main__":
    main()
