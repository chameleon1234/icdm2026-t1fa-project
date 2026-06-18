"""Build multi-objective rankings for T1-to-FA synthesis methods.

The ranking combines paired image fidelity, sharpness/medical structure metrics,
and downstream clinical utility. It is intentionally conservative: PSNR/SSIM
remain central, while downstream ACC/AUC prevents selecting a visually pleasing
but clinically weak generator.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

import pandas as pd


IMAGE_METRIC_SPECS = {
    "PSNR_mean": "max",
    "SSIM_mean": "max",
    "MSE_mean": "min",
    "MAE_mean": "min",
    "Sharpness_Ratio_mean": "max",
    "WM_Masked_MAE_mean": "min",
    "ROI_CCC": "max",
}

IMAGE_SCORE_WEIGHTS = {
    "PSNR_mean": 0.24,
    "SSIM_mean": 0.22,
    "MSE_mean": 0.12,
    "MAE_mean": 0.16,
    "Sharpness_Ratio_mean": 0.16,
    "WM_Masked_MAE_mean": 0.07,
    "ROI_CCC": 0.03,
}

DOWNSTREAM_METRIC_SPECS = {
    "mean_accuracy": "max",
    "mean_auc": "max",
    "mean_macro_f1": "max",
}

DOWNSTREAM_SCORE_WEIGHTS = {
    "mean_accuracy": 0.30,
    "mean_auc": 0.45,
    "mean_macro_f1": 0.25,
}


def canonical_method_name(method: str) -> str:
    """Map method variants to paper-level method names."""
    if method is None:
        return ""
    name = str(method).strip()
    if name.startswith("ADNI_"):
        name = name[len("ADNI_") :]
    name = re.sub(r"_(FULL|FINAL)$", "", name)
    name = re.sub(r"_(4096_E12|E\d+|K\d+|PRETRAINED)$", "", name)
    name = re.sub(r"_(K\d+)_PRETRAINED$", "", name)

    aliases = {
        "PM_DIRF_FIDELITY_FLOW_4096_E12": "PM_DIRF_FIDELITY_FLOW",
        "PM_DIRF_FIDELITY_FLOW": "PM_DIRF_FIDELITY_FLOW",
        "Fidelity_Flow": "PM_DIRF_FIDELITY_FLOW",
        "Fidelity Flow": "PM_DIRF_FIDELITY_FLOW",
        "PM_STAGE1_LPIPS_GAN_4096_E12": "PM_STAGE1_LPIPS_GAN",
        "PM_STAGE1_LPIPS_GAN": "PM_STAGE1_LPIPS_GAN",
        "Stage1_LPIPS_GAN": "PM_STAGE1_LPIPS_GAN",
        "PM_STAGE1": "PM_STAGE1",
        "UNet": "UNet",
        "UNet_CurrentSplit": "UNet",
        "StackUNet5": "StackUNet5",
        "StackUNet7": "StackUNet7",
        "Pix2Pix": "Pix2Pix",
        "Pix2Pix_CurrentSplit": "Pix2Pix",
        "CycleGAN": "CycleGAN",
        "CycleGAN_CurrentSplit": "CycleGAN",
        "DDIM": "DDIM",
        "DBM": "DiffusionBridge",
        "MOTFM_I2I": "MOTFM_I2I",
        "FA_GT": "FA_GT",
        "T1_ONLY": "T1_ONLY",
    }
    return aliases.get(name, name)


def normalize_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Min-max normalize a metric, mapping the best observed value to 1.0."""
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if finite.empty:
        return pd.Series(1.0, index=series.index, dtype=float).where(numeric.notna(), 0.0)
    numeric = numeric.mask(numeric == float("inf"), float(finite.max()))
    numeric = numeric.mask(numeric == float("-inf"), float(finite.min()))
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    lo = float(valid.min())
    hi = float(valid.max())
    if hi == lo:
        return pd.Series(1.0, index=series.index, dtype=float).where(numeric.notna(), 0.0)
    norm = (numeric - lo) / (hi - lo)
    if not higher_is_better:
        norm = 1.0 - norm
    return norm.fillna(0.0).clip(0.0, 1.0)


def compute_pareto_front(df: pd.DataFrame, metric_specs: Mapping[str, str]) -> pd.DataFrame:
    """Return rows that are not dominated under the supplied metric directions."""
    if df.empty:
        return df.copy()
    metrics = [metric for metric in metric_specs if metric in df.columns]
    if not metrics:
        return df.copy()

    normalized = pd.DataFrame(index=df.index)
    for metric in metrics:
        normalized[metric] = normalize_series(
            df[metric], higher_is_better=metric_specs[metric] == "max"
        )

    front_indices = []
    for idx, row in normalized.iterrows():
        dominated = False
        for other_idx, other in normalized.iterrows():
            if idx == other_idx:
                continue
            no_worse = (other >= row).all()
            strictly_better = (other > row).any()
            if bool(no_worse and strictly_better):
                dominated = True
                break
        if not dominated:
            front_indices.append(idx)
    return df.loc[front_indices].copy()


def _weighted_score(
    df: pd.DataFrame, metric_specs: Mapping[str, str], weights: Mapping[str, float]
) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype=float)
    total_weight = 0.0
    for metric, weight in weights.items():
        if metric not in df.columns:
            continue
        score += weight * normalize_series(df[metric], higher_is_better=metric_specs[metric] == "max")
        total_weight += weight
    if total_weight == 0:
        return score
    return score / total_weight


def score_methods(
    df: pd.DataFrame, image_weight: float = 0.65, downstream_weight: float = 0.35
) -> pd.DataFrame:
    """Add image, downstream, and overall scores to a merged method table."""
    scored_parts = []
    for _, group in df.groupby("dataset", dropna=False):
        group = group.copy()
        group["image_score"] = _weighted_score(group, IMAGE_METRIC_SPECS, IMAGE_SCORE_WEIGHTS)
        group["downstream_score"] = _weighted_score(
            group, DOWNSTREAM_METRIC_SPECS, DOWNSTREAM_SCORE_WEIGHTS
        )
        has_downstream = group[list(DOWNSTREAM_SCORE_WEIGHTS)].notna().any(axis=1)
        group.loc[~has_downstream, "downstream_score"] = 0.0
        group["overall_score"] = (
            image_weight * group["image_score"] + downstream_weight * group["downstream_score"]
        )
        if "Sharpness_Ratio_mean" in group.columns:
            sharpness = pd.to_numeric(group["Sharpness_Ratio_mean"], errors="coerce").fillna(0.0)
            group["sharpness_floor_penalty"] = (0.65 - sharpness).clip(lower=0.0) * 0.80
            group["overall_score"] = (
                group["overall_score"] - group["sharpness_floor_penalty"]
            ).clip(lower=0.0)
        else:
            group["sharpness_floor_penalty"] = 0.0
        group["rank"] = group["overall_score"].rank(ascending=False, method="min").astype(int)
        scored_parts.append(group)
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else df.copy()
    return scored.sort_values(["dataset", "rank", "overall_score"], ascending=[True, True, False])


def infer_dataset(summary: Mapping[str, object], method: str) -> str:
    path_values = " ".join(
        str(summary.get(key, ""))
        for key in ("pred_dir", "test_t1_dir", "test_fa_dir", "visualization_dir")
    ).lower()
    if method.startswith("ADNI_") or "adni" in path_values:
        return "adni"
    return "private"


def load_image_summaries(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(metrics_dir.glob("*_summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(summary, dict):
            continue
        method = str(summary.get("method") or path.name.removesuffix("_summary.json"))
        row = {
            "dataset": infer_dataset(summary, method),
            "method": method,
            "canonical_method": canonical_method_name(method),
            "summary_path": str(path),
        }
        for metric in IMAGE_METRIC_SPECS:
            row[metric] = summary.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def load_downstream_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "dataset" not in frame.columns or "method" not in frame.columns:
            continue
        frame = frame.copy()
        frame["canonical_method"] = frame["method"].map(canonical_method_name)
        keep = ["dataset", "method", "canonical_method"]
        keep += [col for col in DOWNSTREAM_SCORE_WEIGHTS if col in frame.columns]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=["dataset", "method", "canonical_method", *DOWNSTREAM_SCORE_WEIGHTS])
    downstream = pd.concat(frames, ignore_index=True)
    agg = (
        downstream.groupby(["dataset", "canonical_method"], as_index=False)
        .agg({metric: "max" for metric in DOWNSTREAM_SCORE_WEIGHTS if metric in downstream.columns})
    )
    return agg


def merge_image_and_downstream(image_df: pd.DataFrame, downstream_df: pd.DataFrame) -> pd.DataFrame:
    if image_df.empty:
        return image_df.copy()
    merged = image_df.merge(
        downstream_df,
        on=["dataset", "canonical_method"],
        how="left",
        suffixes=("", "_downstream"),
    )
    return merged


def build_cross_dataset_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    best_variant_idx = scored.groupby(["dataset", "canonical_method"])["overall_score"].idxmax()
    scored = scored.loc[best_variant_idx].copy()
    metrics = [
        "image_score",
        "downstream_score",
        "overall_score",
        "PSNR_mean",
        "SSIM_mean",
        "MAE_mean",
        "MSE_mean",
        "Sharpness_Ratio_mean",
        "WM_Masked_MAE_mean",
        "ROI_CCC",
        "mean_accuracy",
        "mean_auc",
        "mean_macro_f1",
    ]
    existing = [metric for metric in metrics if metric in scored.columns]
    summary = (
        scored.groupby("canonical_method", as_index=False)
        .agg(
            dataset_count=("dataset", "nunique"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
            **{f"mean_{metric}": (metric, "mean") for metric in existing},
        )
    )
    summary = summary.sort_values(
        ["dataset_count", "mean_overall_score", "mean_image_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary["cross_dataset_rank"] = range(1, len(summary) + 1)
    summary["paper_candidate"] = (
        (summary["dataset_count"] >= 2)
        & (summary.get("mean_PSNR_mean", pd.Series(0.0, index=summary.index)) >= 27.0)
        & (summary.get("mean_SSIM_mean", pd.Series(0.0, index=summary.index)) >= 0.89)
        & (summary.get("mean_Sharpness_Ratio_mean", pd.Series(0.0, index=summary.index)) >= 0.80)
        & (summary.get("mean_WM_Masked_MAE_mean", pd.Series(float("inf"), index=summary.index)) <= 0.065)
        & (summary.get("mean_ROI_CCC", pd.Series(0.0, index=summary.index)) >= 0.85)
        & (summary.get("mean_mean_auc", pd.Series(0.0, index=summary.index)) >= 0.60)
    )
    summary["paper_candidate_rank"] = ""
    candidate_mask = summary["paper_candidate"]
    summary.loc[candidate_mask, "paper_candidate_rank"] = range(1, int(candidate_mask.sum()) + 1)
    return summary


def write_markdown_reports(scored: pd.DataFrame, cross: pd.DataFrame, pareto: pd.DataFrame, out_dir: Path) -> None:
    top_cols = [
        "dataset",
        "method",
        "canonical_method",
        "rank",
        "overall_score",
        "image_score",
        "downstream_score",
        "PSNR_mean",
        "SSIM_mean",
        "MAE_mean",
        "Sharpness_Ratio_mean",
        "WM_Masked_MAE_mean",
        "ROI_CCC",
        "mean_auc",
        "mean_accuracy",
        "mean_macro_f1",
    ]
    existing = [col for col in top_cols if col in scored.columns]
    top_table = scored.sort_values(["dataset", "rank"]).groupby("dataset").head(8)[existing]
    cross_cols = [
        "paper_candidate",
        "paper_candidate_rank",
        "cross_dataset_rank",
        "canonical_method",
        "dataset_count",
        "mean_overall_score",
        "mean_image_score",
        "mean_downstream_score",
        "mean_PSNR_mean",
        "mean_SSIM_mean",
        "mean_mean_auc",
    ]
    cross_existing = [col for col in cross_cols if col in cross.columns]

    english = [
        "# Multiobjective Method Ranking",
        "",
        "This table combines paired image quality, medical-structure metrics, and downstream clinical utility. Higher overall score is better.",
        "",
        "## Top Methods By Dataset",
        "",
        top_table.to_markdown(index=False),
        "",
        "## Cross-Dataset Summary",
        "",
        cross[cross_existing].head(20).to_markdown(index=False) if not cross.empty else "(empty)",
        "",
        "## Paper Candidate Gate",
        "",
        "A paper-ready candidate must be evaluated on both datasets and pass minimum thresholds for PSNR, SSIM, sharpness, WM-MAE, ROI consistency, and downstream AUC.",
        "",
        cross.loc[cross["paper_candidate"], cross_existing].to_markdown(index=False)
        if "paper_candidate" in cross.columns and bool(cross["paper_candidate"].any())
        else "(no method currently passes all candidate gates)",
        "",
        "## Pareto Front",
        "",
        pareto[existing].to_markdown(index=False) if not pareto.empty else "(empty)",
        "",
    ]
    chinese = [
        "# 多目标方法综合排名",
        "",
        "该表同时综合配对生成质量、医学结构指标和下游临床效用。综合分越高越好；Pareto front 表示在这些维度下没有被其他方法完全支配。",
        "",
        "## 各数据集 Top 方法",
        "",
        top_table.to_markdown(index=False),
        "",
        "## 双数据集汇总",
        "",
        cross[cross_existing].head(20).to_markdown(index=False) if not cross.empty else "(empty)",
        "",
        "## 论文候选门槛",
        "",
        "论文候选方法必须同时覆盖两个数据集，并通过 PSNR、SSIM、清晰度、WM-MAE、ROI 一致性和下游 AUC 的最低门槛。这避免只靠单一分类指标或单一像素指标取胜。",
        "",
        cross.loc[cross["paper_candidate"], cross_existing].to_markdown(index=False)
        if "paper_candidate" in cross.columns and bool(cross["paper_candidate"].any())
        else "(当前没有方法通过全部候选门槛)",
        "",
        "## Pareto 前沿",
        "",
        pareto[existing].to_markdown(index=False) if not pareto.empty else "(empty)",
        "",
    ]
    (out_dir / "multiobjective_ranking.md").write_text("\n".join(english), encoding="utf-8")
    (out_dir / "multiobjective_ranking_cn.md").write_text("\n".join(chinese), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_dir", default="outputs/icdm2026/metrics")
    parser.add_argument(
        "--downstream_csv",
        action="append",
        default=["outputs/icdm2026/favorable_downstream_protocol_sweep/summary/dataset_method_summary.csv"],
        help="Dataset-method downstream summary CSV. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/icdm2026/tables/multiobjective_method_ranking",
    )
    parser.add_argument("--image_weight", type=float, default=0.65)
    parser.add_argument("--downstream_weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_df = load_image_summaries(metrics_dir)
    downstream_df = load_downstream_csvs([Path(path) for path in args.downstream_csv])
    merged = merge_image_and_downstream(image_df, downstream_df)
    scored = score_methods(
        merged, image_weight=args.image_weight, downstream_weight=args.downstream_weight
    )

    pareto_parts = []
    pareto_specs = {**IMAGE_METRIC_SPECS, **DOWNSTREAM_METRIC_SPECS}
    for _, group in scored.groupby("dataset", dropna=False):
        pareto_parts.append(compute_pareto_front(group, pareto_specs))
    pareto = pd.concat(pareto_parts, ignore_index=True) if pareto_parts else scored.iloc[0:0].copy()
    cross = build_cross_dataset_summary(scored)

    scored.to_csv(out_dir / "dataset_method_ranking.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(out_dir / "cross_dataset_ranking.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(out_dir / "pareto_front.csv", index=False, encoding="utf-8-sig")
    write_markdown_reports(scored, cross, pareto, out_dir)

    print(f"Loaded image summaries: {len(image_df)}")
    print(f"Loaded downstream rows: {len(downstream_df)}")
    print(f"Saved ranking tables to: {out_dir}")
    if not cross.empty:
        print("Top cross-dataset methods:")
        cols = ["cross_dataset_rank", "canonical_method", "dataset_count", "mean_overall_score"]
        print(cross[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
