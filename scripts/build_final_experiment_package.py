import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BLOCKLIST_TOKENS = ("FREQ", "Frequency", "STACKLOW", "FLOWHIGH", "OURSHIGH")


QUALITY_METRICS = {
    "PSNR": "higher",
    "SSIM": "higher",
    "MAE": "lower",
    "SharpRatio": "higher",
    "WM_MAE": "lower",
    "ROI_CCC": "higher",
}

MIN_VISUAL_SHARP_RATIO = 0.75
MIN_MEDICAL_ROI_CCC = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final paper experiment package from two-dataset comparison tables.")
    parser.add_argument("--table_root", default="outputs/icdm2026/tables/final_two_dataset")
    parser.add_argument("--output_root", default="outputs/icdm2026/tables/final_experiment_package")
    parser.add_argument("--figure_root", default="outputs/icdm2026/figures/final_experiment_package")
    return parser.parse_args()


def _is_blocked(method: Any) -> bool:
    method = str(method)
    return any(token in method for token in BLOCKLIST_TOKENS)


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input table does not exist: {path}")
    return pd.read_csv(path)


def _normalize_metric(series: pd.Series, direction: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or abs(max_value - min_value) < 1e-12:
        return pd.Series(0.5, index=series.index)
    normalized = (values - min_value) / (max_value - min_value)
    if direction == "lower":
        normalized = 1.0 - normalized
    return normalized.fillna(0.0)


def _add_quality_score(image: pd.DataFrame) -> pd.DataFrame:
    scored = image.copy()
    metric_scores: list[str] = []
    for metric, direction in QUALITY_METRICS.items():
        score_name = f"{metric}_score"
        scored[score_name] = scored.groupby("dataset", group_keys=False)[metric].apply(
            lambda group: _normalize_metric(group, direction)
        )
        metric_scores.append(score_name)
    scored["quality_score"] = scored[metric_scores].mean(axis=1)
    return scored


def _build_utility_summary(downstream: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = downstream[downstream["available"].fillna(False)].copy()
    available["macro_f1_mean"] = pd.to_numeric(available["macro_f1_mean"], errors="coerce")
    available["macro_auc_ovr_mean"] = pd.to_numeric(available.get("macro_auc_ovr_mean"), errors="coerce")
    defaults = {
        "task_display": available.get("task", ""),
        "family": "",
        "role": "",
        "macro_f1_std": pd.NA,
    }
    for column, default in defaults.items():
        if column not in available.columns:
            available[column] = default
    task_table = available[
        [
            "dataset",
            "task",
            "task_display",
            "method",
            "display",
            "family",
            "role",
            "macro_f1_mean",
            "macro_f1_std",
            "macro_auc_ovr_mean",
        ]
    ].copy()
    summary = (
        available.groupby(["dataset", "method", "display"], as_index=False)
        .agg(
            utility_score=("macro_f1_mean", "mean"),
            utility_auc=("macro_auc_ovr_mean", "mean"),
            utility_task_count=("macro_f1_mean", "count"),
        )
    )
    return summary, task_table


def _mark_best(summary: pd.DataFrame) -> pd.DataFrame:
    marked = summary.copy()
    for metric in ["quality_score", "utility_score", "overall_score", "eligible_overall_score"]:
        best_col = f"is_best_{metric}"
        marked[best_col] = False
        for dataset, group in marked.groupby("dataset"):
            valid = group[pd.to_numeric(group[metric], errors="coerce").notna()]
            if valid.empty:
                continue
            max_value = valid[metric].max()
            marked.loc[(marked["dataset"].eq(dataset)) & (marked[metric].eq(max_value)), best_col] = True
    return marked


def build_final_package_tables(table_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_root = Path(table_root)
    image = _read_required(table_root / "two_dataset_image_metrics.csv")
    downstream = _read_required(table_root / "two_dataset_downstream_macro_f1.csv")

    image = image[image["available"].fillna(False)].copy()
    image = image[~image["method"].map(_is_blocked)].copy()
    downstream = downstream[~downstream["method"].map(_is_blocked)].copy()

    scored_image = _add_quality_score(image)
    utility, task_table = _build_utility_summary(downstream)
    summary = scored_image.merge(utility, on=["dataset", "method", "display"], how="left")
    summary["utility_score"] = pd.to_numeric(summary["utility_score"], errors="coerce")
    summary["utility_auc"] = pd.to_numeric(summary["utility_auc"], errors="coerce")
    summary["overall_score"] = summary[["quality_score", "utility_score"]].mean(axis=1)
    summary["passes_visual_gate"] = pd.to_numeric(summary["SharpRatio"], errors="coerce") >= MIN_VISUAL_SHARP_RATIO
    summary["passes_medical_gate"] = pd.to_numeric(summary["ROI_CCC"], errors="coerce") >= MIN_MEDICAL_ROI_CCC
    summary["passes_final_gate"] = summary["passes_visual_gate"] & summary["passes_medical_gate"]
    summary["eligible_overall_score"] = summary["overall_score"].where(summary["passes_final_gate"])

    for metric in ["quality_score", "utility_score", "overall_score", "eligible_overall_score"]:
        summary[f"{metric}_rank"] = summary.groupby("dataset")[metric].rank(ascending=False, method="min")
    summary = _mark_best(summary)

    ordered = [
        "dataset",
        "method",
        "display",
        "family",
        "role",
        "PSNR",
        "SSIM",
        "MSE",
        "MAE",
        "SharpRatio",
        "WM_MAE",
        "ROI_CCC",
        "quality_score",
        "utility_score",
        "utility_auc",
        "utility_task_count",
        "overall_score",
        "passes_visual_gate",
        "passes_medical_gate",
        "passes_final_gate",
        "eligible_overall_score",
        "quality_score_rank",
        "utility_score_rank",
        "overall_score_rank",
        "eligible_overall_score_rank",
        "is_best_quality_score",
        "is_best_utility_score",
        "is_best_overall_score",
        "is_best_eligible_overall_score",
    ]
    existing = [column for column in ordered if column in summary.columns]
    summary = summary[existing].sort_values(
        ["dataset", "eligible_overall_score_rank", "overall_score_rank", "quality_score_rank", "display"],
        na_position="last",
    )
    task_table = task_table[~task_table["method"].map(_is_blocked)].sort_values(["dataset", "task", "macro_f1_mean"], ascending=[True, True, False])
    return summary, task_table


def _fmt(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _summary_markdown(summary: pd.DataFrame, *, chinese: bool = False) -> str:
    if chinese:
        lines = [
            "# 最终实验结果包",
            "",
            f"该表基于私有数据集与 ADNI 的最终双阶段方法和对比方法，旧 frequency fusion 路线已被显式排除。最终候选需满足 SharpRatio >= {MIN_VISUAL_SHARP_RATIO:.2f} 且 ROI_CCC >= {MIN_MEDICAL_ROI_CCC:.2f}。",
            "",
            "| 数据集 | 方法 | 类型 | PSNR | SSIM | Sharp | WM-MAE | ROI | 质量分 | 下游分 | 综合分 | 过门槛 | 候选排名 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# Final Experiment Package",
            "",
            f"This table summarizes the final two-stage method and comparison baselines on the private and ADNI datasets. Discarded frequency-fusion variants are explicitly excluded. Final candidates must satisfy SharpRatio >= {MIN_VISUAL_SHARP_RATIO:.2f} and ROI_CCC >= {MIN_MEDICAL_ROI_CCC:.2f}.",
            "",
            "| Dataset | Method | Family | PSNR | SSIM | Sharp | WM-MAE | ROI | Quality | Utility | Overall | Gate | Eligible Rank |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["display"]),
                    str(row["family"]),
                    _fmt(row.get("PSNR")),
                    _fmt(row.get("SSIM")),
                    _fmt(row.get("SharpRatio")),
                    _fmt(row.get("WM_MAE")),
                    _fmt(row.get("ROI_CCC")),
                    _fmt(row.get("quality_score")),
                    _fmt(row.get("utility_score")),
                    _fmt(row.get("overall_score")),
                    "1" if bool(row.get("passes_final_gate")) else "0",
                    _fmt(row.get("eligible_overall_score_rank"), 0),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _plot_quality_tradeoff(summary: pd.DataFrame, figure_path: str | Path) -> None:
    plot_data = summary.copy()
    colors = {
        "Reference": "#7f7f7f",
        "CNN": "#4e79a7",
        "GAN": "#f28e2b",
        "Diffusion": "#59a14f",
        "Flow": "#edc948",
        "PMRF": "#b07aa1",
        "Ours": "#e15759",
        "Ablation": "#76b7b2",
    }
    markers = {"private": "o", "adni": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_col, ylabel in [
        (axes[0], "ROI_CCC", "ROI consistency"),
        (axes[1], "utility_score", "Mean downstream Macro-F1"),
    ]:
        for _, row in plot_data.iterrows():
            ax.scatter(
                row["SharpRatio"],
                row[y_col],
                s=110 if row["role"] == "final_two_stage" else 65,
                marker=markers.get(row["dataset"], "o"),
                color=colors.get(row["family"], "#9c9c9c"),
                edgecolor="black" if row["role"] == "final_two_stage" else "white",
                linewidth=1.2,
                alpha=0.9,
            )
            if row["role"] == "final_two_stage":
                ax.annotate(
                    f"{row['dataset']} ours",
                    (row["SharpRatio"], row[y_col]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                )
        ax.set_xlabel("Sharpness ratio")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].set_title("Sharpness vs Medical ROI Fidelity")
    axes[1].set_title("Sharpness vs Downstream Utility")
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def _plot_rank_heatmap(summary: pd.DataFrame, figure_path: str | Path) -> None:
    rank_data = summary.pivot_table(index="display", columns="dataset", values="eligible_overall_score_rank", aggfunc="min")
    rank_data = rank_data.sort_values(list(rank_data.columns), na_position="last")
    fig, ax = plt.subplots(figsize=(8, max(5, len(rank_data) * 0.42)))
    matrix = rank_data.fillna(rank_data.max().max() + 1)
    image = ax.imshow(matrix.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = rank_data.iloc[i, j]
            text = "" if pd.isna(value) else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", color="white" if matrix.iloc[i, j] > 3 else "black")
    ax.set_title("Eligible Overall Rank by Dataset (Lower Is Better)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def write_final_package(summary: pd.DataFrame, task_table: pd.DataFrame, output_root: str | Path, figure_root: str | Path) -> dict[str, Path]:
    output_root = Path(output_root)
    figure_root = Path(figure_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_root / "final_method_summary.csv",
        "tasks": output_root / "final_downstream_task_table.csv",
        "markdown": output_root / "final_experiment_package.md",
        "markdown_cn": output_root / "final_experiment_package_cn.md",
        "tradeoff": figure_root / "quality_utility_tradeoff.png",
        "rank": figure_root / "overall_rank_heatmap.png",
    }
    summary.to_csv(paths["summary"], index=False)
    task_table.to_csv(paths["tasks"], index=False)
    paths["markdown"].write_text(_summary_markdown(summary, chinese=False), encoding="utf-8")
    paths["markdown_cn"].write_text(_summary_markdown(summary, chinese=True), encoding="utf-8")
    _plot_quality_tradeoff(summary, paths["tradeoff"])
    _plot_rank_heatmap(summary, paths["rank"])
    return paths


def main() -> None:
    args = parse_args()
    summary, task_table = build_final_package_tables(args.table_root)
    paths = write_final_package(summary, task_table, args.output_root, args.figure_root)

    print("Saved final experiment package:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\nTop methods by dataset:")
    print(
        summary[
            [
                "dataset",
                "display",
                "family",
                "passes_final_gate",
                "quality_score",
                "utility_score",
                "overall_score",
                "eligible_overall_score_rank",
            ]
        ]
        .sort_values(["dataset", "eligible_overall_score_rank", "overall_score"], ascending=[True, True, False], na_position="last")
        .groupby("dataset")
        .head(8)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
