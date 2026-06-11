import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FINAL_METHOD_BLOCKLIST = ("FREQ", "Frequency", "STACKLOW", "FLOWHIGH", "OURSHIGH")


PRIVATE_METHODS: dict[str, dict[str, Any]] = {
    "T1_ONLY": {"display": "T1 only", "family": "Reference", "role": "input_reference", "downstream_aliases": ["T1_ONLY"]},
    "FA_GT": {"display": "Real FA", "family": "Reference", "role": "upper_bound", "downstream_aliases": ["FA_GT"]},
    "UNet_CurrentSplit_E100": {"display": "U-Net", "family": "CNN", "role": "comparison"},
    "Pix2Pix_CurrentSplit_E100": {"display": "Pix2Pix", "family": "GAN", "role": "comparison"},
    "CycleGAN_CurrentSplit_E100": {"display": "CycleGAN", "family": "GAN", "role": "comparison"},
    "DDIM_E100_K50": {"display": "DDIM", "family": "Diffusion", "role": "comparison", "downstream_aliases": ["DDIM_E100_K50"]},
    "DIRF_V5_3SLICE_K6": {"display": "DIRF V5", "family": "Flow", "role": "comparison"},
    "PM_STAGE1": {"display": "PM Stage1", "family": "PMRF", "role": "ablation"},
    "PM_STAGE1_LPIPS_GAN_5SLICE_FINAL": {
        "display": "Stage1 LPIPS+GAN",
        "family": "Ours",
        "role": "stage1_ablation",
        "downstream_aliases": ["Stage1_LPIPS_GAN"],
    },
    "PM_DIRF_FIDELITY_FLOW_FULL": {
        "display": "Ours Fidelity Flow",
        "family": "Ours",
        "role": "final_two_stage",
        "downstream_aliases": ["Fidelity_Flow"],
    },
    "PM_DIRF_FIDELITY_DIRECT_FULL": {
        "display": "Fidelity Direct",
        "family": "Ablation",
        "role": "stage2_ablation",
        "downstream_aliases": ["Fidelity_Direct"],
    },
}


ADNI_METHODS: dict[str, dict[str, Any]] = {
    "ADNI_T1_ONLY": {"display": "T1 only", "family": "Reference", "role": "input_reference", "downstream_aliases": ["T1_ONLY"]},
    "ADNI_FA_GT": {"display": "Real FA", "family": "Reference", "role": "upper_bound", "downstream_aliases": ["FA_GT"]},
    "ADNI_UNet_E50": {"display": "U-Net", "family": "CNN", "role": "comparison"},
    "ADNI_StackUNet5_E12": {"display": "Stack U-Net 5", "family": "CNN", "role": "comparison"},
    "ADNI_Pix2Pix_E50": {"display": "Pix2Pix", "family": "GAN", "role": "comparison"},
    "ADNI_CycleGAN_E50": {"display": "CycleGAN", "family": "GAN", "role": "comparison"},
    "ADNI_DDIM_E100_K50_PRETRAINED": {"display": "DDIM", "family": "Diffusion", "role": "comparison"},
    "ADNI_DBM_E100_K40_PRETRAINED": {"display": "Diffusion Bridge", "family": "Diffusion", "role": "comparison"},
    "ADNI_MOTFM_I2I_K10_PRETRAINED": {"display": "MOTFM", "family": "Flow", "role": "comparison"},
    "ADNI_PM_STAGE1_LPIPS_GAN_FULL": {"display": "Stage1 LPIPS+GAN", "family": "Ours", "role": "stage1_ablation"},
    "ADNI_PM_DIRF_FIDELITY_FLOW_FULL": {"display": "Ours Fidelity Flow", "family": "Ours", "role": "final_two_stage"},
}


PRIVATE_TASKS = {
    "cn_scd_vs_mci_ad": "Private CN+SCD vs MCI+AD",
    "cn_vs_mci": "Private CN vs MCI",
    "cn_vs_ad": "Private CN vs AD",
    "mci_vs_ad": "Private MCI vs AD",
}


ADNI_TASKS = {
    "cn_vs_mci_spectrum": "ADNI CN vs MCI-spectrum",
    "cn_vs_ad": "ADNI CN vs AD",
    "mci_spectrum_vs_ad": "ADNI MCI-spectrum vs AD",
}


DEFAULT_PRIVATE_DOWNSTREAM = [
    "outputs/icdm2026/downstream_binary_tasks/classification_repeated_summary.csv",
    "outputs/icdm2026/downstream_comparison/classification_repeated_summary.csv",
    "outputs/icdm2026/downstream_finalpdf_binary_private/classification_repeated_summary.csv",
    "outputs/icdm2026/downstream_unet_current_split/classification_repeated_summary.csv",
    "outputs/icdm2026/downstream_pix2pix_current_e100/classification_repeated_summary.csv",
    "outputs/icdm2026/downstream_cyclegan_current_e100/classification_repeated_summary.csv",
]


DEFAULT_ADNI_DOWNSTREAM = [
    "outputs/icdm2026/downstream_adni_two_stage_full/classification_repeated_summary.csv",
    "outputs/icdm2026/downstream_adni_full_comparison_v2/classification_repeated_summary.csv",
    "outputs/icdm2026/downstream_adni_heavy_baselines/classification_repeated_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final private/ADNI comparison tables and downstream plots.")
    parser.add_argument("--metrics_root", default="outputs/icdm2026/metrics")
    parser.add_argument("--private_downstream", nargs="*", default=DEFAULT_PRIVATE_DOWNSTREAM)
    parser.add_argument("--adni_downstream", nargs="*", default=DEFAULT_ADNI_DOWNSTREAM)
    parser.add_argument("--output_root", default="outputs/icdm2026/tables/final_two_dataset")
    parser.add_argument("--figure_root", default="outputs/icdm2026/figures/final_two_dataset")
    return parser.parse_args()


def _is_blocked_method(method: str) -> bool:
    return any(token in method for token in FINAL_METHOD_BLOCKLIST)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def _method_config(dataset: str) -> dict[str, dict[str, Any]]:
    if dataset == "private":
        return PRIVATE_METHODS
    if dataset == "adni":
        return ADNI_METHODS
    raise ValueError(f"Unknown dataset: {dataset}")


def _task_config(dataset: str) -> dict[str, str]:
    if dataset == "private":
        return PRIVATE_TASKS
    if dataset == "adni":
        return ADNI_TASKS
    raise ValueError(f"Unknown dataset: {dataset}")


def build_image_table(metrics_root: str | Path, *, dataset: str) -> pd.DataFrame:
    metrics_root = Path(metrics_root)
    rows: list[dict[str, Any]] = []
    for method, meta in _method_config(dataset).items():
        if _is_blocked_method(method):
            continue
        summary = _load_json(metrics_root / f"{method}_summary.json")
        row = {
            "dataset": dataset,
            "method": method,
            "display": meta["display"],
            "family": meta["family"],
            "role": meta["role"],
            "available": summary is not None,
        }
        if summary is not None:
            row.update(
                {
                    "PSNR": summary.get("PSNR_mean"),
                    "SSIM": summary.get("SSIM_mean"),
                    "MSE": summary.get("MSE_mean"),
                    "MAE": summary.get("MAE_mean"),
                    "SharpRatio": summary.get("Sharpness_Ratio_mean"),
                    "WM_MAE": _first_present(summary, "WM_Masked_MAE_mean", "WM_MAE_mean"),
                    "ROI_CCC": _first_present(summary, "ROI_CCC_mean", "ROI_CCC"),
                    "pred_dir": summary.get("pred_dir"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_downstream(path: str | Path, priority: int) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    df["source_priority"] = priority
    return df


def _metric_value(row: pd.Series, repeated_name: str, single_name: str) -> Any:
    if repeated_name in row and pd.notna(row[repeated_name]):
        return row[repeated_name]
    if single_name in row and pd.notna(row[single_name]):
        return row[single_name]
    return None


def build_downstream_table(downstream_paths: list[str | Path], *, dataset: str) -> pd.DataFrame:
    frames = [_read_downstream(path, idx) for idx, path in enumerate(downstream_paths)]
    frames = [frame for frame in frames if not frame.empty]
    source = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tasks = _task_config(dataset)
    rows: list[dict[str, Any]] = []

    if not source.empty:
        source = source[~source["method"].astype(str).map(_is_blocked_method)].copy()

    for method, meta in _method_config(dataset).items():
        if _is_blocked_method(method):
            continue
        names = [method, *meta.get("downstream_aliases", [])]
        if method.startswith("ADNI_"):
            names.append(method.removeprefix("ADNI_"))
        for task, task_display in tasks.items():
            row = {
                "dataset": dataset,
                "task": task,
                "task_display": task_display,
                "method": method,
                "display": meta["display"],
                "family": meta["family"],
                "role": meta["role"],
                "available": False,
            }
            if not source.empty:
                candidates = source[source["method"].isin(names) & source["task"].eq(task)].sort_values("source_priority")
                if not candidates.empty:
                    item = candidates.iloc[0]
                    row.update(
                        {
                            "available": True,
                            "n_subjects": item.get("n_subjects"),
                            "macro_f1_mean": _metric_value(item, "macro_f1_mean", "macro_f1"),
                            "macro_f1_std": item.get("macro_f1_std"),
                            "balanced_accuracy_mean": _metric_value(item, "balanced_accuracy_mean", "balanced_accuracy"),
                            "macro_auc_ovr_mean": _metric_value(item, "macro_auc_ovr_mean", "macro_auc_ovr"),
                            "source_file": item.get("source_file"),
                        }
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _markdown_table(df: pd.DataFrame, *, title: str) -> str:
    columns = ["dataset", "task_display", "display", "family", "macro_f1_mean", "macro_f1_std", "macro_auc_ovr_mean"]
    available = df[df["available"]].copy()
    if available.empty:
        return f"## {title}\n\nNo available rows.\n"
    available = available[columns]
    lines = [f"## {title}", "", "| Dataset | Task | Method | Family | Macro-F1 | Std | AUC |", "|---|---|---|---|---:|---:|---:|"]
    for _, row in available.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["task_display"]),
                    str(row["display"]),
                    str(row["family"]),
                    _fmt(row["macro_f1_mean"]),
                    _fmt(row["macro_f1_std"]),
                    _fmt(row["macro_auc_ovr_mean"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_table_cn(df: pd.DataFrame, *, title: str) -> str:
    columns = ["dataset", "task_display", "display", "family", "macro_f1_mean", "macro_f1_std", "macro_auc_ovr_mean"]
    available = df[df["available"]].copy()
    if available.empty:
        return f"## {title}\n\n没有可用结果。\n"
    available = available[columns]
    lines = [f"## {title}", "", "| 数据集 | 任务 | 方法 | 类型 | Macro-F1 | 标准差 | AUC |", "|---|---|---|---|---:|---:|---:|"]
    for _, row in available.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["task_display"]),
                    str(row["display"]),
                    str(row["family"]),
                    _fmt(row["macro_f1_mean"]),
                    _fmt(row["macro_f1_std"]),
                    _fmt(row["macro_auc_ovr_mean"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def plot_downstream_grid(downstream: pd.DataFrame, figure_path: str | Path) -> None:
    plot_data = downstream[downstream["available"]].copy()
    preferred_order = [
        "T1 only",
        "Real FA",
        "U-Net",
        "Stack U-Net 5",
        "Pix2Pix",
        "CycleGAN",
        "DDIM",
        "Diffusion Bridge",
        "MOTFM",
        "PM Stage1",
        "Stage1 LPIPS+GAN",
        "Ours Fidelity Flow",
    ]
    task_grid = [
        ("private", "cn_vs_mci", "CN vs MCI"),
        ("private", "cn_vs_ad", "CN vs AD"),
        ("private", "mci_vs_ad", "MCI vs AD"),
        ("adni", "cn_vs_mci_spectrum", "CN vs MCI-spectrum"),
        ("adni", "cn_vs_ad", "CN vs AD"),
        ("adni", "mci_spectrum_vs_ad", "MCI-spectrum vs AD"),
    ]

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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    for ax, (dataset, task, title) in zip(axes.ravel(), task_grid):
        subset = plot_data[(plot_data["dataset"].eq(dataset)) & (plot_data["task"].eq(task))].copy()
        subset["order"] = subset["display"].map({name: idx for idx, name in enumerate(preferred_order)}).fillna(999)
        subset = subset.sort_values(["order", "display"])
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        y_pos = range(len(subset))
        bar_colors = [colors.get(family, "#9c9c9c") for family in subset["family"]]
        xerr = subset["macro_f1_std"].fillna(0.0)
        ax.barh(y_pos, subset["macro_f1_mean"], xerr=xerr, color=bar_colors, alpha=0.9)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(subset["display"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0.0, max(0.75, min(1.0, subset["macro_f1_mean"].max() + 0.18)))
        ax.set_xlabel("Macro-F1")
        ax.set_title(f"{dataset.upper()} | {title}", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        for idx, value in enumerate(subset["macro_f1_mean"]):
            ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=8)

    fig.suptitle("Downstream Binary Classification Across Private and ADNI Datasets", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def write_outputs(
    private_image: pd.DataFrame,
    adni_image: pd.DataFrame,
    private_downstream: pd.DataFrame,
    adni_downstream: pd.DataFrame,
    output_root: str | Path,
    figure_root: str | Path,
) -> dict[str, Path]:
    output_root = Path(output_root)
    figure_root = Path(figure_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    downstream = pd.concat([private_downstream, adni_downstream], ignore_index=True)
    image = pd.concat([private_image, adni_image], ignore_index=True)

    paths = {
        "private_image": output_root / "private_image_metrics.csv",
        "adni_image": output_root / "adni_image_metrics.csv",
        "all_image": output_root / "two_dataset_image_metrics.csv",
        "private_downstream": output_root / "private_downstream_macro_f1.csv",
        "adni_downstream": output_root / "adni_downstream_macro_f1.csv",
        "all_downstream": output_root / "two_dataset_downstream_macro_f1.csv",
        "markdown": output_root / "two_dataset_downstream_macro_f1.md",
        "markdown_cn": output_root / "two_dataset_downstream_macro_f1_cn.md",
        "figure": figure_root / "downstream_macro_f1_by_dataset.png",
    }
    private_image.to_csv(paths["private_image"], index=False)
    adni_image.to_csv(paths["adni_image"], index=False)
    image.to_csv(paths["all_image"], index=False)
    private_downstream.to_csv(paths["private_downstream"], index=False)
    adni_downstream.to_csv(paths["adni_downstream"], index=False)
    downstream.to_csv(paths["all_downstream"], index=False)

    md = _markdown_table(downstream, title="Final Two-Dataset Downstream Comparison")
    md_cn = _markdown_table_cn(downstream, title="最终双数据集下游分类对比")
    paths["markdown"].write_text(md, encoding="utf-8")
    paths["markdown_cn"].write_text(md_cn, encoding="utf-8")
    plot_downstream_grid(downstream, paths["figure"])
    return paths


def main() -> None:
    args = parse_args()
    private_image = build_image_table(args.metrics_root, dataset="private")
    adni_image = build_image_table(args.metrics_root, dataset="adni")
    private_downstream = build_downstream_table(args.private_downstream, dataset="private")
    adni_downstream = build_downstream_table(args.adni_downstream, dataset="adni")
    paths = write_outputs(private_image, adni_image, private_downstream, adni_downstream, args.output_root, args.figure_root)

    print("Saved final two-dataset comparison outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    available = pd.concat([private_downstream, adni_downstream], ignore_index=True)
    available = available[available["available"]]
    if not available.empty:
        print("\nAvailable downstream rows:")
        print(
            available[
                ["dataset", "task", "display", "macro_f1_mean", "macro_f1_std", "macro_auc_ovr_mean", "source_file"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
