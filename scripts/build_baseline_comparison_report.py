import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_METHODS = [
    ("UNet_E99", "CNN", "U-Net", "Supervised CNN baseline"),
    ("Pix2Pix_E100", "GAN", "Pix2Pix", "Paired GAN"),
    ("CycleGAN_E100", "GAN", "CycleGAN", "Unpaired GAN"),
    ("DDIM_E100_K50", "Diffusion", "DDIM", "50-step diffusion"),
    ("DIRF_V5_3SLICE_K6", "Flow", "DIRF V5", "Legacy flow baseline"),
    ("PM_STAGE1", "PMRF", "PM Stage1", "Posterior-mean predictor"),
    ("PM_STAGE1_LPIPS_GAN_5SLICE_FINAL", "Ours/Ablation", "Stage1 LPIPS+GAN", "Sharp Stage1"),
    ("PM_DIRF_FIDELITY_FLOW_FULL", "Ours/Ablation", "Fidelity Flow", "Medical correction flow"),
    ("FREQ_FLOWBASE_B035", "Ours", "FREQ_FLOWBASE_B035", "Balanced frequency-preserving method"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a comparison-method report from unified metric and downstream outputs.")
    parser.add_argument("--metrics_root", default="outputs/icdm2026/metrics")
    parser.add_argument("--downstream_root", default="outputs/icdm2026/downstream_legacy_baselines")
    parser.add_argument("--output_root", default="outputs/icdm2026/tables")
    parser.add_argument("--main_task", default="cn_scd_vs_mci_ad")
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_metric_row(metrics_root: Path, method_key: str) -> dict[str, Any]:
    path = metrics_root / f"{method_key}_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "PSNR": _safe_float(data.get("PSNR_mean")),
        "SSIM": _safe_float(data.get("SSIM_mean")),
        "MSE": _safe_float(data.get("MSE_mean")),
        "MAE": _safe_float(data.get("MAE_mean")),
        "SharpRatio": _safe_float(data.get("Sharpness_Ratio_mean")),
        "WM_MAE": _safe_float(data.get("WM_Masked_MAE_mean")),
        "ROI_CCC": _safe_float(data.get("ROI_CCC")),
    }


def _load_repeated_downstream(downstream_root: Path) -> pd.DataFrame:
    path = downstream_root / "classification_repeated_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _markdown_table(df: pd.DataFrame, *, cn: bool = False) -> list[str]:
    intro = (
        "说明：PSNR/SSIM/SharpRatio/ROI-CCC/Macro-F1 越高越好，MSE/MAE/WM-MAE 越低越好。"
        if cn
        else "Note: higher is better for PSNR, SSIM, SharpRatio, ROI-CCC, and Macro-F1; lower is better for MSE, MAE, and WM-MAE."
    )
    lines = [
        intro,
        "",
        "| Category | Method | PSNR | SSIM | MSE | MAE | Sharp | WM-MAE | ROI | Main Macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["category"]),
                    str(row["display_name"]),
                    _fmt(row.get("PSNR"), 2),
                    _fmt(row.get("SSIM"), 4),
                    _fmt(row.get("MSE"), 5),
                    _fmt(row.get("MAE"), 5),
                    _fmt(row.get("SharpRatio"), 3),
                    _fmt(row.get("WM_MAE"), 3),
                    _fmt(row.get("ROI_CCC"), 3),
                    f"{_fmt(row.get('Macro_F1'), 3)} +/- {_fmt(row.get('Macro_F1_Std'), 3)}",
                ]
            )
            + " |"
        )
    return lines


def build_report(args: argparse.Namespace) -> dict[str, Path]:
    metrics_root = Path(args.metrics_root)
    downstream_root = Path(args.downstream_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    downstream_df = _load_repeated_downstream(downstream_root)
    main_downstream = {}
    if not downstream_df.empty:
        task_df = downstream_df[downstream_df["task"] == args.main_task]
        main_downstream = {str(row["method"]): row.to_dict() for _, row in task_df.iterrows()}

    rows = []
    for method_key, category, display_name, notes in DEFAULT_METHODS:
        metric_row = _read_metric_row(metrics_root, method_key)
        downstream_key = {
            "PM_STAGE1_LPIPS_GAN_5SLICE_FINAL": "LPIPS_GAN",
            "PM_DIRF_FIDELITY_FLOW_FULL": "Fidelity_Flow",
            "FREQ_FLOWBASE_B035": "FREQ_FLOWBASE_B035",
        }.get(method_key, method_key)
        downstream_row = main_downstream.get(downstream_key, {})
        rows.append(
            {
                "method_key": method_key,
                "display_name": display_name,
                "category": category,
                "notes": notes,
                **metric_row,
                "Macro_F1": _safe_float(downstream_row.get("macro_f1_mean")),
                "Macro_F1_Std": _safe_float(downstream_row.get("macro_f1_std")),
            }
        )
    comparison_df = pd.DataFrame(rows)

    csv_path = output_root / "legacy_baseline_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8")

    pivot_path = output_root / "legacy_baseline_downstream_all_tasks.csv"
    if not downstream_df.empty:
        pivot = downstream_df.pivot(index="method", columns="task", values="macro_f1_mean")
        pivot.to_csv(pivot_path, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(pivot_path, index=False)

    md_path = output_root / "legacy_baseline_comparison.md"
    cn_path = output_root / "legacy_baseline_comparison_cn.md"
    md_lines = [
        "# Legacy Baseline Comparison",
        "",
        "This table uses the same test PNG folders, image metrics, fixed-slice visualization protocol, and repeated subject-level downstream classifier.",
        "",
        *_markdown_table(comparison_df, cn=False),
        "",
        "## Experiment Design",
        "",
        "- CNN: U-Net represents deterministic supervised regression.",
        "- GAN: Pix2Pix tests paired adversarial translation; CycleGAN tests unpaired adversarial translation.",
        "- Diffusion: DDIM tests iterative conditional sampling.",
        "- Flow: DIRF/PMRF/Fidelity/Frequency variants test the flow and posterior-mean family motivated by the local papers.",
    ]
    cn_lines = [
        "# 传统对比方法实验汇总",
        "",
        "本表所有方法都使用同一测试集 PNG、同一图像指标、同一固定切片可视化顺序，以及同一套 repeated subject-level 下游分类器。",
        "",
        *_markdown_table(comparison_df, cn=True),
        "",
        "## 实验设计",
        "",
        "- CNN：U-Net 代表确定性监督回归方法。",
        "- GAN：Pix2Pix 代表 paired adversarial translation，CycleGAN 代表 unpaired adversarial translation。",
        "- Diffusion：DDIM 代表条件扩散迭代采样方法。",
        "- Flow：DIRF、PMRF、Fidelity、Frequency 系列对应 papers 文件夹中 PMRF、flow matching、少步/蒸馏思想相关路线。",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    cn_path.write_text("\n".join(cn_lines) + "\n", encoding="utf-8")
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved downstream pivot: {pivot_path}")
    print(f"Saved report: {md_path}")
    print(f"Saved Chinese report: {cn_path}")
    return {"csv": csv_path, "pivot": pivot_path, "md": md_path, "cn": cn_path}


if __name__ == "__main__":
    build_report(parse_args())
