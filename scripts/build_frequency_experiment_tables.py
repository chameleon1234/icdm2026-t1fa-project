import argparse
from pathlib import Path

import pandas as pd


TASK_ORDER = ["cn_scd_vs_mci_ad", "cn_vs_ad", "cn_vs_mci", "scd_vs_mci", "mci_vs_ad"]
METHOD_ORDER = [
    "PM_STAGE1",
    "Stage1_LPIPS_GAN",
    "Fidelity_Flow",
    "FREQ_PMLOW_GANHIGH",
    "FREQ_PMLOW_FLOWHIGH",
    "FA_GT",
]
VIEW_METHODS = ["PM_STAGE1", "Stage1_LPIPS_GAN", "Fidelity_Flow", "Stage1_GAN_ONLY", "FA_GT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frequency decomposition and fusion downstream tables.")
    parser.add_argument("--fusion_root", default="outputs/icdm2026/downstream_frequency_fusion")
    parser.add_argument("--views_root", default="outputs/icdm2026/downstream_frequency_views")
    parser.add_argument("--output_root", default="outputs/icdm2026/tables")
    return parser.parse_args()


def _fmt(mean: float, std: float | None = None) -> str:
    if pd.isna(mean):
        return "-"
    if std is None or pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_label(task: str) -> str:
    return {
        "cn_scd_vs_mci_ad": "CN+SCD vs MCI+AD",
        "cn_vs_ad": "CN vs AD",
        "cn_vs_mci": "CN vs MCI",
        "scd_vs_mci": "SCD vs MCI",
        "mci_vs_ad": "MCI vs AD",
    }.get(task, task)


def build_fusion_table(fusion_root: Path, output_root: Path) -> None:
    df = pd.read_csv(fusion_root / "classification_repeated_summary.csv")
    rows = df[df["method"].isin(METHOD_ORDER) & df["task"].isin(TASK_ORDER)].copy()
    pivot_mean = rows.pivot(index="method", columns="task", values="macro_f1_mean").reindex(METHOD_ORDER)
    pivot_std = rows.pivot(index="method", columns="task", values="macro_f1_std").reindex(METHOD_ORDER)
    headers = ["Method"] + [_task_label(task) for task in TASK_ORDER]
    lines = [
        "# Frequency Fusion Downstream Table",
        "",
        "Macro-F1 is reported as mean +/- std over repeated 5-fold CV.",
        "",
        "| " + " | ".join(headers) + " |",
        "|---" + "|---:" * len(TASK_ORDER) + "|",
    ]
    cn_lines = [
        "# 频率融合下游分类表",
        "",
        "Macro-F1 以 repeated 5-fold CV 的 mean +/- std 汇报。",
        "",
        "| " + " | ".join(headers) + " |",
        "|---" + "|---:" * len(TASK_ORDER) + "|",
    ]
    for method in METHOD_ORDER:
        values = [_fmt(pivot_mean.loc[method, task], pivot_std.loc[method, task]) for task in TASK_ORDER]
        line = "| " + " | ".join([method] + values) + " |"
        lines.append(line)
        cn_lines.append(line)
    _write(output_root / "frequency_fusion_downstream_table.md", lines)
    _write(output_root / "frequency_fusion_downstream_table_cn.md", cn_lines)


def build_view_table(views_root: Path, output_root: Path) -> None:
    lines = [
        "# Frequency View Downstream Table",
        "",
        "Values are Macro-F1 means over repeated 5-fold CV. Full/lowpass/highpass use the same prediction PNGs but different feature views.",
        "",
        "| Method | Task | Full | Lowpass | Highpass |",
        "|---|---|---:|---:|---:|",
    ]
    cn_lines = [
        "# 频率视图下游分类表",
        "",
        "数值为 repeated 5-fold CV 的 Macro-F1 mean。Full/lowpass/highpass 使用相同预测 PNG，但提取不同频段特征。",
        "",
        "| Method | Task | Full | Lowpass | Highpass |",
        "|---|---|---:|---:|---:|",
    ]
    view_tables = {}
    for view in ["full", "lowpass", "highpass"]:
        df = pd.read_csv(views_root / view / "classification_repeated_summary.csv")
        view_tables[view] = df.pivot(index="method", columns="task", values="macro_f1_mean")
    for method in VIEW_METHODS:
        for task in TASK_ORDER:
            values = [_fmt(view_tables[view].loc[method, task]) for view in ["full", "lowpass", "highpass"]]
            line = "| " + " | ".join([method, _task_label(task)] + values) + " |"
            lines.append(line)
            cn_lines.append(line)
    _write(output_root / "frequency_view_downstream_table.md", lines)
    _write(output_root / "frequency_view_downstream_table_cn.md", cn_lines)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    build_fusion_table(Path(args.fusion_root), output_root)
    build_view_table(Path(args.views_root), output_root)
    print(f"Saved frequency experiment tables to: {output_root}")


if __name__ == "__main__":
    main()
