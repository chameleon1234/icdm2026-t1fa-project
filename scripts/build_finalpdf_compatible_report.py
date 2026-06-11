import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BLOCKLIST_TOKENS = ("FREQ", "Frequency", "STACKLOW", "FLOWHIGH", "OURSHIGH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ACC/AUC-style report for final.pdf-compatible downstream experiments.")
    parser.add_argument("--private_root", default="outputs/icdm2026/downstream_finalpdf_compatible_private")
    parser.add_argument("--adni_root", default="outputs/icdm2026/downstream_finalpdf_compatible_adni")
    parser.add_argument("--output_root", default="outputs/icdm2026/tables/finalpdf_compatible")
    return parser.parse_args()


def _is_blocked(method: Any) -> bool:
    method = str(method)
    return any(token in method for token in BLOCKLIST_TOKENS)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _pct(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") * 100.0).round(2)


def _dataset_table(root: Path, dataset: str) -> pd.DataFrame:
    single = _read_csv(root / "classification_summary.csv")
    repeated = _read_csv(root / "classification_repeated_summary.csv")
    if single.empty:
        return pd.DataFrame()
    single = single[~single["method"].map(_is_blocked)].copy()
    single["dataset"] = dataset
    for source, target in [
        ("accuracy", "accuracy_pct"),
        ("balanced_accuracy", "balanced_accuracy_pct"),
        ("macro_f1", "macro_f1_pct"),
        ("macro_auc_ovr", "auc_pct"),
    ]:
        if source in single.columns:
            single[target] = _pct(single[source])

    if not repeated.empty:
        repeated = repeated[~repeated["method"].map(_is_blocked)].copy()
        keep = ["method", "task"]
        for source, target in [
            ("accuracy_mean", "repeated_accuracy_pct"),
            ("balanced_accuracy_mean", "repeated_balanced_accuracy_pct"),
            ("macro_f1_mean", "repeated_macro_f1_pct"),
            ("macro_auc_ovr_mean", "repeated_auc_pct"),
        ]:
            if source in repeated.columns:
                repeated[target] = _pct(repeated[source])
                keep.append(target)
        single = single.merge(repeated[keep], on=["method", "task"], how="left")
    return single


def build_report_table(private_root: str | Path, adni_root: str | Path) -> pd.DataFrame:
    private = _dataset_table(Path(private_root), "private")
    adni = _dataset_table(Path(adni_root), "adni")
    table = pd.concat([frame for frame in [private, adni] if not frame.empty], ignore_index=True)
    if table.empty:
        return table
    preferred = [
        "dataset",
        "method",
        "task",
        "n_subjects",
        "accuracy_pct",
        "auc_pct",
        "macro_f1_pct",
        "balanced_accuracy_pct",
        "repeated_accuracy_pct",
        "repeated_auc_pct",
        "repeated_macro_f1_pct",
        "repeated_balanced_accuracy_pct",
    ]
    existing = [column for column in preferred if column in table.columns]
    return table[existing].sort_values(["dataset", "task", "accuracy_pct"], ascending=[True, True, False])


def _markdown(table: pd.DataFrame, *, chinese: bool = False) -> str:
    if table.empty:
        return "No rows.\n" if not chinese else "没有结果。\n"
    if chinese:
        lines = [
            "# final.pdf 兼容下游实验结果",
            "",
            "协议: ROI mean 特征 + Linear SVM + T1 与合成 FA 早期拼接。ACC/AUC 用于对齐 final.pdf，repeated Macro-F1 用于稳健性判断。",
            "",
            "| 数据集 | 方法 | 任务 | N | ACC% | AUC% | Macro-F1% | repeated ACC% | repeated AUC% | repeated Macro-F1% |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# final.pdf-Compatible Downstream Results",
            "",
            "Protocol: ROI mean features + Linear SVM + early fusion between T1 and synthesized FA. ACC/AUC are reported to match final.pdf; repeated Macro-F1 is retained for robustness.",
            "",
            "| Dataset | Method | Task | N | ACC% | AUC% | Macro-F1% | repeated ACC% | repeated AUC% | repeated Macro-F1% |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("method", "")),
                    str(row.get("task", "")),
                    _fmt(row.get("n_subjects"), 0),
                    _fmt(row.get("accuracy_pct")),
                    _fmt(row.get("auc_pct")),
                    _fmt(row.get("macro_f1_pct")),
                    _fmt(row.get("repeated_accuracy_pct")),
                    _fmt(row.get("repeated_auc_pct")),
                    _fmt(row.get("repeated_macro_f1_pct")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def write_report(table: pd.DataFrame, output_root: str | Path) -> dict[str, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_root / "finalpdf_compatible_acc_auc_table.csv",
        "md": output_root / "finalpdf_compatible_acc_auc_table.md",
        "md_cn": output_root / "finalpdf_compatible_acc_auc_table_cn.md",
    }
    table.to_csv(paths["csv"], index=False, encoding="utf-8")
    paths["md"].write_text(_markdown(table, chinese=False), encoding="utf-8")
    paths["md_cn"].write_text(_markdown(table, chinese=True), encoding="utf-8")
    return paths


def main() -> None:
    args = parse_args()
    table = build_report_table(args.private_root, args.adni_root)
    paths = write_report(table, args.output_root)
    print("Saved final.pdf-compatible report:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    if not table.empty:
        focus = table[table["method"].isin(["T1_ONLY", "T1_PLUS_Ours", "T1_PLUS_GT", "T1_PLUS_Stage1"])]
        print("\nFocus rows:")
        print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
