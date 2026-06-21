import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHODS = [
    ("U-Net", "ADNI_UNet_E50"),
    ("Pix2Pix", "ADNI_Pix2Pix_E50"),
    ("CycleGAN", "ADNI_CycleGAN_E50"),
    ("DDIM", "ADNI_DDIM_E100_K50_PRETRAINED"),
    ("Diffusion Bridge", "ADNI_DBM_E100_K40_PRETRAINED"),
    ("MOTFM", "ADNI_MOTFM_I2I_K10_PRETRAINED"),
    ("StackUNet5", "ADNI_StackUNet5_E12"),
    ("StackUNet7", "ADNI_StackUNet7_E50"),
    ("Restormer-linear", "ADNI_Restormer_Single_4096_E6"),
    ("Restormer-tanh", "ADNI_Restormer_Single_Tanh_4096_E6"),
    ("Old 5-slice Fidelity Flow", "ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Stage1 Single Sharp", "ADNI_STAGE1_SINGLE_SHARP_FULL_E30"),
    ("Single Fidelity Flow", "ADNI_SINGLE_FIDELITY_FLOW_SHARP_STAGE1_PROBE_4096_E5"),
    ("Ours Single DS Multihead", "ADNI_SINGLE_DS_MULTIHEAD_FINAL_12000_E8"),
]


SUMMARY_KEYS = {
    "PSNR": "PSNR_mean",
    "SSIM": "SSIM_mean",
    "MSE": "MSE_mean",
    "MAE": "MAE_mean",
    "WM-MAE": "WM_Masked_MAE_mean",
    "ROI-CCC": "ROI_CCC",
    "Sharpness": "Sharpness_Ratio_mean",
    "WM-Hist-W": "WM_Hist_Wasserstein_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final ADNI comparison table with task-sensitive metrics.")
    parser.add_argument("--metrics_root", default="outputs/icdm2026/metrics")
    parser.add_argument("--masked_psnr_csv", default="outputs/icdm2026/metrics/adni_masked_psnr_table.csv")
    parser.add_argument("--output_csv", default="outputs/icdm2026/metrics/adni_final_metric_comparison.csv")
    parser.add_argument("--output_md", default="outputs/icdm2026/metrics/adni_final_metric_comparison.md")
    parser.add_argument("--output_cn_md", default="outputs/icdm2026/metrics/adni_final_metric_comparison_cn.md")
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_masked_psnr(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        values = {
            "Brain-PSNR": safe_float(row.get("brain_psnr")),
            "WM-PSNR": safe_float(row.get("wm_psnr")),
        }
        out[str(row["method"])] = values
        out[str(row["folder"])] = values
    return out


def build_frame(args: argparse.Namespace) -> pd.DataFrame:
    metrics_root = Path(args.metrics_root)
    masked = load_masked_psnr(Path(args.masked_psnr_csv))
    rows = []
    for name, method_id in METHODS:
        summary_path = metrics_root / f"{method_id}_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {"Method": name, "MethodID": method_id}
        for out_key, summary_key in SUMMARY_KEYS.items():
            row[out_key] = safe_float(summary.get(summary_key))
        row.update(masked.get(method_id, masked.get(name, {})))
        row["ROI-Inconsistency"] = 1.0 - row["ROI-CCC"] if math.isfinite(row["ROI-CCC"]) else float("nan")
        row["Clear-ROI-Fidelity"] = row["ROI-CCC"] * row["Sharpness"]
        row["WM-Detail-Fidelity"] = row.get("WM-PSNR", float("nan")) * row["Sharpness"]
        row["Brain-Detail-Fidelity"] = row.get("Brain-PSNR", float("nan")) * row["Sharpness"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    add_balanced_scores(frame)
    return frame


def minmax_score(series: pd.Series, lower_better: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if lower_better:
        values = -values
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or abs(float(hi - lo)) < 1e-12:
        return pd.Series([float("nan")] * len(series), index=series.index)
    return (values - lo) / (hi - lo)


def sharpness_adequacy(sharpness: float, low: float = 0.85, high: float = 1.50) -> float:
    if not math.isfinite(sharpness) or sharpness <= 0:
        return float("nan")
    if sharpness < low:
        return max(0.0, sharpness / low)
    if sharpness <= high:
        return 1.0
    return max(0.0, high / sharpness)


def geometric_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].astype(float).clip(lower=1e-6)
    return np.exp(np.log(values).mean(axis=1))


def add_balanced_scores(frame: pd.DataFrame) -> None:
    frame["Sharpness-Adequacy"] = frame["Sharpness"].map(sharpness_adequacy)
    frame["_PSNRScore"] = minmax_score(frame["PSNR"])
    frame["_WMPSNRScore"] = minmax_score(frame["WM-PSNR"])
    frame["_ROIScore"] = minmax_score(frame["ROI-CCC"])
    frame["_WMHistScore"] = minmax_score(frame["WM-Hist-W"], lower_better=True)
    frame["_WMMAEScore"] = minmax_score(frame["WM-MAE"], lower_better=True)
    frame["Balanced-Clinical-Fidelity"] = geometric_mean(
        frame,
        ["_PSNRScore", "_WMPSNRScore", "_ROIScore", "_WMHistScore", "_WMMAEScore", "Sharpness-Adequacy"],
    )


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def add_rank_markers(frame: pd.DataFrame, columns: list[str], lower_better: set[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        best = values.min() if col in lower_better else values.max()
        out[col] = [
            f"**{fmt(v)}**" if math.isfinite(safe_float(v)) and abs(safe_float(v) - best) < 1e-12 else fmt(safe_float(v))
            for v in values
        ]
    return out


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "Method",
        "PSNR",
        "SSIM",
        "MSE",
        "MAE",
        "WM-PSNR",
        "WM-MAE",
        "ROI-CCC",
        "ROI-Inconsistency",
        "Sharpness",
        "Sharpness-Adequacy",
        "WM-Hist-W",
        "Balanced-Clinical-Fidelity",
    ]
    lower = {"MSE", "MAE", "WM-MAE", "ROI-Inconsistency", "WM-Hist-W"}
    shown = add_rank_markers(frame[columns], columns[1:], lower)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---" if col == "Method" else "---:" for col in columns]) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_reports(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_cn_md = Path(args.output_cn_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    table = markdown_table(frame)
    output_md.write_text(
        "# ADNI Final Metric Comparison\n\n"
        + table
        + "\n\n"
        + "Bold values indicate the best method for each metric. Composite metrics are task-sensitive analysis metrics, not replacements for standard PSNR/SSIM.\n",
        encoding="utf-8",
    )
    output_cn_md.write_text(
        "# ADNI 最终指标对比表\n\n"
        + table
        + "\n\n"
        + "加粗表示该指标最优。复合指标是任务敏感分析指标，不替代标准 PSNR/SSIM，而是用于体现清晰度、白质保真和 ROI 医学一致性的综合优势。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    frame = build_frame(args)
    write_reports(frame, args)
    print(Path(args.output_cn_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
