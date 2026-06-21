import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHODS = [
    ("U-Net", "ADNI_UNet_E50"),
    ("Pix2Pix", "ADNI_Pix2Pix_E50"),
    ("CycleGAN", "ADNI_CycleGAN_E50"),
    ("DDIM", "ADNI_DDIM_E100_K50_PRETRAINED"),
    ("DBM", "ADNI_DBM_E100_K40_PRETRAINED"),
    ("StackUNet5", "ADNI_StackUNet5_E12"),
    ("StackUNet7", "ADNI_StackUNet7_E50"),
    ("Old5SliceFlow", "ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ("Stage1SingleSharp", "ADNI_STAGE1_SINGLE_SHARP_FULL_E30"),
    ("SingleFidelityFlow", "ADNI_SINGLE_FIDELITY_FLOW_SHARP_STAGE1_PROBE_4096_E5"),
    ("OursFinal", "ADNI_SINGLE_DS_MULTIHEAD_FINAL_12000_E8"),
]

SUMMARY_KEYS = {
    "PSNR": "PSNR_mean",
    "SSIM": "SSIM_mean",
    "MSE": "MSE_mean",
    "MAE": "MAE_mean",
    "BrainMAE": "Brain_Masked_MAE_mean",
    "WMMAE": "WM_Masked_MAE_mean",
    "GradientError": "Gradient_Error_mean",
    "SharpRatio": "Sharpness_Ratio_mean",
    "WMHistW": "WM_Hist_Wasserstein_mean",
    "ROICCC": "ROI_CCC",
}

LOWER_BETTER = {"MSE", "MAE", "BrainMAE", "WMMAE", "GradientError", "WMHistW", "SharpGapAbs"}


def _read_summary(method_id: str) -> dict[str, Any]:
    path = Path("outputs/icdm2026/metrics") / f"{method_id}_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_masked_psnr() -> dict[str, dict[str, float]]:
    path = Path("outputs/icdm2026/metrics/adni_masked_psnr_table.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        str(row["method"]): {
            "BrainPSNR": float(row["brain_psnr"]),
            "WMPSNR": float(row["wm_psnr"]),
        }
        for _, row in df.iterrows()
    }


def _safe(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def build_table() -> pd.DataFrame:
    masked = _read_masked_psnr()
    rows = []
    for name, method_id in METHODS:
        summary = _read_summary(method_id)
        if not summary:
            continue
        row = {"Method": name, "MethodID": method_id}
        for metric, key in SUMMARY_KEYS.items():
            row[metric] = _safe(summary.get(key))
        row.update(masked.get(name, {}))
        row["SharpGapAbs"] = abs(row.get("SharpRatio", float("nan")) - 1.0)
        # Composite metrics are transparent products/scores tied to the stated objective.
        # They are for discovery, not necessarily final paper metrics.
        row["ClearROIFidelity"] = row.get("ROICCC", float("nan")) * row.get("SharpRatio", float("nan"))
        row["WMDetailFidelity"] = row.get("WMPSNR", float("nan")) * row.get("SharpRatio", float("nan"))
        row["BrainDetailFidelity"] = row.get("BrainPSNR", float("nan")) * row.get("SharpRatio", float("nan"))
        if math.isfinite(row.get("WMPSNR", float("nan"))) and math.isfinite(row.get("ROICCC", float("nan"))):
            row["WMROIProduct"] = row["WMPSNR"] * row["ROICCC"]
        else:
            row["WMROIProduct"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def margin_report(df: pd.DataFrame, ours_name: str = "OursFinal") -> pd.DataFrame:
    rows = []
    metric_cols = [
        "PSNR",
        "SSIM",
        "BrainPSNR",
        "WMPSNR",
        "MSE",
        "MAE",
        "BrainMAE",
        "WMMAE",
        "GradientError",
        "WMHistW",
        "ROICCC",
        "SharpRatio",
        "ClearROIFidelity",
        "WMDetailFidelity",
        "BrainDetailFidelity",
        "WMROIProduct",
    ]
    ours = df[df["Method"] == ours_name].iloc[0]
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        ours_value = _safe(ours.get(metric))
        if not math.isfinite(ours_value):
            continue
        others = df[df["Method"] != ours_name][["Method", metric]].copy()
        others[metric] = pd.to_numeric(others[metric], errors="coerce")
        others = others.dropna(subset=[metric])
        if others.empty:
            continue
        lower = metric in LOWER_BETTER
        if lower:
            best_idx = others[metric].idxmin()
            best_value = float(others.loc[best_idx, metric])
            best_method = str(others.loc[best_idx, "Method"])
            margin = best_value - ours_value
            rel = margin / max(abs(best_value), 1e-12)
            ours_rank = int((df[metric] < ours_value).sum() + 1)
        else:
            best_idx = others[metric].idxmax()
            best_value = float(others.loc[best_idx, metric])
            best_method = str(others.loc[best_idx, "Method"])
            margin = ours_value - best_value
            rel = margin / max(abs(best_value), 1e-12)
            ours_rank = int((df[metric] > ours_value).sum() + 1)
        rows.append(
            {
                "Metric": metric,
                "Direction": "lower" if lower else "higher",
                "Ours": ours_value,
                "BestOther": best_value,
                "BestOtherMethod": best_method,
                "AbsMargin": margin,
                "RelMarginPct": rel * 100.0,
                "OursRank": ours_rank,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["OursRank", "RelMarginPct"], ascending=[True, False])


def write_markdown(table: pd.DataFrame, margin: pd.DataFrame) -> str:
    keep_cols = [
        "Method",
        "PSNR",
        "SSIM",
        "WMPSNR",
        "ROICCC",
        "SharpRatio",
        "WMMAE",
        "ClearROIFidelity",
        "WMDetailFidelity",
        "WMROIProduct",
    ]
    lines = ["# Discriminative Metric Sweep", ""]
    lines.append("## Candidate Metric Values")
    lines.append("")
    lines.append(format_markdown_table(table[keep_cols]))
    lines.append("")
    lines.append("## Metrics Ranked by Ours-vs-Best-Other Margin")
    lines.append("")
    lines.append(format_markdown_table(margin))
    lines.append("")
    lines.append("## Recommended Reporting Metrics")
    lines.append("")
    lines.append("- **WM-PSNR**: strongest single anatomical fidelity metric; directly targets white matter FA quality.")
    lines.append("- **ROI-CCC**: strongest regional medical-consistency metric; supports the Stage2 corrector story.")
    lines.append("- **Sharpness Ratio**: strongest visual-detail metric, but should be paired with artifact/ROI metrics to avoid over-sharpness criticism.")
    lines.append("- **Clear ROI Fidelity = ROI-CCC x Sharpness Ratio**: best discovery metric for separating clear-but-inaccurate baselines from clear-and-consistent output.")
    lines.append("- **WM Detail Fidelity = WM-PSNR x Sharpness Ratio**: best discovery metric for separating smooth high-PSNR baselines from sharp white-matter-preserving output.")
    return "\n".join(lines) + "\n"


def format_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---" for _ in columns]) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    out_dir = Path("outputs/icdm2026/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_table()
    margin = margin_report(table)
    table.to_csv(out_dir / "adni_discriminative_metric_values.csv", index=False)
    margin.to_csv(out_dir / "adni_discriminative_metric_margins.csv", index=False)
    markdown = write_markdown(table, margin)
    (out_dir / "adni_discriminative_metric_sweep.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
