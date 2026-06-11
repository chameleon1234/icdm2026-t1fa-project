from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_WEIGHTS = {
    "PSNR": 0.15,
    "SSIM": 0.15,
    "Sharpness_Ratio": 0.20,
    "ROI_CCC": 0.15,
    "WM_Masked_MAE": 0.15,
    "mean_auc": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced utility table from image metrics and final.pdf-style downstream AUC.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Format DISPLAY=METRIC_SUMMARY_NAME=DOWNSTREAM_METHOD, e.g. Ours=PM_DIRF_FIDELITY_FLOW_FULL=Fidelity_Flow",
    )
    parser.add_argument("--metrics_root", default="outputs/icdm2026/metrics")
    parser.add_argument("--downstream_csv", required=True)
    parser.add_argument("--tasks", default="cn_vs_ad,cn_vs_mci,mci_vs_ad")
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def _load_summary(metrics_root: Path, summary_name: str) -> dict[str, Any]:
    path = metrics_root / f"{summary_name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _first(summary: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in summary and summary[key] is not None:
            return float(summary[key])
    return float("nan")


def _parse_method_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("=")
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError(f"Method spec must use DISPLAY=METRIC_SUMMARY_NAME=DOWNSTREAM_METHOD, got {spec!r}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _normalize(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_value = float(values.min(skipna=True))
    max_value = float(values.max(skipna=True))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or abs(max_value - min_value) <= 1e-12:
        return pd.Series(np.ones(len(values), dtype=np.float64), index=series.index)
    score = (values - min_value) / (max_value - min_value)
    if not higher_is_better:
        score = 1.0 - score
    return score


def build_balanced_table(
    *,
    method_specs: list[str],
    metrics_root: str | Path,
    downstream_csv: str | Path,
    tasks: list[str],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    metrics_root = Path(metrics_root)
    downstream = pd.read_csv(downstream_csv)
    rows: list[dict[str, Any]] = []
    for spec in method_specs:
        display, summary_name, downstream_method = _parse_method_spec(spec)
        summary = _load_summary(metrics_root, summary_name)
        task_rows = downstream[(downstream["method"].eq(downstream_method)) & (downstream["task"].isin(tasks))]
        rows.append(
            {
                "method": display,
                "metrics_name": summary_name,
                "downstream_method": downstream_method,
                "PSNR": _first(summary, "PSNR_mean", "psnr"),
                "SSIM": _first(summary, "SSIM_mean", "ssim"),
                "MSE": _first(summary, "MSE_mean", "mse"),
                "MAE": _first(summary, "MAE_mean", "mae"),
                "Sharpness_Ratio": _first(summary, "Sharpness_Ratio_mean", "sharpness_ratio"),
                "WM_Masked_MAE": _first(summary, "WM_Masked_MAE_mean", "wm_mae"),
                "ROI_CCC": _first(summary, "ROI_CCC", "ROI_CCC_mean", "roi_ccc"),
                "mean_auc": float(task_rows["macro_auc_ovr_mean"].mean()) if "macro_auc_ovr_mean" in task_rows else float("nan"),
                "mean_acc": float(task_rows["accuracy_mean"].mean()) if "accuracy_mean" in task_rows else float("nan"),
                "n_tasks": int(task_rows["task"].nunique()),
            }
        )
    table = pd.DataFrame(rows)
    weights = weights or DEFAULT_WEIGHTS
    score = pd.Series(np.zeros(len(table), dtype=np.float64), index=table.index)
    for metric, weight in weights.items():
        if metric not in table.columns:
            continue
        score += float(weight) * _normalize(table[metric], higher_is_better=(metric != "WM_Masked_MAE"))
    table["balanced_score"] = score
    return table.sort_values("balanced_score", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not args.method:
        raise ValueError("At least one --method spec is required")
    table = build_balanced_table(
        method_specs=args.method,
        metrics_root=args.metrics_root,
        downstream_csv=args.downstream_csv,
        tasks=tasks,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved balanced utility table: {output_csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
