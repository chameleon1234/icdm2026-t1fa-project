import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PURE_BASELINE_METHODS = {
    "UNet_CurrentSplit_E100": {
        "family": "CNN",
        "role": "pure_comparison",
        "status": "fair_retrained_current_split",
        "summary_name": "UNet_CurrentSplit_E100",
    },
    "Pix2Pix_CurrentSplit_E100": {
        "family": "GAN",
        "role": "pure_comparison",
        "status": "fair_retrained_current_split",
        "summary_name": "Pix2Pix_CurrentSplit_E100",
    },
    "CycleGAN_CurrentSplit_E100": {
        "family": "GAN",
        "role": "pure_comparison",
        "status": "fair_retrain_running",
        "summary_name": "CycleGAN_CurrentSplit_E100",
    },
    "DIRF_V5_3SLICE_K6": {
        "family": "Flow",
        "role": "pure_comparison",
        "status": "project_flow_baseline",
        "summary_name": "DIRF_V5_3SLICE_K6",
        "downstream_aliases": ["DIRF_V5_K6"],
    },
}

REFERENCE_METHODS = {
    "T1_ONLY": {"family": "Reference", "role": "reference"},
    "FA_GT": {"family": "Reference", "role": "upper_bound"},
}

OURS_METHODS = {
    "FREQ_FLOWBASE_B035": {
        "family": "Ours",
        "role": "main_candidate",
        "summary_name": "FREQ_FLOWBASE_PMLOW_B035",
        "downstream_aliases": ["FREQ_FLOWBASE_PMLOW_B035"],
    }
}

EXCLUDED_METHODS = {
    "UNet_E99": "legacy/unverified split; suspiciously high PSNR, exclude from pure comparison",
    "Pix2Pix_E100": "legacy/unverified split, exclude from pure comparison",
    "CycleGAN_E100": "legacy CycleGAN; replace with CycleGAN_CurrentSplit_E100 when finished",
    "DDIM_E100_K50": "legacy or unverified split; retrain before pure comparison",
    "FREQ_FLOWBASE_PMLOW_B035": "ours/fusion method, not a pure baseline",
    "PM_STAGE1": "ours/PMRF ablation, not a pure external baseline",
    "PM_STAGE1_LPIPS_GAN_5SLICE_FINAL": "ours/stage1 ablation, not a pure external baseline",
    "PM_DIRF_FIDELITY_FLOW_FULL": "ours/stage2 ablation, not a pure external baseline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean pure-comparison tables without legacy/fusion contamination.")
    parser.add_argument("--metrics_root", default="outputs/icdm2026/metrics")
    parser.add_argument("--downstream_roots", nargs="*", default=list_default_downstream_roots())
    parser.add_argument("--output_root", default="outputs/icdm2026/tables")
    parser.add_argument("--task", default="cn_scd_vs_mci_ad")
    parser.add_argument("--include_ours", action="store_true", help="Append current main candidate after pure baselines.")
    parser.add_argument("--include_references", action="store_true", help="Append T1_ONLY and FA_GT reference rows.")
    return parser.parse_args()


def list_default_downstream_roots() -> list[str]:
    return [
        "outputs/icdm2026/downstream_pix2pix_current_e100/classification_summary.csv",
        "outputs/icdm2026/downstream_unet_current_split/classification_summary.csv",
        "outputs/icdm2026/downstream_robustness_core/classification_summary.csv",
        "outputs/icdm2026/downstream_freq_flowbase_b035_full/classification_summary.csv",
    ]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _method_catalog(include_ours: bool, include_references: bool) -> dict[str, dict[str, Any]]:
    catalog = dict(PURE_BASELINE_METHODS)
    if include_ours:
        catalog.update(OURS_METHODS)
    if include_references:
        catalog.update(REFERENCE_METHODS)
    return catalog


def build_image_table(metrics_root: str | Path, *, include_ours: bool = False, include_references: bool = False) -> pd.DataFrame:
    metrics_root = Path(metrics_root)
    rows: list[dict[str, Any]] = []
    for method, meta in _method_catalog(include_ours, include_references).items():
        summary_name = meta.get("summary_name", method)
        summary = _load_json(metrics_root / f"{summary_name}_summary.json")
        row = {
            "method": method,
            "family": meta["family"],
            "role": meta["role"],
            "status": meta.get("status", "available" if summary else "missing"),
            "available": summary is not None,
        }
        if summary:
            row.update(
                {
                    "PSNR": summary.get("PSNR_mean"),
                    "SSIM": summary.get("SSIM_mean"),
                    "MSE": summary.get("MSE_mean"),
                    "MAE": summary.get("MAE_mean"),
                    "SharpRatio": summary.get("Sharpness_Ratio_mean"),
                    "WM_MAE": summary.get("WM_Masked_MAE_mean"),
                    "ROI_CCC": summary.get("ROI_CCC"),
                    "pred_dir": summary.get("pred_dir"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _load_downstream_frames(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _downstream_names_for(method: str, meta: dict[str, Any]) -> list[str]:
    return [method] + list(meta.get("downstream_aliases", []))


def build_downstream_table(
    downstream_roots: list[str | Path],
    *,
    task: str = "cn_scd_vs_mci_ad",
    include_ours: bool = False,
    include_references: bool = False,
) -> pd.DataFrame:
    raw = _load_downstream_frames(downstream_roots)
    rows: list[dict[str, Any]] = []
    catalog = _method_catalog(include_ours, include_references)
    for method, meta in catalog.items():
        if raw.empty:
            subset = pd.DataFrame()
        else:
            names = _downstream_names_for(method, meta)
            subset = raw[raw["method"].isin(names) & raw["task"].eq(task)].copy()
        row = {
            "method": method,
            "family": meta["family"],
            "role": meta["role"],
            "status": meta.get("status", "available" if not subset.empty else "missing"),
            "available": not subset.empty,
        }
        if not subset.empty:
            selected = subset.iloc[-1]
            row.update(
                {
                    "task": selected["task"],
                    "n_subjects": int(selected["n_subjects"]),
                    "macro_f1": float(selected["macro_f1"]),
                    "balanced_accuracy": float(selected["balanced_accuracy"]),
                    "auc": float(selected["macro_auc_ovr"]),
                    "source_file": selected["source_file"],
                    "source_method": selected["method"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image = build_image_table(
        args.metrics_root,
        include_ours=args.include_ours,
        include_references=args.include_references,
    )
    downstream = build_downstream_table(
        args.downstream_roots,
        task=args.task,
        include_ours=args.include_ours,
        include_references=args.include_references,
    )
    excluded = pd.DataFrame([{"method": method, "reason": reason} for method, reason in EXCLUDED_METHODS.items()])

    image_path = output_root / "pure_comparison_image_metrics.csv"
    downstream_path = output_root / "pure_comparison_downstream.csv"
    excluded_path = output_root / "pure_comparison_excluded_methods.csv"
    image.to_csv(image_path, index=False)
    downstream.to_csv(downstream_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    _write_markdown_table(output_root / "pure_comparison_image_metrics.md", image)
    _write_markdown_table(output_root / "pure_comparison_downstream.md", downstream)
    _write_markdown_table(output_root / "pure_comparison_excluded_methods.md", excluded)

    print(f"Saved image table: {image_path}")
    print(f"Saved downstream table: {downstream_path}")
    print(f"Saved excluded-method table: {excluded_path}")
    print("\nImage metrics:")
    print(image.to_string(index=False))
    print("\nDownstream:")
    print(downstream.to_string(index=False))


if __name__ == "__main__":
    main()
