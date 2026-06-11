import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ADNI_METHODS = {
    "ADNI_T1_ONLY": {
        "family": "Reference",
        "role": "reference",
        "status": "available_reference",
        "downstream_aliases": ["T1_ONLY"],
    },
    "ADNI_FA_GT": {
        "family": "Reference",
        "role": "upper_bound",
        "status": "available_reference",
        "downstream_aliases": ["FA_GT"],
    },
    "ADNI_UNet_E50": {
        "family": "CNN",
        "role": "pure_comparison",
        "status": "training_or_available",
    },
    "ADNI_Pix2Pix_E50": {
        "family": "GAN",
        "role": "pure_comparison",
        "status": "planned",
    },
    "ADNI_CycleGAN_E50": {
        "family": "GAN",
        "role": "pure_comparison",
        "status": "planned",
    },
    "ADNI_PM_DIRF_FIDELITY_FLOW_FULL": {
        "family": "Ours",
        "role": "final_two_stage",
        "status": "available",
        "downstream_aliases": ["ADNI_PM_DIRF_FIDELITY_FLOW_FULL", "Fidelity_Flow"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ADNI image/downstream comparison tables.")
    parser.add_argument("--metrics_root", default="outputs/icdm2026/metrics")
    parser.add_argument("--downstream_roots", nargs="*", default=list_default_downstream_roots())
    parser.add_argument("--output_root", default="outputs/icdm2026/tables/adni_comparison")
    parser.add_argument("--task", default="cn_vs_mci_spectrum_ad")
    return parser.parse_args()


def list_default_downstream_roots() -> list[str]:
    return [
        "outputs/icdm2026/downstream_adni_public_refs/classification_summary.csv",
        "outputs/icdm2026/adni_downstream_unet_e50/classification_summary.csv",
        "outputs/icdm2026/adni_downstream_pix2pix_e50/classification_summary.csv",
        "outputs/icdm2026/adni_downstream_cyclegan_e50/classification_summary.csv",
        "outputs/icdm2026/downstream_adni_two_stage_full/classification_summary.csv",
    ]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def build_image_table(metrics_root: str | Path) -> pd.DataFrame:
    metrics_root = Path(metrics_root)
    rows: list[dict[str, Any]] = []
    for method, meta in ADNI_METHODS.items():
        summary_name = meta.get("summary_name", method)
        summary = _load_json(metrics_root / f"{summary_name}_summary.json")
        row = {
            "method": method,
            "family": meta["family"],
            "role": meta["role"],
            "status": meta["status"],
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
                    "ROI_CCC": _first_present(summary, "ROI_CCC_mean", "ROI_CCC"),
                    "pred_dir": summary.get("pred_dir"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_downstream(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    return df


def build_downstream_table(
    downstream_roots: list[str | Path],
    *,
    task: str = "cn_vs_mci_spectrum_ad",
) -> pd.DataFrame:
    frames = [_read_downstream(path) for path in downstream_roots]
    df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, meta in ADNI_METHODS.items():
        row = {
            "method": method,
            "family": meta["family"],
            "role": meta["role"],
            "status": meta["status"],
            "available": False,
        }
        if not df.empty:
            method_names = [method, *meta.get("downstream_aliases", [])]
            candidates = df[(df["method"].isin(method_names)) & (df["task"].eq(task))]
            if not candidates.empty:
                item = candidates.iloc[0]
                row.update(
                    {
                        "available": True,
                        "task": item.get("task"),
                        "n_subjects": item.get("n_subjects"),
                        "macro_f1": item.get("macro_f1"),
                        "balanced_accuracy": item.get("balanced_accuracy"),
                        "auc": item.get("macro_auc_ovr"),
                        "source_file": item.get("source_file"),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image = build_image_table(args.metrics_root)
    downstream = build_downstream_table(args.downstream_roots, task=args.task)

    image_path = output_root / "adni_comparison_image_metrics.csv"
    downstream_path = output_root / "adni_comparison_downstream.csv"
    image.to_csv(image_path, index=False)
    downstream.to_csv(downstream_path, index=False)

    print(f"Saved ADNI image table: {image_path}")
    print(f"Saved ADNI downstream table: {downstream_path}")
    print("\nADNI image metrics:")
    print(image.to_string(index=False))
    print("\nADNI downstream:")
    print(downstream.to_string(index=False))


if __name__ == "__main__":
    main()
