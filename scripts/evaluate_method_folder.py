import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import load_subject_index, normalize_subject_id
from src.eval.image_metrics import (
    build_brain_mask,
    build_wm_mask,
    collect_spatial_roi_means,
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_ssim,
    gradient_error,
    histogram_wasserstein_distance,
    masked_laplacian_variance,
    masked_mae,
    roi_ccc,
)


METRIC_COLUMNS = [
    "PSNR",
    "SSIM",
    "MSE",
    "MAE",
    "Brain_Masked_MAE",
    "WM_Masked_MAE",
    "Gradient_Error",
    "Sharpness",
    "Target_Sharpness",
    "Sharpness_Ratio",
    "WM_Hist_Wasserstein",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a T1-to-FA prediction folder on the common test split.")
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted FA PNG files.")
    parser.add_argument("--method", required=True, help="Method name used in output files.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--metrics_root", default="", help="Override metrics output directory from config.")
    parser.add_argument("--test_t1_dir", default="", help="Override test T1 slice directory from config.")
    parser.add_argument("--test_fa_dir", default="", help="Override test FA slice directory from config.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N target slices for smoke tests.")
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing predictions instead of failing.")
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--roi_rows", type=int, default=2)
    parser.add_argument("--roi_cols", type=int, default=3)
    parser.add_argument("--roi_min_pixels", type=int, default=32)
    parser.add_argument("--hist_bins", type=int, default=64)
    return parser.parse_args()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_grayscale_tensor(path: str | Path, target_shape: tuple[int, int] | None = None) -> torch.Tensor:
    path = Path(path)
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if target_shape is not None and image.shape != target_shape:
        image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.clamp(0.0, 1.0)


def _parse_slice_id(filename: str) -> int:
    match = re.search(r"_z(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Cannot parse slice id from {filename}")
    return int(match.group(1))


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def _nanstd(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0 if arr.size > 0 and np.all(np.isinf(arr)) else float("nan")
    return float(np.nanstd(finite))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _subject_metadata_lookup(subject_index: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        normalize_subject_id(row["subject_id"]): row
        for row in subject_index.to_dict(orient="records")
    }


def evaluate_folder(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_yaml(args.config)
    processed_root = Path(config["data"]["processed_root"])
    pred_dir = Path(args.pred_dir)
    test_t1_dir = Path(args.test_t1_dir) if args.test_t1_dir else processed_root / "test" / "t1_slices"
    test_fa_dir = Path(args.test_fa_dir) if args.test_fa_dir else processed_root / "test" / "fa_slices"
    metrics_root = Path(args.metrics_root) if args.metrics_root else Path(config["outputs"]["metrics_root"])
    metrics_root.mkdir(parents=True, exist_ok=True)

    subject_index = load_subject_index(
        excel_path=config["data"]["excel_path"],
        split_json=config["data"]["split_json"],
        sheet_name=config["data"].get("excel_sheet", "re_order"),
    )
    metadata_lookup = _subject_metadata_lookup(subject_index)

    target_files = sorted(test_fa_dir.glob("*.png"))
    if args.limit > 0:
        target_files = target_files[: args.limit]
    if not target_files:
        raise FileNotFoundError(f"No target PNG files found in {test_fa_dir}")

    rows: list[dict[str, Any]] = []
    roi_pred_values: list[float] = []
    roi_target_values: list[float] = []
    missing_predictions: list[str] = []

    for target_path in target_files:
        filename = target_path.name
        pred_path = pred_dir / filename
        if not pred_path.exists():
            missing_predictions.append(filename)
            if args.allow_missing:
                continue
            continue

        t1_path = test_t1_dir / filename
        if not t1_path.exists():
            raise FileNotFoundError(f"Missing paired T1 slice: {t1_path}")

        target = _read_grayscale_tensor(target_path)
        pred = _read_grayscale_tensor(pred_path, target_shape=tuple(target.shape[-2:]))
        t1 = _read_grayscale_tensor(t1_path, target_shape=tuple(target.shape[-2:]))

        brain_mask = build_brain_mask(t1, target, args.brain_t1_threshold, args.brain_fa_threshold)
        wm_mask = build_wm_mask(target, brain_mask, args.wm_quantile, args.wm_min_threshold)
        slice_roi_pred, slice_roi_target = collect_spatial_roi_means(
            pred,
            target,
            brain_mask,
            wm_mask,
            roi_rows=args.roi_rows,
            roi_cols=args.roi_cols,
            min_pixels=args.roi_min_pixels,
        )
        roi_pred_values.extend(slice_roi_pred)
        roi_target_values.extend(slice_roi_target)

        pred_sharp = masked_laplacian_variance(pred, brain_mask)
        target_sharp = masked_laplacian_variance(target, brain_mask)
        subject_id = normalize_subject_id(filename.split("_z", 1)[0])
        if subject_id not in metadata_lookup:
            raise KeyError(f"Subject {subject_id} missing from subject index")
        metadata = metadata_lookup[subject_id]

        rows.append(
            {
                "method": args.method,
                "fname": filename,
                "subject_id": subject_id,
                "slice_id": _parse_slice_id(filename),
                "group_id": int(metadata["group_id"]),
                "group_name": str(metadata["group_name"]),
                "age": float(metadata["age"]),
                "gender": int(metadata["gender"]),
                "edu": float(metadata["edu"]),
                "MMSE": float(metadata["MMSE"]),
                "split": str(metadata["split"]),
                "PSNR": compute_psnr(pred, target),
                "SSIM": compute_ssim(pred, target),
                "MSE": compute_mse(pred, target),
                "MAE": compute_mae(pred, target),
                "Brain_Masked_MAE": masked_mae(pred, target, brain_mask),
                "WM_Masked_MAE": masked_mae(pred, target, wm_mask),
                "Gradient_Error": gradient_error(pred, target, brain_mask),
                "Sharpness": pred_sharp,
                "Target_Sharpness": target_sharp,
                "Sharpness_Ratio": pred_sharp / max(target_sharp, 1e-8) if math.isfinite(target_sharp) else float("nan"),
                "WM_Hist_Wasserstein": histogram_wasserstein_distance(pred[wm_mask], target[wm_mask], bins=args.hist_bins),
            }
        )

    if missing_predictions and not args.allow_missing:
        preview = ", ".join(missing_predictions[:10])
        raise FileNotFoundError(f"Missing {len(missing_predictions)} predictions in {pred_dir}: {preview}")
    if not rows:
        raise RuntimeError("No slices were evaluated.")

    slice_df = pd.DataFrame(rows)
    subject_df = (
        slice_df.groupby(["method", "subject_id", "group_id", "group_name", "split"], as_index=False)
        .agg(
            n_slices=("fname", "count"),
            age=("age", "first"),
            gender=("gender", "first"),
            edu=("edu", "first"),
            MMSE=("MMSE", "first"),
            **{metric: (metric, "mean") for metric in METRIC_COLUMNS},
        )
        .sort_values(["subject_id"])
        .reset_index(drop=True)
    )

    summary: dict[str, Any] = {
        "method": args.method,
        "pred_dir": str(pred_dir),
        "test_t1_dir": str(test_t1_dir),
        "test_fa_dir": str(test_fa_dir),
        "n_slices": int(len(slice_df)),
        "n_subjects": int(subject_df["subject_id"].nunique()),
        "missing_prediction_count": int(len(missing_predictions)),
        "ROI_CCC": roi_ccc(roi_pred_values, roi_target_values),
    }
    for metric in METRIC_COLUMNS:
        values = slice_df[metric].astype(float).tolist()
        summary[f"{metric}_mean"] = _nanmean(values)
        summary[f"{metric}_std"] = _nanstd(values)
    summary["group_metrics"] = {
        group_name: {
            metric: _nanmean(group_df[metric].astype(float).tolist())
            for metric in ["PSNR", "SSIM", "MSE", "MAE", "Brain_Masked_MAE", "WM_Masked_MAE"]
        }
        for group_name, group_df in slice_df.groupby("group_name")
    }

    slice_path = metrics_root / f"{args.method}_slice_metrics.csv"
    subject_path = metrics_root / f"{args.method}_subject_metrics.csv"
    summary_path = metrics_root / f"{args.method}_summary.json"
    slice_df.to_csv(slice_path, index=False, encoding="utf-8")
    subject_df.to_csv(subject_path, index=False, encoding="utf-8")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, ensure_ascii=False)

    print(
        f"{args.method}: n_slices={summary['n_slices']} n_subjects={summary['n_subjects']} "
        f"PSNR={summary['PSNR_mean']:.3f} SSIM={summary['SSIM_mean']:.4f} "
        f"MSE={summary['MSE_mean']:.6f} MAE={summary['MAE_mean']:.6f}"
    )
    print(f"Saved slice metrics to: {slice_path}")
    print(f"Saved subject metrics to: {subject_path}")
    print(f"Saved summary to: {summary_path}")
    return summary


if __name__ == "__main__":
    evaluate_folder(parse_args())

