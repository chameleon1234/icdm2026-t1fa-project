from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply T1-guided ROI mean calibration to an existing prediction folder.")
    parser.add_argument("--input_dir", required=True, help="Existing prediction PNG folder to calibrate.")
    parser.add_argument("--t1_dir", required=True, help="T1 PNG folder matching input_dir filenames.")
    parser.add_argument("--train_t1_dir", required=True, help="Training T1 PNG folder used to fit ROI mean mapping.")
    parser.add_argument("--train_fa_dir", required=True, help="Training FA GT PNG folder used to fit ROI mean mapping.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--roi_rows", type=int, default=2)
    parser.add_argument("--roi_cols", type=int, default=3)
    parser.add_argument("--brain_threshold", type=float, default=0.02)
    parser.add_argument("--ridge_alpha", type=float, default=0.1)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--max_delta", type=float, default=0.08)
    parser.add_argument("--smooth_sigma", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _read_png01(path: Path) -> np.ndarray:
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def _write_png01(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", np.clip(image * 255.0, 0, 255).round().astype(np.uint8))
    if not encoded_ok:
        raise ValueError(f"Failed to encode PNG: {path}")
    encoded.tofile(str(path))


def _roi_slices(shape: tuple[int, int], roi_rows: int, roi_cols: int) -> list[tuple[int, int, slice, slice]]:
    height, width = shape
    slices = []
    for row_idx in range(roi_rows):
        y0 = int(round(row_idx * height / roi_rows))
        y1 = int(round((row_idx + 1) * height / roi_rows))
        for col_idx in range(roi_cols):
            x0 = int(round(col_idx * width / roi_cols))
            x1 = int(round((col_idx + 1) * width / roi_cols))
            slices.append((row_idx, col_idx, slice(y0, y1), slice(x0, x1)))
    return slices


def _roi_mean(image: np.ndarray, y_slice: slice, x_slice: slice, brain_threshold: float) -> float:
    patch = image[y_slice, x_slice]
    mask = patch > brain_threshold
    if not np.any(mask):
        return float(np.mean(patch))
    return float(np.mean(patch[mask]))


def _fit_roi_models(
    train_t1_dir: Path,
    train_fa_dir: Path,
    roi_rows: int,
    roi_cols: int,
    brain_threshold: float,
    ridge_alpha: float,
) -> dict[str, Ridge]:
    t1_files = sorted(train_t1_dir.glob("*.png"))
    if not t1_files:
        raise FileNotFoundError(f"No training T1 PNG files found in {train_t1_dir}")
    first = _read_png01(t1_files[0])
    roi_defs = _roi_slices(first.shape, roi_rows, roi_cols)
    features_by_roi: dict[str, list[list[float]]] = {f"r{r}_c{c}": [] for r, c, _, _ in roi_defs}
    targets_by_roi: dict[str, list[float]] = {f"r{r}_c{c}": [] for r, c, _, _ in roi_defs}
    for t1_path in tqdm(t1_files, desc="Fit ROI calibration"):
        fa_path = train_fa_dir / t1_path.name
        if not fa_path.exists():
            continue
        t1 = _read_png01(t1_path)
        fa = _read_png01(fa_path)
        if fa.shape != t1.shape:
            fa = cv2.resize(fa, (t1.shape[1], t1.shape[0]), interpolation=cv2.INTER_CUBIC)
        for row_idx, col_idx, y_slice, x_slice in roi_defs:
            key = f"r{row_idx}_c{col_idx}"
            t1_mean = _roi_mean(t1, y_slice, x_slice, brain_threshold)
            t1_patch = t1[y_slice, x_slice]
            t1_std = float(np.std(t1_patch[t1_patch > brain_threshold])) if np.any(t1_patch > brain_threshold) else float(np.std(t1_patch))
            features_by_roi[key].append([t1_mean, t1_std])
            targets_by_roi[key].append(_roi_mean(fa, y_slice, x_slice, brain_threshold))
    models: dict[str, Ridge] = {}
    for key, x_rows in features_by_roi.items():
        if len(x_rows) < 2:
            raise ValueError(f"Not enough samples to fit ROI model {key}")
        model = Ridge(alpha=ridge_alpha)
        model.fit(np.asarray(x_rows, dtype=np.float32), np.asarray(targets_by_roi[key], dtype=np.float32))
        models[key] = model
    return models


def _predict_roi_target(model: Ridge, t1: np.ndarray, y_slice: slice, x_slice: slice, brain_threshold: float) -> float:
    t1_mean = _roi_mean(t1, y_slice, x_slice, brain_threshold)
    patch = t1[y_slice, x_slice]
    mask = patch > brain_threshold
    t1_std = float(np.std(patch[mask])) if np.any(mask) else float(np.std(patch))
    return float(model.predict(np.asarray([[t1_mean, t1_std]], dtype=np.float32))[0])


def calibrate_image(
    pred: np.ndarray,
    t1: np.ndarray,
    models: dict[str, Ridge],
    roi_rows: int,
    roi_cols: int,
    brain_threshold: float,
    gain: float,
    max_delta: float,
    smooth_sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if t1.shape != pred.shape:
        t1 = cv2.resize(t1, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
    delta_map = np.zeros_like(pred, dtype=np.float32)
    stats: dict[str, float] = {}
    for row_idx, col_idx, y_slice, x_slice in _roi_slices(pred.shape, roi_rows, roi_cols):
        key = f"r{row_idx}_c{col_idx}"
        desired = _predict_roi_target(models[key], t1, y_slice, x_slice, brain_threshold)
        current = _roi_mean(pred, y_slice, x_slice, brain_threshold)
        delta = float(np.clip(desired - current, -max_delta, max_delta) * gain)
        delta_map[y_slice, x_slice] = delta
        stats[f"{key}_desired"] = desired
        stats[f"{key}_current"] = current
        stats[f"{key}_delta"] = delta
    if smooth_sigma > 0:
        delta_map = cv2.GaussianBlur(delta_map, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    brain_mask = (pred > brain_threshold).astype(np.float32)
    calibrated = np.clip(pred + delta_map * brain_mask, 0.0, 1.0)
    stats["mean_abs_delta"] = float(np.mean(np.abs(delta_map[brain_mask > 0.5]))) if np.any(brain_mask > 0.5) else 0.0
    return calibrated, stats


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    t1_dir = Path(args.t1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _fit_roi_models(
        train_t1_dir=Path(args.train_t1_dir),
        train_fa_dir=Path(args.train_fa_dir),
        roi_rows=args.roi_rows,
        roi_cols=args.roi_cols,
        brain_threshold=args.brain_threshold,
        ridge_alpha=args.ridge_alpha,
    )
    input_files = sorted(input_dir.glob("*.png"))
    if args.limit > 0:
        input_files = input_files[: args.limit]
    rows = []
    for pred_path in tqdm(input_files, desc="Apply ROI calibration"):
        t1_path = t1_dir / pred_path.name
        if not t1_path.exists():
            raise FileNotFoundError(f"Missing T1 file for {pred_path.name}: {t1_path}")
        pred = _read_png01(pred_path)
        t1 = _read_png01(t1_path)
        calibrated, stats = calibrate_image(
            pred=pred,
            t1=t1,
            models=models,
            roi_rows=args.roi_rows,
            roi_cols=args.roi_cols,
            brain_threshold=args.brain_threshold,
            gain=args.gain,
            max_delta=args.max_delta,
            smooth_sigma=args.smooth_sigma,
        )
        out_path = output_dir / pred_path.name
        _write_png01(out_path, calibrated)
        rows.append({"fname": pred_path.name, "input_path": str(pred_path), "t1_path": str(t1_path), "output_path": str(out_path), **stats})
    pd.DataFrame(rows).to_csv(output_dir / "export_manifest.csv", index=False, encoding="utf-8")
    model_payload = {
        "roi_rows": args.roi_rows,
        "roi_cols": args.roi_cols,
        "brain_threshold": args.brain_threshold,
        "ridge_alpha": args.ridge_alpha,
        "gain": args.gain,
        "max_delta": args.max_delta,
        "smooth_sigma": args.smooth_sigma,
        "models": {
            key: {"coef": model.coef_.astype(float).tolist(), "intercept": float(model.intercept_)}
            for key, model in models.items()
        },
    }
    with open(output_dir / "roi_calibration_model.json", "w", encoding="utf-8") as handle:
        json.dump(model_payload, handle, indent=2, ensure_ascii=False)
    print(f"ROI_CALIBRATION: exported={len(rows)} output_dir={output_dir}")
    print(f"Saved manifest to: {output_dir / 'export_manifest.csv'}")


if __name__ == "__main__":
    main()
