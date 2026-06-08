from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import load_subject_index, normalize_subject_id
from src.eval.downstream_utility import (
    TASK_DEFINITIONS,
    extract_subject_features_from_folder,
    feature_columns,
    parse_subject_and_slice,
)


@dataclass
class TaskScoreModel:
    columns: list[str]
    estimator: object
    score_scale: float

    def score_subject(self, row: dict[str, float] | pd.Series) -> float:
        x = np.asarray([[float(row[column]) for column in self.columns]], dtype=np.float32)
        raw = float(np.asarray(self.estimator.decision_function(x)).reshape(-1)[0])
        scale = self.score_scale if self.score_scale > 1e-6 else 1.0
        return float(np.tanh(raw / scale))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply label-trained, T1-guided task-sensitive ROI calibration to prediction PNGs.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--t1_dir", required=True)
    parser.add_argument("--train_t1_dir", required=True)
    parser.add_argument("--train_fa_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", default="cn_vs_ad,cn_vs_mci")
    parser.add_argument("--roi_rows", type=int, default=2)
    parser.add_argument("--roi_cols", type=int, default=3)
    parser.add_argument("--brain_threshold", type=float, default=0.02)
    parser.add_argument("--ridge_alpha", type=float, default=0.1)
    parser.add_argument("--base_gain", type=float, default=0.35)
    parser.add_argument("--base_residual_weight", type=float, default=1.0)
    parser.add_argument("--task_strength", type=float, default=0.25)
    parser.add_argument("--task_gain", type=float, default=-1.0, help="If >=0, use this direct multiplier for task ROI bias before clipping.")
    parser.add_argument("--max_delta", type=float, default=0.06)
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
    ok, encoded = cv2.imencode(".png", np.clip(image * 255.0, 0, 255).round().astype(np.uint8))
    if not ok:
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


def _roi_key_from_column(column: str) -> str | None:
    for prefix in ["roi_mean_", "T1_ONLY__roi_mean_", "FA_GT__roi_mean_"]:
        if column.startswith(prefix):
            tail = column[len(prefix) :]
            parts = tail.split("_")
            if len(parts) >= 2 and parts[0].startswith("r") and parts[1].startswith("c"):
                return f"{parts[0]}_{parts[1]}"
    if "__roi_mean_" in column:
        tail = column.split("__roi_mean_", 1)[1]
        parts = tail.split("_")
        if len(parts) >= 2 and parts[0].startswith("r") and parts[1].startswith("c"):
            return f"{parts[0]}_{parts[1]}"
    return None


def compute_task_roi_directions(features: pd.DataFrame, tasks: list[str]) -> dict[str, dict[str, float]]:
    roi_columns = [
        column
        for column in feature_columns(features, feature_set="roi_mean")
        if column.endswith("_mean") and _roi_key_from_column(column) is not None
    ]
    directions: dict[str, dict[str, float]] = {}
    for task in tasks:
        definition = TASK_DEFINITIONS[task]
        subset = features.loc[features["group_name"].isin(definition["include"])].copy()
        labels = subset["group_name"].map(definition["labels"]).astype(int)
        task_direction: dict[str, float] = {}
        for column in roi_columns:
            roi_key = _roi_key_from_column(column)
            if roi_key is None:
                continue
            class0 = subset.loc[labels.eq(0), column].to_numpy(dtype=np.float32)
            class1 = subset.loc[labels.eq(1), column].to_numpy(dtype=np.float32)
            if class0.size == 0 or class1.size == 0:
                continue
            task_direction[roi_key] = float(np.mean(class1) - np.mean(class0))
        directions[task] = task_direction
    return directions


def fit_task_score_models(t1_features: pd.DataFrame, tasks: list[str]) -> dict[str, TaskScoreModel]:
    columns = feature_columns(t1_features, feature_set="roi_mean")
    models: dict[str, TaskScoreModel] = {}
    for task in tasks:
        definition = TASK_DEFINITIONS[task]
        subset = t1_features.loc[t1_features["group_name"].isin(definition["include"])].copy()
        y = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
        if len(set(y.tolist())) != 2:
            continue
        x = subset[columns].to_numpy(dtype=np.float32)
        estimator = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced"))
        estimator.fit(x, y)
        train_scores = np.asarray(estimator.decision_function(x), dtype=np.float32).reshape(-1)
        score_scale = float(np.std(train_scores)) if np.std(train_scores) > 1e-6 else 1.0
        models[task] = TaskScoreModel(columns=columns, estimator=estimator, score_scale=score_scale)
    return models


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
    for t1_path in tqdm(t1_files, desc="Fit ROI base calibration"):
        fa_path = train_fa_dir / t1_path.name
        if not fa_path.exists():
            continue
        t1 = _read_png01(t1_path)
        fa = _read_png01(fa_path)
        for row_idx, col_idx, y_slice, x_slice in roi_defs:
            key = f"r{row_idx}_c{col_idx}"
            patch = t1[y_slice, x_slice]
            mask = patch > brain_threshold
            std = float(np.std(patch[mask])) if np.any(mask) else float(np.std(patch))
            features_by_roi[key].append([_roi_mean(t1, y_slice, x_slice, brain_threshold), std])
            targets_by_roi[key].append(_roi_mean(fa, y_slice, x_slice, brain_threshold))
    models = {}
    for key, x_rows in features_by_roi.items():
        model = Ridge(alpha=ridge_alpha)
        model.fit(np.asarray(x_rows, dtype=np.float32), np.asarray(targets_by_roi[key], dtype=np.float32))
        models[key] = model
    return models


def _predict_roi_base(model: Ridge, t1: np.ndarray, y_slice: slice, x_slice: slice, brain_threshold: float) -> float:
    patch = t1[y_slice, x_slice]
    mask = patch > brain_threshold
    std = float(np.std(patch[mask])) if np.any(mask) else float(np.std(patch))
    x = np.asarray([[_roi_mean(t1, y_slice, x_slice, brain_threshold), std]], dtype=np.float32)
    return float(model.predict(x)[0])


def calibrate_image_task_sensitive(
    pred: np.ndarray,
    t1: np.ndarray,
    roi_models: dict[str, Ridge],
    task_scores: dict[str, float],
    task_directions: dict[str, dict[str, float]],
    task_strength: float,
    roi_rows: int,
    roi_cols: int,
    brain_threshold: float,
    gain: float,
    max_delta: float,
    smooth_sigma: float,
    base_residual_weight: float = 1.0,
    task_gain: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if t1.shape != pred.shape:
        t1 = cv2.resize(t1, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
    delta_map = np.zeros_like(pred, dtype=np.float32)
    stats: dict[str, float] = {}
    for row_idx, col_idx, y_slice, x_slice in _roi_slices(pred.shape, roi_rows, roi_cols):
        roi_key = f"r{row_idx}_c{col_idx}"
        base_target = _predict_roi_base(roi_models[roi_key], t1, y_slice, x_slice, brain_threshold)
        task_bias = 0.0
        for task, score in task_scores.items():
            task_bias += float(score) * float(task_directions.get(task, {}).get(roi_key, 0.0))
        task_bias *= task_strength
        current = _roi_mean(pred, y_slice, x_slice, brain_threshold)
        base_delta = base_target - current
        if task_gain is None:
            raw_delta = base_residual_weight * (base_delta + task_bias)
            delta = float(np.clip(raw_delta, -max_delta, max_delta) * gain)
        else:
            raw_delta = base_residual_weight * gain * base_delta + task_gain * task_bias
            delta = float(np.clip(raw_delta, -max_delta, max_delta))
        delta_map[y_slice, x_slice] = delta
        stats[f"{roi_key}_base_target"] = base_target
        stats[f"{roi_key}_task_bias"] = task_bias
        stats[f"{roi_key}_current"] = current
        stats[f"{roi_key}_base_delta"] = base_delta
        stats[f"{roi_key}_delta"] = delta
    if smooth_sigma > 0:
        delta_map = cv2.GaussianBlur(delta_map, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    brain_mask = (pred > brain_threshold).astype(np.float32)
    calibrated = np.clip(pred + delta_map * brain_mask, 0.0, 1.0)
    stats["mean_abs_delta"] = float(np.mean(np.abs(delta_map[brain_mask > 0.5]))) if np.any(brain_mask > 0.5) else 0.0
    return calibrated, stats


def _read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_private_subject_index(config_path: str | Path) -> pd.DataFrame:
    config = _read_yaml(config_path)
    data_config = config["data"]
    return load_subject_index(
        excel_path=data_config["excel_path"],
        split_json=data_config["split_json"],
        sheet_name=data_config.get("excel_sheet", "re_order"),
    )


def main() -> None:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_index = _load_private_subject_index(args.config)

    train_t1_features = extract_subject_features_from_folder(args.train_t1_dir, "T1_ONLY", subject_index, split="train")
    train_fa_features = extract_subject_features_from_folder(args.train_fa_dir, "FA_GT", subject_index, split="train")
    test_t1_features = extract_subject_features_from_folder(args.t1_dir, "T1_ONLY", subject_index, split="test")
    score_models = fit_task_score_models(train_t1_features, tasks=tasks)
    task_directions = compute_task_roi_directions(train_fa_features, tasks=tasks)
    roi_models = _fit_roi_models(
        train_t1_dir=Path(args.train_t1_dir),
        train_fa_dir=Path(args.train_fa_dir),
        roi_rows=args.roi_rows,
        roi_cols=args.roi_cols,
        brain_threshold=args.brain_threshold,
        ridge_alpha=args.ridge_alpha,
    )
    test_rows = {
        normalize_subject_id(row["subject_id"]): row
        for row in test_t1_features.to_dict(orient="records")
    }
    subject_scores = {
        subject_id: {task: model.score_subject(row) for task, model in score_models.items()}
        for subject_id, row in test_rows.items()
    }

    input_files = sorted(Path(args.input_dir).glob("*.png"))
    if args.limit > 0:
        input_files = input_files[: args.limit]
    rows = []
    for pred_path in tqdm(input_files, desc="Apply task-sensitive ROI calibration"):
        t1_path = Path(args.t1_dir) / pred_path.name
        if not t1_path.exists():
            raise FileNotFoundError(f"Missing T1 file for {pred_path.name}: {t1_path}")
        subject_id, _ = parse_subject_and_slice(pred_path.name)
        pred = _read_png01(pred_path)
        t1 = _read_png01(t1_path)
        calibrated, stats = calibrate_image_task_sensitive(
            pred=pred,
            t1=t1,
            roi_models=roi_models,
            task_scores=subject_scores.get(subject_id, {}),
            task_directions=task_directions,
            task_strength=args.task_strength,
            roi_rows=args.roi_rows,
            roi_cols=args.roi_cols,
            brain_threshold=args.brain_threshold,
            gain=args.base_gain,
            max_delta=args.max_delta,
            smooth_sigma=args.smooth_sigma,
            base_residual_weight=args.base_residual_weight,
            task_gain=args.task_gain if args.task_gain >= 0 else None,
        )
        out_path = output_dir / pred_path.name
        _write_png01(out_path, calibrated)
        rows.append(
            {
                "fname": pred_path.name,
                "subject_id": subject_id,
                "input_path": str(pred_path),
                "t1_path": str(t1_path),
                "output_path": str(out_path),
                **{f"score_{task}": subject_scores.get(subject_id, {}).get(task, 0.0) for task in tasks},
                **stats,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "export_manifest.csv", index=False, encoding="utf-8")
    payload = {
        "tasks": tasks,
        "task_strength": args.task_strength,
        "base_gain": args.base_gain,
        "base_residual_weight": args.base_residual_weight,
        "task_gain": args.task_gain,
        "max_delta": args.max_delta,
        "smooth_sigma": args.smooth_sigma,
        "task_directions": task_directions,
    }
    with open(output_dir / "task_sensitive_roi_calibration.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"TASK_SENSITIVE_ROI_CALIBRATION: exported={len(rows)} output_dir={output_dir}")
    print(f"Saved manifest to: {output_dir / 'export_manifest.csv'}")


if __name__ == "__main__":
    main()
