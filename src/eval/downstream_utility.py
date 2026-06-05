from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.subject_index import normalize_subject_id


METADATA_COLUMNS = {
    "method",
    "subject_id",
    "group_id",
    "group_name",
    "split",
    "n_slices",
}

TASK_DEFINITIONS = {
    "four_class": {"include": {"CN", "SCD", "MCI", "AD"}, "labels": {"CN": 0, "SCD": 1, "MCI": 2, "AD": 3}},
    "cn_vs_ad": {"include": {"CN", "AD"}, "labels": {"CN": 0, "AD": 1}},
    "cn_vs_mci_ad": {"include": {"CN", "MCI", "AD"}, "labels": {"CN": 0, "MCI": 1, "AD": 1}},
    "cn_scd_vs_mci_ad": {"include": {"CN", "SCD", "MCI", "AD"}, "labels": {"CN": 0, "SCD": 0, "MCI": 1, "AD": 1}},
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    image_dir: Path


def parse_method_specs(specs: Iterable[str]) -> list[MethodSpec]:
    parsed: list[MethodSpec] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Method spec must use NAME=DIR format, got {spec!r}")
        name, raw_dir = spec.split("=", 1)
        name = name.strip()
        image_dir = Path(raw_dir.strip())
        if not name:
            raise ValueError(f"Missing method name in spec {spec!r}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Method directory does not exist: {image_dir}")
        parsed.append(MethodSpec(name=name, image_dir=image_dir))
    if not parsed:
        raise ValueError("At least one method spec is required")
    return parsed


def read_grayscale_01(path: str | Path) -> np.ndarray:
    path = Path(path)
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def parse_subject_and_slice(filename: str) -> tuple[str, int]:
    match = re.match(r"(.+?)_z(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Cannot parse subject and slice id from {filename!r}")
    return normalize_subject_id(match.group(1)), int(match.group(2))


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {key: 0.0 for key in ["mean", "std", "p10", "p25", "p50", "p75", "p90", "skew", "kurtosis"]}
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    skew = float(np.mean(centered**3) / ((std**3) + 1e-8))
    kurtosis = float(np.mean(centered**4) / ((std**4) + 1e-8))
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "mean": mean,
        "std": std,
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "skew": skew,
        "kurtosis": kurtosis,
    }


def extract_slice_features(image_01: np.ndarray, brain_threshold: float = 0.02, wm_quantile: float = 0.65) -> dict[str, float]:
    image = np.asarray(image_01, dtype=np.float32)
    brain_mask = image > brain_threshold
    brain_values = image[brain_mask]
    if brain_values.size == 0:
        brain_mask = image >= 0.0
        brain_values = image.reshape(-1)

    wm_threshold = max(0.20, float(np.quantile(brain_values, wm_quantile)))
    wm_mask = brain_mask & (image >= wm_threshold)
    wm_values = image[wm_mask]

    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)
    highpass = image - blur
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)

    features: dict[str, float] = {
        "brain_area": float(np.mean(brain_mask.astype(np.float32))),
        "wm_area": float(np.mean(wm_mask.astype(np.float32))),
        "highpass_energy": float(np.mean(np.abs(highpass[brain_mask]))),
        "laplacian_energy": float(np.var(laplacian[brain_mask])),
        "gradient_energy": float(np.mean(grad_mag[brain_mask])),
        "threshold_frac_025": float(np.mean((brain_values >= 0.25).astype(np.float32))),
        "threshold_frac_050": float(np.mean((brain_values >= 0.50).astype(np.float32))),
        "threshold_frac_075": float(np.mean((brain_values >= 0.75).astype(np.float32))),
    }
    for prefix, values in [("intensity", brain_values), ("wm", wm_values)]:
        for key, value in _safe_stats(values).items():
            features[f"{prefix}_{key}"] = value
    return features


def _aggregate_slice_features(slice_features: list[dict[str, float]]) -> dict[str, float]:
    if not slice_features:
        raise ValueError("Cannot aggregate an empty slice feature list")
    keys = sorted(slice_features[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.asarray([features[key] for features in slice_features], dtype=np.float32)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    return aggregated


def extract_subject_features_from_folder(
    image_dir: str | Path,
    method: str,
    subject_index: pd.DataFrame,
    split: str = "test",
    brain_threshold: float = 0.02,
    wm_quantile: float = 0.65,
) -> pd.DataFrame:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    split_subjects = subject_index.loc[subject_index["split"] == split].copy()
    if split_subjects.empty:
        raise ValueError(f"No subjects found for split {split!r}")

    metadata = {
        normalize_subject_id(row["subject_id"]): row
        for row in split_subjects.to_dict(orient="records")
    }
    by_subject: dict[str, list[Path]] = {subject_id: [] for subject_id in metadata}
    for path in sorted(image_dir.glob("*.png")):
        subject_id, _ = parse_subject_and_slice(path.name)
        if subject_id in by_subject:
            by_subject[subject_id].append(path)

    rows: list[dict[str, float | int | str]] = []
    for subject_id in sorted(by_subject):
        paths = sorted(by_subject[subject_id], key=lambda p: parse_subject_and_slice(p.name)[1])
        if not paths:
            continue
        slice_features = [
            extract_slice_features(read_grayscale_01(path), brain_threshold=brain_threshold, wm_quantile=wm_quantile)
            for path in paths
        ]
        row: dict[str, float | int | str] = {
            "method": method,
            "subject_id": subject_id,
            "group_id": int(metadata[subject_id]["group_id"]),
            "group_name": str(metadata[subject_id]["group_name"]),
            "split": str(metadata[subject_id]["split"]),
            "n_slices": int(len(paths)),
        }
        row.update(_aggregate_slice_features(slice_features))
        rows.append(row)

    if not rows:
        raise ValueError(f"No matching PNG files found in {image_dir} for split {split!r}")
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def _prepare_task(features: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task {task!r}. Available: {sorted(TASK_DEFINITIONS)}")
    definition = TASK_DEFINITIONS[task]
    subset = features.loc[features["group_name"].isin(definition["include"])].copy()
    labels = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
    if subset.empty:
        raise ValueError(f"No subjects available for task {task!r}")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError(f"Task {task!r} requires at least two classes")
    return subset.reset_index(drop=True), labels, classes


def feature_columns(features: pd.DataFrame) -> list[str]:
    columns = []
    for column in features.columns:
        if column in METADATA_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(features[column]):
            columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns found")
    return columns


def run_classification_cv(
    features: pd.DataFrame,
    method: str,
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    subset, y, classes = _prepare_task(features, task)
    columns = feature_columns(subset)
    x = subset[columns].to_numpy(dtype=np.float32)
    min_class_count = int(pd.Series(y).value_counts().min())
    actual_splits = min(max(2, n_splits), min_class_count)
    if actual_splits < 2:
        raise ValueError(f"Task {task!r} has too few subjects per class for cross-validation")

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
    )
    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    y_pred = np.zeros_like(y)
    y_prob = np.zeros((len(y), len(classes)), dtype=np.float32)
    class_to_col = {label: idx for idx, label in enumerate(classes)}

    for train_idx, test_idx in cv.split(x, y):
        classifier.fit(x[train_idx], y[train_idx])
        fold_pred = classifier.predict(x[test_idx])
        fold_prob = classifier.predict_proba(x[test_idx])
        y_pred[test_idx] = fold_pred
        for local_col, class_label in enumerate(classifier.classes_):
            y_prob[test_idx, class_to_col[int(class_label)]] = fold_prob[:, local_col]

    macro_auc = float("nan")
    try:
        if len(classes) == 2:
            macro_auc = float(roc_auc_score(y, y_prob[:, class_to_col[classes[1]]]))
        else:
            macro_auc = float(roc_auc_score(y, y_prob, multi_class="ovr", average="macro", labels=classes))
    except ValueError:
        macro_auc = float("nan")

    summary: dict[str, float | int | str] = {
        "method": method,
        "task": task,
        "n_subjects": int(len(y)),
        "n_features": int(len(columns)),
        "n_splits": int(actual_splits),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": macro_auc,
    }
    metadata_columns = ["method", "subject_id", "group_id", "group_name"]
    if "split" in subset.columns:
        metadata_columns.append("split")
    predictions = subset[metadata_columns].copy()
    predictions["task"] = task
    predictions["y_true"] = y
    predictions["y_pred"] = y_pred
    for class_label in classes:
        predictions[f"prob_class_{class_label}"] = y_prob[:, class_to_col[class_label]]
    return summary, predictions


def confusion_matrix_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(set(predictions["y_true"].astype(int).tolist()) | set(predictions["y_pred"].astype(int).tolist()))
    matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=labels)
    return pd.DataFrame(matrix, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])
