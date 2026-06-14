from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import load_subject_index, normalize_subject_id
from src.eval.downstream_utility import (
    TASK_DEFINITIONS,
    extract_subject_features_from_folder,
    parse_subject_and_slice,
)


@dataclass(frozen=True)
class MethodPair:
    name: str
    train_dir: str
    test_dir: str


def parse_method_pair(spec: str) -> MethodPair:
    if "=" not in spec:
        raise ValueError(f"Method spec must use NAME=TRAIN_DIR:TEST_DIR format, got {spec!r}")
    name, raw_pair = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing method name in {spec!r}")

    if "|" in raw_pair:
        train_dir, test_dir = raw_pair.split("|", 1)
    elif ";" in raw_pair:
        train_dir, test_dir = raw_pair.split(";", 1)
    else:
        # Relative paths are common in this project. For absolute Windows paths,
        # prefer NAME=TRAIN_DIR|TEST_DIR to avoid ambiguity with drive letters.
        train_dir, test_dir = raw_pair.split(":", 1)
    train_dir = train_dir.strip()
    test_dir = test_dir.strip()
    if not train_dir or not test_dir:
        raise ValueError(f"Invalid method pair {spec!r}")
    return MethodPair(name=name, train_dir=train_dir, test_dir=test_dir)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _load_subject_index(config: dict[str, Any], subject_index_csv: str = "") -> pd.DataFrame:
    if subject_index_csv:
        return pd.read_csv(subject_index_csv)
    data_config = config["data"]
    return load_subject_index(
        excel_path=data_config["excel_path"],
        split_json=data_config["split_json"],
        sheet_name=data_config.get("excel_sheet", "re_order"),
    )


def _default_split_dir(config: dict[str, Any], split: str, modality: str) -> str:
    return str(Path(config["data"]["processed_root"]) / split / f"{modality}_slices")


def _read_grayscale(path: Path) -> np.ndarray:
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def _read_label_mask(path: Path) -> np.ndarray:
    stream = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read atlas mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def _atlas_path_for_image(atlas_dir: Path, image_path: Path) -> Path:
    subject_id, slice_idx = parse_subject_and_slice(image_path.name)
    candidates = [
        atlas_dir / image_path.name,
        atlas_dir / f"atlas_z{slice_idx:03d}.png",
        atlas_dir / f"mask_z{slice_idx:03d}.png",
        atlas_dir / f"{subject_id}_z{slice_idx:03d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No atlas mask found for {image_path.name}. Tried: {', '.join(str(item) for item in candidates)}"
    )


def extract_subject_features_with_atlas(
    image_dir: str | Path,
    atlas_dir: str | Path,
    method: str,
    subject_index: pd.DataFrame,
    split: str,
    min_pixels: int = 8,
) -> pd.DataFrame:
    image_dir = Path(image_dir)
    atlas_dir = Path(atlas_dir)
    split_subjects = subject_index.loc[subject_index["split"].eq(split)].copy()
    metadata = {
        normalize_subject_id(row["subject_id"]): row
        for row in split_subjects.to_dict(orient="records")
    }
    per_subject: dict[str, list[Path]] = {subject_id: [] for subject_id in metadata}
    for path in sorted(image_dir.glob("*.png")):
        subject_id, _ = parse_subject_and_slice(path.name)
        if subject_id in per_subject:
            per_subject[subject_id].append(path)

    rows: list[dict[str, Any]] = []
    for subject_id, paths in sorted(per_subject.items()):
        if not paths:
            continue
        roi_values: dict[int, list[float]] = {}
        for image_path in paths:
            image = _read_grayscale(image_path)
            mask = _read_label_mask(_atlas_path_for_image(atlas_dir, image_path))
            if mask.shape != image.shape:
                mask = cv2.resize(mask.astype(np.float32), image.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.int32)
            for label in sorted(int(item) for item in np.unique(mask) if int(item) > 0):
                label_mask = mask == label
                if int(label_mask.sum()) < min_pixels:
                    continue
                roi_values.setdefault(label, []).append(float(np.mean(image[label_mask])))
        meta = metadata[subject_id]
        row: dict[str, Any] = {
            "method": method,
            "subject_id": subject_id,
            "group_id": int(meta["group_id"]),
            "group_name": str(meta["group_name"]),
            "split": str(meta["split"]),
            "n_slices": int(len(paths)),
        }
        for optional in ["gender", "age", "edu", "MMSE"]:
            if optional in meta:
                row[optional] = meta[optional]
        for label, values in roi_values.items():
            arr = np.asarray(values, dtype=np.float32)
            row[f"roi_label_{label}_mean"] = float(np.mean(arr))
            row[f"roi_label_{label}_std"] = float(np.std(arr))
        rows.append(row)
    if not rows:
        raise ValueError(f"No atlas ROI features extracted from {image_dir} for split {split!r}")
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def extract_slice_features_with_atlas(
    image_dir: str | Path,
    atlas_dir: str | Path,
    method: str,
    subject_index: pd.DataFrame,
    split: str,
    min_pixels: int = 8,
) -> pd.DataFrame:
    image_dir = Path(image_dir)
    atlas_dir = Path(atlas_dir)
    split_subjects = subject_index.loc[subject_index["split"].eq(split)].copy()
    metadata = {
        normalize_subject_id(row["subject_id"]): row
        for row in split_subjects.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.glob("*.png")):
        subject_id, slice_idx = parse_subject_and_slice(image_path.name)
        if subject_id not in metadata:
            continue
        image = _read_grayscale(image_path)
        mask = _read_label_mask(_atlas_path_for_image(atlas_dir, image_path))
        if mask.shape != image.shape:
            mask = cv2.resize(mask.astype(np.float32), image.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.int32)

        meta = metadata[subject_id]
        row: dict[str, Any] = {
            "method": method,
            "sample_id": f"{subject_id}_z{slice_idx:03d}",
            "subject_id": subject_id,
            "slice_idx": int(slice_idx),
            "group_id": int(meta["group_id"]),
            "group_name": str(meta["group_name"]),
            "split": str(meta["split"]),
        }
        for optional in ["gender", "age", "edu", "MMSE"]:
            if optional in meta:
                row[optional] = meta[optional]
        for label in sorted(int(item) for item in np.unique(mask) if int(item) > 0):
            label_mask = mask == label
            if int(label_mask.sum()) < min_pixels:
                continue
            values = image[label_mask]
            row[f"roi_label_{label}_mean"] = float(np.mean(values))
            row[f"roi_label_{label}_std"] = float(np.std(values))
        rows.append(row)
    if not rows:
        raise ValueError(f"No slice atlas ROI features extracted from {image_dir} for split {split!r}")
    return pd.DataFrame(rows).sort_values(["subject_id", "slice_idx"]).reset_index(drop=True)


def _prepare_task(features: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task {task!r}; available tasks: {sorted(TASK_DEFINITIONS)}")
    definition = TASK_DEFINITIONS[task]
    subset = features.loc[features["group_name"].isin(definition["include"])].copy()
    y = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        raise ValueError(f"Task {task!r} requires at least two classes")
    return subset.reset_index(drop=True), y, classes


def _build_estimator(classifier: str, n_features: int, max_features: int):
    selected = n_features if max_features <= 0 else min(n_features, int(max_features))
    steps = []
    if selected < n_features:
        steps.append(("select", SelectKBest(score_func=f_classif, k=selected)))
    steps.append(("scale", StandardScaler()))
    if classifier == "linear_svm":
        clf = SVC(kernel="linear", class_weight="balanced")
    elif classifier == "rbf_svm":
        clf = SVC(kernel="rbf", class_weight="balanced", gamma="scale")
    else:
        raise ValueError("classifier must be one of: linear_svm, rbf_svm")
    steps.append(("clf", clf))
    return make_pipeline(*[step for _, step in steps])


def _roi_feature_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "full":
        columns = [
            column
            for column in features.columns
            if column not in {"method", "subject_id", "group_id", "group_name", "split", "n_slices", "gender", "age", "edu", "MMSE"}
            and pd.api.types.is_numeric_dtype(features[column])
            and not features[column].isna().all()
        ]
    elif feature_set == "roi_mean":
        columns = [
            column
            for column in features.columns
            if (
                column.startswith("roi_mean_")
                or "__roi_mean_" in column
                or (column.startswith("roi_label_") and column.endswith("_mean"))
                or "__roi_label_" in column
            )
            and pd.api.types.is_numeric_dtype(features[column])
            and not features[column].isna().all()
        ]
    else:
        raise ValueError("feature_set must be one of: roi_mean, full")
    if not columns:
        raise ValueError(f"No {feature_set} feature columns found")
    return columns


def _decision_scores(estimator, x: np.ndarray, classes: list[int]) -> np.ndarray:
    raw = estimator.decision_function(x)
    if len(classes) == 2:
        return np.asarray(raw, dtype=np.float32).reshape(-1)
    return np.asarray(raw, dtype=np.float32)


def evaluate_train_test_classifier(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    method: str,
    task: str,
    classifier: str = "linear_svm",
    feature_set: str = "roi_mean",
    max_features: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_subset, y_train, classes = _prepare_task(train_features, task)
    test_subset, y_test, test_classes = _prepare_task(test_features, task)
    if classes != test_classes:
        raise ValueError(f"Train/test classes differ for {task}: train={classes}, test={test_classes}")

    train_columns = _roi_feature_columns(train_subset, feature_set=feature_set)
    test_columns = _roi_feature_columns(test_subset, feature_set=feature_set)
    columns = [column for column in train_columns if column in set(test_columns)]
    if not columns:
        raise ValueError(f"No shared numeric {feature_set} features between train and test for {method}")
    x_train = train_subset[columns].fillna(0.0).to_numpy(dtype=np.float32)
    x_test = test_subset[columns].fillna(0.0).to_numpy(dtype=np.float32)
    estimator = _build_estimator(classifier, n_features=len(columns), max_features=max_features)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    scores = _decision_scores(estimator, x_test, classes)

    try:
        if len(classes) == 2:
            auc = float(roc_auc_score(y_test, scores))
        else:
            auc = float(roc_auc_score(y_test, scores, labels=classes, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    selected = len(columns) if max_features <= 0 else min(len(columns), int(max_features))
    result: dict[str, Any] = {
        "method": method,
        "task": task,
        "classifier": classifier,
        "feature_set": feature_set,
        "n_train_subjects": int(train_subset["subject_id"].nunique()) if "subject_id" in train_subset else int(len(y_train)),
        "n_test_subjects": int(test_subset["subject_id"].nunique()) if "subject_id" in test_subset else int(len(y_test)),
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "n_features": int(len(columns)),
        "n_selected_features": int(selected),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": auc,
    }
    prediction_columns = ["method", "sample_id", "subject_id", "slice_idx", "group_id", "group_name", "split"]
    predictions = pd.DataFrame(index=test_subset.index)
    for column in prediction_columns:
        if column in test_subset.columns:
            predictions[column] = test_subset[column]
        elif column == "method":
            predictions[column] = method
        elif column == "split":
            predictions[column] = "test"
        else:
            predictions[column] = ""
    predictions["task"] = task
    predictions["y_true"] = y_test
    predictions["y_pred"] = y_pred
    if len(classes) == 2:
        predictions["decision_score"] = scores
    return result, predictions


def _parse_fusion(spec: str) -> tuple[str, str, str]:
    if "=" not in spec or "+" not in spec:
        raise ValueError(f"Fusion spec must use NAME=LEFT+RIGHT, got {spec!r}")
    name, pair = spec.split("=", 1)
    left, right = pair.split("+", 1)
    return name.strip(), left.strip(), right.strip()


def build_fused_feature_table_local(
    features: pd.DataFrame,
    fused_name: str,
    left_method: str,
    right_method: str,
    feature_set: str = "full",
) -> pd.DataFrame:
    left = features.loc[features["method"] == left_method].copy()
    right = features.loc[features["method"] == right_method].copy()
    if left.empty:
        raise ValueError(f"Missing left method for fusion: {left_method}")
    if right.empty:
        raise ValueError(f"Missing right method for fusion: {right_method}")

    join_columns = [
        column
        for column in ["subject_id", "group_id", "group_name", "split", "gender", "age", "edu", "MMSE", "n_slices"]
        if column in left.columns and column in right.columns
    ]
    left_features = _roi_feature_columns(left, feature_set=feature_set)
    right_features = _roi_feature_columns(right, feature_set=feature_set)

    left_prefixed = left[join_columns + left_features].rename(
        columns={column: f"{left_method}__{column}" for column in left_features}
    )
    right_prefixed = right[["subject_id"] + right_features].rename(
        columns={column: f"{right_method}__{column}" for column in right_features}
    )
    fused = left_prefixed.merge(right_prefixed, on="subject_id", how="inner", validate="one_to_one")
    if fused.empty:
        raise ValueError(f"No overlapping subjects for fusion {left_method}+{right_method}")
    fused.insert(0, "method", fused_name)
    return fused.sort_values("subject_id").reset_index(drop=True)


def build_fused_slice_feature_table_local(
    features: pd.DataFrame,
    fused_name: str,
    left_method: str,
    right_method: str,
    feature_set: str = "full",
) -> pd.DataFrame:
    left = features.loc[features["method"] == left_method].copy()
    right = features.loc[features["method"] == right_method].copy()
    if left.empty:
        raise ValueError(f"Missing left method for fusion: {left_method}")
    if right.empty:
        raise ValueError(f"Missing right method for fusion: {right_method}")
    for column in ["sample_id", "subject_id", "slice_idx"]:
        if column not in left.columns or column not in right.columns:
            raise ValueError(f"Slice-level fusion requires {column!r} in both feature tables")

    join_columns = [
        column
        for column in ["sample_id", "subject_id", "slice_idx", "group_id", "group_name", "split", "gender", "age", "edu", "MMSE"]
        if column in left.columns and column in right.columns
    ]
    left_features = _roi_feature_columns(left, feature_set=feature_set)
    right_features = _roi_feature_columns(right, feature_set=feature_set)

    left_prefixed = left[join_columns + left_features].rename(
        columns={column: f"{left_method}__{column}" for column in left_features}
    )
    right_prefixed = right[["sample_id"] + right_features].rename(
        columns={column: f"{right_method}__{column}" for column in right_features}
    )
    fused = left_prefixed.merge(right_prefixed, on="sample_id", how="inner", validate="one_to_one")
    if fused.empty:
        raise ValueError(f"No overlapping slices for fusion {left_method}+{right_method}")
    fused.insert(0, "method", fused_name)
    return fused.sort_values(["subject_id", "slice_idx"]).reset_index(drop=True)


def _extract_features(
    image_dir: str,
    atlas_dir: str,
    roi_mode: str,
    method: str,
    subject_index: pd.DataFrame,
    split: str,
    brain_threshold: float,
    wm_quantile: float,
    sample_level: str = "subject",
) -> pd.DataFrame:
    if roi_mode == "atlas":
        if sample_level == "slice":
            return extract_slice_features_with_atlas(
                image_dir=image_dir,
                atlas_dir=atlas_dir,
                method=method,
                subject_index=subject_index,
                split=split,
            )
        return extract_subject_features_with_atlas(
            image_dir=image_dir,
            atlas_dir=atlas_dir,
            method=method,
            subject_index=subject_index,
            split=split,
        )
    if roi_mode == "grid":
        if sample_level == "slice":
            raise ValueError("--sample_level slice currently requires --roi_mode atlas")
        return extract_subject_features_from_folder(
            image_dir=image_dir,
            method=method,
            subject_index=subject_index,
            split=split,
            brain_threshold=brain_threshold,
            wm_quantile=wm_quantile,
        )
    raise ValueError("roi_mode must be one of: grid, atlas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="final.pdf-style downstream evaluation: train-split SVM -> test-split ACC/AUC on ROI features."
    )
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--subject_index_csv", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--method", action="append", default=[], help="NAME=TRAIN_DIR:TEST_DIR. Windows abs paths can use NAME=TRAIN_DIR|TEST_DIR.")
    parser.add_argument("--fusion", action="append", default=[], help="NAME=LEFT+RIGHT using available method names.")
    parser.add_argument("--include_t1", action="store_true")
    parser.add_argument("--include_fa_gt", action="store_true")
    parser.add_argument("--tasks", default="cn_vs_ad,cn_vs_mci,mci_vs_ad")
    parser.add_argument("--classifier", choices=["linear_svm", "rbf_svm"], default="linear_svm")
    parser.add_argument("--feature_set", choices=["roi_mean", "full"], default="roi_mean")
    parser.add_argument("--max_features", type=int, default=0)
    parser.add_argument("--roi_mode", choices=["grid", "atlas"], default="grid")
    parser.add_argument("--sample_level", choices=["subject", "slice"], default="subject")
    parser.add_argument("--atlas_dir", default="", help="Directory of atlas label PNG masks. Required for --roi_mode atlas.")
    parser.add_argument("--brain_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--output_root", default="outputs/icdm2026/downstream_finalpdf_train_test_roi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.roi_mode == "atlas" and not args.atlas_dir:
        raise ValueError("--atlas_dir is required when --roi_mode atlas")
    config = _read_yaml(args.config)
    subject_index = _load_subject_index(config, args.subject_index_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs: list[MethodPair] = []
    if args.include_t1:
        pairs.append(
            MethodPair(
                "T1_ONLY",
                _default_split_dir(config, args.train_split, "t1"),
                _default_split_dir(config, args.test_split, "t1"),
            )
        )
    if args.include_fa_gt:
        pairs.append(
            MethodPair(
                "FA_GT",
                _default_split_dir(config, args.train_split, "fa"),
                _default_split_dir(config, args.test_split, "fa"),
            )
        )
    pairs.extend(parse_method_pair(spec) for spec in args.method)
    if not pairs:
        raise ValueError("No methods selected. Use --include_t1, --include_fa_gt, or --method.")

    train_tables: dict[str, pd.DataFrame] = {}
    test_tables: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        train_tables[pair.name] = _extract_features(
            pair.train_dir,
            args.atlas_dir,
            args.roi_mode,
            pair.name,
            subject_index,
            args.train_split,
            args.brain_threshold,
            args.wm_quantile,
            args.sample_level,
        )
        test_tables[pair.name] = _extract_features(
            pair.test_dir,
            args.atlas_dir,
            args.roi_mode,
            pair.name,
            subject_index,
            args.test_split,
            args.brain_threshold,
            args.wm_quantile,
            args.sample_level,
        )

    if args.fusion:
        train_base = pd.concat(train_tables.values(), ignore_index=True)
        test_base = pd.concat(test_tables.values(), ignore_index=True)
        for spec in args.fusion:
            name, left, right = _parse_fusion(spec)
            if args.sample_level == "slice":
                train_tables[name] = build_fused_slice_feature_table_local(train_base, name, left, right, args.feature_set)
                test_tables[name] = build_fused_slice_feature_table_local(test_base, name, left, right, args.feature_set)
            else:
                train_tables[name] = build_fused_feature_table_local(train_base, name, left, right, args.feature_set)
                test_tables[name] = build_fused_feature_table_local(test_base, name, left, right, args.feature_set)

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    summaries: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for method in sorted(train_tables):
        for task in tasks:
            summary, pred = evaluate_train_test_classifier(
                train_tables[method],
                test_tables[method],
                method=method,
                task=task,
                classifier=args.classifier,
                feature_set=args.feature_set,
                max_features=args.max_features,
            )
            summary["roi_mode"] = args.roi_mode
            summary["sample_level"] = args.sample_level
            summaries.append(summary)
            predictions.append(pred)

    train_features = pd.concat(train_tables.values(), ignore_index=True)
    test_features = pd.concat(test_tables.values(), ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values(["task", "accuracy", "macro_auc_ovr"], ascending=[True, False, False])
    pred_df = pd.concat(predictions, ignore_index=True)
    train_features.to_csv(output_root / "train_subject_features.csv", index=False, encoding="utf-8")
    test_features.to_csv(output_root / "test_subject_features.csv", index=False, encoding="utf-8")
    summary_df.to_csv(output_root / "classification_train_test_summary.csv", index=False, encoding="utf-8")
    pred_df.to_csv(output_root / "classification_train_test_predictions.csv", index=False, encoding="utf-8")
    with open(output_root / "classification_train_test_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summaries), handle, indent=2, ensure_ascii=False)

    print(f"Saved final.pdf-style train/test ROI summary to: {output_root / 'classification_train_test_summary.csv'}")
    print(summary_df[["method", "task", "n_train_samples", "n_test_samples", "n_train_subjects", "n_test_subjects", "accuracy", "macro_auc_ovr", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
