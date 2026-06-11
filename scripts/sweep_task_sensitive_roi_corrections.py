import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import load_subject_index
from src.eval.downstream_utility import (
    TASK_DEFINITIONS,
    extract_subject_features_from_folder,
    parse_subject_and_slice,
    read_grayscale_01,
)


ROI_ROWS = 2
ROI_COLS = 3
ROI_KEYS = [f"roi_mean_r{row}_c{col}_mean" for row in range(ROI_ROWS) for col in range(ROI_COLS)]


@dataclass(frozen=True)
class CorrectionSpec:
    name: str
    mode: str
    alpha: float
    top_k: int = 0
    task: str = "cn_scd_vs_mci_ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fast task-sensitive ROI-corrected image candidates from a base prediction folder. "
            "All calibration uses train FA statistics only; test labels are not used."
        )
    )
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--base_pred_dir", default="outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL")
    parser.add_argument("--method_prefix", default="ROI_TASK")
    parser.add_argument("--output_root", default="outputs/icdm2026/predictions/task_sensitive_roi_sweep")
    parser.add_argument("--split", default="test")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--tasks", default="cn_scd_vs_mci_ad,cn_vs_mci,mci_vs_ad")
    parser.add_argument("--global_alphas", default="0.25,0.5")
    parser.add_argument("--contrast_alphas", default="0.25,0.5,0.75")
    parser.add_argument("--prototype_alphas", default="0.25,0.5")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--mask_threshold", type=float, default=0.02)
    parser.add_argument("--max_abs_shift", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_subject_index(config_path: str | Path) -> pd.DataFrame:
    config = _read_yaml(config_path)
    data_config = config["data"]
    return load_subject_index(
        excel_path=data_config["excel_path"],
        split_json=data_config["split_json"],
        sheet_name=data_config.get("excel_sheet", "re_order"),
    )


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _roi_feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [column for column in ROI_KEYS if column in frame.columns]
    if len(columns) != len(ROI_KEYS):
        raise ValueError(f"Missing ROI feature columns: {sorted(set(ROI_KEYS) - set(columns))}")
    return frame[columns].to_numpy(dtype=np.float32), columns


def _task_labels(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task: {task}")
    definition = TASK_DEFINITIONS[task]
    subset = frame.loc[frame["group_name"].isin(definition["include"])].copy()
    labels = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
    if len(set(labels.tolist())) != 2:
        raise ValueError(f"Task {task!r} is not binary after filtering.")
    return subset, labels


def _direction_from_train_fa(train_features: pd.DataFrame, task: str, top_k: int) -> np.ndarray:
    subset, labels = _task_labels(train_features, task)
    values, _ = _roi_feature_matrix(subset)
    direction = values[labels == 1].mean(axis=0) - values[labels == 0].mean(axis=0)
    if top_k > 0:
        keep = np.argsort(np.abs(direction))[-top_k:]
        masked = np.zeros_like(direction)
        masked[keep] = direction[keep]
        direction = masked
    return direction.astype(np.float32)


def _class_prototypes(train_features: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    subset, labels = _task_labels(train_features, task)
    values, _ = _roi_feature_matrix(subset)
    return values[labels == 0].mean(axis=0).astype(np.float32), values[labels == 1].mean(axis=0).astype(np.float32)


def _fit_task_classifier(train_features: pd.DataFrame, task: str) -> Any:
    subset, labels = _task_labels(train_features, task)
    values, _ = _roi_feature_matrix(subset)
    classifier = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced"))
    classifier.fit(values, labels)
    return classifier


def _subject_roi_values(test_features: pd.DataFrame) -> dict[str, np.ndarray]:
    values, _ = _roi_feature_matrix(test_features)
    return {str(row.subject_id): values[idx] for idx, row in enumerate(test_features.itertuples(index=False))}


def _compute_shifts(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    spec: CorrectionSpec,
    max_abs_shift: float,
) -> dict[str, np.ndarray]:
    test_values = _subject_roi_values(test_features)
    shifts: dict[str, np.ndarray] = {}
    if spec.mode == "global_match":
        train_values, _ = _roi_feature_matrix(train_features)
        target = train_values.mean(axis=0).astype(np.float32)
        for subject_id, current in test_values.items():
            shifts[subject_id] = spec.alpha * (target - current)
    elif spec.mode == "contrast_amplify":
        direction = _direction_from_train_fa(train_features, spec.task, spec.top_k)
        classifier = _fit_task_classifier(train_features, spec.task)
        ordered_subjects = list(test_values)
        values = np.stack([test_values[subject_id] for subject_id in ordered_subjects], axis=0)
        pseudo = classifier.predict(values).astype(np.float32)
        signs = np.where(pseudo > 0, 1.0, -1.0).astype(np.float32)
        for subject_id, sign in zip(ordered_subjects, signs):
            shifts[subject_id] = spec.alpha * sign * direction
    elif spec.mode == "prototype_pull":
        proto0, proto1 = _class_prototypes(train_features, spec.task)
        classifier = _fit_task_classifier(train_features, spec.task)
        ordered_subjects = list(test_values)
        values = np.stack([test_values[subject_id] for subject_id in ordered_subjects], axis=0)
        pseudo = classifier.predict(values).astype(np.int64)
        for subject_id, label in zip(ordered_subjects, pseudo):
            target = proto1 if label == 1 else proto0
            shifts[subject_id] = spec.alpha * (target - test_values[subject_id])
    else:
        raise ValueError(f"Unsupported correction mode: {spec.mode}")

    return {key: np.clip(value, -max_abs_shift, max_abs_shift).astype(np.float32) for key, value in shifts.items()}


def _apply_roi_shift(image: np.ndarray, shift: np.ndarray, mask_threshold: float) -> np.ndarray:
    out = image.astype(np.float32).copy()
    height, width = out.shape
    for row in range(ROI_ROWS):
        y0 = int(round(row * height / ROI_ROWS))
        y1 = int(round((row + 1) * height / ROI_ROWS))
        for col in range(ROI_COLS):
            x0 = int(round(col * width / ROI_COLS))
            x1 = int(round((col + 1) * width / ROI_COLS))
            idx = row * ROI_COLS + col
            region = out[y0:y1, x0:x1]
            mask = region > mask_threshold
            if np.any(mask):
                region[mask] = np.clip(region[mask] + float(shift[idx]), 0.0, 1.0)
            out[y0:y1, x0:x1] = region
    return out


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise ValueError(f"Failed to encode PNG: {path}")
    encoded.tofile(str(path))


def _copy_manifest(output_dir: Path, method: str, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "export_manifest.csv", index=False, encoding="utf-8")
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"method": method, "n_slices": len(rows)}, handle, indent=2)


def _export_corrected_dir(base_pred_dir: Path, output_dir: Path, method: str, shifts: dict[str, np.ndarray], args: argparse.Namespace) -> int:
    paths = sorted(path for path in base_pred_dir.glob("*.png"))
    if args.limit > 0:
        paths = paths[: args.limit]
    rows: list[dict[str, str]] = []
    for path in tqdm(paths, desc=f"Exporting {method}"):
        subject_id, _ = parse_subject_and_slice(path.name)
        shift = shifts.get(subject_id)
        if shift is None:
            continue
        corrected = _apply_roi_shift(read_grayscale_01(path), shift, args.mask_threshold)
        out_path = output_dir / path.name
        _save_png(out_path, corrected)
        rows.append({"method": method, "fname": path.name, "output_path": str(out_path)})
    _copy_manifest(output_dir, method, rows)
    return len(rows)


def build_specs(args: argparse.Namespace) -> list[CorrectionSpec]:
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    specs: list[CorrectionSpec] = []
    for alpha in _parse_float_list(args.global_alphas):
        specs.append(CorrectionSpec(name=f"{args.method_prefix}_GLOBAL_A{alpha:g}", mode="global_match", alpha=alpha))
    for task in tasks:
        short = task.upper().replace("_", "")
        for alpha in _parse_float_list(args.contrast_alphas):
            specs.append(
                CorrectionSpec(
                    name=f"{args.method_prefix}_CONTRAST_{short}_A{alpha:g}_K{args.top_k}",
                    mode="contrast_amplify",
                    alpha=alpha,
                    top_k=args.top_k,
                    task=task,
                )
            )
        for alpha in _parse_float_list(args.prototype_alphas):
            specs.append(
                CorrectionSpec(
                    name=f"{args.method_prefix}_PROTO_{short}_A{alpha:g}",
                    mode="prototype_pull",
                    alpha=alpha,
                    task=task,
                )
            )
    return specs


def main() -> None:
    args = parse_args()
    subject_index = _load_subject_index(args.config)
    train_features = extract_subject_features_from_folder(args.train_fa_dir, "FA_GT", subject_index, split="train")
    test_features = extract_subject_features_from_folder(args.base_pred_dir, "BASE", subject_index, split=args.split)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for spec in build_specs(args):
        shifts = _compute_shifts(train_features, test_features, spec, args.max_abs_shift)
        output_dir = output_root / spec.name
        count = _export_corrected_dir(Path(args.base_pred_dir), output_dir, spec.name, shifts, args)
        manifest_rows.append(
            {
                "method": spec.name,
                "mode": spec.mode,
                "task": spec.task,
                "alpha": spec.alpha,
                "top_k": spec.top_k,
                "output_dir": str(output_dir),
                "n_slices": count,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "sweep_manifest.csv", index=False, encoding="utf-8")
    print(manifest.to_string(index=False))
    print(f"Saved sweep manifest to: {output_root / 'sweep_manifest.csv'}")


if __name__ == "__main__":
    main()
