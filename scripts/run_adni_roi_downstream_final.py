from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subject_index import normalize_subject_id  # noqa: E402


TASKS: dict[str, dict[str, Any]] = {
    "cn_vs_mci_spectrum": {
        "include": {"CN", "MCI_spectrum"},
        "labels": {"CN": 0, "MCI_spectrum": 1},
    },
    "cn_vs_ad": {
        "include": {"CN", "AD"},
        "labels": {"CN": 0, "AD": 1},
    },
    "mci_spectrum_vs_ad": {
        "include": {"MCI_spectrum", "AD"},
        "labels": {"MCI_spectrum": 0, "AD": 1},
    },
}


@dataclass(frozen=True)
class MethodPair:
    name: str
    train_dir: Path
    test_dir: Path
    runnable: bool
    note: str = ""


def read_gray01(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def parse_subject(filename: str) -> str:
    if "_z" not in filename:
        raise ValueError(f"Cannot parse subject from {filename!r}")
    return normalize_subject_id(filename.split("_z", 1)[0].replace("sub-", ""))


def parse_slice(filename: str) -> int:
    stem = Path(filename).stem
    if "_z" not in stem:
        raise ValueError(f"Cannot parse slice from {filename!r}")
    return int(stem.rsplit("_z", 1)[1])


def load_adni_subject_index(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    required = {"subject", "split", "normalized_group", "filename"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ADNI manifest missing columns: {sorted(missing)}")
    subjects = frame[["subject", "split", "normalized_group"]].drop_duplicates("subject").copy()
    subjects["subject_id"] = subjects["subject"].map(normalize_subject_id)
    subjects["group_name"] = subjects["normalized_group"].astype(str)
    group_ids = {"CN": 1, "MCI_spectrum": 2, "AD": 3, "UNLABELED": 0, "EXCLUDE": 99}
    subjects["group_id"] = subjects["group_name"].map(group_ids).fillna(99).astype(int)
    return subjects[["subject_id", "group_id", "group_name", "split"]].sort_values("subject_id")


def manifest_split_counts(manifest_path: Path, image_dir: Path) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path)
    name_to_split = dict(zip(manifest["filename"], manifest["split"]))
    counts: dict[str, int] = {}
    subjects: set[str] = set()
    unknown = 0
    for path in image_dir.glob("*.png"):
        split = name_to_split.get(path.name)
        if split is None:
            unknown += 1
            continue
        counts[split] = counts.get(split, 0) + 1
        subjects.add(parse_subject(path.name))
    return {
        "exists": image_dir.exists(),
        "png_count": int(sum(counts.values()) + unknown),
        "split_counts": counts,
        "unknown_count": int(unknown),
        "subject_count": int(len(subjects)),
    }


def safe_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "count": 0.0,
            "grad_mean": np.nan,
            "local_var": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "count": float(values.size),
    }


def roi_slice_features(
    image: np.ndarray,
    rows: int,
    cols: int,
    brain_threshold: float,
    wm_quantile: float,
    min_pixels: int,
) -> dict[str, float]:
    image = np.asarray(image, dtype=np.float32)
    brain = image > brain_threshold
    if int(brain.sum()) < min_pixels:
        brain = image >= 0.0
    brain_values = image[brain]
    wm_thr = max(0.20, float(np.quantile(brain_values, wm_quantile))) if brain_values.size else 0.20
    wm = brain & (image >= wm_thr)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    local_mean = cv2.blur(image, (5, 5))
    local_var = cv2.blur((image - local_mean) ** 2, (5, 5))

    coords = np.argwhere(brain)
    if coords.size:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
    else:
        y_min, x_min = 0, 0
        y_max, x_max = image.shape
    y_edges = np.linspace(y_min, y_max, rows + 1).round().astype(int)
    x_edges = np.linspace(x_min, x_max, cols + 1).round().astype(int)
    result: dict[str, float] = {}
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            region = image[y0:y1, x0:x1]
            region_brain = brain[y0:y1, x0:x1]
            region_wm = wm[y0:y1, x0:x1]
            region_grad = grad[y0:y1, x0:x1]
            region_var = local_var[y0:y1, x0:x1]
            for prefix, mask in (("brain", region_brain), ("wm", region_wm)):
                key = f"roi_r{r}_c{c}_{prefix}"
                stats = safe_stats(region[mask])
                if int(stats["count"]) < min_pixels:
                    stats = safe_stats(np.array([], dtype=np.float32))
                for name, value in stats.items():
                    result[f"{key}_{name}"] = value
                if int(np.nansum(mask)) >= min_pixels:
                    result[f"{key}_grad_mean"] = float(np.mean(region_grad[mask]))
                    result[f"{key}_local_var"] = float(np.mean(region_var[mask]))
                else:
                    result[f"{key}_grad_mean"] = np.nan
                    result[f"{key}_local_var"] = np.nan
    return result


def aggregate_subject_features(slice_rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(k for k in slice_rows[0] if k not in {"slice_idx"})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in slice_rows], dtype=np.float32)
        finite = values[np.isfinite(values)]
        out[f"{key}_slice_mean"] = float(np.mean(finite)) if finite.size else np.nan
        out[f"{key}_slice_std"] = float(np.std(finite)) if finite.size else np.nan
    return out


def extract_roi_features(
    image_dir: Path,
    method: str,
    subject_index: pd.DataFrame,
    split: str,
    rows: int,
    cols: int,
    brain_threshold: float,
    wm_quantile: float,
    min_pixels: int,
) -> pd.DataFrame:
    split_subjects = subject_index.loc[subject_index["split"].eq(split)].copy()
    metadata = {str(row.subject_id): row for row in split_subjects.itertuples(index=False)}
    by_subject: dict[str, list[Path]] = {sid: [] for sid in metadata}
    for path in sorted(image_dir.glob("*.png")):
        sid = parse_subject(path.name)
        if sid in by_subject:
            by_subject[sid].append(path)
    rows_out: list[dict[str, Any]] = []
    for subject_id, paths in sorted(by_subject.items()):
        if not paths:
            continue
        slice_features: list[dict[str, float]] = []
        for path in sorted(paths, key=lambda p: parse_slice(p.name)):
            feats = roi_slice_features(read_gray01(path), rows, cols, brain_threshold, wm_quantile, min_pixels)
            feats["slice_idx"] = float(parse_slice(path.name))
            slice_features.append(feats)
        if not slice_features:
            continue
        meta = metadata[subject_id]
        row: dict[str, Any] = {
            "method": method,
            "subject_id": subject_id,
            "group_id": int(meta.group_id),
            "group_name": str(meta.group_name),
            "split": str(meta.split),
            "n_slices": int(len(slice_features)),
        }
        row.update(aggregate_subject_features(slice_features))
        rows_out.append(row)
    if not rows_out:
        raise ValueError(f"No ROI features extracted from {image_dir} split={split}")
    return pd.DataFrame(rows_out).sort_values("subject_id").reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    meta = {"method", "subject_id", "group_id", "group_name", "split", "n_slices"}
    return [
        col
        for col in frame.columns
        if col not in meta and pd.api.types.is_numeric_dtype(frame[col]) and not frame[col].isna().all()
    ]


def prepare_task(train_df: pd.DataFrame, test_df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    spec = TASKS[task]
    train = train_df.loc[train_df["group_name"].isin(spec["include"])].copy().reset_index(drop=True)
    test = test_df.loc[test_df["group_name"].isin(spec["include"])].copy().reset_index(drop=True)
    y_train = train["group_name"].map(spec["labels"]).astype(int).to_numpy()
    y_test = test["group_name"].map(spec["labels"]).astype(int).to_numpy()
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise ValueError(f"Task {task} lacks at least two classes in train/test")
    return train, test, y_train, y_test


def build_classifier(name: str):
    if name == "logistic":
        estimator = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=2026)
    elif name == "linear_svm":
        estimator = SVC(kernel="linear", class_weight="balanced", probability=False, random_state=2026)
    else:
        raise ValueError(f"Unsupported classifier: {name}")
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), estimator)


def decision_scores(model: Any, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        score = model.decision_function(x_test)
        return np.asarray(score, dtype=np.float32).reshape(-1)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float32)
    raise ValueError("Classifier has neither decision_function nor predict_proba")


def evaluate_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    task: str,
    classifier: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    train, test, y_train, y_test = prepare_task(train_df, test_df, task)
    cols = [col for col in feature_columns(train) if col in set(feature_columns(test))]
    x_train = train[cols].to_numpy(dtype=np.float32)
    x_test = test[cols].to_numpy(dtype=np.float32)
    model = build_classifier(classifier)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    scores = decision_scores(model, x_test)
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auc = float("nan")
    per_class = f1_score(y_test, y_pred, average=None, labels=sorted(set(y_test) | set(y_pred)), zero_division=0)
    labels = sorted(set(y_test) | set(y_pred))
    summary = {
        "method": method,
        "task": task,
        "classifier": classifier,
        "protocol": "coarse_grid_roi_train_test",
        "roi_resource": "2x3_coarse_grid_brain_wm_mask",
        "n_train_subjects": int(train["subject_id"].nunique()),
        "n_test_subjects": int(test["subject_id"].nunique()),
        "n_features": int(len(cols)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_auc": auc,
    }
    for label, score in zip(labels, per_class):
        summary[f"class_{label}_f1"] = float(score)
    pred_df = test[["method", "subject_id", "group_id", "group_name", "split"]].copy()
    pred_df["task"] = task
    pred_df["classifier"] = classifier
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_df["decision_score"] = scores
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    top_rows: list[dict[str, Any]] = []
    final_est = model.steps[-1][1]
    coefs = getattr(final_est, "coef_", None)
    if coefs is not None:
        coef = np.ravel(coefs)
        order = np.argsort(np.abs(coef))[::-1][:20]
        for idx in order:
            top_rows.append(
                {
                    "method": method,
                    "task": task,
                    "classifier": classifier,
                    "feature": cols[int(idx)],
                    "coefficient": float(coef[int(idx)]),
                    "abs_coefficient": float(abs(coef[int(idx)])),
                }
            )
    return summary, pred_df, cm_df, top_rows


def write_table_md(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    lines = [markdown_table(display)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def report_text(
    average: pd.DataFrame,
    summary: pd.DataFrame,
    resource_note: str,
    language: str,
) -> str:
    if language == "cn":
        title = "ADNI ROI-level 下游分类补充评估"
        body = [
            f"# {title}",
            "",
            "## 资源与协议",
            "",
            resource_note,
            "",
            "- 本实验没有训练新的 T1-to-FA 生成模型。",
            "- 本实验没有改变最终主方法 A080+DS Full。",
            "- 使用 ADNI manifest 的 subject-level train/test split；同一 subject 的切片不会跨 split。",
            "- ROI 协议为项目已有 2×3 coarse grid + brain/WM mask 统计特征，不是 AAL/Neuromorphometrics 解剖 atlas。",
            "- 该结果是 ROI-level supplementary utility evidence，不替代原有 image-level MIL downstream。",
            "",
            "## 方法平均结果",
            "",
            markdown_table(average),
            "",
            "## 分任务结果",
            "",
            markdown_table(summary),
            "",
        ]
    else:
        title = "ADNI ROI-level downstream supplementary evaluation"
        body = [
            f"# {title}",
            "",
            "## Resource and protocol",
            "",
            resource_note,
            "",
            "- No new T1-to-FA image-generation model was trained.",
            "- The final selected method A080+DS Full was not changed.",
            "- The ADNI manifest subject-level train/test split was used; slices from the same subject were not split across train and test.",
            "- The ROI protocol uses the existing project-native 2x3 coarse grid with brain/WM mask statistics; it is not an AAL/Neuromorphometrics anatomical atlas.",
            "- This result is supplementary ROI-level utility evidence and does not replace the image-level MIL downstream evaluation.",
            "",
            "## Method average results",
            "",
            markdown_table(average),
            "",
            "## Per-task results",
            "",
            markdown_table(summary),
            "",
        ]
    return "\n".join(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADNI ROI-level downstream classification for final methods.")
    parser.add_argument("--manifest", default="data/adni_processed/adni_slice_manifest.csv")
    parser.add_argument("--output_dir", default="outputs/icdm2026/downstream_adni_roi_final")
    parser.add_argument("--final_selection_dir", default="outputs/icdm2026/final_selection_a080_ds_full")
    parser.add_argument("--roi_rows", type=int, default=2)
    parser.add_argument("--roi_cols", type=int, default=3)
    parser.add_argument("--brain_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--min_pixels", type=int, default=8)
    parser.add_argument("--classifiers", default="logistic,linear_svm")
    parser.add_argument("--tasks", default="cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    final_selection_dir = Path(args.final_selection_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_selection_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    subject_index = load_adni_subject_index(manifest)

    methods = [
        MethodPair("T1_ONLY", Path("data/adni_processed/train/t1_slices"), Path("data/adni_processed/test/t1_slices"), True),
        MethodPair("FA_GT", Path("data/adni_processed/train/fa_slices"), Path("data/adni_processed/test/fa_slices"), True),
        MethodPair(
            "Old Fidelity Flow",
            Path("outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL_TRAIN_FULL"),
            Path("outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
            True,
        ),
        MethodPair(
            "A080 Base",
            Path("outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL"),
            Path("outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080"),
            True,
        ),
        MethodPair(
            "A080+DS Full",
            Path("outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_TRAIN_FULL"),
            Path("outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST"),
            True,
        ),
    ]
    audit_rows = []
    for pair in methods:
        train_info = manifest_split_counts(manifest, pair.train_dir)
        test_info = manifest_split_counts(manifest, pair.test_dir)
        runnable = pair.train_dir.exists() and pair.test_dir.exists() and train_info["split_counts"].get("train", 0) > 0 and test_info["split_counts"].get("test", 0) > 0
        audit_rows.append(
            {
                "method": pair.name,
                "train_dir": str(pair.train_dir),
                "test_dir": str(pair.test_dir),
                "runnable": bool(runnable),
                "train_png": train_info["png_count"],
                "train_split_counts": json.dumps(train_info["split_counts"], ensure_ascii=False),
                "train_subjects": train_info["subject_count"],
                "test_png": test_info["png_count"],
                "test_split_counts": json.dumps(test_info["split_counts"], ensure_ascii=False),
                "test_subjects": test_info["subject_count"],
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "roi_resource_audit.csv", index=False, encoding="utf-8")
    usable = [row for row, pair in zip(audit_rows, methods) if row["runnable"]]
    runnable_methods = [pair for pair, row in zip(methods, audit_rows) if row["runnable"]]
    resource_note_cn = (
        "未发现可直接证明适配 ADNI 切片空间的 anatomical atlas mask。项目内存在 AAL3 原始文件和 private_aal3 mask，"
        "但这些不是本次 ADNI final prediction folders 的逐切片配准 mask。因此本次安全版本使用项目已有的 2×3 coarse grid "
        "brain/WM ROI 特征，不将其表述为解剖 atlas。"
    )
    resource_note_en = (
        "No anatomical atlas mask could be verified as directly aligned to the ADNI final prediction folders. "
        "The repository contains raw AAL3 resources and private_aal3 masks, but they are not verified per-slice ADNI masks. "
        "Therefore, this safe first version uses the project-native 2x3 coarse grid brain/WM ROI features and does not describe them as an anatomical atlas."
    )
    audit_cn = [
        "# ADNI ROI 资源审计",
        "",
        resource_note_cn,
        "",
        "## 目录覆盖",
        "",
        markdown_table(audit),
        "",
    ]
    audit_en = [
        "# ADNI ROI resource audit",
        "",
        resource_note_en,
        "",
        "## Folder coverage",
        "",
        markdown_table(audit),
        "",
    ]
    (output_dir / "roi_resource_audit_cn.md").write_text("\n".join(audit_cn), encoding="utf-8")
    (output_dir / "roi_resource_audit.md").write_text("\n".join(audit_en), encoding="utf-8")

    config = {
        "manifest": str(manifest),
        "roi_protocol": "2x3 coarse grid within brain bounding box, with brain and WM masks",
        "roi_rows": args.roi_rows,
        "roi_cols": args.roi_cols,
        "brain_threshold": args.brain_threshold,
        "wm_quantile": args.wm_quantile,
        "min_pixels": args.min_pixels,
        "classifiers": [item.strip() for item in args.classifiers.split(",") if item.strip()],
        "tasks": [item.strip() for item in args.tasks.split(",") if item.strip()],
        "methods": audit_rows,
    }
    (output_dir / "roi_feature_extraction_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    train_tables: dict[str, pd.DataFrame] = {}
    test_tables: dict[str, pd.DataFrame] = {}
    for pair in runnable_methods:
        train_df = extract_roi_features(
            pair.train_dir,
            pair.name,
            subject_index,
            "train",
            args.roi_rows,
            args.roi_cols,
            args.brain_threshold,
            args.wm_quantile,
            args.min_pixels,
        )
        test_df = extract_roi_features(
            pair.test_dir,
            pair.name,
            subject_index,
            "test",
            args.roi_rows,
            args.roi_cols,
            args.brain_threshold,
            args.wm_quantile,
            args.min_pixels,
        )
        train_tables[pair.name] = train_df
        test_tables[pair.name] = test_df
        safe_name = pair.name.replace("+", "_").replace(" ", "_").replace("-", "_")
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined.to_csv(output_dir / f"roi_feature_table_{safe_name}.csv", index=False, encoding="utf-8")

    summaries: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    confusion_json: dict[str, Any] = {}
    top_features: list[dict[str, Any]] = []
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    classifiers = [item.strip() for item in args.classifiers.split(",") if item.strip()]
    for method in sorted(train_tables):
        for task in tasks:
            for classifier in classifiers:
                summary, pred_df, cm_df, top_rows = evaluate_pair(
                    train_tables[method],
                    test_tables[method],
                    method,
                    task,
                    classifier,
                )
                summaries.append(summary)
                predictions.append(pred_df)
                confusion_json[f"{method}::{task}::{classifier}"] = cm_df.to_dict()
                top_features.extend(top_rows)
    summary_df = pd.DataFrame(summaries)
    pred_all = pd.concat(predictions, ignore_index=True)
    summary_df.to_csv(output_dir / "roi_classification_subject_summary.csv", index=False, encoding="utf-8")
    pred_all.to_csv(output_dir / "roi_classification_subject_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "roi_confusion_matrices.json").write_text(json.dumps(confusion_json, indent=2, ensure_ascii=False), encoding="utf-8")
    top_df = pd.DataFrame(top_features)
    if not top_df.empty:
        top_df.to_csv(output_dir / "roi_top_features.csv", index=False, encoding="utf-8")

    avg_df = (
        summary_df.groupby("method", as_index=False)
        .agg(
            rows=("method", "size"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_auc=("macro_auc", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_n_train_subjects=("n_train_subjects", "mean"),
            mean_n_test_subjects=("n_test_subjects", "mean"),
        )
        .sort_values(["mean_macro_auc", "mean_macro_f1", "mean_accuracy"], ascending=False)
    )
    task_df = (
        summary_df.groupby(["method", "task"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_auc=("macro_auc", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values(["task", "mean_macro_auc"], ascending=[True, False])
    )
    avg_df.to_csv(output_dir / "roi_method_average_summary.csv", index=False, encoding="utf-8")
    task_df.to_csv(output_dir / "roi_per_task_summary.csv", index=False, encoding="utf-8")
    final_table = avg_df.copy()
    final_table.insert(0, "dataset", "ADNI")
    final_table.insert(1, "protocol", "coarse_grid_roi_train_test")
    final_table.to_csv(final_selection_dir / "final_downstream_roi_table.csv", index=False, encoding="utf-8")
    write_table_md(final_table, final_selection_dir / "final_downstream_roi_table.md")
    report_cn = report_text(avg_df, task_df, resource_note_cn, "cn")
    report_en = report_text(avg_df, task_df, resource_note_en, "en")
    (output_dir / "downstream_adni_roi_report_cn.md").write_text(report_cn, encoding="utf-8")
    (output_dir / "downstream_adni_roi_report.md").write_text(report_en, encoding="utf-8")
    run_log = [
        "ADNI ROI-level downstream final run",
        f"manifest={manifest}",
        f"output_dir={output_dir}",
        "image_generation_training=no",
        "final_method_changed=no",
        f"runnable_methods={', '.join([p.name for p in runnable_methods])}",
        f"tasks={', '.join(tasks)}",
        f"classifiers={', '.join(classifiers)}",
    ]
    (output_dir / "run_log.txt").write_text("\n".join(run_log) + "\n", encoding="utf-8")
    print(f"Saved ROI classification summary to: {output_dir / 'roi_classification_subject_summary.csv'}")
    print(f"Saved final ROI table to: {final_selection_dir / 'final_downstream_roi_table.csv'}")
    print(avg_df.to_string(index=False))


if __name__ == "__main__":
    main()
