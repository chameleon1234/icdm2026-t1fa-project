from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
import torch


ROI_FEATURE_RE = re.compile(r"(?:^|__)roi_mean_r(?P<row>\d+)_c(?P<col>\d+)_mean$")

TASK_DEFINITIONS = {
    "four_class": {"include": {"CN", "SCD", "MCI", "AD"}, "labels": {"CN": 0, "SCD": 1, "MCI": 2, "AD": 3}},
    "cn_vs_scd": {"include": {"CN", "SCD"}, "labels": {"CN": 0, "SCD": 1}},
    "cn_vs_mci": {"include": {"CN", "MCI"}, "labels": {"CN": 0, "MCI": 1}},
    "cn_vs_ad": {"include": {"CN", "AD"}, "labels": {"CN": 0, "AD": 1}},
    "scd_vs_mci": {"include": {"SCD", "MCI"}, "labels": {"SCD": 0, "MCI": 1}},
    "mci_vs_ad": {"include": {"MCI", "AD"}, "labels": {"MCI": 0, "AD": 1}},
    "cn_vs_mci_ad": {"include": {"CN", "MCI", "AD"}, "labels": {"CN": 0, "MCI": 1, "AD": 1}},
    "cn_scd_vs_ad": {"include": {"CN", "SCD", "AD"}, "labels": {"CN": 0, "SCD": 0, "AD": 1}},
    "cn_scd_vs_mci_ad": {"include": {"CN", "SCD", "MCI", "AD"}, "labels": {"CN": 0, "SCD": 0, "MCI": 1, "AD": 1}},
    "cn_scd_mci_vs_ad": {"include": {"CN", "SCD", "MCI", "AD"}, "labels": {"CN": 0, "SCD": 0, "MCI": 0, "AD": 1}},
    "cn_vs_mci_spectrum": {
        "include": {"CN", "SCD", "MCI", "MCI_spectrum"},
        "labels": {"CN": 0, "SCD": 1, "MCI": 1, "MCI_spectrum": 1},
    },
    "cn_vs_mci_spectrum_ad": {
        "include": {"CN", "SCD", "MCI", "MCI_spectrum", "AD"},
        "labels": {"CN": 0, "SCD": 1, "MCI": 1, "MCI_spectrum": 1, "AD": 1},
    },
    "mci_spectrum_vs_ad": {
        "include": {"SCD", "MCI", "MCI_spectrum", "AD"},
        "labels": {"SCD": 0, "MCI": 0, "MCI_spectrum": 0, "AD": 1},
    },
    "adni_three_class": {"include": {"CN", "MCI_spectrum", "AD"}, "labels": {"CN": 0, "MCI_spectrum": 1, "AD": 2}},
}

METADATA_COLUMNS = {
    "method",
    "subject_id",
    "group_id",
    "group_name",
    "split",
    "n_slices",
    "gender",
    "age",
    "edu",
    "MMSE",
    "feature_view",
}


def feature_columns(frame: pd.DataFrame, feature_set: str = "full") -> list[str]:
    if feature_set not in {"full", "roi_mean"}:
        raise ValueError("feature_set must be one of: full, roi_mean")
    columns: list[str] = []
    for column in frame.columns:
        if column in METADATA_COLUMNS:
            continue
        if feature_set == "roi_mean" and "__roi_mean_" not in column and not column.startswith("roi_mean_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    if feature_set == "roi_mean" and not columns:
        raise ValueError("No ROI mean feature columns found")
    return columns


def parse_roi_feature(column: str) -> tuple[str, int, int] | None:
    match = ROI_FEATURE_RE.search(column)
    if not match:
        return None
    row = int(match.group("row"))
    col = int(match.group("col"))
    return f"r{row}_c{col}", row, col


def _cohens_abs(values0: np.ndarray, values1: np.ndarray) -> float:
    values0 = np.asarray(values0, dtype=np.float64)
    values1 = np.asarray(values1, dtype=np.float64)
    if values0.size < 2 or values1.size < 2:
        return 0.0
    pooled = np.sqrt(
        ((values0.size - 1) * np.var(values0, ddof=1) + (values1.size - 1) * np.var(values1, ddof=1))
        / max(values0.size + values1.size - 2, 1)
    )
    if pooled <= 1e-12:
        return 0.0
    return float(abs(np.mean(values1) - np.mean(values0)) / pooled)


def build_disease_roi_weights(
    features: pd.DataFrame,
    tasks: Iterable[str],
    method: str = "FA_GT",
    top_k: int = 0,
) -> pd.DataFrame:
    frame = features.loc[features["method"].eq(method)].copy() if "method" in features.columns else features.copy()
    if frame.empty:
        raise ValueError(f"No rows found for method={method!r}")
    roi_columns = [column for column in feature_columns(frame, feature_set="roi_mean") if parse_roi_feature(column)]
    if not roi_columns:
        raise ValueError("No roi_mean_r*_c*_mean columns found")

    rows: list[dict[str, float | int | str]] = []
    task_list = [task.strip() for task in tasks if task.strip()]
    for task in task_list:
        if task not in TASK_DEFINITIONS:
            raise ValueError(f"Unknown downstream task: {task}")
        definition = TASK_DEFINITIONS[task]
        subset = frame.loc[frame["group_name"].isin(definition["include"])].copy()
        if subset.empty:
            continue
        labels = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
        if len(set(labels.tolist())) != 2:
            continue
        for column in roi_columns:
            parsed = parse_roi_feature(column)
            if parsed is None:
                continue
            roi_key, row_idx, col_idx = parsed
            values = subset[column].to_numpy(dtype=np.float64)
            class0 = values[labels == 0]
            class1 = values[labels == 1]
            abs_delta = float(abs(np.mean(class1) - np.mean(class0))) if class0.size and class1.size else 0.0
            abs_d = _cohens_abs(class0, class1)
            rows.append(
                {
                    "task": task,
                    "feature": column,
                    "roi_key": roi_key,
                    "row": row_idx,
                    "col": col_idx,
                    "abs_delta": abs_delta,
                    "abs_cohens_d": abs_d,
                    "raw_score": abs_delta * (1.0 + abs_d),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No binary task ROI weights could be computed")
    grouped = (
        detail.groupby(["roi_key", "row", "col"], as_index=False)
        .agg(raw_score=("raw_score", "mean"), abs_delta=("abs_delta", "mean"), abs_cohens_d=("abs_cohens_d", "mean"))
        .sort_values("raw_score", ascending=False)
    )
    if top_k > 0:
        keep = set(grouped.head(top_k)["roi_key"])
        grouped.loc[~grouped["roi_key"].isin(keep), "raw_score"] = 0.0
    max_score = float(grouped["raw_score"].max())
    grouped["weight"] = 0.0 if max_score <= 1e-12 else grouped["raw_score"] / max_score
    return grouped.sort_values(["row", "col"]).reset_index(drop=True)


def roi_weight_tensor_from_frame(frame: pd.DataFrame, roi_rows: int, roi_cols: int) -> torch.Tensor:
    tensor = torch.zeros((roi_rows, roi_cols), dtype=torch.float32)
    for row in frame.to_dict(orient="records"):
        row_idx = int(row["row"])
        col_idx = int(row["col"])
        if 0 <= row_idx < roi_rows and 0 <= col_idx < roi_cols:
            tensor[row_idx, col_idx] = float(row["weight"])
    return tensor


def weighted_grid_roi_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    roi_weights: torch.Tensor,
    min_pixels: int = 16,
) -> torch.Tensor:
    if roi_weights.dim() != 2:
        raise ValueError("roi_weights must have shape [roi_rows, roi_cols]")
    roi_rows, roi_cols = int(roi_weights.shape[0]), int(roi_weights.shape[1])
    height, width = pred.shape[-2:]
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for row_idx in range(roi_rows):
        y0 = int(round(row_idx * height / roi_rows))
        y1 = int(round((row_idx + 1) * height / roi_rows))
        for col_idx in range(roi_cols):
            weight = roi_weights[row_idx, col_idx].to(device=pred.device, dtype=pred.dtype)
            if float(weight.detach().cpu()) <= 0.0:
                continue
            x0 = int(round(col_idx * width / roi_cols))
            x1 = int(round((col_idx + 1) * width / roi_cols))
            region_mask = mask[:, :, y0:y1, x0:x1].to(dtype=pred.dtype)
            denom = region_mask.flatten(1).sum(dim=1)
            valid = denom >= float(min_pixels)
            if not bool(valid.any()):
                continue
            pred_mean = (pred[:, :, y0:y1, x0:x1] * region_mask).flatten(1).sum(dim=1) / denom.clamp_min(1.0)
            target_mean = (target[:, :, y0:y1, x0:x1] * region_mask).flatten(1).sum(dim=1) / denom.clamp_min(1.0)
            losses.append(torch.abs(pred_mean[valid] - target_mean[valid]).mean() * weight)
            weights.append(weight)
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)
