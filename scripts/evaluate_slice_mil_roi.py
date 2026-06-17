from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_finalpdf_train_test_roi import (
    MethodPair,
    _build_estimator,
    _default_split_dir,
    _decision_scores,
    _json_ready,
    _load_subject_index,
    _parse_fusion,
    _prepare_task,
    _read_yaml,
    _roi_feature_columns,
    build_fused_slice_feature_table_local,
    extract_slice_features_with_atlas,
    parse_method_pair,
)
from src.eval.downstream_utility import TASK_DEFINITIONS


@dataclass(frozen=True)
class SubjectBag:
    subject_id: str
    group_name: str
    y: int
    x: np.ndarray


class AttentionMIL(torch.nn.Module):
    def __init__(self, input_dim: int, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(inplace=True),
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(width, width // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(width // 2, 1),
        )
        self.classifier = torch.nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=0)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=0)
        return self.classifier(pooled).squeeze(0)


class LateFusionAttentionMIL(torch.nn.Module):
    def __init__(self, left_dim: int, right_dim: int, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.left = AttentionBranch(left_dim, width, dropout)
        self.right = AttentionBranch(right_dim, width, dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(width * 2, width),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )

    def forward(self, left_x: torch.Tensor, right_x: torch.Tensor) -> torch.Tensor:
        left_pooled = self.left(left_x)
        right_pooled = self.right(right_x)
        return self.classifier(torch.cat([left_pooled, right_pooled], dim=0)).squeeze(0)


class MultiTaskAttentionMIL(torch.nn.Module):
    def __init__(self, input_dim: int, tasks: list[str], width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.branch = AttentionBranch(input_dim, width, dropout)
        self.heads = torch.nn.ModuleDict({task: torch.nn.Linear(width, 1) for task in tasks})

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        pooled = self.branch(x)
        return self.heads[task](pooled).squeeze(0)


class AttentionBranch(torch.nn.Module):
    def __init__(self, input_dim: int, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(inplace=True),
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(width, width // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(width // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=0)
        return torch.sum(h * weights.unsqueeze(-1), dim=0)


def aggregate_slice_scores_by_subject(
    slice_predictions: pd.DataFrame,
    method: str,
    task: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for subject_id, group in slice_predictions.groupby("subject_id", sort=True):
        y_values = group["y_true"].astype(int).unique().tolist()
        if len(y_values) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels: {y_values}")
        score = float(group["decision_score"].mean())
        rows.append(
            {
                "method": method,
                "task": task,
                "subject_id": subject_id,
                "group_name": str(group["group_name"].iloc[0]),
                "y_true": int(y_values[0]),
                "subject_score": score,
                "n_slices": int(len(group)),
            }
        )
    subject_predictions = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    y_true = subject_predictions["y_true"].to_numpy(dtype=np.int64)
    scores = subject_predictions["subject_score"].to_numpy(dtype=np.float32)
    y_pred = (scores >= 0.0).astype(np.int64)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    summary = {
        "method": method,
        "task": task,
        "protocol": "slice_svm_vote",
        "n_test_subjects": int(subject_predictions["subject_id"].nunique()),
        "n_test_slices": int(slice_predictions.shape[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": auc,
    }
    subject_predictions["y_pred"] = y_pred
    return summary, subject_predictions


def build_subject_bags(
    features: pd.DataFrame,
    columns: list[str],
    labels: dict[str, int],
) -> list[SubjectBag]:
    bags: list[SubjectBag] = []
    for subject_id, group in features.groupby("subject_id", sort=True):
        group = group.sort_values("slice_idx")
        group_names = group["group_name"].unique().tolist()
        if len(group_names) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent groups: {group_names}")
        group_name = str(group_names[0])
        if group_name not in labels:
            continue
        x = group[columns].fillna(0.0).to_numpy(dtype=np.float32)
        bags.append(SubjectBag(subject_id=str(subject_id), group_name=group_name, y=int(labels[group_name]), x=x))
    if not bags:
        raise ValueError("No subject bags were built")
    return bags


def build_multitask_subject_bags(
    features: pd.DataFrame,
    columns: list[str],
    tasks: list[str],
) -> list[tuple[str, SubjectBag]]:
    task_bags: list[tuple[str, SubjectBag]] = []
    for task in tasks:
        if task not in TASK_DEFINITIONS:
            raise ValueError(f"Unknown task {task!r}")
        labels = TASK_DEFINITIONS[task]["labels"]
        included = TASK_DEFINITIONS[task]["include"]
        subset = features.loc[features["group_name"].isin(included)].copy()
        for bag in build_subject_bags(subset, columns, labels):
            task_bags.append((task, bag))
    if not task_bags:
        raise ValueError("No multi-task subject bags were built")
    return task_bags


def split_late_fusion_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    prefixed = [column for column in columns if "__" in column]
    prefixes = []
    for column in prefixed:
        prefix = column.split("__", 1)[0]
        if prefix not in prefixes:
            prefixes.append(prefix)
    if len(prefixes) != 2:
        raise ValueError(f"Late fusion requires exactly two feature prefixes, got {prefixes}")
    left = [column for column in columns if column.startswith(prefixes[0] + "__")]
    right = [column for column in columns if column.startswith(prefixes[1] + "__")]
    if not left or not right:
        raise ValueError("Late fusion branch columns cannot be empty")
    return left, right


def select_shared_disease_roi_columns(
    features: pd.DataFrame,
    feature_set: str,
    tasks: list[str],
    top_k: int,
) -> list[str]:
    columns = _roi_feature_columns(features, feature_set)
    if top_k <= 0 or top_k >= len(columns):
        return columns

    score_sum = pd.Series(0.0, index=columns, dtype=np.float64)
    used_tasks = 0
    for task in tasks:
        if task not in TASK_DEFINITIONS:
            raise ValueError(f"Unknown task {task!r}")
        definition = TASK_DEFINITIONS[task]
        subset = features.loc[features["group_name"].isin(definition["include"])].copy()
        if subset.empty:
            continue
        y = subset["group_name"].map(definition["labels"]).astype(int).to_numpy()
        if len(np.unique(y)) != 2:
            continue
        x = subset[columns].fillna(0.0).to_numpy(dtype=np.float32)
        scores, _ = f_classif(x, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        max_score = float(np.max(scores)) if scores.size else 0.0
        if max_score > 0:
            scores = scores / max_score
        score_sum += pd.Series(scores, index=columns)
        used_tasks += 1

    if used_tasks == 0:
        return columns[: int(top_k)]
    selected = score_sum.sort_values(ascending=False).head(int(top_k)).index.tolist()
    return selected


def _select_and_scale(
    train_subset: pd.DataFrame,
    test_subset: pd.DataFrame,
    feature_set: str,
    max_features: int,
    preset_columns: list[str] | None = None,
    preserve_column_names: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if preset_columns is None:
        train_columns = _roi_feature_columns(train_subset, feature_set)
        test_columns = _roi_feature_columns(test_subset, feature_set)
        columns = [column for column in train_columns if column in set(test_columns)]
    else:
        columns = [column for column in preset_columns if column in train_subset.columns and column in test_subset.columns]
    if not columns:
        raise ValueError("No shared feature columns for MIL/voting")
    x_train = train_subset[columns].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_subset["y_label"].to_numpy(dtype=np.int64)
    x_test = test_subset[columns].fillna(0.0).to_numpy(dtype=np.float32)

    selected_columns = columns
    if preset_columns is None and max_features > 0 and max_features < len(columns):
        selector = SelectKBest(score_func=f_classif, k=int(max_features))
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        selected_columns = [columns[idx] for idx in selector.get_support(indices=True)]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    out_columns = selected_columns if preserve_column_names else [f"feature_{idx:03d}" for idx in range(x_train.shape[1])]
    train_meta = train_subset[["method", "sample_id", "subject_id", "slice_idx", "group_name", "y_label"]].reset_index(drop=True)
    test_meta = test_subset[["method", "sample_id", "subject_id", "slice_idx", "group_name", "y_label"]].reset_index(drop=True)
    train_out = pd.concat([train_meta, pd.DataFrame(x_train, columns=out_columns)], axis=1)
    test_out = pd.concat([test_meta, pd.DataFrame(x_test, columns=out_columns)], axis=1)
    return train_out, test_out, out_columns


def _task_slice_subset(features: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task {task!r}")
    definition = TASK_DEFINITIONS[task]
    subset = features.loc[features["group_name"].isin(definition["include"])].copy()
    subset["y_label"] = subset["group_name"].map(definition["labels"]).astype(int)
    if subset["y_label"].nunique() != 2:
        raise ValueError(f"This MIL script currently supports binary tasks only, got {task!r}")
    return subset.reset_index(drop=True)


def evaluate_slice_svm_vote(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    method: str,
    task: str,
    classifier: str,
    feature_set: str,
    max_features: int,
    preset_columns: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_subset = _task_slice_subset(train_features, task)
    test_subset = _task_slice_subset(test_features, task)
    train_scaled, test_scaled, columns = _select_and_scale(train_subset, test_subset, feature_set, max_features, preset_columns)
    estimator = _build_estimator(classifier, n_features=len(columns), max_features=0)
    x_train = train_scaled[columns].to_numpy(dtype=np.float32)
    y_train = train_scaled["y_label"].to_numpy(dtype=np.int64)
    x_test = test_scaled[columns].to_numpy(dtype=np.float32)
    estimator.fit(x_train, y_train)
    scores = _decision_scores(estimator, x_test, [0, 1])
    slice_predictions = test_scaled[["method", "sample_id", "subject_id", "slice_idx", "group_name", "y_label"]].rename(
        columns={"y_label": "y_true"}
    )
    slice_predictions["decision_score"] = scores
    summary, subject_predictions = aggregate_slice_scores_by_subject(slice_predictions, method, task)
    summary.update(
        {
            "classifier": classifier,
            "feature_set": feature_set,
            "n_train_subjects": int(train_scaled["subject_id"].nunique()),
            "n_train_slices": int(train_scaled.shape[0]),
            "n_features": int(len(columns)),
            "n_selected_features": int(len(columns)),
        }
    )
    return summary, subject_predictions


def _train_attention_mil(
    train_bags: list[SubjectBag],
    test_bags: list[SubjectBag],
    input_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    width: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = AttentionMIL(input_dim=input_dim, width=width).to(resolved_device)
    positives = sum(bag.y for bag in train_bags)
    negatives = len(train_bags) - positives
    pos_weight = torch.tensor([max(1.0, negatives / max(1, positives))], device=resolved_device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(int(epochs)):
        model.train()
        order = np.random.permutation(len(train_bags))
        for idx in order:
            bag = train_bags[int(idx)]
            x = torch.from_numpy(bag.x).to(resolved_device)
            y = torch.tensor(float(bag.y), device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logit = model(x)
            loss = criterion(logit.view(1), y.view(1))
            loss.backward()
            optimizer.step()

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for bag in test_bags:
            x = torch.from_numpy(bag.x).to(resolved_device)
            logit = model(x)
            scores.append(float(logit.detach().cpu()))
            labels.append(int(bag.y))
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def _train_late_fusion_mil(
    train_bags: list[tuple[SubjectBag, SubjectBag]],
    test_bags: list[tuple[SubjectBag, SubjectBag]],
    left_dim: int,
    right_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    width: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = LateFusionAttentionMIL(left_dim=left_dim, right_dim=right_dim, width=width).to(resolved_device)
    positives = sum(left_bag.y for left_bag, _ in train_bags)
    negatives = len(train_bags) - positives
    pos_weight = torch.tensor([max(1.0, negatives / max(1, positives))], device=resolved_device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(int(epochs)):
        model.train()
        order = np.random.permutation(len(train_bags))
        for idx in order:
            left_bag, right_bag = train_bags[int(idx)]
            left_x = torch.from_numpy(left_bag.x).to(resolved_device)
            right_x = torch.from_numpy(right_bag.x).to(resolved_device)
            y = torch.tensor(float(left_bag.y), device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logit = model(left_x, right_x)
            loss = criterion(logit.view(1), y.view(1))
            loss.backward()
            optimizer.step()

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for left_bag, right_bag in test_bags:
            left_x = torch.from_numpy(left_bag.x).to(resolved_device)
            right_x = torch.from_numpy(right_bag.x).to(resolved_device)
            scores.append(float(model(left_x, right_x).detach().cpu()))
            labels.append(int(left_bag.y))
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def _train_multitask_mil(
    train_task_bags: list[tuple[str, SubjectBag]],
    test_task_bags: list[tuple[str, SubjectBag]],
    tasks: list[str],
    input_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    width: int,
    seed: int,
    device: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[SubjectBag]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = MultiTaskAttentionMIL(input_dim=input_dim, tasks=tasks, width=width).to(resolved_device)
    pos_weights: dict[str, torch.Tensor] = {}
    for task in tasks:
        task_train = [bag for item_task, bag in train_task_bags if item_task == task]
        positives = sum(bag.y for bag in task_train)
        negatives = len(task_train) - positives
        pos_weights[task] = torch.tensor([max(1.0, negatives / max(1, positives))], device=resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(int(epochs)):
        model.train()
        order = np.random.permutation(len(train_task_bags))
        for idx in order:
            task, bag = train_task_bags[int(idx)]
            x = torch.from_numpy(bag.x).to(resolved_device)
            y = torch.tensor(float(bag.y), device=resolved_device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights[task])
            optimizer.zero_grad(set_to_none=True)
            logit = model(x, task)
            loss = criterion(logit.view(1), y.view(1))
            loss.backward()
            optimizer.step()

    outputs: dict[str, tuple[np.ndarray, np.ndarray, list[SubjectBag]]] = {}
    model.eval()
    with torch.no_grad():
        for task in tasks:
            labels: list[int] = []
            scores: list[float] = []
            bags: list[SubjectBag] = []
            for item_task, bag in test_task_bags:
                if item_task != task:
                    continue
                x = torch.from_numpy(bag.x).to(resolved_device)
                scores.append(float(model(x, task).detach().cpu()))
                labels.append(int(bag.y))
                bags.append(bag)
            outputs[task] = (np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32), bags)
    return outputs


def _prepare_multitask_scaled(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_set: str,
    max_features: int,
    tasks: list[str],
    preset_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if preset_columns is not None:
        base_columns = [column for column in preset_columns if column in train_features.columns and column in test_features.columns]
    else:
        train_columns = _roi_feature_columns(train_features, feature_set)
        test_columns = _roi_feature_columns(test_features, feature_set)
        common = [column for column in train_columns if column in set(test_columns)]
        if max_features > 0 and max_features < len(common):
            selected = select_shared_disease_roi_columns(train_features, feature_set, tasks, max_features)
            base_columns = [column for column in selected if column in common]
        else:
            base_columns = common
    if not base_columns:
        raise ValueError("No shared multi-task feature columns")

    included_groups = set()
    for task in tasks:
        included_groups.update(TASK_DEFINITIONS[task]["include"])
    train_subset = train_features.loc[train_features["group_name"].isin(included_groups)].copy()
    test_subset = test_features.loc[test_features["group_name"].isin(included_groups)].copy()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_subset[base_columns].fillna(0.0).to_numpy(dtype=np.float32)).astype(np.float32)
    x_test = scaler.transform(test_subset[base_columns].fillna(0.0).to_numpy(dtype=np.float32)).astype(np.float32)
    columns = [f"feature_{idx:03d}" for idx in range(x_train.shape[1])]
    keep = ["method", "sample_id", "subject_id", "slice_idx", "group_name"]
    train_meta = train_subset[keep].reset_index(drop=True)
    test_meta = test_subset[keep].reset_index(drop=True)
    train_out = pd.concat([train_meta, pd.DataFrame(x_train, columns=columns)], axis=1)
    test_out = pd.concat([test_meta, pd.DataFrame(x_test, columns=columns)], axis=1)
    return train_out, test_out, columns


def evaluate_attention_mil(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    method: str,
    task: str,
    feature_set: str,
    max_features: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    width: int,
    seed: int,
    device: str,
    preset_columns: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_subset = _task_slice_subset(train_features, task)
    test_subset = _task_slice_subset(test_features, task)
    train_scaled, test_scaled, columns = _select_and_scale(train_subset, test_subset, feature_set, max_features, preset_columns)
    labels = TASK_DEFINITIONS[task]["labels"]
    train_bags = build_subject_bags(train_scaled, columns, labels)
    test_bags = build_subject_bags(test_scaled, columns, labels)
    y_true, scores = _train_attention_mil(
        train_bags,
        test_bags,
        input_dim=len(columns),
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        width=width,
        seed=seed,
        device=device,
    )
    y_pred = (scores >= 0.0).astype(np.int64)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    predictions = pd.DataFrame(
        {
            "method": method,
            "task": task,
            "subject_id": [bag.subject_id for bag in test_bags],
            "group_name": [bag.group_name for bag in test_bags],
            "y_true": y_true,
            "y_pred": y_pred,
            "subject_score": scores,
            "n_slices": [int(bag.x.shape[0]) for bag in test_bags],
        }
    )
    summary = {
        "method": method,
        "task": task,
        "protocol": "attention_mil",
        "classifier": "attention_mil",
        "feature_set": feature_set,
        "n_train_subjects": int(len(train_bags)),
        "n_test_subjects": int(len(test_bags)),
        "n_train_slices": int(train_scaled.shape[0]),
        "n_test_slices": int(test_scaled.shape[0]),
        "n_features": int(len(columns)),
        "n_selected_features": int(len(columns)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": auc,
        "epochs": int(epochs),
        "seed": int(seed),
    }
    return summary, predictions


def evaluate_multi_task_mil(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    method: str,
    tasks: list[str],
    feature_set: str,
    max_features: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    width: int,
    seed: int,
    device: str,
    preset_columns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train_scaled, test_scaled, columns = _prepare_multitask_scaled(
        train_features,
        test_features,
        feature_set,
        max_features,
        tasks,
        preset_columns,
    )
    train_task_bags = build_multitask_subject_bags(train_scaled, columns, tasks)
    test_task_bags = build_multitask_subject_bags(test_scaled, columns, tasks)
    outputs = _train_multitask_mil(
        train_task_bags,
        test_task_bags,
        tasks=tasks,
        input_dim=len(columns),
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        width=width,
        seed=seed,
        device=device,
    )
    summaries: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    for task in tasks:
        y_true, scores, bags = outputs[task]
        y_pred = (scores >= 0.0).astype(np.int64)
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = float("nan")
        predictions = pd.DataFrame(
            {
                "method": method,
                "task": task,
                "subject_id": [bag.subject_id for bag in bags],
                "group_name": [bag.group_name for bag in bags],
                "y_true": y_true,
                "y_pred": y_pred,
                "subject_score": scores,
                "n_slices": [int(bag.x.shape[0]) for bag in bags],
            }
        )
        summaries.append(
            {
                "method": method,
                "task": task,
                "protocol": "multi_task_mil",
                "classifier": "multi_task_mil",
                "feature_set": feature_set,
                "n_train_subjects": int(len({bag.subject_id for _, bag in train_task_bags})),
                "n_test_subjects": int(len(bags)),
                "n_train_slices": int(train_scaled.shape[0]),
                "n_test_slices": int(test_scaled.shape[0]),
                "n_features": int(len(columns)),
                "n_selected_features": int(len(columns)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_auc_ovr": auc,
                "epochs": int(epochs),
                "seed": int(seed),
            }
        )
        prediction_tables.append(predictions)
    return summaries, pd.concat(prediction_tables, ignore_index=True)


def evaluate_late_fusion_mil(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    method: str,
    task: str,
    feature_set: str,
    max_features: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    width: int,
    seed: int,
    device: str,
    preset_columns: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_subset = _task_slice_subset(train_features, task)
    test_subset = _task_slice_subset(test_features, task)
    train_scaled, test_scaled, columns = _select_and_scale(
        train_subset,
        test_subset,
        feature_set,
        max_features,
        preset_columns,
        preserve_column_names=True,
    )
    left_columns, right_columns = split_late_fusion_columns(columns)
    labels = TASK_DEFINITIONS[task]["labels"]
    train_left = build_subject_bags(train_scaled, left_columns, labels)
    train_right = build_subject_bags(train_scaled, right_columns, labels)
    test_left = build_subject_bags(test_scaled, left_columns, labels)
    test_right = build_subject_bags(test_scaled, right_columns, labels)
    train_bags = _pair_bags(train_left, train_right)
    test_bags = _pair_bags(test_left, test_right)
    y_true, scores = _train_late_fusion_mil(
        train_bags,
        test_bags,
        left_dim=len(left_columns),
        right_dim=len(right_columns),
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        width=width,
        seed=seed,
        device=device,
    )
    y_pred = (scores >= 0.0).astype(np.int64)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    predictions = pd.DataFrame(
        {
            "method": method,
            "task": task,
            "subject_id": [left_bag.subject_id for left_bag, _ in test_bags],
            "group_name": [left_bag.group_name for left_bag, _ in test_bags],
            "y_true": y_true,
            "y_pred": y_pred,
            "subject_score": scores,
            "n_slices": [int(left_bag.x.shape[0]) for left_bag, _ in test_bags],
        }
    )
    summary = {
        "method": method,
        "task": task,
        "protocol": "late_fusion_mil",
        "classifier": "late_fusion_mil",
        "feature_set": feature_set,
        "n_train_subjects": int(len(train_bags)),
        "n_test_subjects": int(len(test_bags)),
        "n_train_slices": int(train_scaled.shape[0]),
        "n_test_slices": int(test_scaled.shape[0]),
        "n_features": int(len(columns)),
        "n_selected_features": int(len(columns)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_auc_ovr": auc,
        "epochs": int(epochs),
        "seed": int(seed),
    }
    return summary, predictions


def _pair_bags(left: list[SubjectBag], right: list[SubjectBag]) -> list[tuple[SubjectBag, SubjectBag]]:
    right_by_subject = {bag.subject_id: bag for bag in right}
    paired: list[tuple[SubjectBag, SubjectBag]] = []
    for left_bag in left:
        right_bag = right_by_subject.get(left_bag.subject_id)
        if right_bag is None:
            continue
        if left_bag.y != right_bag.y or left_bag.x.shape[0] != right_bag.x.shape[0]:
            raise ValueError(f"Mismatched late-fusion bags for {left_bag.subject_id}")
        paired.append((left_bag, right_bag))
    if not paired:
        raise ValueError("No paired bags for late fusion")
    return paired


def _extract_slice_features(
    pair: MethodPair,
    atlas_dir: str,
    subject_index: pd.DataFrame,
    train_split: str,
    test_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = extract_slice_features_with_atlas(
        image_dir=pair.train_dir,
        atlas_dir=atlas_dir,
        method=pair.name,
        subject_index=subject_index,
        split=train_split,
    )
    test = extract_slice_features_with_atlas(
        image_dir=pair.test_dir,
        atlas_dir=atlas_dir,
        method=pair.name,
        subject_index=subject_index,
        split=test_split,
    )
    return train, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice-level ROI features with subject-level vote/MIL evaluation.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--subject_index_csv", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--method", action="append", default=[], help="NAME=TRAIN_DIR|TEST_DIR")
    parser.add_argument("--fusion", action="append", default=[], help="NAME=LEFT+RIGHT")
    parser.add_argument("--include_t1", action="store_true")
    parser.add_argument("--include_fa_gt", action="store_true")
    parser.add_argument("--atlas_dir", required=True)
    parser.add_argument("--tasks", default="cn_vs_ad,cn_vs_mci,mci_vs_ad")
    parser.add_argument("--protocols", default="slice_svm_vote,attention_mil")
    parser.add_argument("--classifier", choices=["linear_svm", "rbf_svm"], default="rbf_svm")
    parser.add_argument("--feature_set", choices=["roi_mean", "full"], default="roi_mean")
    parser.add_argument("--max_features", type=int, default=24)
    parser.add_argument("--shared_roi_top_k", type=int, default=0, help="Select one train-only disease-sensitive ROI subset shared by all tasks for each method.")
    parser.add_argument("--shared_roi_tasks", default="", help="Comma-separated tasks used to choose the shared ROI subset. Defaults to --tasks.")
    parser.add_argument("--mil_epochs", type=int, default=160)
    parser.add_argument("--mil_lr", type=float, default=1e-3)
    parser.add_argument("--mil_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mil_width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", default="outputs/icdm2026/downstream_slice_mil_roi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)
    subject_index = _load_subject_index(config, args.subject_index_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs: list[MethodPair] = []
    if args.include_t1:
        pairs.append(MethodPair("T1_ONLY", _default_split_dir(config, args.train_split, "t1"), _default_split_dir(config, args.test_split, "t1")))
    if args.include_fa_gt:
        pairs.append(MethodPair("FA_GT", _default_split_dir(config, args.train_split, "fa"), _default_split_dir(config, args.test_split, "fa")))
    pairs.extend(parse_method_pair(spec) for spec in args.method)
    if not pairs:
        raise ValueError("No methods selected")

    train_tables: dict[str, pd.DataFrame] = {}
    test_tables: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        train_tables[pair.name], test_tables[pair.name] = _extract_slice_features(
            pair, args.atlas_dir, subject_index, args.train_split, args.test_split
        )

    if args.fusion:
        train_base = pd.concat(train_tables.values(), ignore_index=True)
        test_base = pd.concat(test_tables.values(), ignore_index=True)
        for spec in args.fusion:
            name, left, right = _parse_fusion(spec)
            train_tables[name] = build_fused_slice_feature_table_local(train_base, name, left, right, args.feature_set)
            test_tables[name] = build_fused_slice_feature_table_local(test_base, name, left, right, args.feature_set)

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    shared_roi_tasks = [item.strip() for item in (args.shared_roi_tasks or args.tasks).split(",") if item.strip()]
    shared_columns_by_method: dict[str, list[str] | None] = {}
    if args.shared_roi_top_k > 0:
        selected_rows: list[dict[str, Any]] = []
        for method, table in sorted(train_tables.items()):
            selected = select_shared_disease_roi_columns(
                table,
                feature_set=args.feature_set,
                tasks=shared_roi_tasks,
                top_k=args.shared_roi_top_k,
            )
            shared_columns_by_method[method] = selected
            for rank, column in enumerate(selected, start=1):
                selected_rows.append({"method": method, "rank": rank, "feature": column})
        pd.DataFrame(selected_rows).to_csv(output_root / "shared_disease_roi_columns.csv", index=False, encoding="utf-8")
    else:
        shared_columns_by_method = {method: None for method in train_tables}

    summaries: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for method in sorted(train_tables):
        if "multi_task_mil" in protocols:
            multi_summaries, multi_pred = evaluate_multi_task_mil(
                train_tables[method],
                test_tables[method],
                method,
                tasks=tasks,
                feature_set=args.feature_set,
                max_features=args.max_features,
                epochs=args.mil_epochs,
                lr=args.mil_lr,
                weight_decay=args.mil_weight_decay,
                width=args.mil_width,
                seed=args.seed,
                device=args.device,
                preset_columns=shared_columns_by_method.get(method),
            )
            for summary in multi_summaries:
                summary["shared_roi_top_k"] = int(args.shared_roi_top_k)
            summaries.extend(multi_summaries)
            predictions.append(multi_pred)
        for task in tasks:
            if "slice_svm_vote" in protocols:
                summary, pred = evaluate_slice_svm_vote(
                    train_tables[method],
                    test_tables[method],
                    method,
                    task,
                    classifier=args.classifier,
                    feature_set=args.feature_set,
                    max_features=args.max_features,
                    preset_columns=shared_columns_by_method.get(method),
                )
                summary["shared_roi_top_k"] = int(args.shared_roi_top_k)
                summaries.append(summary)
                predictions.append(pred)
            if "attention_mil" in protocols:
                summary, pred = evaluate_attention_mil(
                    train_tables[method],
                    test_tables[method],
                    method,
                    task,
                    feature_set=args.feature_set,
                    max_features=args.max_features,
                    epochs=args.mil_epochs,
                    lr=args.mil_lr,
                    weight_decay=args.mil_weight_decay,
                    width=args.mil_width,
                    seed=args.seed,
                    device=args.device,
                    preset_columns=shared_columns_by_method.get(method),
                )
                summary["shared_roi_top_k"] = int(args.shared_roi_top_k)
                summaries.append(summary)
                predictions.append(pred)
            if "late_fusion_mil" in protocols:
                try:
                    summary, pred = evaluate_late_fusion_mil(
                        train_tables[method],
                        test_tables[method],
                        method,
                        task,
                        feature_set=args.feature_set,
                        max_features=args.max_features,
                        epochs=args.mil_epochs,
                        lr=args.mil_lr,
                        weight_decay=args.mil_weight_decay,
                        width=args.mil_width,
                        seed=args.seed,
                        device=args.device,
                        preset_columns=shared_columns_by_method.get(method),
                    )
                except ValueError as exc:
                    print(f"Skipping late_fusion_mil for {method}/{task}: {exc}")
                else:
                    summary["shared_roi_top_k"] = int(args.shared_roi_top_k)
                    summaries.append(summary)
                    predictions.append(pred)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["task", "protocol", "accuracy", "macro_auc_ovr"],
        ascending=[True, True, False, False],
    )
    pred_df = pd.concat(predictions, ignore_index=True)
    summary_df.to_csv(output_root / "classification_subject_summary.csv", index=False, encoding="utf-8")
    pred_df.to_csv(output_root / "classification_subject_predictions.csv", index=False, encoding="utf-8")
    with open(output_root / "classification_subject_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(summaries), handle, indent=2, ensure_ascii=False)
    print(f"Saved slice-to-subject ROI summary to: {output_root / 'classification_subject_summary.csv'}")
    print(summary_df[["protocol", "method", "task", "n_train_subjects", "n_test_subjects", "n_train_slices", "n_test_slices", "accuracy", "macro_auc_ovr", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
