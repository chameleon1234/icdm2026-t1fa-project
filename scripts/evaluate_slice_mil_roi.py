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


def _select_and_scale(
    train_subset: pd.DataFrame,
    test_subset: pd.DataFrame,
    feature_set: str,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_columns = _roi_feature_columns(train_subset, feature_set)
    test_columns = _roi_feature_columns(test_subset, feature_set)
    columns = [column for column in train_columns if column in set(test_columns)]
    if not columns:
        raise ValueError("No shared feature columns for MIL/voting")
    x_train = train_subset[columns].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_subset["y_label"].to_numpy(dtype=np.int64)
    x_test = test_subset[columns].fillna(0.0).to_numpy(dtype=np.float32)

    selected_columns = columns
    if max_features > 0 and max_features < len(columns):
        selector = SelectKBest(score_func=f_classif, k=int(max_features))
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        selected_columns = [columns[idx] for idx in selector.get_support(indices=True)]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    out_columns = [f"feature_{idx:03d}" for idx in range(x_train.shape[1])]
    train_out = train_subset[["method", "sample_id", "subject_id", "slice_idx", "group_name", "y_label"]].copy()
    test_out = test_subset[["method", "sample_id", "subject_id", "slice_idx", "group_name", "y_label"]].copy()
    train_out[out_columns] = x_train
    test_out[out_columns] = x_test
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
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_subset = _task_slice_subset(train_features, task)
    test_subset = _task_slice_subset(test_features, task)
    train_scaled, test_scaled, columns = _select_and_scale(train_subset, test_subset, feature_set, max_features)
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
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_subset = _task_slice_subset(train_features, task)
    test_subset = _task_slice_subset(test_features, task)
    train_scaled, test_scaled, columns = _select_and_scale(train_subset, test_subset, feature_set, max_features)
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
    summaries: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for method in sorted(train_tables):
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
                )
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
                )
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
