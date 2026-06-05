from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image.astype(np.uint8))
    assert ok
    encoded.tofile(str(path))


def test_extract_subject_features_aggregates_slices_without_slice_leakage(tmp_path):
    from src.eval.downstream_utility import extract_subject_features_from_folder

    image_dir = tmp_path / "pred"
    for subject_idx, subject_id in enumerate(["sub-001", "sub-002"]):
        for z in [20, 21, 22]:
            image = np.full((16, 16), 30 + subject_idx * 60 + z, dtype=np.uint8)
            _write_png(image_dir / f"{subject_id}_z{z:03d}.png", image)

    subject_index = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-002"],
            "group_id": [1, 4],
            "group_name": ["CN", "AD"],
            "split": ["test", "test"],
        }
    )

    features = extract_subject_features_from_folder(
        image_dir=image_dir,
        method="toy",
        subject_index=subject_index,
        split="test",
    )

    assert features.shape[0] == 2
    assert set(features["subject_id"]) == {"sub-001", "sub-002"}
    assert features["n_slices"].tolist() == [3, 3]
    assert "intensity_mean_mean" in features.columns
    assert "highpass_energy_mean" in features.columns
    assert features.loc[features["subject_id"] == "sub-002", "intensity_mean_mean"].item() > features.loc[
        features["subject_id"] == "sub-001", "intensity_mean_mean"
    ].item()


def test_run_classification_cv_reports_subject_level_metrics():
    from src.eval.downstream_utility import run_classification_cv

    rows = []
    for class_id, group_name in enumerate(["CN", "SCD", "MCI", "AD"], start=1):
        for idx in range(4):
            rows.append(
                {
                    "method": "toy",
                    "subject_id": f"sub-{class_id}{idx:02d}",
                    "group_id": class_id,
                    "group_name": group_name,
                    "n_slices": 3,
                    "f0": float(class_id) + idx * 0.01,
                    "f1": float(class_id * 2) + idx * 0.01,
                }
            )
    features = pd.DataFrame(rows)

    summary, predictions = run_classification_cv(
        features,
        method="toy",
        task="four_class",
        n_splits=4,
        random_state=7,
        max_features=1,
    )

    assert summary["method"] == "toy"
    assert summary["task"] == "four_class"
    assert summary["n_subjects"] == 16
    assert summary["n_features"] == 2
    assert summary["n_selected_features"] == 1
    assert 0.0 <= summary["macro_f1"] <= 1.0
    assert 0.0 <= summary["balanced_accuracy"] <= 1.0
    assert "macro_auc_ovr" in summary
    assert set(["subject_id", "y_true", "y_pred"]).issubset(predictions.columns)


def test_load_method_specs_accepts_named_directories(tmp_path):
    from src.eval.downstream_utility import parse_method_specs

    method_dir = tmp_path / "method"
    method_dir.mkdir()
    specs = parse_method_specs([f"Flow={method_dir}"])

    assert len(specs) == 1
    assert specs[0].name == "Flow"
    assert specs[0].image_dir == method_dir


def test_build_fused_feature_table_prefixes_modalities_without_leakage():
    from src.eval.downstream_utility import build_fused_feature_table

    rows = []
    for method in ["T1_ONLY", "Fidelity_Flow"]:
        for idx, group_name in enumerate(["CN", "AD"], start=1):
            rows.append(
                {
                    "method": method,
                    "subject_id": f"sub-{idx:03d}",
                    "group_id": idx,
                    "group_name": group_name,
                    "split": "test",
                    "MMSE": 28 - idx,
                    "n_slices": 3,
                    "intensity_mean_mean": idx * (1.0 if method == "T1_ONLY" else 2.0),
                    "highpass_energy_mean": idx * (0.1 if method == "T1_ONLY" else 0.2),
                }
            )
    features = pd.DataFrame(rows)

    fused = build_fused_feature_table(
        features,
        fused_name="T1_PLUS_FLOW",
        left_method="T1_ONLY",
        right_method="Fidelity_Flow",
    )

    assert fused["method"].unique().tolist() == ["T1_PLUS_FLOW"]
    assert fused.shape[0] == 2
    assert "T1_ONLY__intensity_mean_mean" in fused.columns
    assert "Fidelity_Flow__intensity_mean_mean" in fused.columns
    assert "intensity_mean_mean" not in fused.columns
    assert fused.loc[fused["subject_id"] == "sub-002", "Fidelity_Flow__highpass_energy_mean"].item() == 0.4


def test_run_regression_cv_reports_mmse_metrics():
    from src.eval.downstream_utility import run_regression_cv

    rows = []
    for idx in range(12):
        rows.append(
            {
                "method": "toy",
                "subject_id": f"sub-{idx:03d}",
                "group_id": 1 + (idx % 4),
                "group_name": ["CN", "SCD", "MCI", "AD"][idx % 4],
                "split": "test",
                "MMSE": 30.0 - idx * 0.5,
                "n_slices": 3,
                "f0": float(idx),
                "f1": float(idx % 3),
            }
        )
    features = pd.DataFrame(rows)

    summary, predictions = run_regression_cv(
        features,
        method="toy",
        target="MMSE",
        n_splits=3,
        random_state=3,
        max_features=1,
        clip_range=(0.0, 30.0),
    )

    assert summary["method"] == "toy"
    assert summary["target"] == "MMSE"
    assert summary["n_subjects"] == 12
    assert summary["n_features"] == 2
    assert summary["n_selected_features"] == 1
    assert summary["mae"] >= 0.0
    assert summary["rmse"] >= 0.0
    assert "pearson_r" in summary
    assert "spearman_r" in summary
    assert set(["subject_id", "y_true", "y_pred"]).issubset(predictions.columns)
    assert predictions["y_pred"].between(0.0, 30.0).all()
