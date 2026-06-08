from __future__ import annotations

import numpy as np
import pandas as pd


def test_task_roi_direction_uses_gt_fa_class_difference():
    from scripts.apply_task_sensitive_roi_calibration import compute_task_roi_directions

    rows = []
    for group_name, value in [("CN", 0.30), ("MCI", 0.55)]:
        for idx in range(4):
            rows.append(
                {
                    "method": "FA_GT",
                    "subject_id": f"{group_name}-{idx}",
                    "group_name": group_name,
                    "roi_mean_r0_c0_mean": value + idx * 0.001,
                    "roi_mean_r0_c1_mean": 0.50,
                }
            )
    directions = compute_task_roi_directions(pd.DataFrame(rows), tasks=["cn_vs_mci"])

    assert directions["cn_vs_mci"]["r0_c0"] > 0.20
    assert abs(directions["cn_vs_mci"]["r0_c1"]) < 1e-6


def test_task_score_model_scores_class_one_higher_from_t1_roi_features():
    from scripts.apply_task_sensitive_roi_calibration import fit_task_score_models

    rows = []
    for group_name, value in [("CN", 0.20), ("MCI", 0.80)]:
        for idx in range(6):
            rows.append(
                {
                    "method": "T1_ONLY",
                    "subject_id": f"{group_name}-{idx}",
                    "group_name": group_name,
                    "roi_mean_r0_c0_mean": value + idx * 0.001,
                    "roi_mean_r0_c0_std": 0.01,
                }
            )
    models = fit_task_score_models(pd.DataFrame(rows), tasks=["cn_vs_mci"])
    model = models["cn_vs_mci"]
    cn_score = model.score_subject({"roi_mean_r0_c0_mean": 0.20, "roi_mean_r0_c0_std": 0.01})
    mci_score = model.score_subject({"roi_mean_r0_c0_mean": 0.80, "roi_mean_r0_c0_std": 0.01})

    assert mci_score > cn_score
    assert mci_score > 0


def test_task_sensitive_calibration_pushes_roi_toward_task_direction():
    from sklearn.linear_model import Ridge

    from scripts.apply_task_sensitive_roi_calibration import calibrate_image_task_sensitive

    model = Ridge(alpha=0.0)
    model.fit(np.asarray([[0.5, 0.0], [0.6, 0.0]], dtype=np.float32), np.asarray([0.4, 0.4], dtype=np.float32))
    roi_models = {f"r{r}_c{c}": model for r in range(2) for c in range(3)}
    pred = np.full((12, 18), 0.40, dtype=np.float32)
    t1 = np.full((12, 18), 0.50, dtype=np.float32)
    task_scores = {"cn_vs_mci": 1.0}
    task_directions = {"cn_vs_mci": {"r0_c0": 0.10}}

    calibrated, stats = calibrate_image_task_sensitive(
        pred=pred,
        t1=t1,
        roi_models=roi_models,
        task_scores=task_scores,
        task_directions=task_directions,
        task_strength=0.5,
        roi_rows=2,
        roi_cols=3,
        brain_threshold=0.02,
        gain=1.0,
        max_delta=0.20,
        smooth_sigma=0.0,
    )

    assert stats["r0_c0_task_bias"] > 0.04
    assert float(np.mean(calibrated[:6, :6])) > float(np.mean(pred[:6, :6]))
    assert np.allclose(calibrated[:6, 6:12], pred[:6, 6:12])


def test_task_gain_can_override_base_roi_regression_when_base_is_smoothing():
    from sklearn.linear_model import Ridge

    from scripts.apply_task_sensitive_roi_calibration import calibrate_image_task_sensitive

    model = Ridge(alpha=0.0)
    model.fit(np.asarray([[0.5, 0.0], [0.6, 0.0]], dtype=np.float32), np.asarray([0.20, 0.20], dtype=np.float32))
    roi_models = {f"r{r}_c{c}": model for r in range(2) for c in range(3)}
    pred = np.full((12, 18), 0.40, dtype=np.float32)
    t1 = np.full((12, 18), 0.50, dtype=np.float32)

    calibrated, stats = calibrate_image_task_sensitive(
        pred=pred,
        t1=t1,
        roi_models=roi_models,
        task_scores={"cn_vs_mci": 1.0},
        task_directions={"cn_vs_mci": {"r0_c0": 0.10}},
        task_strength=0.5,
        roi_rows=2,
        roi_cols=3,
        brain_threshold=0.02,
        gain=1.0,
        max_delta=0.20,
        smooth_sigma=0.0,
        base_residual_weight=0.0,
        task_gain=1.0,
    )

    assert stats["r0_c0_base_delta"] < 0
    assert stats["r0_c0_delta"] > 0
    assert float(np.mean(calibrated[:6, :6])) > 0.40
