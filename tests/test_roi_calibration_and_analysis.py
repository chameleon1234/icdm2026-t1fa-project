from __future__ import annotations

import numpy as np
import pandas as pd


def test_analyze_roi_discriminative_features_finds_strong_binary_roi():
    from scripts.analyze_roi_discriminative_features import analyze_features

    rows = []
    for label, group in [(0, "CN"), (1, "MCI")]:
        for idx in range(6):
            rows.append(
                {
                    "method": "T1_PLUS_FA",
                    "subject_id": f"sub-{label}{idx}",
                    "group_id": label,
                    "group_name": group,
                    "split": "test",
                    "n_slices": 3,
                    "roi_mean_r0_c0_mean": 0.2 + 0.5 * label + idx * 0.001,
                    "roi_mean_r0_c1_mean": 0.4 + idx * 0.001,
                }
            )
    detail, summary = analyze_features(pd.DataFrame(rows), tasks=["cn_vs_mci"], methods=["T1_PLUS_FA"])

    assert not detail.empty
    assert summary.loc[0, "top_feature"] == "roi_mean_r0_c0_mean"
    assert summary.loc[0, "top_feature_auc"] == 1.0


def test_roi_calibration_moves_prediction_toward_t1_guided_target():
    from sklearn.linear_model import Ridge

    from scripts.apply_roi_calibration import calibrate_image

    model = Ridge(alpha=0.0)
    model.fit(np.asarray([[0.20, 0.0], [0.60, 0.0]], dtype=np.float32), np.asarray([0.30, 0.70], dtype=np.float32))
    models = {f"r{r}_c{c}": model for r in range(2) for c in range(3)}
    pred = np.full((12, 18), 0.20, dtype=np.float32)
    t1 = np.full((12, 18), 0.60, dtype=np.float32)

    calibrated, stats = calibrate_image(
        pred=pred,
        t1=t1,
        models=models,
        roi_rows=2,
        roi_cols=3,
        brain_threshold=0.02,
        gain=0.5,
        max_delta=0.2,
        smooth_sigma=0.0,
    )

    assert float(np.mean(calibrated)) > float(np.mean(pred))
    assert 0.09 <= stats["r0_c0_delta"] <= 0.11
    assert 0.09 <= stats["mean_abs_delta"] <= 0.11
