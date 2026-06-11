import numpy as np
import pandas as pd
import torch


def test_build_disease_roi_weights_selects_discriminative_roi():
    from src.eval.disease_sensitive_roi import build_disease_roi_weights

    rows = []
    for group_name, value in [("CN", 0.20), ("MCI", 0.80)]:
        for idx in range(6):
            rows.append(
                {
                    "method": "FA_GT",
                    "subject_id": f"{group_name}-{idx}",
                    "group_name": group_name,
                    "roi_mean_r0_c0_mean": value + idx * 0.001,
                    "roi_mean_r0_c1_mean": 0.50 + idx * 0.001,
                }
            )
    weights = build_disease_roi_weights(pd.DataFrame(rows), tasks=["cn_vs_mci"], method="FA_GT")

    strong = weights[weights["roi_key"].eq("r0_c0")].iloc[0]
    weak = weights[weights["roi_key"].eq("r0_c1")].iloc[0]
    assert strong["weight"] > 0.95
    assert weak["weight"] < 0.05


def test_weighted_roi_l1_uses_high_weight_regions_more():
    from src.eval.disease_sensitive_roi import weighted_grid_roi_l1

    pred = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target = torch.zeros_like(pred)
    target[:, :, :4, :4] = 1.0
    target[:, :, 4:, 4:] = 1.0
    mask = torch.ones_like(pred)
    weights = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

    loss = weighted_grid_roi_l1(pred, target, mask, weights, min_pixels=1)

    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


def test_roi_weight_tensor_respects_requested_grid_shape():
    from src.eval.disease_sensitive_roi import roi_weight_tensor_from_frame

    frame = pd.DataFrame(
        [
            {"roi_key": "r0_c0", "row": 0, "col": 0, "weight": 0.8},
            {"roi_key": "r1_c2", "row": 1, "col": 2, "weight": 0.4},
        ]
    )
    tensor = roi_weight_tensor_from_frame(frame, roi_rows=2, roi_cols=3)

    assert tuple(tensor.shape) == (2, 3)
    assert np.isclose(float(tensor[0, 0]), 0.8)
    assert np.isclose(float(tensor[1, 2]), 0.4)
