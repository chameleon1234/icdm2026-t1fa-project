import numpy as np
import pandas as pd

from scripts.sweep_task_sensitive_roi_corrections import (
    CorrectionSpec,
    _apply_roi_shift,
    _compute_shifts,
)


def _feature_row(subject_id: str, group: str, values: list[float], split: str = "train") -> dict:
    row = {
        "method": "FA_GT",
        "subject_id": subject_id,
        "group_name": group,
        "group_id": {"CN": 1, "SCD": 2, "MCI": 3, "AD": 4}[group],
        "split": split,
    }
    for idx, value in enumerate(values):
        row[f"roi_mean_r{idx // 3}_c{idx % 3}_mean"] = value
    return row


def test_apply_roi_shift_changes_only_masked_pixels():
    image = np.full((4, 6), 0.5, dtype=np.float32)
    image[0, 0] = 0.0
    shift = np.array([0.1, 0.0, 0.0, 0.0, -0.2, 0.0], dtype=np.float32)

    corrected = _apply_roi_shift(image, shift, mask_threshold=0.02)

    assert corrected[0, 0] == 0.0
    assert np.isclose(corrected[1, 1], 0.6)
    assert np.isclose(corrected[3, 3], 0.3)
    assert np.isclose(corrected[0, 3], 0.5)


def test_contrast_shift_uses_task_direction_and_top_k():
    train = pd.DataFrame(
        [
            _feature_row("sub-001", "CN", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            _feature_row("sub-002", "CN", [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]),
            _feature_row("sub-003", "AD", [0.8, 0.1, 0.5, 0.1, 0.1, 0.1]),
            _feature_row("sub-004", "AD", [0.9, 0.1, 0.6, 0.1, 0.1, 0.1]),
        ]
    )
    test = pd.DataFrame([_feature_row("sub-101", "CN", [0.7, 0.1, 0.4, 0.1, 0.1, 0.1], split="test")])

    shifts = _compute_shifts(
        train,
        test,
        CorrectionSpec(name="probe", mode="contrast_amplify", alpha=0.5, top_k=1, task="cn_vs_ad"),
        max_abs_shift=1.0,
    )

    shift = shifts["sub-101"]
    assert np.count_nonzero(np.abs(shift) > 1e-8) == 1
    assert shift[0] > 0.0


def test_prototype_shift_moves_toward_predicted_class_prototype():
    train = pd.DataFrame(
        [
            _feature_row("sub-001", "CN", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            _feature_row("sub-002", "CN", [0.2, 0.1, 0.1, 0.1, 0.1, 0.1]),
            _feature_row("sub-003", "AD", [0.8, 0.1, 0.5, 0.1, 0.1, 0.1]),
            _feature_row("sub-004", "AD", [0.9, 0.1, 0.6, 0.1, 0.1, 0.1]),
        ]
    )
    test = pd.DataFrame([_feature_row("sub-101", "CN", [0.7, 0.1, 0.4, 0.1, 0.1, 0.1], split="test")])

    shifts = _compute_shifts(
        train,
        test,
        CorrectionSpec(name="probe", mode="prototype_pull", alpha=0.5, task="cn_vs_ad"),
        max_abs_shift=1.0,
    )

    assert shifts["sub-101"][0] > 0.0
    assert shifts["sub-101"][2] > 0.0
