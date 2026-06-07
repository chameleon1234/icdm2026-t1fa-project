from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def test_subject_slice_filename_preserves_subject_for_downstream_grouping():
    from scripts.preprocess_adni_slices import make_slice_filename

    name = make_slice_filename("002_S_0413", 20)

    assert name == "sub-002_S_0413_z020.png"
    assert name.split("_z")[0] == "sub-002_S_0413"


def test_select_slice_indices_clips_requested_range_to_volume_shape():
    from scripts.preprocess_adni_slices import select_slice_indices

    indices = select_slice_indices((91, 109, 91), axis=2, slice_start=20, slice_end=95)

    assert indices[0] == 20
    assert indices[-1] == 90
    assert len(indices) == 71


def test_normalization_and_resize_outputs_uint8_png_ready_images():
    from scripts.preprocess_adni_slices import normalize_fa_slice, normalize_t1_slice, resize_and_pad_uint8

    t1 = np.zeros((5, 7), dtype=np.float32)
    t1[1:4, 2:6] = np.linspace(10, 120, 12, dtype=np.float32).reshape(3, 4)
    fa = np.array([[-0.1, 0.0, 0.5], [0.8, 1.2, np.nan]], dtype=np.float32)

    t1_norm = normalize_t1_slice(t1)
    fa_norm = normalize_fa_slice(fa)
    t1_png = resize_and_pad_uint8(t1_norm, 16)
    fa_png = resize_and_pad_uint8(fa_norm, 16)

    assert t1_norm.min() >= 0.0
    assert t1_norm.max() <= 1.0
    assert fa_norm.min() >= 0.0
    assert fa_norm.max() <= 1.0
    assert t1_png.shape == (16, 16)
    assert fa_png.shape == (16, 16)
    assert t1_png.dtype == np.uint8
    assert fa_png.dtype == np.uint8


def test_write_subject_slice_pairs_creates_matching_t1_fa_pngs(tmp_path):
    from scripts.preprocess_adni_slices import write_subject_slice_pairs

    shape = (8, 10, 6)
    t1 = np.zeros(shape, dtype=np.float32)
    fa = np.zeros(shape, dtype=np.float32)
    t1[:, :, 2:4] = 50.0
    fa[:, :, 2:4] = 0.4
    manifest = pd.DataFrame(
        [
            {
                "subject": "002_S_0413",
                "split": "train",
                "normalized_group": "CN",
            }
        ]
    )

    rows = write_subject_slice_pairs(
        subject_row=manifest.iloc[0],
        t1_volume=t1,
        fa_volume=fa,
        output_root=tmp_path,
        axis=2,
        slice_start=2,
        slice_end=4,
        target_size=16,
        min_brain_fraction=0.0,
    )

    assert [row["filename"] for row in rows] == ["sub-002_S_0413_z002.png", "sub-002_S_0413_z003.png"]
    for row in rows:
        t1_path = tmp_path / "train" / "t1_slices" / row["filename"]
        fa_path = tmp_path / "train" / "fa_slices" / row["filename"]
        assert t1_path.exists()
        assert fa_path.exists()
        assert cv2.imread(str(t1_path), cv2.IMREAD_GRAYSCALE).shape == (16, 16)
        assert cv2.imread(str(fa_path), cv2.IMREAD_GRAYSCALE).shape == (16, 16)
