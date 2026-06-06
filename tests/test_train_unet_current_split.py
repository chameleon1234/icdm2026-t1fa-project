from pathlib import Path


def test_collect_subjects_from_slice_folder(tmp_path):
    from scripts.train_unet_current_split import collect_subjects_from_slice_dir

    slice_dir = tmp_path / "t1_slices"
    slice_dir.mkdir()
    for name in ["sub-001_z020.png", "sub-001_z021.png", "sub-002_z020.png"]:
        (slice_dir / name).write_bytes(b"x")

    assert collect_subjects_from_slice_dir(slice_dir) == {"sub-001", "sub-002"}


def test_assert_disjoint_subject_splits_rejects_overlap():
    from scripts.train_unet_current_split import assert_disjoint_subject_splits

    try:
        assert_disjoint_subject_splits({"sub-001"}, {"sub-001"}, {"sub-002"})
    except ValueError as exc:
        assert "train/val" in str(exc)
    else:
        raise AssertionError("Expected overlap to be rejected")


def test_assert_disjoint_subject_splits_accepts_current_split():
    from scripts.train_unet_current_split import assert_disjoint_subject_splits

    assert_disjoint_subject_splits({"sub-001"}, {"sub-002"}, {"sub-003"})
