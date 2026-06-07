from pathlib import Path

import pytest
import torch


def test_cyclegan_current_split_subject_leakage_detection(tmp_path: Path):
    from scripts.train_cyclegan_current_split import collect_subjects_from_slice_dir, assert_disjoint_subject_splits

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    test_dir = tmp_path / "test"
    for directory in (train_dir, val_dir, test_dir):
        directory.mkdir()

    (train_dir / "sub-101_z020.png").write_bytes(b"")
    (val_dir / "sub-102_z020.png").write_bytes(b"")
    (test_dir / "sub-103_z020.png").write_bytes(b"")

    assert collect_subjects_from_slice_dir(train_dir) == {"sub-101"}
    assert_disjoint_subject_splits(
        collect_subjects_from_slice_dir(train_dir),
        collect_subjects_from_slice_dir(val_dir),
        collect_subjects_from_slice_dir(test_dir),
    )

    (val_dir / "sub-101_z030.png").write_bytes(b"")
    with pytest.raises(ValueError, match="Subject leakage"):
        assert_disjoint_subject_splits(
            collect_subjects_from_slice_dir(train_dir),
            collect_subjects_from_slice_dir(val_dir),
            collect_subjects_from_slice_dir(test_dir),
        )


def test_cyclegan_models_forward_shapes():
    from scripts.train_cyclegan_current_split import Discriminator, ResNetGenerator

    generator = ResNetGenerator(input_nc=1, output_nc=1, ngf=8, n_blocks=1)
    discriminator = Discriminator(input_nc=1, ndf=8, n_layers=2)

    x = torch.randn(2, 1, 64, 64)
    y = generator(x)
    pred = discriminator(y)

    assert y.shape == (2, 1, 64, 64)
    assert pred.shape[0] == 2
    assert pred.shape[1] == 1


def test_cyclegan_parse_args_defaults():
    from scripts.train_cyclegan_current_split import parse_args

    args = parse_args([])

    assert args.train_t1_dir == "data/processed/train/t1_slices"
    assert args.val_t1_dir == "data/processed/val/t1_slices"
    assert args.run_name == "cyclegan_current_split_e100"
    assert args.lambda_cycle == 10.0
    assert args.lambda_identity == 5.0
