from pathlib import Path

import pytest
import torch


def test_pix2pix_current_split_subject_leakage_detection(tmp_path: Path):
    from scripts.train_pix2pix_current_split import collect_subjects_from_slice_dir, assert_disjoint_subject_splits

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    test_dir = tmp_path / "test"
    for directory in (train_dir, val_dir, test_dir):
        directory.mkdir()

    (train_dir / "sub-001_z020.png").write_bytes(b"")
    (val_dir / "sub-002_z020.png").write_bytes(b"")
    (test_dir / "sub-003_z020.png").write_bytes(b"")

    assert collect_subjects_from_slice_dir(train_dir) == {"sub-001"}
    assert_disjoint_subject_splits(
        collect_subjects_from_slice_dir(train_dir),
        collect_subjects_from_slice_dir(val_dir),
        collect_subjects_from_slice_dir(test_dir),
    )

    (test_dir / "sub-001_z021.png").write_bytes(b"")
    with pytest.raises(ValueError, match="Subject leakage"):
        assert_disjoint_subject_splits(
            collect_subjects_from_slice_dir(train_dir),
            collect_subjects_from_slice_dir(val_dir),
            collect_subjects_from_slice_dir(test_dir),
        )


def test_pix2pix_models_forward_shapes():
    from scripts.train_pix2pix_current_split import ConditionalDiscriminator, ResNetGenerator

    generator = ResNetGenerator(input_nc=1, output_nc=1, ngf=8, n_blocks=1)
    discriminator = ConditionalDiscriminator(input_nc=2, ndf=8, n_layers=2)

    t1 = torch.randn(2, 1, 64, 64)
    fake_fa = generator(t1)
    pred = discriminator(torch.cat([t1, fake_fa], dim=1))

    assert fake_fa.shape == (2, 1, 64, 64)
    assert pred.shape[0] == 2
    assert pred.shape[1] == 1


def test_pix2pix_parse_args_defaults():
    from scripts.train_pix2pix_current_split import parse_args

    args = parse_args([])

    assert args.train_t1_dir == "data/processed/train/t1_slices"
    assert args.val_t1_dir == "data/processed/val/t1_slices"
    assert args.run_name == "pix2pix_current_split_e100"
    assert args.lambda_l1 == 100.0
