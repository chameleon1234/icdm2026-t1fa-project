import torch


def test_stack_unet_accepts_five_slice_input_and_outputs_single_fa_channel():
    from scripts.train_stack_unet_current_split import StackUNet

    model = StackUNet(in_channels=5, out_channels=1, base_channels=8)
    output = model(torch.randn(2, 5, 64, 64))

    assert output.shape == (2, 1, 64, 64)
    assert output.min().item() >= -1.0
    assert output.max().item() <= 1.0


def test_stack_unet_uses_stack_dataset_for_context_slices(tmp_path):
    from scripts.train_stack_unet_current_split import build_dataset
    from src.data.t1fa_stack_dataset import T1FAStackDataset

    t1_dir = tmp_path / "t1"
    fa_dir = tmp_path / "fa"
    t1_dir.mkdir()
    fa_dir.mkdir()
    for name in ["sub-001_z020.png", "sub-001_z021.png"]:
        (t1_dir / name).write_bytes(b"x")
        (fa_dir / name).write_bytes(b"x")

    dataset = build_dataset(t1_dir, fa_dir, context_slices=5)

    assert isinstance(dataset, T1FAStackDataset)
    assert dataset.context_slices == 5
