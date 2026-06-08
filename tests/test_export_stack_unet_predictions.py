import torch


def test_load_stack_unet_checkpoint_reads_training_summary(tmp_path):
    from scripts.export_stack_unet_predictions import load_stack_unet_model
    from scripts.train_stack_unet_current_split import StackUNet

    run_dir = tmp_path / "stack_run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    model = StackUNet(in_channels=5, out_channels=1, base_channels=8)
    ckpt = ckpt_dir / "best_stack_unet.pt"
    torch.save(model.state_dict(), ckpt)
    (run_dir / "train_summary.json").write_text(
        '{"args": {"context_slices": 5, "width": 8}}',
        encoding="utf-8",
    )

    loaded, meta = load_stack_unet_model(ckpt, torch.device("cpu"), context_slices=0, width=0)

    assert meta["context_slices"] == 5
    assert meta["width"] == 8
    assert loaded(torch.randn(1, 5, 32, 32)).shape == (1, 1, 32, 32)
