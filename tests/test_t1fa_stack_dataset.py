from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _write_gray_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 6), value, dtype=np.uint8), mode="L").save(path)


def _to_norm(value: int) -> float:
    return value / 127.5 - 1.0


def test_stack_dataset_returns_center_target_with_clamped_context(tmp_path):
    from src.data.t1fa_stack_dataset import T1FAStackDataset

    t1_dir = tmp_path / "t1"
    fa_dir = tmp_path / "fa"
    for z, value in enumerate([10, 20, 30]):
        name = f"sub-001_z{z:03d}.png"
        _write_gray_png(t1_dir / name, value)
        _write_gray_png(fa_dir / name, 100 + z)

    dataset = T1FAStackDataset(t1_dir, fa_dir, context_slices=3, target_size=(4, 4))

    edge = dataset[0]
    middle = dataset[1]

    assert edge["fname"] == "sub-001_z000.png"
    assert edge["t1_slice"].shape == torch.Size([3, 4, 4])
    assert edge["fa_slice"].shape == torch.Size([1, 4, 4])
    assert [round(float(edge["t1_slice"][i, 0, 0]), 4) for i in range(3)] == [
        round(_to_norm(10), 4),
        round(_to_norm(10), 4),
        round(_to_norm(20), 4),
    ]
    assert [round(float(middle["t1_slice"][i, 0, 0]), 4) for i in range(3)] == [
        round(_to_norm(10), 4),
        round(_to_norm(20), 4),
        round(_to_norm(30), 4),
    ]


def test_predict_stage1_residual_uses_center_slice_for_stacked_input():
    from pmrf_t1fa.models.pmrf_t1fa import predict_stage1_fa

    class ZeroResidual(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))

    t1_stack = torch.tensor([[[[-0.5]], [[0.25]], [[0.75]]]])

    coarse = predict_stage1_fa(ZeroResidual(), t1_stack, prediction_mode="residual")

    assert coarse.shape == torch.Size([1, 1, 1, 1])
    assert coarse.item() == 0.25
