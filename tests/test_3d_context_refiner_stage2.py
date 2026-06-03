import torch

from pmrf_t1fa.train_pmrf_t1fa_stage2_3d_context_refiner import (
    build_3d_context_refiner_input,
    make_neighbor_stage1_inputs,
)


def test_make_neighbor_stage1_inputs_builds_left_center_right_windows():
    t1_stack = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)

    windows = make_neighbor_stage1_inputs(t1_stack)

    assert windows.shape == (1, 3, 5, 1, 1)
    assert torch.equal(windows[0, 0, :, 0, 0], torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(windows[0, 1, :, 0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(windows[0, 2, :, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0]))


def test_build_3d_context_refiner_input_uses_t1_neighbor_coarse_and_edges():
    t1_stack = torch.rand((2, 5, 16, 16))
    coarse_triplet = torch.rand((2, 3, 16, 16))

    model_input = build_3d_context_refiner_input(t1_stack, coarse_triplet)

    assert model_input.shape == (2, 10, 16, 16)
