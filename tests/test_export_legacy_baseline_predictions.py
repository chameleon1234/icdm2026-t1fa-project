import numpy as np
import torch


def test_strip_orig_mod_prefix_removes_compiled_prefix():
    from scripts.export_legacy_baseline_predictions import strip_state_dict_prefix

    state = {
        "_orig_mod.inc.weight": torch.ones(1),
        "_orig_mod.outc.bias": torch.zeros(1),
    }

    stripped = strip_state_dict_prefix(state)

    assert set(stripped) == {"inc.weight", "outc.bias"}


def test_to_gray01_averages_rgb_tensor_from_minus_one_to_one():
    from scripts.export_legacy_baseline_predictions import tensor_to_gray01

    tensor = torch.tensor(
        [[[[ -1.0, 1.0]], [[1.0, -1.0]], [[0.0, 0.0]]]],
        dtype=torch.float32,
    )

    gray = tensor_to_gray01(tensor)

    assert gray.shape == (1, 1, 1, 2)
    np.testing.assert_allclose(gray.numpy(), np.array([[[[0.5, 0.5]]]], dtype=np.float32))


def test_write_png01_round_trips_unicode_path(tmp_path):
    import cv2
    from scripts.export_legacy_baseline_predictions import write_png01

    path = tmp_path / "中文" / "sub-001_z020.png"
    image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    write_png01(path, image)

    stream = np.fromfile(str(path), dtype=np.uint8)
    decoded = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    assert decoded.tolist() == [[0, 128, 255]]


def test_resolve_method_name_defaults_to_checkpoint_stem():
    from scripts.export_legacy_baseline_predictions import resolve_method_name

    assert resolve_method_name("", "cyclegan", "checkpoints_cyclegan/epoch_100.pt") == "CycleGAN_epoch_100"
    assert resolve_method_name("Pix2Pix_E100", "pix2pix", "x.pt") == "Pix2Pix_E100"
