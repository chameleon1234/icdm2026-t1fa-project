from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_t1_highpass_injection import alpha_to_tag, inject_t1_highpass, parse_alpha_list


def test_parse_alpha_list_accepts_comma_separated_values():
    assert parse_alpha_list("0.2,0.4, 0.6") == [0.2, 0.4, 0.6]


def test_alpha_to_tag_is_filesystem_friendly():
    assert alpha_to_tag(0.2) == "0p2"
    assert alpha_to_tag(1.25) == "1p25"


def test_t1_highpass_injection_changes_edges_but_preserves_shape_and_range():
    pred = torch.full((1, 1, 16, 16), 0.5)
    t1 = torch.zeros((1, 1, 16, 16))
    t1[:, :, :, 8:] = 1.0

    injected, highpass = inject_t1_highpass(pred, t1, alpha=0.5, sigma=1.0)

    assert injected.shape == pred.shape
    assert highpass.shape == pred.shape
    assert torch.all(injected >= 0.0)
    assert torch.all(injected <= 1.0)
    assert not torch.allclose(injected, pred)
    assert torch.abs(highpass).max() > 0.0


def test_t1_highpass_injection_alpha_zero_is_identity():
    pred = torch.rand((1, 1, 8, 8))
    t1 = torch.rand((1, 1, 8, 8))

    injected, _ = inject_t1_highpass(pred, t1, alpha=0.0, sigma=1.0)

    assert torch.allclose(injected, pred)
