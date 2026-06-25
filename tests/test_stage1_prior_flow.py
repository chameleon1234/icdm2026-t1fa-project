import torch


def test_prior_flow_state_uses_template_to_target_line():
    from pmrf_t1fa.models.pmrf_t1fa import build_prior_flow_state

    template = torch.zeros((2, 1, 8, 8))
    target = torch.ones((2, 1, 8, 8))
    t = torch.tensor([0.25, 0.75])

    x_t, v_target = build_prior_flow_state(template, target, t)

    assert torch.allclose(x_t[0], torch.full((1, 8, 8), 0.25))
    assert torch.allclose(x_t[1], torch.full((1, 8, 8), 0.75))
    assert torch.allclose(v_target, torch.ones_like(v_target))


def test_template_prior_flow_net_preserves_spatial_shape():
    from pmrf_t1fa.models.pmrf_t1fa import TemplatePriorFlowNet

    model = TemplatePriorFlowNet(t1_channels=3, width=8, num_blocks=2)
    x_t = torch.rand((2, 1, 32, 32))
    template = torch.rand((2, 1, 32, 32))
    t1 = torch.rand((2, 3, 32, 32))
    wm_prob = torch.rand((2, 1, 32, 32))
    z_map = torch.rand((2, 1, 32, 32))
    t = torch.tensor([0.1, 0.9])

    velocity = model(x_t, template, t1, wm_prob, z_map, t)

    assert velocity.shape == (2, 1, 32, 32)


def test_prior_flow_euler_constant_velocity_reaches_target():
    from pmrf_t1fa.models.pmrf_t1fa import euler_sample_prior_flow

    class ConstantVelocity(torch.nn.Module):
        def forward(self, x_t, template, t1, wm_prob, z_map, t):
            return torch.ones_like(x_t) * 0.5

    template = torch.zeros((1, 1, 8, 8))
    t1 = torch.zeros((1, 1, 8, 8))
    wm_prob = torch.ones((1, 1, 8, 8))
    z_map = torch.zeros((1, 1, 8, 8))

    pred = euler_sample_prior_flow(ConstantVelocity(), template, t1, wm_prob, z_map, steps=4)

    assert torch.allclose(pred, torch.full_like(pred, 0.5), atol=1e-6)


def test_prepare_batch_t1_source_uses_center_t1_channel():
    from pmrf_t1fa.train_pmrf_t1fa_stage1_prior_flow import prepare_batch

    batch = {
        "t1_slice": torch.tensor(
            [
                [
                    [[-1.0, -0.5], [0.0, 0.5]],
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.0], [-0.5, -1.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        "fa_slice": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        "slice_id": torch.tensor([42]),
        "fname": ["sub-001_z042.png"],
    }
    template = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    t1_img, _target, x0, _wm_prob, _z_map = prepare_batch(
        batch,
        torch.device("cpu"),
        context_slices=3,
        template_by_slice={42: template},
        default_template=template,
        z_min=42,
        z_max=42,
        flow_source="t1",
    )

    assert torch.allclose(x0, t1_img[:, 1:2])


def test_export_prior_flow_parser_accepts_t1_source(monkeypatch):
    from scripts.export_prior_flow_stage1_predictions import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_prior_flow_stage1_predictions.py",
            "--ckpt",
            "checkpoint.pt",
            "--output_dir",
            "predictions",
            "--flow_source",
            "t1",
        ],
    )

    args = parse_args()

    assert args.flow_source == "t1"
