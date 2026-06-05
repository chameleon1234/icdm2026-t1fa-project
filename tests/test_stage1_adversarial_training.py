import torch


def test_stage1_patch_discriminator_preserves_batch_and_outputs_patch_map():
    from pmrf_t1fa.train_pmrf_t1fa_stage1 import Stage1PatchDiscriminator

    disc = Stage1PatchDiscriminator(in_channels=6, width=8)
    condition = torch.rand((2, 5, 32, 32))
    image = torch.rand((2, 1, 32, 32))

    logits = disc(condition, image)

    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
    assert logits.shape[-1] < image.shape[-1]
    assert logits.shape[-2] < image.shape[-2]


def test_lsgan_losses_are_small_for_correct_logits():
    from pmrf_t1fa.train_pmrf_t1fa_stage1 import lsgan_discriminator_loss, lsgan_generator_loss

    real_logits = torch.ones((2, 1, 4, 4))
    fake_logits = torch.zeros((2, 1, 4, 4))

    disc_loss = lsgan_discriminator_loss(real_logits, fake_logits)
    gen_loss = lsgan_generator_loss(real_logits)

    assert torch.allclose(disc_loss, torch.zeros_like(disc_loss))
    assert torch.allclose(gen_loss, torch.zeros_like(gen_loss))


def test_sharp_adversarial_preset_enables_lpips_and_gan():
    from pmrf_t1fa.train_pmrf_t1fa_stage1 import apply_stage1_training_preset

    class Args:
        stage1_training_preset = "sharp_adversarial"
        stage1_model_variant = "single"
        stage1_detail_scale = 0.0
        lpips_max_weight = 0.03
        adv_weight = 0.0
        disc_lr = 2e-5
        best_metric = "paired"
        paired_sharp_weight = 6.0
        hf_weight = 0.02
        detail_hf_weight = 0.0
        detail_lap_weight = 0.0
        grad_weight = 0.05
        wm_l1_weight = 0.0
        wm_grad_weight = 0.0
        roi_consistency_weight = 0.0

    args = apply_stage1_training_preset(Args())

    assert args.stage1_model_variant == "detail"
    assert args.lpips_max_weight >= 0.1
    assert args.adv_weight >= 0.01
    assert args.best_metric == "detail_paired"


def test_sharp_adversarial_preset_preserves_existing_reconstruction_weights():
    from pmrf_t1fa.train_pmrf_t1fa_stage1 import apply_stage1_training_preset

    class Args:
        stage1_training_preset = "sharp_adversarial"
        stage1_model_variant = "detail"
        stage1_detail_scale = 0.4
        lpips_max_weight = 0.0
        adv_weight = 0.0
        disc_lr = 1e-5
        best_metric = "paired"
        paired_sharp_weight = 6.0
        hf_weight = 0.10
        detail_hf_weight = 0.20
        detail_lap_weight = 0.12
        grad_weight = 0.12
        wm_l1_weight = 0.16
        wm_grad_weight = 0.08
        roi_consistency_weight = 0.04

    args = apply_stage1_training_preset(Args())

    assert args.stage1_detail_scale == 0.4
    assert args.hf_weight == 0.10
    assert args.detail_hf_weight == 0.20
    assert args.detail_lap_weight == 0.12
    assert args.grad_weight == 0.12
    assert args.wm_l1_weight == 0.16
    assert args.wm_grad_weight == 0.08
    assert args.roi_consistency_weight == 0.04


def test_add_stage1_adversarial_checkpoint_state_includes_disc_when_enabled():
    from pmrf_t1fa.train_pmrf_t1fa_stage1 import (
        Stage1PatchDiscriminator,
        add_stage1_adversarial_checkpoint_state,
    )

    disc = Stage1PatchDiscriminator(in_channels=6, width=8)
    optimizer = torch.optim.AdamW(disc.parameters(), lr=2e-5)
    checkpoint = {"model": {}, "optimizer": {}}

    add_stage1_adversarial_checkpoint_state(checkpoint, disc, optimizer)

    assert "discriminator" in checkpoint
    assert "disc_optimizer" in checkpoint
    assert checkpoint["discriminator"]
    assert checkpoint["disc_optimizer"]


def test_add_stage1_adversarial_checkpoint_state_skips_when_disabled():
    from pmrf_t1fa.train_pmrf_t1fa_stage1 import add_stage1_adversarial_checkpoint_state

    checkpoint = {"model": {}, "optimizer": {}}

    add_stage1_adversarial_checkpoint_state(checkpoint, None, None)

    assert "discriminator" not in checkpoint
    assert "disc_optimizer" not in checkpoint
