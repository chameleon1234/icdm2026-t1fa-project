# Comparison Baseline Experiment Design

## Goal

All comparison methods must be converted to the same test-set PNG format and evaluated by the same scripts:

1. Export predictions to `outputs/icdm2026/predictions/<METHOD>`.
2. Run `scripts/evaluate_method_folder.py` for PSNR, SSIM, MSE, MAE, sharpness, WM-MAE, ROI-CCC, and fixed-slice visualization.
3. Run `scripts/evaluate_downstream_classification.py` for repeated subject-level disease utility.

This avoids mixing FID-style natural-image evaluation with paired medical-image evaluation.

## Implemented Baselines

| Category | Method | Checkpoint | Rationale |
|---|---|---|---|
| CNN | U-Net | `outputs/unet_training/checkpoints/unet_epoch_0099.pth` | Deterministic supervised regression baseline. |
| GAN | Pix2Pix | `checkpoints_pix2pix/epoch_100.pt` | Paired adversarial image-to-image translation. |
| GAN | CycleGAN | `checkpoints_cyclegan/epoch_100.pt` | Unpaired adversarial translation baseline. |
| Diffusion | DDIM | `checkpoints_baseline_ddim_steps1/epoch_100.pt` | Iterative conditional diffusion sampling, exported with 50 DDIM steps. |
| Flow | DIRF V5 | Existing prediction folder | Legacy flow baseline. |
| PMRF | PM Stage1 / PM-DIRF family | Existing prediction folders | Posterior-mean and rectified-flow family motivated by `papers/PMRF.pdf`. |
| Ours | Frequency-preserving fidelity variants | Existing prediction folders | Current balanced route for sharpness, fidelity, and downstream utility. |

## Current Command Pattern

```powershell
conda activate dinov3test

python scripts/export_legacy_baseline_predictions.py --model_type cyclegan --ckpt checkpoints_cyclegan/epoch_100.pt --output_dir outputs/icdm2026/predictions/CycleGAN_E100 --method CycleGAN_E100 --device cuda --batch_size 16
python scripts/export_legacy_baseline_predictions.py --model_type pix2pix --ckpt checkpoints_pix2pix/epoch_100.pt --output_dir outputs/icdm2026/predictions/Pix2Pix_E100 --method Pix2Pix_E100 --device cuda --batch_size 16
python scripts/export_legacy_baseline_predictions.py --model_type unet --ckpt outputs/unet_training/checkpoints/unet_epoch_0099.pth --output_dir outputs/icdm2026/predictions/UNet_E99 --method UNet_E99 --device cuda --batch_size 16
python scripts/export_legacy_baseline_predictions.py --model_type ddim --ckpt checkpoints_baseline_ddim_steps1/epoch_100.pt --output_dir outputs/icdm2026/predictions/DDIM_E100_K50 --method DDIM_E100_K50 --device cuda --batch_size 8 --sample_steps 50
```

Then run the unified image and downstream evaluation scripts on the exported folders.

## Second-Batch Baselines

The LDM/Diffusion Bridge checkpoint exists under `outputs/ldm_bridge_training_vgg15/checkpoints`, but there is no stable export script in the current repository. It should be handled as a separate second-batch baseline after implementing and validating a VAE+LDM sampler. It should not be mixed into the table until it produces the same 1900 test PNGs and passes the same evaluation pipeline.
