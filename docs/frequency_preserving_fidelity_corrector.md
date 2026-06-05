# Frequency-Preserving Fidelity Corrector

## Purpose

The project no longer asks Stage 2 to reconstruct high-frequency FA texture that a posterior-mean Stage 1 has already discarded.

The new division of responsibility is:

```text
T1 5-slice
  -> sharp perceptual-adversarial Stage 1
  -> frequency-preserving medical fidelity corrector
  -> final FA and downstream utility analysis
```

Stage 1 is frozen and supplies the visible anatomy and texture. Stage 2 may only modify a configured low-frequency band. This lets Stage 2 correct FA intensity, WM error, and regional consistency without regenerating or smoothing Stage 1 texture.

## Architecture

The corrector condition is:

```text
[T1 5-slice, Stage1 FA, lowpass(Stage1 FA), highpass(Stage1 FA)]
```

Both corrector variants use the same NAF-style backbone and frequency composition:

- `flow`: learns a conditional residual-flow trajectory in low-frequency correction space.
- `direct`: directly predicts the low-frequency correction and serves as the required Flow ablation.

The final output is:

```text
final = Stage1 FA + FFT_lowpass(raw_correction)
```

The forbidden high-frequency band is therefore inherited from Stage 1. `HFLeak` measures only energy written into this forbidden band.

## Losses And Selection

Training uses velocity supervision for Flow, low-frequency correction L1, final-image L1/MSE/SSIM, target-derived proxy WM L1, grid-based ROI consistency, correction magnitude, background correction, and high-frequency preservation.

The training ROI loss is a grid-based regional proxy. The final paper metric remains test-set `ROI-CCC`; the proxy must not be described as anatomical ROI ground truth.

A checkpoint becomes `best_fidelity_corrector.pt` only when:

- Sharpness retention is at least `0.97`.
- WM L1 does not degrade.
- ROI proxy error does not degrade.
- PSNR degradation is no worse than `0.03 dB`.
- SSIM degradation is no worse than `0.002`.

## Verified Probe

The 256-slice, 3-epoch probes showed:

| Variant | Sharp Retention | Best Delta WM L1 | Best Delta ROI | Best Delta PSNR | Forbidden HF Leak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flow | 1.0004 | +0.001674 | +0.004872 | -0.0177 | 0.00000002 |
| Direct | 0.9998 | +0.000786 | +0.002036 | -0.0005 | 0.00000166 |

These results validate the architecture-level hypothesis: Stage 2 can improve medical regional errors without removing Stage 1 high-frequency content. Full-data training and test-set ROI-CCC are still required before claiming a final model improvement.

## Formal Training

Activate the GPU environment first:

```powershell
conda activate dinov3test
```

Flow main experiment:

```powershell
python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --run_name pmrf_t1fa_stage2_fidelity_flow_full `
  --corrector_mode flow `
  --epochs 40 `
  --batch_size 2 `
  --width 48 `
  --num_blocks 8 `
  --eval_steps 4 `
  --mixed_precision bf16
```

Direct-corrector ablation:

```powershell
python -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --run_name pmrf_t1fa_stage2_fidelity_direct_full `
  --corrector_mode direct `
  --epochs 40 `
  --batch_size 2 `
  --width 48 `
  --num_blocks 8 `
  --mixed_precision bf16
```

Resume by repeating the original command and adding:

```powershell
--resume outputs/<run_name>/checkpoints/latest_fidelity_corrector.pt
```

Resume validation is strict. Missing or changed architecture, frequency, loss, mask, or selection parameters are rejected.

## Export And Evaluate

```powershell
python scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt `
  --fidelity_corrector_ckpt outputs/pmrf_t1fa_stage2_fidelity_flow_full/checkpoints/best_fidelity_corrector.pt `
  --output_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL `
  --device cuda `
  --batch_size 8

python scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL `
  --method PM_DIRF_FIDELITY_FLOW_FULL `
  --visualize_count 8
```

If no best checkpoint is produced, the run failed the medical-fidelity gate and must not be promoted as the main method.
