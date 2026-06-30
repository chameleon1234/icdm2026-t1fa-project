# Final Method Story: A080+DS Full

## 1. Final selected method

The final main method is fixed as **ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST**.

Its design is **frequency-aware sharp FA prior + disease-sensitive high-frequency-preserving correction**. The method first constructs a clear but controlled FA prior from a frequency-aware texture source, then applies a disease-sensitive corrector to improve white-matter and ROI-level medical consistency while preserving high-frequency texture.

## 2. Why not continue new branches

The project has moved from exploration to final validation. We therefore stop opening new Stage1/Stage2 branches and stop tuning GAN, LPIPS, template PriorFlow, T1-start flow, or new hybrid sources. Previous experiments showed that posterior/template priors tend to be smooth, LPIPS/GAN can sharpen images but may introduce local white-matter over-bright or false texture, and overly conservative correction can suppress real detail. The most defensible path is to fix A080+DS Full and complete unified validation.

## 3. Main quantitative results

ADNI final main table:

| Method | PSNR | SSIM | WM_MAE | ROI_CCC | SharpRatio | Revised_Composite_Score |
|---|---|---|---|---|---|---|
| A080+DS Full (Ours) | 28.816 | 0.914 | 0.055 | 0.842 | 1.023 | 0.953 |
| Old Fidelity Flow | 28.092 | 0.905 | 0.057 | 0.867 | 0.889 | 0.853 |
| PriorFlow Safe Probe | 28.068 | 0.894 | 0.057 | 0.843 | 0.980 | 0.758 |
| A080 Base | 28.371 | 0.909 | 0.061 | 0.768 | 0.996 | 0.737 |
| Pix2Pix | 28.011 | 0.901 | 0.061 | 0.827 | 0.913 | 0.711 |
| U-Net | 28.444 | 0.907 | 0.058 | 0.812 | 0.519 | 0.623 |
| CycleGAN | 26.256 | 0.871 | 0.078 | 0.691 | 0.877 | 0.337 |
| DBM | 25.880 | 0.877 | 0.093 | 0.506 | 0.489 | 0.165 |
| DDIM | 25.783 | 0.862 | 0.088 | 0.580 | 0.569 | 0.164 |
| T1_ONLY | 14.052 | 0.748 | 0.317 | 0.051 | 3.616 |  |
| FA_GT |  | 1.000 |  | 1.000 | 1.000 |  |

Private final main table:

| Method | PSNR | SSIM | WM_MAE | ROI_CCC | SharpRatio | Revised_Composite_Score |
|---|---|---|---|---|---|---|
| Private Stage1 Sharp | 27.280 | 0.890 | 0.062 | 0.861 | 0.937 |  |
| Private DS Hybrid (Main) | 28.014 | 0.902 | 0.053 | 0.892 | 1.010 |  |
| Private DS Atlas Probe | 27.071 | 0.886 | 0.061 | 0.828 | 0.655 |  |
| Private DS Hybrid Smoke | 27.089 | 0.885 | 0.061 | 0.830 | 0.722 |  |
| Private Stage1 Sharp Full | 27.491 | 0.892 | 0.062 | 0.835 | 0.821 |  |
| Private Fidelity Flow | 27.581 | 0.892 | 0.059 | 0.867 | 0.869 |  |
| Private Stage1 Baseline | 27.972 | 0.903 | 0.063 | 0.800 | 0.519 |  |
| Private Stage1 LPIPS+GAN | 27.562 | 0.891 | 0.062 | 0.834 | 0.882 |  |

## 4. Why A080+DS Full is selected

A080+DS Full is selected as the final main method because it provides the best overall image-quality balance among reconstruction fidelity, white-matter fidelity, and sharpness preservation, while also providing competitive downstream utility evidence. Compared with A080 base, it improves PSNR/SSIM, reduces WM-MAE, substantially improves ROI-CCC, and keeps SharpRatio close to real FA instead of smoothing the texture away.

This is not a claim that every single metric is the best. It is a best-overall-balance claim.

## 5. Comparison with Old Fidelity Flow

Old Fidelity Flow remains a strong ROI-consistency baseline. Its ROI-CCC is higher than A080+DS Full. However, A080+DS Full provides a better image-quality trade-off across PSNR, SSIM, MAE, WM-MAE, and SharpRatio, while maintaining competitive downstream performance, making it more suitable as the final integrated method.

## 6. Ablation interpretation

Ablation table:

| Method | PSNR | SSIM | WM_MAE | ROI_CCC | SharpRatio | Revised_Composite_Score |
|---|---|---|---|---|---|---|
| LPIPS+GAN sharp Stage1 | 27.831 | 0.902 | 0.065 | 0.803 | 0.914 |  |
| CleanBase / LowGuard | 28.190 | 0.905 | 0.062 | 0.775 | 0.822 |  |
| PriorFlow Safe Probe | 28.068 | 0.894 | 0.057 | 0.843 | 0.980 |  |
| Old Fidelity Flow | 28.092 | 0.905 | 0.057 | 0.867 | 0.889 |  |
| A080 Base | 28.371 | 0.909 | 0.061 | 0.768 | 0.996 |  |
| A080+DS 4096 | 28.679 | 0.913 | 0.057 | 0.828 | 1.023 |  |
| A080+DS Full (Ours) | 28.816 | 0.914 | 0.055 | 0.842 | 1.023 |  |

The ablation supports the final design: a sharp prior alone is insufficient, conservative correction tends to smooth details, and disease-sensitive high-frequency-preserving correction improves WM/ROI consistency without sacrificing sharpness.

## 7. Downstream utility

Downstream classification is used as utility evidence rather than the sole selection criterion. A080+DS Full remains stable under train/test MIL evaluation and does not collapse after image-quality improvement. Some comparison methods may win a single CV task, but they do not provide the same joint balance of reconstruction fidelity, white-matter fidelity, and sharpness.

## 8. Limitations

The generated image should be described as a complementary FA-like representation derived from T1, not as a complete replacement for real DTI/FA. T1 cannot directly recover all microscopic diffusion information. The paper should emphasize complementary representation, white-matter consistency, ROI-level consistency, and downstream utility evidence.

## 9. Paper-ready conclusion

A080+DS Full is selected as the final main method because it provides the best overall balance among reconstruction fidelity, white-matter fidelity, sharpness preservation, and downstream utility. Although Old Fidelity Flow remains strong in ROI-level consistency, A080+DS Full achieves a more favorable overall trade-off for T1-to-FA synthesis.

## Output paths

- ADNI final main table: `outputs\icdm2026\final_selection_a080_ds_full\adni_final_main_table.csv`
- Private final main table: `outputs\icdm2026\final_selection_a080_ds_full\private_final_main_table.csv`
- Ablation table: `outputs\icdm2026\final_selection_a080_ds_full\final_ablation_table.csv`
- Final figures: `outputs\icdm2026\final_selection_a080_ds_full\figures`

## Downstream audit update

A080+DS Full provides downstream utility evidence and maintains competitive classification performance, while its main advantage lies in reconstruction fidelity, white-matter fidelity, and sharpness preservation. This audit confirms that the fair ADNI train/test MIL protocol only includes methods with both train and test prediction folders. U-Net, Pix2Pix, and CycleGAN lack train-full prediction folders and are therefore included only as subject-level test-CV supplements, not as fair MIL main conclusions.

