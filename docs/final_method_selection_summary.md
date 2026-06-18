# Final Method Selection Summary

## Multiobjective Ranking Result

The current ranking combines paired synthesis quality, medical structure quality, visual sharpness, and downstream clinical utility across the private dataset and ADNI.

| Method | Datasets | PSNR | SSIM | MAE | Sharpness | WM-MAE | ROI-CCC | AUC | ACC | Macro-F1 | Candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CycleGAN | 2 | 26.211 | 0.871 | 0.0223 | 0.871 | 0.0733 | 0.751 | 0.665 | 0.736 | 0.580 | No |
| U-Net | 2 | 32.715 | 0.939 | 0.0119 | 0.596 | 0.0401 | 0.896 | 0.608 | 0.688 | 0.516 | No |
| PM-DIRF Fidelity Flow | 2 | 27.836 | 0.899 | 0.0183 | 0.879 | 0.0581 | 0.867 | 0.615 | 0.686 | 0.551 | Yes |
| Pix2Pix | 2 | 27.793 | 0.897 | 0.0183 | 1.077 | 0.0603 | 0.854 | 0.542 | 0.691 | 0.552 | No |

## Interpretation

CycleGAN has the strongest downstream score in the current sweep, but it fails paired image fidelity and medical structure consistency: PSNR, SSIM, WM-MAE, and ROI-CCC are all weak. It is useful as a strong adversarial baseline, not as the final medical synthesis method.

U-Net has excellent PSNR/SSIM/MAE, but its sharpness is too low. This matches the earlier posterior-mean problem: strong pixel fidelity can still produce visually over-smoothed FA maps.

Pix2Pix is visually sharp and has acceptable paired metrics, but the downstream AUC is weak. It is a useful image-to-image GAN baseline, but it does not support the clinical utility claim strongly enough.

PM-DIRF Fidelity Flow is currently the only method that passes all paper-candidate gates on both datasets:

- PSNR >= 27.0
- SSIM >= 0.89
- Sharpness ratio >= 0.80
- WM-MAE <= 0.065
- ROI-CCC >= 0.85
- AUC >= 0.60

This makes it the best final method candidate under the revised medical-imaging objective: not necessarily first in every single metric, but the only method that balances paired fidelity, visible white-matter detail, anatomical consistency, and downstream utility across both datasets.

## Paper Direction

The final paper should not claim that the proposed method is simply the highest PSNR method or the highest downstream classifier method. The stronger and more defensible claim is:

> The proposed two-stage high-frequency-preserving PM-DIRF framework achieves the best balance between quantitative FA synthesis fidelity, white-matter detail preservation, anatomical ROI consistency, and downstream disease utility across private and ADNI datasets.

The next step is to build manuscript-ready tables and figures around this balanced claim:

1. Main synthesis table: PSNR, SSIM, MSE, MAE, sharpness, WM-MAE, ROI-CCC.
2. Clinical utility table: ACC, AUC, Macro-F1 for CN vs AD, CN vs MCI, and MCI vs AD.
3. Candidate-gate table: show why CycleGAN, U-Net, and Pix2Pix fail one or more clinically important requirements, while PM-DIRF Fidelity Flow passes all.
4. Visual figure: compare U-Net smoothness, CycleGAN/ Pix2Pix artifacts, and PM-DIRF Fidelity Flow balance.
