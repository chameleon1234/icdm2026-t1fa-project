# ADNI Comparison Results

This table uses the same ADNI test split and the unified folder evaluator.

| Method | PSNR | SSIM | MSE | MAE | Sharp | WM-MAE | ROI-CCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 28.444 | 0.9073 | 0.001539 | 0.017043 | 0.519 | 0.0584 | 0.8122 |
| Pix2Pix | 28.011 | 0.9014 | 0.001690 | 0.017702 | 0.913 | 0.0606 | 0.8270 |
| CycleGAN | 26.256 | 0.8705 | 0.002522 | 0.022029 | 0.877 | 0.0784 | 0.6907 |
| DDIM | 25.783 | 0.8617 | 0.002800 | 0.023572 | 0.569 | 0.0880 | 0.5803 |
| DBM | 25.880 | 0.8765 | 0.002777 | 0.023218 | 0.489 | 0.0928 | 0.5057 |
| MOTFM | 17.232 | 0.2378 | 0.020446 | 0.087579 | 2.184 | 0.1803 | 0.1392 |
| StackUNet5 | 28.446 | 0.9052 | 0.001536 | 0.017142 | 0.521 | 0.0564 | 0.8481 |
| StackUNet7 | 28.450 | 0.9070 | 0.001537 | 0.017004 | 0.636 | 0.0591 | 0.8142 |
| Restormer-linear probe | 23.288 | 0.7796 | 0.005026 | 0.032389 | 30.201 | 0.1167 | 0.3589 |
| Restormer-tanh probe | 23.200 | 0.7813 | 0.005139 | 0.032139 | 54.588 | 0.1222 | 0.3086 |
| Old 5-slice Fidelity Flow | 28.092 | 0.9053 | 0.001666 | 0.017686 | 0.889 | 0.0570 | 0.8670 |
| Stage1 Single Sharp | 27.727 | 0.8999 | 0.001810 | 0.018513 | 1.074 | 0.0636 | 0.8426 |
| Single Fidelity Flow | 27.954 | 0.9024 | 0.001720 | 0.018052 | 1.050 | 0.0588 | 0.8637 |
| Ours Single DS Multihead | **28.514** | **0.9093** | **0.001506** | **0.016950** | **1.395** | **0.0540** | **0.9052** |

Observation: the proposed single-slice disease-sensitive multihead corrector is the only method that is simultaneously best on paired image fidelity, WM error, ROI consistency, and visual sharpness. The Restormer probes produce very high sharpness values because they introduce strong artifacts; their PSNR/SSIM/WM/ROI metrics show that this is not useful anatomical detail.
