# Results Section Draft

## ADNI Image Quality

On ADNI, A080+DS Full achieves a strong balance between reconstruction fidelity and medical consistency. Compared with Old Fidelity Flow and A080 Base, it maintains competitive PSNR/SSIM while improving the balance among white-matter error, ROI consistency, and texture realism. Pix2Pix, U-Net, and CycleGAN remain competitive on some individual metrics, so the manuscript should not claim that A080+DS Full is the best on every single metric.

## ADNI Downstream Utility

Under strict subject-level train/test downstream evaluation, A080+DS Full achieves the highest average Macro-AUC (0.7226) among the ADNI fair downstream methods. A080 Base remains competitive for Accuracy and Macro-F1, suggesting a task-dependent trade-off between high-frequency texture and medical consistency.

## Private Dataset Image Quality

On the private dataset, Private DS Hybrid provides stable image quality, white-matter fidelity, ROI consistency, and sharpness preservation. However, this does not mean it dominates all downstream tasks. The result supports synthetic FA as a complementary representation rather than a complete replacement for T1.

## Private Dataset Downstream Utility

The private fair downstream results show that T1_ONLY has the highest average Macro-AUC, while FidelityFlow is strong in average Accuracy. PRIVATE_DS_HYBRID remains competitive but is not universally best. This indicates that private-dataset downstream signal may rely strongly on original T1 structure or low-frequency statistics.

## Integrated Composite Result

The primary integrated composite score uses fixed weights: 25% reconstruction fidelity, 30% medical fidelity, 20% texture realism, and 25% downstream utility. A080+DS Full is selected as the final method because it achieves the best overall integrated balance across reconstruction fidelity, white-matter/ROI medical fidelity, texture realism, and downstream utility, rather than because it is the top method on every individual metric.

## Ablation Interpretation

The ablation results suggest that simply increasing LPIPS or GAN pressure does not reliably recover realistic FA texture; aggressive sharpening can introduce false white-matter texture or local over-bright artifacts. A080+DS is valuable because A080 preserves stable FA-space high-frequency information and the disease-sensitive corrector improves medical consistency.

## Safe Conclusion

The final conclusion should state that A080+DS Full provides dual-dataset reconstruction evidence and ADNI downstream utility evidence, while private downstream results suggest synthetic FA should be interpreted as a complementary representation to T1. Do not claim that generated FA fully replaces T1 or that the method is first on every metric.


## 2026-07-01 Dual-Dataset Fair Full-Heavy Downstream Closure

No new image-generation model was trained in this update. Missing train-full prediction folders were completed where needed, and the private dataset was re-audited using fair subject-level train/test downstream evaluation. ADNI uses the latest `outputs/icdm2026/downstream_adni_fair_full_all_methods/method_average_summary.csv`, while the private dataset uses `outputs/icdm2026/downstream_private_fair_full_heavy_all_methods/method_average_summary.csv`.

On ADNI, `ADNI_A080_DS_FULL` achieves the highest average Macro-AUC. On the private dataset, `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` is not the best single downstream method by Macro-AUC. Therefore, the paper should frame synthetic FA as a complementary representation to T1 and justify final method selection by integrated balance across reconstruction, medical fidelity, texture preservation, and downstream utility, not by claiming every single metric is best.
