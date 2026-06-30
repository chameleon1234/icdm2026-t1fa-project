# Final Dual-Dataset Downstream and Integrated Score Fix Report (Clean)

## 1. Overall Conclusion

This fix resolves the missing downstream metrics for ADNI Pix2Pix / U-Net / CycleGAN in the integrated comparison table. The root cause was an incomplete alias mapping: the image-quality table used `Pix2Pix / U-Net / CycleGAN`, whereas the fair downstream table used `ADNI_PIX2PIX / ADNI_UNET / ADNI_CYCLEGAN`.

After the fix, downstream_ACC, downstream_AUC, and downstream_F1 are correctly merged for ADNI Pix2Pix, U-Net, and CycleGAN.

ADNI: A080+DS Full ranks first by the Primary Integrated Composite Score with a score of 0.9642.

Private dataset: PRIVATE_DS_HYBRID ranks first by the Primary Integrated Composite Score with a score of 0.7784.

Safe conclusion: A080+DS Full is selected as the final method because it achieves the best overall integrated balance across reconstruction fidelity, white-matter/ROI medical fidelity, texture realism, and downstream utility, rather than because it is the top method on every individual metric.

## 2. ADNI Fair Downstream Results

| method | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 | mean_n_train_subjects | mean_n_test_subjects | dataset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ADNI_A080_DS_FULL | 9 | 0.7100 | 0.7226 | 0.5931 | 211.5556 | 52.0000 | ADNI |
| ADNI_OLD_FIDELITY_FLOW | 9 | 0.6776 | 0.7133 | 0.5726 | 211.5556 | 52.0000 | ADNI |
| T1_ONLY | 9 | 0.6898 | 0.7090 | 0.5815 | 211.5556 | 52.0000 | ADNI |
| ADNI_A080_BASE | 9 | 0.7364 | 0.7016 | 0.6004 | 211.5556 | 52.0000 | ADNI |
| ADNI_PIX2PIX | 9 | 0.7120 | 0.6892 | 0.5989 | 211.5556 | 52.0000 | ADNI |
| ADNI_UNET | 9 | 0.6072 | 0.6885 | 0.5134 | 211.5556 | 52.0000 | ADNI |
| FA_GT | 9 | 0.7241 | 0.6757 | 0.5727 | 211.5556 | 52.0000 | ADNI |
| ADNI_CYCLEGAN | 9 | 0.6900 | 0.6726 | 0.5664 | 211.5556 | 52.0000 | ADNI |
| ADNI_LIGHTGUARD_PRIOR_FLOW | 9 | 0.5959 | 0.6533 | 0.4263 | 211.5556 | 52.0000 | ADNI |


## 3. Private Fair Downstream Results

| method | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 | mean_n_train_subjects | mean_n_test_subjects | dataset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1_ONLY | 9 | 0.6659 | 0.7285 | 0.6267 | 129.6667 | 24.6667 | Private |
| Pix2Pix | 9 | 0.6483 | 0.7097 | 0.6049 | 129.6667 | 24.6667 | Private |
| UNet | 9 | 0.6730 | 0.7072 | 0.6229 | 129.6667 | 24.6667 | Private |
| CycleGAN | 9 | 0.6992 | 0.6864 | 0.6509 | 129.6667 | 24.6667 | Private |
| FIDELITY_FLOW | 9 | 0.7050 | 0.6815 | 0.6362 | 129.6667 | 24.6667 | Private |
| FA_GT | 9 | 0.6964 | 0.6639 | 0.6039 | 129.6667 | 24.6667 | Private |
| PRIVATE_DS_HYBRID | 9 | 0.6902 | 0.6493 | 0.6042 | 129.6667 | 24.6667 | Private |


## 4. Protocol Coverage

The strict fair downstream protocol trains classifiers on the train split and evaluates subject-level aggregation on the test split. It does not use test-as-train and does not mix test-CV with fair MIL. ADNI uses `data/adni_processed/adni_slice_manifest.csv`; the private dataset uses `data/processed/train` and `data/processed/test`.

## 5. Methods Not Included in Fair Downstream

- ADNI DDIM / DBM / MOTFM: missing train-full prediction folders.
- Private PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 and Stage1_LPIPS_GAN: missing train-full prediction folders.
- These methods are not deleted; they remain in image-quality, coverage, or fixed integrated tables, but are excluded from strict downstream ranking.

## 6. Integrated Composite Results

The Primary Integrated Composite Score uses fixed weights: 25% reconstruction fidelity, 30% medical fidelity, 20% texture realism, and 25% downstream utility. The weights were not changed to favor A080+DS Full.

| dataset | paper_method | downstream_method | PSNR | SSIM | MAE | WM_MAE | ROI_CCC | SharpRatio | downstream_ACC | downstream_AUC | downstream_F1 | primary_integrated_score | primary_rank_within_dataset | primary_score_eligible | primary_exclusion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADNI | A080+DS Full (Ours) | ADNI_A080_DS_FULL | 28.8156 | 0.9142 | 0.0166 | 0.0545 | 0.8423 | 1.0226 | 0.7100 | 0.7226 | 0.5931 | 0.9642 | 1.0000 | True |  |
| ADNI | Old Fidelity Flow | ADNI_OLD_FIDELITY_FLOW | 28.0915 | 0.9053 | 0.0177 | 0.0570 | 0.8670 | 0.8892 | 0.6776 | 0.7133 | 0.5726 | 0.8199 | 2.0000 | True |  |
| ADNI | Pix2Pix | ADNI_PIX2PIX | 28.0113 | 0.9014 | 0.0177 | 0.0606 | 0.8270 | 0.9126 | 0.7120 | 0.6892 | 0.5989 | 0.7504 | 3.0000 | True |  |
| ADNI | A080 Base | ADNI_A080_BASE | 28.3714 | 0.9086 | 0.0174 | 0.0613 | 0.7677 | 0.9961 | 0.7364 | 0.7016 | 0.6004 | 0.7442 | 4.0000 | True |  |
| ADNI | U-Net | ADNI_UNET | 28.4444 | 0.9073 | 0.0170 | 0.0584 | 0.8122 | 0.5186 | 0.6072 | 0.6885 | 0.5134 | 0.6255 | 5.0000 | True |  |
| ADNI | LightGuard PriorFlow | ADNI_LIGHTGUARD_PRIOR_FLOW | 28.0684 | 0.8940 | 0.0184 | 0.0570 | 0.8434 | 0.9798 | 0.5959 | 0.6533 | 0.4263 | 0.4891 | 6.0000 | True |  |
| ADNI | CycleGAN | ADNI_CYCLEGAN | 26.2564 | 0.8705 | 0.0220 | 0.0784 | 0.6907 | 0.8769 | 0.6900 | 0.6726 | 0.5664 | 0.2211 | 7.0000 | True |  |
| ADNI | DBM | ADNI_DBM | 25.8801 | 0.8765 | 0.0232 | 0.0928 | 0.5057 | 0.4889 |  |  |  |  |  | False | missing strict fair downstream result (train-full prediction folder unavailable or excluded) |
| ADNI | DDIM | ADNI_DDIM | 25.7831 | 0.8617 | 0.0236 | 0.0880 | 0.5803 | 0.5694 |  |  |  |  |  | False | missing strict fair downstream result (train-full prediction folder unavailable or excluded) |
| ADNI | FA_GT | FA_GT |  | 1.0000 |  |  | 1.0000 | 1.0000 | 0.7241 | 0.6757 | 0.5727 |  |  | False | reference/input upper-bound or non-generated baseline; reported but not ranked by generated-method composite |
| ADNI | T1_ONLY | T1_ONLY | 14.0519 | 0.7483 | 0.1052 | 0.3167 | 0.0507 | 3.6161 | 0.6898 | 0.7090 | 0.5815 |  |  | False | reference/input upper-bound or non-generated baseline; reported but not ranked by generated-method composite |
| Private | PRIVATE_DS_HYBRID | PRIVATE_DS_HYBRID | 28.0136 | 0.9023 | 0.0176 | 0.0525 | 0.8916 | 1.0099 | 0.6902 | 0.6493 | 0.6042 | 0.7784 | 1.0000 | True |  |
| Private | FidelityFlow | FIDELITY_FLOW | 27.5812 | 0.8925 | 0.0189 | 0.0592 | 0.8665 | 0.8692 | 0.7050 | 0.6815 | 0.6362 | 0.6828 | 2.0000 | True |  |
| Private | Pix2Pix | Pix2Pix | 28.1136 | 0.9035 | 0.0178 | 0.0578 | 0.8715 | 0.6415 | 0.6483 | 0.7097 | 0.6049 | 0.6667 | 3.0000 | True |  |
| Private | UNet | UNet | 28.2361 | 0.8973 | 0.0178 | 0.0575 | 0.8361 | 0.4561 | 0.6730 | 0.7072 | 0.6229 | 0.6279 | 4.0000 | True |  |
| Private | CycleGAN | CycleGAN | 26.1662 | 0.8711 | 0.0225 | 0.0681 | 0.8115 | 0.8643 | 0.6992 | 0.6864 | 0.6509 | 0.2858 | 5.0000 | True |  |
| Private | Private DS Atlas Probe | Private DS Atlas Probe | 27.0712 | 0.8857 | 0.0205 | 0.0607 | 0.8282 | 0.6548 |  |  |  |  |  | False | missing strict fair downstream result (train-full prediction folder unavailable or excluded) |
| Private | Private StackUNet5 CurrentSplit | Private StackUNet5 CurrentSplit | 28.2577 | 0.9038 | 0.0175 | 0.0575 | 0.8569 | 0.5429 |  |  |  |  |  | False | missing strict fair downstream result (train-full prediction folder unavailable or excluded) |
| Private | Private Stage1 Baseline | Private Stage1 Baseline | 27.9724 | 0.9034 | 0.0182 | 0.0629 | 0.7997 | 0.5187 |  |  |  |  |  | False | missing strict fair downstream result (train-full prediction folder unavailable or excluded) |
| Private | Private Stage1 LPIPS+GAN | Stage1_LPIPS_GAN | 27.5620 | 0.8914 | 0.0189 | 0.0616 | 0.8345 | 0.8822 |  |  |  |  |  | False | missing strict fair downstream result (train-full prediction folder unavailable or excluded) |
| Private | Private Stage1 Sharp | PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | 27.2797 | 0.8896 | 0.0194 | 0.0625 | 0.8607 | 0.9374 |  |  |  |  |  | False | missing strict fair downstream result (train-full prediction folder unavailable or excluded) |


## 7. Sensitivity Analysis

| dataset | score_view | top_method | score |
| --- | --- | --- | --- |
| ADNI | balanced/primary | A080+DS Full (Ours) | 0.9642 |
| ADNI | image-heavy | A080+DS Full (Ours) | 0.9780 |
| ADNI | medical-heavy | A080+DS Full (Ours) | 0.9648 |
| ADNI | downstream-heavy | A080+DS Full (Ours) | 0.9503 |
| Private | balanced/primary | PRIVATE_DS_HYBRID | 0.7784 |
| Private | image-heavy | PRIVATE_DS_HYBRID | 0.8817 |
| Private | medical-heavy | PRIVATE_DS_HYBRID | 0.8562 |
| Private | downstream-heavy | FidelityFlow | 0.7016 |


## 8. Safe Manuscript Language

A080+DS Full is selected as the final method because it achieves the best overall integrated balance across reconstruction fidelity, white-matter/ROI medical fidelity, texture realism, and downstream utility, rather than because it is the top method on every individual metric.
