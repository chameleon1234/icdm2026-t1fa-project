# ADNI ????????

???subject-level train/test??? `data/adni_processed/adni_slice_manifest.csv`??? T1_ONLY?FA_GT?Old Fidelity Flow?A080 Base?A080+DS Full?UNet?Pix2Pix?CycleGAN?LightGuard PriorFlow?

## ??????
| method | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 | mean_n_train_subjects | mean_n_test_subjects |
| --- | --- | --- | --- | --- | --- | --- |
| ADNI_A080_DS_FULL | 9 | 0.7100 | 0.7226 | 0.5931 | 211.5556 | 52.0000 |
| ADNI_OLD_FIDELITY_FLOW | 9 | 0.6776 | 0.7133 | 0.5726 | 211.5556 | 52.0000 |
| T1_ONLY | 9 | 0.6898 | 0.7090 | 0.5815 | 211.5556 | 52.0000 |
| ADNI_A080_BASE | 9 | 0.7364 | 0.7016 | 0.6004 | 211.5556 | 52.0000 |
| ADNI_PIX2PIX | 9 | 0.7120 | 0.6892 | 0.5989 | 211.5556 | 52.0000 |
| ADNI_UNET | 9 | 0.6072 | 0.6885 | 0.5134 | 211.5556 | 52.0000 |
| FA_GT | 9 | 0.7241 | 0.6757 | 0.5727 | 211.5556 | 52.0000 |
| ADNI_CYCLEGAN | 9 | 0.6900 | 0.6726 | 0.5664 | 211.5556 | 52.0000 |
| ADNI_LIGHTGUARD_PRIOR_FLOW | 9 | 0.5959 | 0.6533 | 0.4263 | 211.5556 | 52.0000 |

## ??????
| method | task | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| ADNI_OLD_FIDELITY_FLOW | cn_vs_ad | 3 | 0.8403 | 0.8233 | 0.6779 |
| T1_ONLY | cn_vs_ad | 3 | 0.8681 | 0.8233 | 0.7006 |
| ADNI_A080_DS_FULL | cn_vs_ad | 3 | 0.7569 | 0.8171 | 0.5843 |
| ADNI_A080_BASE | cn_vs_ad | 3 | 0.7639 | 0.7876 | 0.6041 |
| ADNI_UNET | cn_vs_ad | 3 | 0.7431 | 0.7814 | 0.5689 |
| ADNI_CYCLEGAN | cn_vs_ad | 3 | 0.7986 | 0.7659 | 0.5941 |
| ADNI_PIX2PIX | cn_vs_ad | 3 | 0.7639 | 0.7612 | 0.5908 |
| FA_GT | cn_vs_ad | 3 | 0.8125 | 0.7457 | 0.6113 |
| ADNI_LIGHTGUARD_PRIOR_FLOW | cn_vs_ad | 3 | 0.5903 | 0.7302 | 0.4020 |
| ADNI_A080_DS_FULL | cn_vs_mci_spectrum | 3 | 0.6301 | 0.6863 | 0.6076 |
| ADNI_A080_BASE | cn_vs_mci_spectrum | 3 | 0.6073 | 0.6817 | 0.5966 |
| ADNI_OLD_FIDELITY_FLOW | cn_vs_mci_spectrum | 3 | 0.6210 | 0.6811 | 0.5993 |
| ADNI_PIX2PIX | cn_vs_mci_spectrum | 3 | 0.6484 | 0.6731 | 0.6316 |
| ADNI_UNET | cn_vs_mci_spectrum | 3 | 0.6119 | 0.6506 | 0.5994 |
| FA_GT | cn_vs_mci_spectrum | 3 | 0.6073 | 0.6437 | 0.5895 |
| T1_ONLY | cn_vs_mci_spectrum | 3 | 0.5251 | 0.6372 | 0.5216 |
| ADNI_CYCLEGAN | cn_vs_mci_spectrum | 3 | 0.5571 | 0.6209 | 0.5500 |
| ADNI_LIGHTGUARD_PRIOR_FLOW | cn_vs_mci_spectrum | 3 | 0.6164 | 0.5630 | 0.4646 |
| ADNI_LIGHTGUARD_PRIOR_FLOW | mci_spectrum_vs_ad | 3 | 0.5810 | 0.6667 | 0.4123 |
| T1_ONLY | mci_spectrum_vs_ad | 3 | 0.6762 | 0.6667 | 0.5223 |
| ADNI_A080_DS_FULL | mci_spectrum_vs_ad | 3 | 0.7429 | 0.6644 | 0.5873 |
| FA_GT | mci_spectrum_vs_ad | 3 | 0.7524 | 0.6378 | 0.5172 |
| ADNI_A080_BASE | mci_spectrum_vs_ad | 3 | 0.8381 | 0.6356 | 0.6004 |
| ADNI_OLD_FIDELITY_FLOW | mci_spectrum_vs_ad | 3 | 0.5714 | 0.6356 | 0.4405 |
| ADNI_UNET | mci_spectrum_vs_ad | 3 | 0.4667 | 0.6333 | 0.3718 |
| ADNI_PIX2PIX | mci_spectrum_vs_ad | 3 | 0.7238 | 0.6333 | 0.5743 |
| ADNI_CYCLEGAN | mci_spectrum_vs_ad | 3 | 0.7143 | 0.6311 | 0.5551 |
