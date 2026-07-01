# ????/???????????

## ??

- ????????? Python ?????
- ????? `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST` ? ADNI test/full train ?????????
- ???????????? train-full ???`ADNI_DDIM`?`ADNI_DBM`?`ADNI_MOTFM`???? `Stage1_LPIPS_GAN`?
- ????? ADNI ????? train/test ????? DDIM/DBM/MOTFM ??`ADNI_A080_DS_FULL` ???? Macro-AUC ?????

## ??????

| ?? | train-full PNG | ???? |
|---|---:|---|
| ADNI_DDIM | 19552 | `outputs/icdm2026/predictions/ADNI_DDIM_E100_K50_PRETRAINED_TRAIN_FULL` |
| ADNI_DBM | 19552 | `outputs/icdm2026/predictions/ADNI_DBM_E100_K40_PRETRAINED_TRAIN_FULL` |
| ADNI_MOTFM | 19552 | `outputs/icdm2026/predictions/ADNI_MOTFM_I2I_K10_PRETRAINED_TRAIN_FULL` |
| Stage1_LPIPS_GAN private | 8650 | `outputs/icdm2026/predictions/Stage1_LPIPS_GAN_TRAIN_FULL` |

## ?? ADNI ??????

| Method                     | Accuracy | Macro_AUC | Macro_F1 | Rows |
| -------------------------- | -------- | --------- | -------- | ---- |
| ADNI_A080_DS_FULL          | 0.7100   | 0.7226    | 0.5931   | 9    |
| ADNI_OLD_FIDELITY_FLOW     | 0.6776   | 0.7133    | 0.5726   | 9    |
| ADNI_MOTFM                 | 0.7155   | 0.7100    | 0.6062   | 9    |
| T1_ONLY                    | 0.6898   | 0.7090    | 0.5815   | 9    |
| ADNI_A080_BASE             | 0.7364   | 0.7016    | 0.6004   | 9    |
| ADNI_PIX2PIX               | 0.7120   | 0.6892    | 0.5989   | 9    |
| ADNI_UNET                  | 0.6072   | 0.6885    | 0.5134   | 9    |
| FA_GT                      | 0.7241   | 0.6757    | 0.5727   | 9    |
| ADNI_CYCLEGAN              | 0.6900   | 0.6726    | 0.5664   | 9    |
| ADNI_DBM                   | 0.6758   | 0.6574    | 0.5583   | 9    |
| ADNI_LIGHTGUARD_PRIOR_FLOW | 0.5959   | 0.6533    | 0.4263   | 9    |
| ADNI_DDIM                  | 0.7398   | 0.6469    | 0.5872   | 9    |

## ????

- `outputs/icdm2026/downstream_adni_fair_full_all_methods/classification_subject_summary.csv`
- `outputs/icdm2026/downstream_adni_fair_full_all_methods/method_average_summary.csv`
- `outputs/icdm2026/final_dual_downstream_closure/final_integrated_comparison_table_full_heavy.csv`
- `outputs/icdm2026/paper_assets/tables/table1_adni_image_medical_metrics_full_heavy.csv`
- `outputs/icdm2026/paper_assets/tables/table2_adni_fair_downstream_full_heavy.csv`
- `outputs/icdm2026/paper_assets/tables/table5_integrated_composite_score_full_heavy.csv`

## ??

`ADNI_A080_DS_FULL` ???????????????????? Macro-AUC ????????????????????ROI ????????????`MOTFM` ??? Macro-F1 ? Accuracy ?????????? ROI ???????????????????
