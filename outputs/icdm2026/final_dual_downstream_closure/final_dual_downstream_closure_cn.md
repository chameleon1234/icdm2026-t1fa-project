# ???????????????

## ??

- ADNI ????? `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST` ????? subject-level train/test ?????
- ADNI ???????T1_ONLY?FA_GT?Old Fidelity Flow?A080 Base?A080+DS Full?UNet?Pix2Pix?CycleGAN?LightGuard PriorFlow?
- ??????????T1_ONLY?FA_GT?FidelityFlow?PRIVATE_DS_HYBRID?UNet?Pix2Pix?CycleGAN?
- ????????????????ADNI ? A080+DS Full ??? Macro-AUC ??????? T1_ONLY ??? Macro-AUC ???FidelityFlow ? Accuracy ???
- ?????????A080+DS Full ???????????????????????????????????????????????????

## ADNI ????????

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


## ???????????

| method | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 | mean_n_train_subjects | mean_n_test_subjects |
| --- | --- | --- | --- | --- | --- | --- |
| T1_ONLY | 9 | 0.6659 | 0.7285 | 0.6267 | 129.6667 | 24.6667 |
| Pix2Pix | 9 | 0.6483 | 0.7097 | 0.6049 | 129.6667 | 24.6667 |
| UNet | 9 | 0.6730 | 0.7072 | 0.6229 | 129.6667 | 24.6667 |
| CycleGAN | 9 | 0.6992 | 0.6864 | 0.6509 | 129.6667 | 24.6667 |
| FIDELITY_FLOW | 9 | 0.7050 | 0.6815 | 0.6362 | 129.6667 | 24.6667 |
| FA_GT | 9 | 0.6964 | 0.6639 | 0.6039 | 129.6667 | 24.6667 |
| PRIVATE_DS_HYBRID | 9 | 0.6902 | 0.6493 | 0.6042 | 129.6667 | 24.6667 |


## ?????

### ADNI

| method | test_prediction_folder | train_prediction_folder | checkpoint | test_prediction_folder_exists | train_prediction_folder_exists | checkpoint_exists | test_png_count | train_png_count | eligible_for_fair_train_test_MIL | eligible_for_test_CV_only | reason_if_not_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_ONLY | data/adni_processed/test/t1_slices | data/adni_processed/train/t1_slices |  | True | True | False | 5616 | 19552 | True | False |  |
| FA_GT | data/adni_processed/test/fa_slices | data/adni_processed/train/fa_slices |  | True | True | False | 5616 | 19552 | True | False |  |
| ADNI_OLD_FIDELITY_FLOW | outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL | outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL_TRAIN_FULL | outputs/adni_pm_dirf_fidelity_flow_full/checkpoints/best_fidelity_corrector.pt | True | True | False | 5616 | 19552 | True | False |  |
| ADNI_A080_BASE | outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080 | outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080_TRAIN_FULL |  | True | True | False | 5616 | 19552 | True | False |  |
| ADNI_A080_DS_FULL | outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST | outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_TRAIN_FULL | outputs/adni_a080_ds_corrector_diseaseroi_hfpreserve_full_e12/checkpoints/best_ds_corrector_score.pt | True | True | False | 5616 | 19552 | True | False |  |
| ADNI_UNET | outputs/icdm2026/predictions/ADNI_UNet_E50 | outputs/icdm2026/predictions/ADNI_UNet_E50_TRAIN_FULL | outputs/unet_adni_split_e50/checkpoints/best_unet.pt | True | True | True | 5616 | 19552 | True | False |  |
| ADNI_PIX2PIX | outputs/icdm2026/predictions/ADNI_Pix2Pix_E50 | outputs/icdm2026/predictions/ADNI_Pix2Pix_E50_TRAIN_FULL | outputs/pix2pix_adni_split_e50/checkpoints/best_pix2pix_generator.pt | True | True | True | 5616 | 19552 | True | False |  |
| ADNI_CYCLEGAN | outputs/icdm2026/predictions/ADNI_CycleGAN_E50 | outputs/icdm2026/predictions/ADNI_CycleGAN_E50_TRAIN_FULL | outputs/cyclegan_adni_split_e50/checkpoints/best_cyclegan_a2b_generator.pt | True | True | True | 5616 | 19552 | True | False |  |
| ADNI_LIGHTGUARD_PRIOR_FLOW | outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_4096_E8_BEST_K8 | outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_TRAIN_FULL_K8 | outputs/adni_stage1_prior_flow_lightguard_4096_e8/checkpoints/best_prior_flow_stage1.pt | True | True | False | 5616 | 19552 | True | False |  |
| ADNI_DDIM | outputs/icdm2026/predictions/ADNI_DDIM_E100_K50_PRETRAINED | outputs/icdm2026/predictions/ADNI_DDIM_E100_K50_PRETRAINED_TRAIN_FULL | checkpoints_baseline_ddim_steps50/epoch_100.pt | True | False | True | 5616 | 0 | False | True | missing train prediction folder |
| ADNI_DBM | outputs/icdm2026/predictions/ADNI_DBM_E100_K40_PRETRAINED | outputs/icdm2026/predictions/ADNI_DBM_E100_K40_PRETRAINED_TRAIN_FULL | checkpoints_baseline_dbm/epoch_100.pt | True | False | True | 5616 | 0 | False | True | missing train prediction folder |
| ADNI_MOTFM | outputs/icdm2026/predictions/ADNI_MOTFM_I2I_K10_PRETRAINED | outputs/icdm2026/predictions/ADNI_MOTFM_I2I_K10_PRETRAINED_TRAIN_FULL | MOTFM-main/checkpoints_t1_fa_i2i/t1_fa_config/version_1/checkpoints/last.ckpt | True | False | True | 5616 | 0 | False | True | missing train prediction folder |


### ???

| method | test_prediction_folder | train_prediction_folder | checkpoint | test_prediction_folder_exists | train_prediction_folder_exists | checkpoint_exists | test_png_count | train_png_count | eligible_for_fair_train_test_MIL | eligible_for_test_CV_only | reason_if_not_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_ONLY | data/processed/test/t1_slices | data/processed/train/t1_slices |  | True | True | False | 1900 | 8650 | True | False |  |
| FA_GT | data/processed/test/fa_slices | data/processed/train/fa_slices |  | True | True | False | 1900 | 8650 | True | False |  |
| FidelityFlow | outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL | outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL_TRAIN_FULL | outputs/pmrf_t1fa_stage2_fidelity_flow_full/checkpoints/best_fidelity_corrector.pt | True | True | True | 1900 | 8650 | True | False |  |
| PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL | outputs/icdm2026/predictions/PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL | outputs/icdm2026/predictions/PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL_TRAIN_FULL | outputs/pmrf_t1fa_stage2_single_ds_hybrid_from_stage1_e030_full/checkpoints/best_ds_corrector.pt | True | True | True | 1900 | 8650 | True | False |  |
| UNet | outputs/icdm2026/predictions/UNet_CurrentSplit_E100 | outputs/icdm2026/predictions_finalpdf_train/UNet | outputs/unet_current_split_e100/checkpoints/best_unet.pt | True | True | True | 1900 | 8650 | True | False |  |
| Pix2Pix | outputs/icdm2026/predictions/Pix2Pix_CurrentSplit_E100 | outputs/icdm2026/predictions_finalpdf_train/Pix2Pix | outputs/pix2pix_current_split_e100/checkpoints/best_pix2pix_generator.pt | True | True | True | 1900 | 8650 | True | False |  |
| CycleGAN | outputs/icdm2026/predictions/CycleGAN_CurrentSplit_E100 | outputs/icdm2026/predictions_finalpdf_train/CycleGAN | outputs/cyclegan_current_split_e100/checkpoints/best_cyclegan_a2b_generator.pt | True | True | True | 1900 | 8650 | True | False |  |
| PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | outputs/icdm2026/predictions/PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030 | outputs/icdm2026/predictions/PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030_TRAIN_FULL | outputs/pmrf_t1fa_stage1_single_sharp_stripe_full/checkpoints/epoch_030.pt | True | False | True | 1900 | 0 | False | True | missing train prediction folder |
| Stage1_LPIPS_GAN | outputs/icdm2026/predictions/PM_STAGE1_LPIPS_GAN_5SLICE_FINAL | outputs/icdm2026/predictions/Stage1_LPIPS_GAN_TRAIN_FULL | outputs/pmrf_t1fa_stage1_wmroi_detail_5slice_lpips01_gan001/checkpoints/best_stage1.pt | True | False | True | 1900 | 0 | False | True | missing train prediction folder |


## ?????

| dataset | image_method | downstream_method | PSNR | SSIM | MSE | MAE | WM_MAE | ROI_CCC | ROI_Spearman | SharpRatio | Tenengrad_or_EdgeGrad | WM_Skeleton_Error | Slice_Consistency | n_slices | n_subjects | downstream_ACC | downstream_AUC | downstream_F1 | downstream_rows | score_fidelity | score_medical | sharp_closeness | score_texture | score_downstream | integrated_score | rank_within_dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADNI | A080+DS Full (Ours) | ADNI_A080_DS_FULL | 28.8156 | 0.9142 | 0.0014 | 0.0166 | 0.0545 | 0.8423 | 0.8531 | 1.0226 | 0.1141 | 0.0737 | 0.0320 | 5616 | 108 | 0.7100 | 0.7226 | 0.5931 | 9 | 0.8864 | 0.9131 | 0.9774 | 0.9914 | 0.8515 | 0.9067 | 1.0000 |
| ADNI | A080 Base | ADNI_A080_BASE | 28.3714 | 0.9086 | 0.0016 | 0.0174 | 0.0613 | 0.7677 | 0.8420 | 0.9961 | 0.1174 | 0.0827 | 0.0329 | 5616 | 108 | 0.7364 | 0.7016 | 0.6004 | 9 | 0.8656 | 0.8609 | 0.9961 | 0.9985 | 0.8295 | 0.8818 | 2.0000 |
| ADNI | Old Fidelity Flow | ADNI_OLD_FIDELITY_FLOW | 28.0915 | 0.9053 | 0.0017 | 0.0177 | 0.0570 | 0.8670 |  | 0.8892 | 0.1232 |  |  | 5616 | 108 | 0.6776 | 0.7133 | 0.5726 | 9 | 0.8540 | 0.9215 | 0.8892 | 0.9576 | 0.7020 | 0.8570 | 3.0000 |
| ADNI | PriorFlow Safe Probe | ADNI_LIGHTGUARD_PRIOR_FLOW | 28.0684 | 0.8940 | 0.0017 | 0.0184 | 0.0570 | 0.8434 | 0.8209 | 0.9798 | 0.1186 | 0.0784 | 0.0298 | 5616 | 108 | 0.5959 | 0.6533 | 0.4263 | 9 | 0.8357 | 0.9090 | 0.9798 | 0.9923 | 0.0182 | 0.6846 | 4.0000 |
| ADNI | Pix2Pix | Pix2Pix | 28.0113 | 0.9014 | 0.0017 | 0.0177 | 0.0606 | 0.8270 |  | 0.9126 | 0.1272 |  |  | 5616 | 108 |  |  |  | 0 | 0.8469 | 0.8936 | 0.9126 | 0.9666 |  | 0.6731 | 5.0000 |
| ADNI | U-Net | U-Net | 28.4444 | 0.9073 | 0.0015 | 0.0170 | 0.0584 | 0.8122 |  | 0.5186 | 0.1183 |  |  | 5616 | 108 |  |  |  | 0 | 0.8671 | 0.8899 | 0.5186 | 0.8160 |  | 0.6469 | 6.0000 |
| ADNI | CycleGAN | CycleGAN | 26.2564 | 0.8705 | 0.0025 | 0.0220 | 0.0784 | 0.6907 |  | 0.8769 | 0.1497 |  |  | 5616 | 108 |  |  |  | 0 | 0.7502 | 0.7881 | 0.8769 | 0.9529 |  | 0.6146 | 7.0000 |
| ADNI | DDIM | DDIM | 25.7831 | 0.8617 | 0.0028 | 0.0236 | 0.0880 | 0.5803 |  | 0.5694 | 0.1548 |  |  | 5616 | 108 |  |  |  | 0 | 0.7219 | 0.7119 | 0.5694 | 0.8354 |  | 0.5611 | 8.0000 |
| ADNI | DBM | DBM | 25.8801 | 0.8765 | 0.0028 | 0.0232 | 0.0928 | 0.5057 |  | 0.4889 | 0.1408 |  |  | 5616 | 108 |  |  |  | 0 | 0.7452 | 0.6633 | 0.4889 | 0.8046 |  | 0.5462 | 9.0000 |
| ADNI | T1_ONLY | T1_ONLY | 14.0519 | 0.7483 | 0.0428 | 0.1052 | 0.3167 | 0.0507 |  | 3.6161 | 0.3527 |  |  | 5616 | 108 | 0.6898 | 0.7090 | 0.5815 | 9 | 0.0000 | 0.0000 | -1.6161 | 0.0000 | 0.7248 | 0.1812 | 10.0000 |
| ADNI | FA_GT | FA_GT |  | 1.0000 |  |  |  | 1.0000 |  | 1.0000 | 0.0000 |  |  | 5616 | 108 | 0.7241 | 0.6757 | 0.5727 | 9 |  |  | 1.0000 | 1.0000 | 0.6414 |  |  |
| Private | Private Fidelity Flow | FIDELITY_FLOW | 27.5812 | 0.8925 | 0.0018 | 0.0189 | 0.0592 | 0.8665 |  | 0.8692 | 0.1237 |  |  | 1900 | 38 | 0.7050 | 0.6815 | 0.6362 | 9 | 0.8210 | 0.9170 | 0.8692 | 0.9500 | 0.7168 | 0.8495 | 1.0000 |
| Private | Private Pix2Pix CurrentSplit | Pix2Pix | 28.1136 | 0.9035 | 0.0016 | 0.0178 | 0.0578 | 0.8715 |  | 0.6415 | 0.1139 |  |  | 1900 | 38 | 0.6483 | 0.7097 | 0.6049 | 9 | 0.8515 | 0.9223 | 0.6415 | 0.8630 | 0.6640 | 0.8282 | 2.0000 |
| Private | Private U-Net CurrentSplit | UNet | 28.2361 | 0.8973 | 0.0016 | 0.0178 | 0.0575 | 0.8361 |  | 0.4561 | 0.1126 |  |  | 1900 | 38 | 0.6730 | 0.7072 | 0.6229 | 9 | 0.8461 | 0.9043 | 0.4561 | 0.7921 | 0.7382 | 0.8258 | 3.0000 |
| Private | Private CycleGAN CurrentSplit | CycleGAN | 26.1662 | 0.8711 | 0.0026 | 0.0225 | 0.0681 | 0.8115 |  | 0.8643 | 0.1394 |  |  | 1900 | 38 | 0.6992 | 0.6864 | 0.6509 | 9 | 0.7472 | 0.8711 | 0.8643 | 0.9481 | 0.7470 | 0.8245 | 4.0000 |
| Private | Private DS Hybrid (Main) | PRIVATE_DS_HYBRID | 28.0136 | 0.9023 | 0.0017 | 0.0176 | 0.0525 | 0.8916 |  | 1.0099 | 0.1182 |  |  | 1900 | 38 | 0.6902 | 0.6493 | 0.6042 | 9 | 0.8486 | 0.9429 | 0.9901 | 0.9962 | 0.4878 | 0.8162 | 5.0000 |
| Private | Private Stage1 Sharp | Private Stage1 Sharp | 27.2797 | 0.8896 | 0.0020 | 0.0194 | 0.0625 | 0.8607 |  | 0.9374 | 0.1286 |  |  | 1900 | 38 |  |  |  | 0 | 0.8085 | 0.9078 | 0.9374 | 0.9761 |  | 0.6697 | 6.0000 |
| Private | Private Stage1 LPIPS+GAN | Private Stage1 LPIPS+GAN | 27.5620 | 0.8914 | 0.0019 | 0.0189 | 0.0616 | 0.8345 |  | 0.8822 | 0.1245 |  |  | 1900 | 38 |  |  |  | 0 | 0.8190 | 0.8956 | 0.8822 | 0.9550 |  | 0.6644 | 7.0000 |
| Private | Private StackUNet5 CurrentSplit | Private StackUNet5 CurrentSplit | 28.2577 | 0.9038 | 0.0016 | 0.0175 | 0.0575 | 0.8569 |  | 0.5429 | 0.1131 |  |  | 1900 | 38 |  |  |  | 0 | 0.8564 | 0.9152 | 0.5429 | 0.8253 |  | 0.6537 | 8.0000 |
| Private | Private DS Atlas Probe | Private DS Atlas Probe | 27.0712 | 0.8857 | 0.0021 | 0.0205 | 0.0607 | 0.8282 |  | 0.6548 | 0.1264 |  |  | 1900 | 38 |  |  |  | 0 | 0.7946 | 0.8940 | 0.6548 | 0.8681 |  | 0.6404 | 9.0000 |
| Private | Private Stage1 Baseline | Private Stage1 Baseline | 27.9724 | 0.9034 | 0.0017 | 0.0182 | 0.0629 | 0.7997 |  | 0.5187 | 0.1145 |  |  | 1900 | 38 |  |  |  | 0 | 0.8470 | 0.8749 | 0.5187 | 0.8160 |  | 0.6374 | 10.0000 |


## ???

- ADNI DDIM / DBM / MOTFM ???? test ??? checkpoint ????? train-full prediction folder????????? train/test ?????
- ??? `PRIVATE_STAGE1_SINGLE_SHARP_STRIPE_EPOCH030` ? `Stage1_LPIPS_GAN` ?? train-full prediction folder????????? train/test ?????
