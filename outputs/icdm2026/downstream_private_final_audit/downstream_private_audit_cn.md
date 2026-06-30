# ?????????

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

## ??????

| method | rows | mean_accuracy | mean_macro_auc | mean_macro_f1 | mean_n_train_subjects | mean_n_test_subjects |
| --- | --- | --- | --- | --- | --- | --- |
| T1_ONLY | 9 | 0.6659 | 0.7285 | 0.6267 | 129.6667 | 24.6667 |
| Pix2Pix | 9 | 0.6483 | 0.7097 | 0.6049 | 129.6667 | 24.6667 |
| UNet | 9 | 0.6730 | 0.7072 | 0.6229 | 129.6667 | 24.6667 |
| CycleGAN | 9 | 0.6992 | 0.6864 | 0.6509 | 129.6667 | 24.6667 |
| FIDELITY_FLOW | 9 | 0.7050 | 0.6815 | 0.6362 | 129.6667 | 24.6667 |
| FA_GT | 9 | 0.6964 | 0.6639 | 0.6039 | 129.6667 | 24.6667 |
| PRIVATE_DS_HYBRID | 9 | 0.6902 | 0.6493 | 0.6042 | 129.6667 | 24.6667 |
