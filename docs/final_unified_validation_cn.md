# ??????????????

?????2026-06-29

## ????

- ??????????????????? `scripts/evaluate_method_folder.py` ???
- ADNI?5616 ? test slices / 108 subjects?
- ????1900 ? test slices / 38 subjects?
- ???ADNI ?? `scripts/evaluate_slice_mil_roi.py` ??? slice vote / attention MIL / multitask MIL???????????????? MIL ??????????? train split ??????????? train/test?

## ADNI ????

| label | PSNR | SSIM | MAE | WM_MAE | ROI_CCC | SharpRatio | GradErr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ours-A080+DS | 28.6789 | 0.9128 | 0.0168 | 0.0565 | 0.8285 | 1.0228 | 0.1149 |
| StackUNet | 28.4505 | 0.9070 | 0.0170 | 0.0591 | 0.8142 | 0.6363 | 0.1198 |
| U-Net | 28.4444 | 0.9073 | 0.0170 | 0.0584 | 0.8122 | 0.5186 | 0.1183 |
| A080 texture base | 28.3714 | 0.9086 | 0.0174 | 0.0613 | 0.7677 | 0.9961 | 0.1174 |
| Ours-FF | 28.0915 | 0.9053 | 0.0177 | 0.0570 | 0.8670 | 0.8892 | 0.1232 |
| PriorFlow ablation | 28.0684 | 0.8940 | 0.0184 | 0.0570 | 0.8434 | 0.9798 | 0.1186 |
| Pix2Pix | 28.0113 | 0.9014 | 0.0177 | 0.0606 | 0.8270 | 0.9126 | 0.1272 |
| Stage1 LPIPS+GAN | 27.8312 | 0.9022 | 0.0182 | 0.0649 | 0.8035 | 0.9136 | 0.1267 |
| CycleGAN | 26.2564 | 0.8705 | 0.0220 | 0.0784 | 0.6907 | 0.8769 | 0.1497 |
| DBM | 25.8801 | 0.8765 | 0.0232 | 0.0928 | 0.5057 | 0.4889 | 0.1408 |
| DDIM | 25.7831 | 0.8617 | 0.0236 | 0.0880 | 0.5803 | 0.5694 | 0.1548 |
| MOTFM | 17.2323 | 0.2378 | 0.0876 | 0.1803 | 0.1392 | 2.1839 | 0.2681 |

## ???????

| label | PSNR | SSIM | MAE | WM_MAE | ROI_CCC | SharpRatio | GradErr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIRF V5 | 28.3995 | 0.9094 | 0.0172 | 0.0545 | 0.8839 | 0.4551 | 0.1102 |
| StackUNet | 28.2577 | 0.9038 | 0.0175 | 0.0575 | 0.8569 | 0.5429 | 0.1131 |
| U-Net | 28.2361 | 0.8973 | 0.0178 | 0.0575 | 0.8361 | 0.4561 | 0.1126 |
| Pix2Pix | 28.1136 | 0.9035 | 0.0178 | 0.0578 | 0.8715 | 0.6415 | 0.1139 |
| Ours-DS hybrid | 28.0136 | 0.9023 | 0.0176 | 0.0525 | 0.8916 | 1.0099 | 0.1182 |
| PM Stage1 | 27.7409 | 0.8978 | 0.0184 | 0.0597 | 0.8548 | 0.6810 | 0.1191 |
| Ours-FF | 27.5812 | 0.8925 | 0.0189 | 0.0592 | 0.8665 | 0.8692 | 0.1237 |
| DDIM | 27.4790 | 0.8876 | 0.0192 | 0.0629 | 0.8272 | 0.6669 | 0.1280 |
| Stage1 sharp | 27.2797 | 0.8896 | 0.0194 | 0.0625 | 0.8607 | 0.9374 | 0.1286 |
| CycleGAN | 26.1662 | 0.8711 | 0.0225 | 0.0681 | 0.8115 | 0.8643 | 0.1394 |

## ADNI ??????????

| protocol | method | tasks | mean_acc | mean_auc | mean_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| attention_mil | ADNI_A080_DS_HFKeep | 3.0000 | 0.7967 | 0.7012 | 0.5741 |
| attention_mil | ADNI_MOTFM | 3.0000 | 0.7807 | 0.6439 | 0.5032 |
| attention_mil | ADNI_Old_Fidelity | 3.0000 | 0.7785 | 0.6558 | 0.6194 |
| attention_mil | T1_ONLY | 3.0000 | 0.7336 | 0.7804 | 0.6295 |
| attention_mil | ADNI_DDIM | 3.0000 | 0.7277 | 0.5933 | 0.5088 |
| attention_mil | ADNI_DBM | 3.0000 | 0.7275 | 0.6840 | 0.5351 |
| attention_mil | FA_GT | 3.0000 | 0.7239 | 0.5262 | 0.5195 |
| attention_mil | ADNI_A080 | 3.0000 | 0.6424 | 0.6778 | 0.4791 |
| multi_task_mil | FA_GT | 3.0000 | 0.7966 | 0.6050 | 0.6079 |
| multi_task_mil | ADNI_DDIM | 3.0000 | 0.7436 | 0.6154 | 0.5230 |
| multi_task_mil | ADNI_DBM | 3.0000 | 0.7229 | 0.6529 | 0.5650 |
| multi_task_mil | T1_ONLY | 3.0000 | 0.7194 | 0.6722 | 0.5420 |
| multi_task_mil | ADNI_A080_DS_HFKeep | 3.0000 | 0.6771 | 0.6551 | 0.4861 |
| multi_task_mil | ADNI_Old_Fidelity | 3.0000 | 0.6622 | 0.6552 | 0.5634 |
| multi_task_mil | ADNI_MOTFM | 3.0000 | 0.5773 | 0.4714 | 0.4748 |
| multi_task_mil | ADNI_A080 | 3.0000 | 0.5184 | 0.6018 | 0.4758 |
| slice_svm_vote | T1_ONLY | 3.0000 | 0.6906 | 0.7229 | 0.6046 |
| slice_svm_vote | ADNI_MOTFM | 3.0000 | 0.6668 | 0.5587 | 0.5415 |
| slice_svm_vote | ADNI_Old_Fidelity | 3.0000 | 0.6297 | 0.6584 | 0.5432 |
| slice_svm_vote | FA_GT | 3.0000 | 0.6186 | 0.6238 | 0.5122 |
| slice_svm_vote | ADNI_DBM | 3.0000 | 0.4936 | 0.5781 | 0.4306 |
| slice_svm_vote | ADNI_A080 | 3.0000 | 0.4753 | 0.6667 | 0.4203 |
| slice_svm_vote | ADNI_DDIM | 3.0000 | 0.4640 | 0.5914 | 0.4289 |
| slice_svm_vote | ADNI_A080_DS_HFKeep | 3.0000 | 0.4638 | 0.6494 | 0.4114 |

## ???/???????????????

| dataset | protocol | method | tasks | mean_acc | mean_auc | mean_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| ADNI | attention_mil | ADNI_A080_DS_Fair | 3.0000 | 0.7989 | 0.6901 | 0.5232 |
| ADNI | attention_mil | ADNI_A080_DS_DiseaseROI | 3.0000 | 0.7989 | 0.6871 | 0.5258 |
| ADNI | attention_mil | ADNI_A080_DS_HFKeep | 3.0000 | 0.7989 | 0.6862 | 0.5258 |
| ADNI | attention_mil | ADNI_A080 | 3.0000 | 0.7944 | 0.6783 | 0.5214 |
| ADNI | attention_mil | ADNI_Old_Fidelity | 3.0000 | 0.7712 | 0.6197 | 0.5765 |
| ADNI | attention_mil | T1_PLUS_Old_Fidelity | 3.0000 | 0.7293 | 0.6031 | 0.5292 |
| ADNI | attention_mil | T1_PLUS_GT | 3.0000 | 0.7221 | 0.6459 | 0.5662 |
| ADNI | attention_mil | T1_ONLY | 3.0000 | 0.7168 | 0.7149 | 0.6019 |
| ADNI | attention_mil | T1_PLUS_A080_DS_Fair | 3.0000 | 0.7154 | 0.6291 | 0.5600 |
| ADNI | attention_mil | FA_GT | 3.0000 | 0.7003 | 0.5892 | 0.5363 |
| ADNI | attention_mil | T1_PLUS_A080_DS_HFKeep | 3.0000 | 0.6987 | 0.6269 | 0.5489 |
| ADNI | multi_task_mil | ADNI_A080_DS_HFKeep | 3.0000 | 0.7807 | 0.6550 | 0.5077 |
| ADNI | multi_task_mil | T1_PLUS_GT | 3.0000 | 0.7539 | 0.6926 | 0.6268 |
| ADNI | multi_task_mil | T1_PLUS_A080_DS_Fair | 3.0000 | 0.7455 | 0.6568 | 0.6092 |
| ADNI | multi_task_mil | T1_PLUS_A080_DS_HFKeep | 3.0000 | 0.7245 | 0.6421 | 0.5735 |
| ADNI | multi_task_mil | FA_GT | 3.0000 | 0.7140 | 0.6292 | 0.5922 |
| ADNI | multi_task_mil | ADNI_Old_Fidelity | 3.0000 | 0.7134 | 0.6069 | 0.5326 |
| ADNI | multi_task_mil | ADNI_A080_DS_DiseaseROI | 3.0000 | 0.6906 | 0.6798 | 0.5116 |
| ADNI | multi_task_mil | ADNI_A080_DS_Fair | 3.0000 | 0.6813 | 0.7081 | 0.5914 |
| ADNI | multi_task_mil | ADNI_A080 | 3.0000 | 0.6769 | 0.6322 | 0.5077 |
| ADNI | multi_task_mil | T1_ONLY | 3.0000 | 0.6664 | 0.6955 | 0.5235 |
| ADNI | multi_task_mil | T1_PLUS_Old_Fidelity | 3.0000 | 0.5997 | 0.5656 | 0.4778 |
| ADNI | slice_svm_vote | T1_PLUS_A080_DS_Fair | 3.0000 | 0.6676 | 0.5924 | 0.5487 |
| ADNI | slice_svm_vote | T1_PLUS_A080_DS_HFKeep | 3.0000 | 0.6561 | 0.5959 | 0.5374 |
| ADNI | slice_svm_vote | T1_PLUS_Old_Fidelity | 3.0000 | 0.6495 | 0.5586 | 0.5157 |
| ADNI | slice_svm_vote | T1_PLUS_GT | 3.0000 | 0.6410 | 0.6961 | 0.5647 |
| ADNI | slice_svm_vote | T1_ONLY | 3.0000 | 0.5999 | 0.6600 | 0.5276 |
| ADNI | slice_svm_vote | FA_GT | 3.0000 | 0.5809 | 0.6911 | 0.4972 |
| ADNI | slice_svm_vote | ADNI_A080 | 3.0000 | 0.5432 | 0.5518 | 0.4735 |
| ADNI | slice_svm_vote | ADNI_A080_DS_Fair | 3.0000 | 0.5408 | 0.5083 | 0.4725 |
| ADNI | slice_svm_vote | ADNI_A080_DS_DiseaseROI | 3.0000 | 0.5362 | 0.5115 | 0.4686 |
| ADNI | slice_svm_vote | ADNI_A080_DS_HFKeep | 3.0000 | 0.5319 | 0.5162 | 0.4643 |
| ADNI | slice_svm_vote | ADNI_Old_Fidelity | 3.0000 | 0.5170 | 0.5164 | 0.4507 |
| Private | attention_mil | T1_PLUS_GT | 3.0000 | 0.7548 | 0.6373 | 0.6581 |
| Private | attention_mil | T1_PLUS_Ours | 3.0000 | 0.7341 | 0.6765 | 0.6886 |
| Private | attention_mil | FidelityFlow | 3.0000 | 0.6984 | 0.6439 | 0.6304 |
| Private | attention_mil | UNet | 3.0000 | 0.6960 | 0.7313 | 0.6455 |
| Private | attention_mil | FA_GT | 3.0000 | 0.6849 | 0.6609 | 0.5909 |
| Private | attention_mil | T1_PLUS_Stage1 | 3.0000 | 0.6556 | 0.6729 | 0.6244 |
| Private | attention_mil | CycleGAN | 3.0000 | 0.6127 | 0.6092 | 0.5440 |
| Private | attention_mil | T1_PLUS_UNet | 3.0000 | 0.5937 | 0.7326 | 0.5691 |
| Private | attention_mil | Pix2Pix | 3.0000 | 0.5873 | 0.7268 | 0.5473 |
| Private | attention_mil | T1_ONLY | 3.0000 | 0.5548 | 0.7094 | 0.5417 |
| Private | attention_mil | T1_PLUS_CycleGAN | 3.0000 | 0.5492 | 0.6836 | 0.5341 |
| Private | attention_mil | Stage1 | 3.0000 | 0.5484 | 0.6471 | 0.5122 |
| Private | attention_mil | T1_PLUS_Pix2Pix | 3.0000 | 0.5214 | 0.7179 | 0.5101 |
| Private | multi_task_mil | T1_PLUS_UNet | 3.0000 | 0.8310 | 0.7531 | 0.8051 |
| Private | multi_task_mil | T1_PLUS_Pix2Pix | 3.0000 | 0.8270 | 0.7848 | 0.7959 |
| Private | multi_task_mil | T1_PLUS_GT | 3.0000 | 0.8167 | 0.7545 | 0.7551 |
| Private | multi_task_mil | T1_ONLY | 3.0000 | 0.7952 | 0.7179 | 0.7702 |
| Private | multi_task_mil | T1_PLUS_CycleGAN | 3.0000 | 0.7532 | 0.7291 | 0.7175 |
| Private | multi_task_mil | UNet | 3.0000 | 0.7357 | 0.7340 | 0.6549 |
| Private | multi_task_mil | CycleGAN | 3.0000 | 0.7341 | 0.6604 | 0.6775 |
| Private | multi_task_mil | T1_PLUS_Ours | 3.0000 | 0.7333 | 0.6698 | 0.6691 |
| Private | multi_task_mil | Pix2Pix | 3.0000 | 0.7325 | 0.7420 | 0.6610 |
| Private | multi_task_mil | FA_GT | 3.0000 | 0.7206 | 0.5628 | 0.6257 |
| Private | multi_task_mil | T1_PLUS_Stage1 | 3.0000 | 0.7175 | 0.5985 | 0.6557 |
| Private | multi_task_mil | FidelityFlow | 3.0000 | 0.7167 | 0.6823 | 0.6374 |
| Private | multi_task_mil | Stage1 | 3.0000 | 0.7008 | 0.7197 | 0.6333 |
| Private | slice_svm_vote | UNet | 3.0000 | 0.7325 | 0.6373 | 0.6833 |
| Private | slice_svm_vote | T1_ONLY | 3.0000 | 0.6913 | 0.5357 | 0.6279 |
| Private | slice_svm_vote | Pix2Pix | 3.0000 | 0.6690 | 0.6546 | 0.6103 |
| Private | slice_svm_vote | T1_PLUS_UNet | 3.0000 | 0.6635 | 0.5762 | 0.6054 |
| Private | slice_svm_vote | T1_PLUS_Ours | 3.0000 | 0.6556 | 0.5152 | 0.5884 |
| Private | slice_svm_vote | T1_PLUS_Pix2Pix | 3.0000 | 0.6516 | 0.5512 | 0.5947 |
| Private | slice_svm_vote | T1_PLUS_CycleGAN | 3.0000 | 0.6397 | 0.5530 | 0.5838 |
| Private | slice_svm_vote | T1_PLUS_Stage1 | 3.0000 | 0.6333 | 0.5152 | 0.5680 |
| Private | slice_svm_vote | FidelityFlow | 3.0000 | 0.6325 | 0.6225 | 0.5349 |
| Private | slice_svm_vote | Stage1 | 3.0000 | 0.6325 | 0.6225 | 0.5491 |
| Private | slice_svm_vote | CycleGAN | 3.0000 | 0.5865 | 0.5258 | 0.5302 |
| Private | slice_svm_vote | T1_PLUS_GT | 3.0000 | 0.5833 | 0.5312 | 0.5277 |
| Private | slice_svm_vote | FA_GT | 3.0000 | 0.5825 | 0.6225 | 0.5198 |

## ????

1. ADNI ??????`ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_4096_E6` ????????PSNR/SSIM/WM-MAE/SharpRatio ?????????????? disease-sensitive corrector ?? A080 ????????????????
2. ????????U-Net/StackUNet?PSNR ??????? SharpRatio ???????????????Pix2Pix/CycleGAN/DDIM/DBM ??????????????
3. ???? `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` ????????????? `PM_DIRF_FIDELITY_FLOW_FULL` ???????????????????????? DS hybrid ?????
4. `ADNI_PRIOR_FLOW_SAFE_PROBE_FULL` / template-source PriorFlow ????????? template/prior-flow ??????????????????????
5. ??????????????????? ACC/AUC/F1 ???????????????????

## ????

- ???? CSV?`outputs/icdm2026/final_validation/tables/final_unified_image_metrics.csv`
- ADNI ?????`outputs/icdm2026/final_validation/tables/final_adni_downstream_mil_aggregate.csv`
- ?????????`outputs/icdm2026/final_validation/tables/reference_dual_dataset_downstream_mil_summary.csv`
- ADNI ?????`outputs/icdm2026/final_validation/figures/adni_compact_panels`
- ADNI ??????`outputs/icdm2026/final_validation/figures/adni_texture_panels`
- ???????`outputs/icdm2026/final_validation/figures/private_stage2_panels`