# Final Quality Control Checklist

| check | status | evidence |
| --- | --- | --- |
| ADNI manifest used | PASS | ADNI downstream uses data/adni_processed/adni_slice_manifest.csv. |
| ADNI final method is FULL_E12_SCOREBEST | PASS | Prediction folder and method alias ADNI_A080_DS_FULL point to ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST. |
| ADNI core fair baselines present | PASS | T1_ONLY, FA_GT, Old Fidelity Flow, A080 Base, A080+DS Full, UNet, Pix2Pix, CycleGAN are included; LightGuard PriorFlow also included. |
| ADNI heavier baselines DDIM/DBM/MOTFM | PARTIAL | Test folders/checkpoints exist but train-full prediction folders are missing; excluded from strict fair train/test downstream table. |
| Private subject-level split | PASS | Private downstream uses data/processed train/test folders and subject-level aggregation. |
| Private FidelityFlow train export | PASS | PM_DIRF_FIDELITY_FLOW_FULL_TRAIN_FULL exported with 8650 train PNGs. |
| Private core fair baselines present | PASS | T1_ONLY, FA_GT, FidelityFlow, PRIVATE_DS_HYBRID, UNet, Pix2Pix, CycleGAN included. |
| No test-as-train | PASS | All strict fair methods have separate train and test prediction folders. |
| Report honesty | PASS | Reports state A080+DS Full is ADNI Macro-AUC leader but not universally first across all downstream metrics/datasets. |
