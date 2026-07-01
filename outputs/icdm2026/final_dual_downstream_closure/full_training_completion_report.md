# Full Training/Export and Downstream Completion Audit

The missing full train prediction exports were completed for ADNI DDIM, DBM, MOTFM, and private Stage1 LPIPS+GAN. ADNI fair train/test downstream evaluation was rerun with all methods. ADNI_A080_DS_FULL remains ranked first by average Macro-AUC, while MOTFM is strong in Macro-F1/Accuracy but weaker in reconstruction and ROI fidelity.

## Latest ADNI Downstream Averages

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
