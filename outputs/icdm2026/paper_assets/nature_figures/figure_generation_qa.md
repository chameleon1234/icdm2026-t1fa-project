# Nature figure QA summary

- [x] `Fig1_overall_framework.svg` size=10902
- [x] `Fig1_overall_framework.pdf` size=47397
- [x] `Fig1_overall_framework.png` size=240159
- [x] `Fig2_a080_prior_construction.svg` size=12771
- [x] `Fig2_a080_prior_construction.pdf` size=46513
- [x] `Fig2_a080_prior_construction.png` size=354372
- [x] `Fig3_ds_corrector_architecture.svg` size=20505
- [x] `Fig3_ds_corrector_architecture.pdf` size=52888
- [x] `Fig3_ds_corrector_architecture.png` size=301133
- [x] `Fig4_loss_and_selection.svg` size=10119
- [x] `Fig4_loss_and_selection.pdf` size=42550
- [x] `Fig4_loss_and_selection.png` size=273001
- [x] `Fig5_adni_quantitative_summary.pdf` size=52041
- [x] `Fig5_adni_quantitative_summary.png` size=260605
- [x] `Fig6_a080_to_ds_ablation.pdf` size=48331
- [x] `Fig6_a080_to_ds_ablation.png` size=315239
- [x] `Fig7_downstream_protocol_and_fair_mil.pdf` size=50168
- [x] `Fig7_downstream_protocol_and_fair_mil.png` size=232722
- [x] `SuppFig1_design_lessons.pdf` size=41926
- [x] `SuppFig1_design_lessons.png` size=213480
- [x] `SuppFig2_private_results.pdf` size=43536
- [x] `SuppFig2_private_results.png` size=131691

## Required semantic checks

- [x] Fig1/Fig2/Fig3 contain the final method core modules.
- [x] Fig2 contains the A080 formula and alpha/kernel/clip_delta.
- [x] Fig3 contains 9-channel condition, NAFBlock, low/high/stripe heads, and ROI gate.
- [x] Fig5/Fig6 handle WM-MAE as lower-is-better and Sharpness Ratio as closer-to-1.
- [x] Fig7 separates fair MIL from supplementary test-CV.
- [x] No new model training was invoked.
- [x] Final main method was not modified.

Overall file existence status: PASS