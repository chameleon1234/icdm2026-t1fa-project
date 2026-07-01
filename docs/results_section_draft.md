# Results Section Draft

## Dual-Dataset Fair Downstream Closure

Both ADNI and the private dataset have completed fair subject-level train/test downstream evaluation. Downstream classifiers are trained on the training split and evaluated on the held-out test split, using subject-level aggregation or MIL-level outputs. Test predictions are not used as training predictions, and test-CV results are not mixed into the main downstream conclusion.

## ADNI Results

On ADNI, `A080+DS Full` achieves the highest average Macro-AUC (0.7226) in the latest fair all-method downstream evaluation. This supports the downstream utility of the final ADNI method.

However, `A080+DS Full` is not the best method for every downstream metric: DDIM obtains higher Accuracy and MOTFM obtains higher Macro-F1. Therefore, the ADNI downstream conclusion should be stated as:

`A080+DS Full achieves the highest Macro-AUC and competitive Accuracy/F1, supporting downstream utility, not universal downstream superiority.`

From the image and medical-fidelity perspective, `A080+DS Full` provides the most stable integrated balance across reconstruction fidelity, white-matter/ROI consistency, and sharpness preservation. The Macro-AUC result complements, rather than replaces, this image-level evidence.

## Private Dataset Results

On the private dataset, `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` is not the strongest standalone downstream classifier. Its average Macro-AUC is 0.5913, below UNet (0.7244), FA_GT (0.7046), Pix2Pix (0.6997), and Stage1_LPIPS_GAN (0.6950).

The per-task sanity check shows that the weakest task for `PRIVATE_DS_HYBRID` is `CN vs MCI-spectrum`, with Accuracy = 0.4706, Macro-AUC = 0.4406, and Macro-F1 = 0.4112. `CN vs AD` and `MCI-spectrum vs AD` are valid and run normally, but they are also not uniformly leading.

At the same time, `PRIVATE_DS_HYBRID_FROM_STAGE1_E030_FULL` ranks first by the integrated image/medical/texture/downstream score. Therefore, the private-dataset conclusion should emphasize integrated balance rather than downstream dominance.

## Safe Dual-Dataset Interpretation

The dual-dataset results suggest that synthetic FA is better framed as a complementary white-matter representation to T1, rather than a replacement for T1 or real DTI/FA. The final method selection should be justified by integrated balance across reconstruction quality, white-matter/ROI medical consistency, texture preservation, and downstream utility.

The manuscript should not claim that the method is first on every metric, that both datasets achieve downstream superiority, or that generated FA can fully replace real FA.
