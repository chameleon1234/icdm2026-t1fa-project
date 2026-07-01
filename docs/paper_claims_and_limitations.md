# Paper Claims and Limitations

## Safe Claims

1. On ADNI, `A080+DS Full` achieves the highest average Macro-AUC in the latest fair full-heavy downstream audit, but it is not the best in Accuracy or Macro-F1.
2. Both datasets have completed subject-level train/test downstream evaluation, without using test predictions as training predictions.
3. Synthetic FA should be described as a complementary white-matter representation to T1, not as a replacement for T1 or real DTI/FA.
4. The final method selection is based on an integrated balance: image reconstruction fidelity 25%, medical fidelity 30%, texture realism 20%, and downstream utility 25%.

## Required Limitations

1. Downstream classification depends on dataset size, task difficulty, label distribution, feature extraction, and classification protocol.
2. On the private dataset, T1_ONLY / UNet / FA_GT / Pix2Pix are stronger than PRIVATE_DS_HYBRID on some downstream metrics.
3. PRIVATE_DS_HYBRID is not the top private downstream method by Macro-AUC; its value is the integrated balance across image quality, medical consistency, and texture realism.
4. The primary value of generated FA is complementary white-matter structural representation and medical consistency, not guaranteed optimality for every classification task.
5. Synthetic FA cannot fully replace real DTI/FA; it should be used as a complementary representation when DTI/FA is unavailable.
6. The private dataset is relatively small, with an average of about 25.33 test subjects at the subject-level task setting, so classification conclusions should be interpreted cautiously.
