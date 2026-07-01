# Figure usage guide

| Figure | Placement | Supported claim | Do not over-interpret | Reviewer response |
|---|---|---|---|---|
| Fig1 | Main Methods | Final method is A080 prior + DS bounded corrector | Not pure PMRF or pure GAN | Cite export summary and DS corrector code |
| Fig2 | Main Methods | A080 is an FA-space clipped high-frequency prior | Not an online Stage1 checkpoint | Cite blend script and manifest |
| Fig3 | Main Methods | Stage2 is a disease-sensitive bounded corrector | Not a full FA generator | Cite 9-channel condition and multihead correction |
| Fig4 | Methods/Appendix | Checkpoint selection is joint and constrained | Not PSNR-only optimization | Cite build_loss and selection_score |
| Fig5 | Main Results | Integrated balance across metrics | Not best on all metrics | Explain metric direction and missing-data handling |
| Fig6 | Main Results | Stage2 improves fidelity/medical consistency while preserving sharpness | Not texture regeneration from scratch | Use A080 Base -> A080+DS Full contrast |
| Fig7 | Main Results | Fair MIL downstream evidence | Do not mix test-CV supplementary | Label fair train/test MIL |
| SuppFig1 | Supplement | Failed branches are design lessons | Do not draw them as final components | Present as ablation/lesson |
| SuppFig2 | Supplement | Private audited supplementary result | Not fully rerun identical protocol unless proven | Label existing audited results |
