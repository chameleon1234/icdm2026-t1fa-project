# Figure captions

## Figure 1. Overall framework
The final T1-to-FA synthesis framework first constructs an A080 frequency-aware sharp FA prior from a T1 single slice, then applies a disease-sensitive high-frequency-preserving corrector conditioned on T1 and disease-sensitive ROI weights. The resulting synthetic FA is evaluated with reconstruction, medical-consistency, texture, and downstream-utility metrics.

## Figure 2. A080 frequency-aware sharp FA prior construction
The A080 prior blends stable low-frequency FA structure with clipped FA-space high-frequency detail. The high-frequency delta is clipped at 0.06 and added to the base prediction with alpha = 0.8, yielding a frozen frequency-aware prior rather than an online trainable Stage1 checkpoint.

## Figure 3. Disease-sensitive high-frequency-preserving corrector
The Stage2 corrector receives a 9-channel condition tensor and uses a compact NAFBlock backbone to predict low-, high-, and stripe-correction heads plus an uncertainty/log-sigma head. The correction is bounded and ROI-gated, so Stage2 acts as a medical-consistency corrector rather than a full generator.

## Figure 4. Loss and checkpoint selection design
Losses are grouped into reconstruction fidelity, medical fidelity, texture preservation, and artifact control. The final checkpoint is selected by a joint criterion that balances reconstruction quality, white-matter/ROI consistency, sharpness preservation, and artifact suppression.

## Figure 5. ADNI quantitative result summary
This figure summarizes the ADNI final comparison using favorable normalized scores across reconstruction, medical, texture, and downstream metrics. WM-MAE is inverted and Sharpness Ratio is scored by closeness to 1.

## Figure 6. A080 Base to A080+DS Full ablation
This ablation shows that Stage2 improves reconstruction and medical consistency over A080 Base while preserving the sharpness of the A080 prior.

## Figure 7. Downstream protocol and fair MIL comparison
The downstream protocol aggregates slice-level ROI/statistical features at subject level and evaluates fair train/test MIL classification. The main panel includes only T1_ONLY, FA_GT, Old Fidelity Flow, A080 Base, and A080+DS Full.

## Supplementary Figure 1. Exploratory design lessons
This supplementary figure summarizes the main non-final exploratory branches and their limitations. These branches are design lessons or ablations, not final-model components.

## Supplementary Figure 2. Private dataset supplementary results
This supplementary figure reports existing audited private-dataset results. It supports complementary evidence but does not claim a fully rerun identical protocol unless separately verified.
