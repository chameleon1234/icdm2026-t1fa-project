# Final Two-Stage Method

## Selected Final Method

The project is now locked to the direct two-stage model:

```text
T1 5-slice -> Stage 1 sharp FA generator -> Stage 2 fidelity flow corrector -> final FA
```

The selected ADNI output folder is:

```text
outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL
```

Frequency-fusion variants were removed from the final method because the two-stage fidelity-flow result provides the preferred visual quality and a cleaner method story.

## Stage 1: Sharp FA Generator

Stage 1 uses the 5-slice T1 context and a sharp adversarial training preset to avoid posterior-mean smoothing:

```text
T1_{z-2:z+2} -> sharp FA initialization
```

Checkpoint:

```text
outputs/adni_pmrf_stage1_lpips_gan_5slice_full_e80/checkpoints/best_stage1.pt
```

Exported prediction folder:

```text
outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL
```

## Stage 2: Fidelity Flow Corrector

Stage 2 is a flow-based medical-fidelity corrector. It refines the sharp Stage 1 output without introducing the separate frequency-fusion post-processing branch.

Checkpoint:

```text
outputs/adni_pmrf_stage2_fidelity_flow_full_e40/checkpoints/best_fidelity_corrector.pt
```

Exported prediction folder:

```text
outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL
```

## Final Paper Framing

The method should be described as a two-stage T1-to-FA synthesis framework:

1. A high-frequency-preserving Stage 1 generator prevents the posterior-mean blur observed in PMRF-style coarse prediction.
2. A fidelity flow corrector improves medical consistency while preserving the sharp structure provided by Stage 1.
3. Downstream AD classification evaluates whether the generated FA carries disease-relevant utility, not only paired image fidelity.

Do not describe Frequency Balanced Fusion as part of the final method.
