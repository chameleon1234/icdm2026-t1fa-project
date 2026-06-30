# Final Downstream Table

## ADNI latest full-data final method downstream: train/test MIL

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| ADNI_A080_BASE | 0.736 | 0.702 | 0.600 | 9 |
| ADNI_A080_DS_FULL | 0.710 | 0.723 | 0.593 | 9 |
| T1_ONLY | 0.690 | 0.709 | 0.582 | 9 |
| FA_GT | 0.724 | 0.676 | 0.573 | 9 |
| ADNI_OLD_FIDELITY_FLOW | 0.678 | 0.713 | 0.573 | 9 |

## ADNI old/supplemental test split subject-level CV

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| FA_GT | 0.715 | 0.704 | 0.607 | 3 |
| ADNI_PIX2PIX | 0.623 | 0.466 | 0.517 | 3 |
| ADNI_A080_DS_FULL | 0.687 | 0.533 | 0.515 | 3 |
| ADNI_OLD_FIDELITY_FLOW | 0.658 | 0.617 | 0.506 | 3 |
| T1_ONLY | 0.689 | 0.577 | 0.504 | 3 |
| ADNI_UNET | 0.631 | 0.485 | 0.493 | 3 |
| ADNI_A080_BASE | 0.629 | 0.439 | 0.478 | 3 |
| ADNI_CYCLEGAN | 0.583 | 0.423 | 0.453 | 3 |

## Private existing train/test MIL

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| UNet | 0.721 | 0.701 | 0.661 | 9 |
| T1_PLUS_UNet | 0.696 | 0.687 | 0.660 | 9 |
| T1_PLUS_Ours | 0.708 | 0.620 | 0.649 | 9 |
| T1_PLUS_GT | 0.718 | 0.641 | 0.647 | 9 |
| T1_ONLY | 0.680 | 0.654 | 0.647 | 9 |
| T1_PLUS_Pix2Pix | 0.667 | 0.685 | 0.634 | 9 |
| T1_PLUS_Stage1 | 0.669 | 0.596 | 0.616 | 9 |
| T1_PLUS_CycleGAN | 0.647 | 0.655 | 0.612 | 9 |
| Pix2Pix | 0.663 | 0.708 | 0.606 | 9 |
| FidelityFlow | 0.683 | 0.650 | 0.601 | 9 |
| CycleGAN | 0.644 | 0.598 | 0.584 | 9 |
| FA_GT | 0.663 | 0.615 | 0.579 | 9 |
| Stage1 | 0.627 | 0.663 | 0.565 | 9 |

## Private existing repeated subject-level CV

| Method | Mean ACC | Mean AUC | Mean Macro-F1 | Rows |
|---|---:|---:|---:|---:|
| T1_PLUS_Pix2Pix | 0.690 | 0.605 | 0.576 | 3 |
| CycleGAN_CurrentSplit_E100 | 0.638 | 0.532 | 0.558 | 3 |
| Pix2Pix_CurrentSplit_E100 | 0.627 | 0.528 | 0.533 | 3 |
| T1_ONLY | 0.642 | 0.465 | 0.507 | 3 |
| T1_PLUS_UNet | 0.648 | 0.536 | 0.501 | 3 |
| T1_PLUS_GT | 0.634 | 0.407 | 0.482 | 3 |
| T1_PLUS_CycleGAN | 0.619 | 0.495 | 0.478 | 3 |
| UNet_CurrentSplit_E100 | 0.566 | 0.386 | 0.478 | 3 |
| T1_PLUS_Stage1 | 0.624 | 0.500 | 0.468 | 3 |
| T1_PLUS_Ours | 0.623 | 0.501 | 0.468 | 3 |
| FA_GT | 0.575 | 0.366 | 0.466 | 3 |
| Fidelity_Flow | 0.549 | 0.454 | 0.465 | 3 |
| Stage1_LPIPS_GAN | 0.544 | 0.453 | 0.461 | 3 |

Notes:
- ADNI train/test MIL uses `data/adni_processed/adni_slice_manifest.csv` and subject-level aggregation.
- ADNI U-Net/Pix2Pix/CycleGAN do not currently have train-full prediction folders, so they are included only in subject-level test-CV supplement, not in train/test MIL.
- Per-class F1 is not emitted by the current downstream scripts; macro-F1 is available for all protocols.
