# Nature-style Final Model Figure Plan

Generated at: 2026-07-01T14:11:36

Skill: `nature-figure`. Backend: `Python / matplotlib`.

## General rules

- No new model training, no new prediction export, and no final-method modification.
- No fabricated metrics; every number comes from existing CSV, JSON, Markdown, or PNG files.
- Final method: `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`, shorthand `A080+DS Full`.
- Synthetic FA is a complementary white-matter representation, not a replacement for real FA.

## File audit

| Path | Status | Notes |
|---|---|---|
| `scripts/blend_highpass_detail.py` | exists | file |
| `pmrf_t1fa/train_pmrf_t1fa_stage2_ds_corrector.py` | exists | file |
| `scripts/export_ds_corrector_predictions.py` | exists | file |
| `outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST/export_summary.json` | exists | file |
| `outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_summary.json` | exists | file |
| `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080/blend_manifest.csv` | exists | file |
| `outputs/icdm2026/roi_weights/adni_disease_sensitive_roi_weights_2x3_top4.csv` | exists | file |
| `outputs/icdm2026/final_selection_a080_ds_full/adni_final_main_table.csv` | exists | file |
| `outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table.csv` | exists | file |
| `outputs/icdm2026/final_selection_a080_ds_full/figures` | exists | directory |
| `outputs/icdm2026/paper_assets/tables` | exists | directory |
| `outputs/icdm2026/paper_assets/figures` | missing | not used for numeric claims |
| `docs/final_model_architecture_notes_cn.md` | missing | not used for numeric claims |
| `docs/final_model_architecture_notes.md` | missing | not used for numeric claims |
| `docs/model_figure_checklist_cn.md` | missing | not used for numeric claims |
| `outputs/icdm2026/final_selection_a080_ds_full/final_method_story_cn.md` | exists | file |
| `outputs/icdm2026/final_selection_a080_ds_full/final_method_story.md` | exists | file |
| `docs/model_architecture_for_figures_cn.md` | exists | file |
| `docs/model_architecture_for_figures.md` | exists | file |
| `docs/figure_drawing_checklist_cn.md` | exists | file |

## Figure 1: Overall framework

Main path: `T1 slice -> A080 frequency-aware sharp FA prior -> Disease-sensitive high-frequency-preserving corrector -> Synthetic FA`, with T1 and disease-sensitive ROI weights feeding the corrector and Synthetic FA feeding the evaluation module.

## Figure 2: A080 prior construction

Formula: `base_hp = Base - LP(Base)`; `detail_hp = Detail - LP(Detail)`; `delta_hp = clip(detail_hp - base_hp, -0.06, 0.06)`; `A080 = Base + 0.8 * delta_hp`. Mark `alpha=0.8`, `kernel=9`, and `clip_delta=0.06`.

## Figure 3: DS corrector

Show the 9-channel condition, `Conv2d(9 -> 48) + 8 x NAFBlock(48) + Conv2d(48 -> 4)`, low/high/stripe heads, uncertainty head, ROI gate, and `final = clamp(A080 + correction, -1, 1)`.

## Figure 4: Loss and checkpoint selection

Group losses into reconstruction fidelity, medical fidelity, texture preservation, and artifact control. Present selection as a joint criterion rather than PSNR maximization.

## Figure 5: Quantitative result summary

Use `adni_final_main_table.csv` first and final JSON only as fallback. Show PSNR, SSIM, WM-MAE, ROI-CCC, Sharpness Ratio, and downstream ACC/AUC/F1.

## Figure 6: A080 Base vs A080+DS Full ablation

Show that Stage2 preserves A080 sharpness while improving reconstruction and medical consistency.

## Figure 7: Downstream protocol and fair comparison

Main figure contains fair train/test MIL methods only: T1_ONLY, FA_GT, Old Fidelity Flow, A080 Base, A080+DS Full.

## Supplementary Figures

SuppFig1 shows failed-route design lessons. SuppFig2 shows private existing audited results.
