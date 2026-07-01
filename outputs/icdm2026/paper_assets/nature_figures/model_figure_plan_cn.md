# Nature-style 最终模型绘图计划

生成时间：2026-07-01T14:11:36

调用 skill：`nature-figure`。后端：`Python / matplotlib`。

## 总体原则

- 不训练新模型，不导出新预测，不修改最终主方法。
- 不编造指标；所有数值只来自已有 CSV、JSON、Markdown 或 PNG。
- 最终方法固定为 `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`，论文简称 `A080+DS Full`。
- 合成 FA 是 T1 的补充性白质结构表征，不是替代真实 FA。

## 文件审计

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

## Figure 1：Overall framework

主线：`T1 slice -> A080 frequency-aware sharp FA prior -> Disease-sensitive high-frequency-preserving corrector -> Synthetic FA`。同时画出 T1 和 disease-sensitive ROI weights 作为 DS corrector 输入，并把 Synthetic FA 连接到 evaluation module。

## Figure 2：A080 prior construction

公式：`base_hp = Base - LP(Base)`；`detail_hp = Detail - LP(Detail)`；`delta_hp = clip(detail_hp - base_hp, -0.06, 0.06)`；`A080 = Base + 0.8 * delta_hp`。标注 `alpha=0.8`、`kernel=9`、`clip_delta=0.06`。

## Figure 3：DS corrector

画 9-channel condition、`Conv2d(9 -> 48) + 8 x NAFBlock(48) + Conv2d(48 -> 4)`、low/high/stripe heads、uncertainty head、ROI gate 和 `final = clamp(A080 + correction, -1, 1)`。

## Figure 4：Loss and checkpoint selection

四类模块：reconstruction fidelity、medical fidelity、texture preservation、artifact control。checkpoint selection 表述为 joint criterion，不写单纯 PSNR 最大化。

## Figure 5：Quantitative result summary

从 `adni_final_main_table.csv` 优先读取；如缺失则用 final summary JSON。展示 PSNR、SSIM、WM-MAE、ROI-CCC、Sharpness Ratio、Downstream ACC/AUC/F1。

## Figure 6：A080 Base vs A080+DS Full ablation

展示 Stage2 的作用：保留 A080 prior sharpness，同时提升 reconstruction 和 medical consistency。

## Figure 7：Downstream protocol and fair comparison

主图只放 fair train/test MIL：T1_ONLY、FA_GT、Old Fidelity Flow、A080 Base、A080+DS Full。UNet/Pix2Pix/CycleGAN 若是 test-CV supplementary，不混入主图。

## Supplementary Figures

SuppFig1 展示 failed-route design lessons。SuppFig2 展示 private existing audited results。
