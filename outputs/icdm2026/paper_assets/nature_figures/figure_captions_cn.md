# Figure captions（中文）

## Figure 1. Overall framework
本图展示最终 T1-to-FA synthesis 框架：T1 single slice 首先通过 A080 frequency-aware sharp FA prior 获得清晰 FA prior，随后 disease-sensitive high-frequency-preserving corrector 使用 T1 condition 和 disease-sensitive ROI weights 进行有界医学一致性校正，最终输出 synthetic FA 并进入重建、医学一致性、纹理和下游任务评估。

## Figure 2. A080 frequency-aware sharp FA prior construction
本图展示 A080 prior 的频域构建方式。稳定低频 FA source 提供 FA 低频结构和亮度空间，FA-space high-frequency detail source 提供纹理候选；二者的高频差异经 `clip_delta=0.06` 限制后以 `alpha=0.8` 加回 Base，得到 frozen A080 prior。

## Figure 3. Disease-sensitive high-frequency-preserving corrector
本图展示 Stage2 corrector 的结构。Corrector 接收 9-channel condition，经 `Conv2d(9 -> 48) + 8 x NAFBlock(48) + Conv2d(48 -> 4)` 输出 low/high/stripe correction heads 与 uncertainty/log-sigma head。最终校正是有界小幅 correction，而不是完整重新生成。

## Figure 4. Loss and checkpoint selection design
本图将训练目标分为 reconstruction fidelity、medical fidelity、texture preservation 和 artifact control 四类，并说明最终 checkpoint 由综合标准选择，而不是单纯最大化 PSNR。

## Figure 5. ADNI quantitative result summary
本图基于现有 ADNI final table 展示 A080+DS Full 与主要对比方法在重建、医学一致性、纹理和下游指标上的归一化有利分数。WM-MAE 采用 lower-is-better 方向，Sharpness Ratio 采用 closer-to-1-is-better 方向。

## Figure 6. A080 Base to A080+DS Full ablation
本图展示 Stage2 的实际贡献：A080+DS Full 相比 A080 Base 提升 PSNR、SSIM、WM-MAE 和 ROI-CCC，同时保持接近 1 的 Sharpness Ratio。

## Figure 7. Downstream protocol and fair MIL comparison
本图展示 slice-level ROI/statistical features 到 subject-level aggregation/MIL 的下游评估流程，并仅比较 fair train/test MIL 设置下的 T1_ONLY、FA_GT、Old Fidelity Flow、A080 Base 和 A080+DS Full。

## Supplementary Figure 1. Exploratory design lessons
本附录图总结 posterior mean blur、LPIPS/GAN over-bright artifacts、direct T1 high-pass mismatch 和 template/PriorFlow 安全但细节不足等探索分支。这些分支是设计经验或消融，不是最终模型组件。

## Supplementary Figure 2. Private dataset supplementary results
本附录图展示私有集已有审计结果，用于补充说明 PRIVATE_DS_HYBRID 的综合图像/医学/纹理平衡和下游表现。该图不声称与 ADNI 完全相同协议下重新训练。
