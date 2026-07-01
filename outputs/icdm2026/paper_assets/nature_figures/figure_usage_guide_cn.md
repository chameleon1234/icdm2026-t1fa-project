# Figure usage guide（中文）

| Figure | 建议位置 | 支持结论 | 不能过度解读 | 审稿质疑回应 |
|---|---|---|---|---|
| Fig1 | 主文 Methods | 最终方法是 A080 prior + DS bounded corrector | 不能说是纯 PMRF 或纯 GAN | 指向 export summary 和 DS corrector 代码 |
| Fig2 | 主文 Methods | A080 是 FA-space clipped high-frequency prior | 不能说 A080 是在线 Stage1 checkpoint | 指向 blend script 与 manifest |
| Fig3 | 主文 Methods | Stage2 是 disease-sensitive bounded corrector | 不能说 Stage2 完整重新生成 FA | 指向 9-channel condition 与 multihead correction |
| Fig4 | 主文 Methods/Appendix | 选优是综合标准 | 不能说只优化 PSNR | 指向 build_loss 与 selection_score |
| Fig5 | 主文 Results | 展示综合平衡 | 不能写所有指标第一 | 说明方向统一与缺失指标处理 |
| Fig6 | 主文 Results | Stage2 的真实增益 | 不能说 Stage2 生成新纹理 | 用 A080 Base -> A080+DS Full 对比解释 |
| Fig7 | 主文 Results | 下游 fair MIL 证据 | 不能混入 test-CV supplementary | 明确 fair train/test MIL |
| SuppFig1 | 附录 | 失败路线是设计经验 | 不能画进主模型 | 作为 ablation/lesson |
| SuppFig2 | 附录 | 私有集补充结果 | 不能声称完全同协议新跑 | 标注 existing audited results |
