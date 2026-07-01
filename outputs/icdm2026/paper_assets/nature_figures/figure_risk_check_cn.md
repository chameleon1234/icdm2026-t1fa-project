# Figure risk check（中文）

| Risk | Status | Note |
|---|---|---|
| 是否把 A080+DS Full 画成所有指标全面第一 | PASS | 图中使用 integrated balance / normalized favorable score，不写所有指标第一 |
| 是否混淆 fair train/test MIL 和 subject-level test-CV supplementary | PASS | Fig7 只放 fair train/test MIL，supplementary 单独标注 |
| 是否忽略 Old Fidelity Flow 的 ROI-CCC 优势 | PASS_WITH_NOTE | Fig5 保留 Old Fidelity Flow；正文需说明 ROI-CCC 单项优势 |
| 是否把 synthetic FA 描述为真实 FA 替代品 | PASS | caption 和 plan 均写 complementary representation |
| 是否把 disease-sensitive grid ROI 误画成 anatomical atlas | PASS | Fig3 标注 2 x 3 data-driven grid ROI |
| 是否把 A080 画成单一在线神经网络 Stage1 | PASS | Fig2 标注 frozen frequency-aware prior |
| 是否把 Stage2 画成完整生成器 | PASS | Fig3 标注 bounded corrector |
| 是否把 Sharpness Ratio 误画成越高越好 | PASS | Fig5/Fig6 标注 closer to 1 is better |
| 是否使用无法核验数字 | PASS | 量化图从现有 CSV/JSON 读取 |
| 是否把失败探索分支画成最终主方法的一部分 | PASS | SuppFig1 单独作为 design lessons |
