# 最终表格修复质量检查

| 检查项 | 状态 | 证据 |
|---|---|---|
| ADNI Pix2Pix / U-Net / CycleGAN downstream 是否合并成功 | PASS | fixed 表中三者均已有 downstream_ACC、downstream_AUC、downstream_F1 |
| A080+DS Full 是否仍为 ADNI primary integrated score 第一 | PASS | ADNI primary score = 0.9642，rank = 1 |
| 是否混用 fair MIL 和 test-CV | PASS | fixed 表使用 strict train/test subject-level downstream 表 |
| 是否把 test predictions 当作 train predictions | PASS | coverage 表保留独立 train/test prediction folder |
| 中文报告是否无乱码 | PASS | 已生成 `final_dual_downstream_closure_cn_clean.md` |
| 是否保留 raw metrics | PASS | fixed integrated table 保留 PSNR、SSIM、MSE、MAE、WM-MAE、ROI、纹理和下游指标 |
| 是否删除不利结果 | PASS | 私有集中 T1_ONLY Macro-AUC 最高、PRIVATE_DS_HYBRID 非下游全第一等结果均保留 |
| 是否写了“所有指标第一” | PASS | 报告明确禁止该表述，采用综合平衡表述 |
