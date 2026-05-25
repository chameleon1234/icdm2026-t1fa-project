# ICDM 2026 T1-to-FA 论文实施计划

本文件是 `docs/icdm2026_execution_plan.md` 的中文版本。详细任务以英文计划为准，中文版本用于快速阅读和团队沟通。

## 总目标

建立一套可复现的 ICDM Applied Track 论文管线：用医学保真指标评价 T1-to-FA 合成，并用阿尔兹海默症分期验证合成 FA 的数据挖掘价值。

## 当前阶段

优先执行 Task 1-3：

1. 创建 ICDM 项目配置和中英文进度日志。
2. 建立 subject-level 标签索引，固定 `1=CN, 2=SCD, 3=MCI, 4=AD`。
3. 建立带元数据的 T1/FA 切片 dataset，保持原有 `T1FADataset` 兼容。

## 后续阶段

4. 统一图像指标：PSNR、SSIM、MSE、MAE、白质 masked MAE、ROI-CCC 等。
5. 导出 PM-DIRF 预测。
6. 重评估已有 baseline。
7. 做 subject-level 下游分类。
8. 做消融实验。
9. 生成论文表格和图。
10. 重写 `main.tex`，将论文定位改为 missing diffusion biomarker mining for Alzheimer's staging。
11. 做最终可复现检查。

## 执行原则

- 所有划分必须 subject-level，禁止 slice-level 随机划分。
- FID/KID 不再作为主指标。
- 每次新增 `.md` 文件都要同步生成 `_cn.md`。
- 每次有阶段性更新都提交 commit 并同步到 GitHub。

