# E2EDiT 代码审计报告

## 审计范围

项目根目录：`E:\桌面\T1_to_FA_Generation`

本次新主线为 **E2EDiT**，不复用旧 FADiT/A080/PriorFlow/template/offline blend prediction folders。

## ADNI 数据路径

已确认存在：

- `data/adni_processed/adni_slice_manifest.csv`
- `data/adni_processed/train/t1_slices`
- `data/adni_processed/train/fa_slices`
- `data/adni_processed/val/t1_slices`
- `data/adni_processed/val/fa_slices`
- `data/adni_processed/test/t1_slices`
- `data/adni_processed/test/fa_slices`

`data/adni_processed` 下存在 `train`、`val`、`test` 三个 split。E2EDiT smoke 只使用 train/val，不使用 test set 调参。

## 图像值域

现有 `src/datasets.py::T1FADataset` 在初始化时打印：

`Dataset initialized. Found ... images. Range: [-1, 1]`

其读取流程为：

1. 从 PNG 读取图像。
2. resize 到 `224 × 224`。
3. 转为 tensor。
4. 使用 `(img / 127.5) - 1.0` 映射到 `[-1, 1]`。

因此 E2EDiT 训练和值域假设统一为 `[-1, 1]`。导出 PNG 时使用反变换 `(x + 1) * 127.5`。

## 可复用模块

可以复用：

- `src.datasets.T1FADataset`：用于 ADNI train/val/test 读取，默认不 preload RAM。
- `pmrf_t1fa.train_pmrf_t1fa_stage1` 中的设计思想：WM proxy、ROI grid loss、SSIM、masked L1。
- `pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector` 中 bounded corrector 的思想：小幅 correction、zero-init、HF preserve、stripe/ROI/WM 约束。
- `scripts/evaluate_method_folder.py` 的最终文件夹评估逻辑可作为 full 阶段外部评估参考。

## 不可复用或仅可参考模块

不能作为 E2EDiT 输入或训练依赖：

- `outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080`
- `outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_4096_E8_BEST_K8`
- `outputs/icdm2026/predictions/ADNI_STAGE1_PRIOR_FLOW_LIGHTGUARD_4096_E8_BEST_K25`
- `outputs/icdm2026/predictions/*blend_manifest.csv`
- 任何 template / mean FA prediction folder
- 任何 A080 / PriorFlow / K8 / K25 teacher prediction folder

这些只能作为历史方法解释或最终对照，不能进入 E2EDiT 训练路径。

## 新增模块

本次新增：

- `pmrf_t1fa/e2edit.py`
- `scripts/train_e2edit_smoke.py`
- `scripts/evaluate_e2edit.py`
- `scripts/export_e2edit_predictions.py`
- `scripts/train_e2edit_full.py`
- `tests/test_e2edit.py`
- `docs/e2edit_theory_cn.md`
- `docs/e2edit_theory.md`

## 是否发现 A080 / PriorFlow / template 依赖

E2EDiT 新代码不读取 A080、PriorFlow、template 或 offline blend prediction folder。

`train_e2edit_smoke.py` 的输入只有：

- `train_t1_dir`
- `train_fa_dir`
- `val_t1_dir`
- `val_fa_dir`

M3 corrector 的输入来自 M2 Stage1 checkpoint，而不是外部 prediction folder。

## 当前评估能力

E2EDiT 新模块内置 smoke 级指标：

- PSNR
- SSIM
- MAE
- WM-MAE proxy
- ROI-CCC grid proxy
- SharpRatio
- Tenengrad ratio
- HF-Corr
- Overbright proxy
- B/H branch diagnostics
- correction magnitude
- correction high-pass change

这些指标足以完成 smoke go/no-go，但 full paper 阶段仍应继续使用 `scripts/evaluate_method_folder.py` 做标准 test folder 评估。

## 内存安全

`T1FADataset(preload_ram=False)` 被用于 E2EDiT 脚本。`DataLoader(pin_memory=False, num_workers=0)` 是默认安全配置，避免再次出现 RAM 爆满和磁盘 100% 的问题。

## 审计结论

E2EDiT 当前实现路径满足：

- 端到端 T1 -> FA。
- 不使用 A080 / PriorFlow / template / 离线融合。
- 不使用 test set 训练或选择超参。
- smoke 先验证 M0/M1/M2/M3，再根据 go/no-go 决定是否 full。

它仍然只是 smoke pipeline，不应在结果未达标前作为最终主方法。
