# Pix2Pix 当前 split 探针实验

## 目的

在发现旧 U-Net checkpoint 存在 subject leakage 之后，对比方法也需要逐个检查。必要时要在当前 `data/processed/train`、`data/processed/val`、`data/processed/test` 被试划分上重新训练。

这一步新增了一个可复现的 Pix2Pix 当前 split 训练入口，并先跑 10 epoch 探针实验，判断是否值得直接投入 100 epoch 长训。

## 新增代码

- `scripts/train_pix2pix_current_split.py`
- `tests/test_train_pix2pix_current_split.py`

训练脚本会在训练前检查 train/val/test 的 subject 是否互斥，并按验证集 PSNR 保存 `best_pix2pix_generator.pt`。

## 探针训练命令

```powershell
D:\Anaconda3\envs\dinov3test\python.exe -m scripts.train_pix2pix_current_split `
  --run_name pix2pix_current_split_probe_e10 `
  --epochs 10 `
  --batch_size 16 `
  --lr 2e-4 `
  --lambda_l1 100 `
  --mixed_precision bf16 `
  --device cuda `
  --save_every 5
```

最佳验证 checkpoint 出现在第 8 轮：

- `val_PSNR=27.3103`
- checkpoint: `outputs/pix2pix_current_split_probe_e10/checkpoints/best_pix2pix_generator.pt`

## 测试集图像指标

| 方法 | PSNR | SSIM | MAE | SharpRatio | WM-MAE | ROI-CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Pix2Pix_CurrentSplit_E10` | 27.575 | 0.893 | 0.0188 | 1.242 | 0.0600 | 0.882 |
| `Pix2Pix_E100` | 27.760 | 0.895 | 0.0183 | 0.891 | 0.0601 | 0.856 |
| `UNet_CurrentSplit_E100` | 28.236 | 0.897 | 0.0178 | 0.456 | 0.0575 | 0.836 |
| `FREQ_FLOWBASE_B035` | 28.180 | 0.902 | 0.0176 | 0.842 | 0.0571 | 0.850 |

## 下游 utility

主任务：`CN+SCD vs MCI+AD`，20 个随机种子重复 5 折交叉验证。

| 方法 | Macro-F1 |
| --- | ---: |
| `Pix2Pix_CurrentSplit_E10` | 0.441 +/- 0.048 |
| `Pix2Pix_E100` | 0.530 +/- 0.055 |
| `UNet_CurrentSplit_E100` | 0.610 +/- 0.054 |
| `FREQ_FLOWBASE_B035` | 0.572 +/- 0.048 |
| `T1_ONLY` | 0.519 +/- 0.057 |
| `FA_GT` | 0.519 +/- 0.043 |

## 解释

10 epoch 当前 split Pix2Pix 在测试集上非常锐，`SharpRatio=1.242`，但是下游 utility 明显崩掉，`Macro-F1=0.441`。这很可能是早期 GAN 不稳定状态：模型很快学会了增加高频对比，但这些高频并没有和真实 FA 结构、白质疾病相关信号对齐。

这个结果对论文叙事是有用的：清晰度本身不够，必须同时评估 paired fidelity、医学 ROI/WM 指标和下游疾病 utility。我们的频率/Flow 平衡路线需要和 GAN 方法在这些指标上一起比较。

## 下一步

如果需要一行可发表的当前 split GAN baseline，再跑完整 Pix2Pix：

```powershell
D:\Anaconda3\envs\dinov3test\python.exe -m scripts.train_pix2pix_current_split `
  --run_name pix2pix_current_split_e100 `
  --epochs 100 `
  --batch_size 16 `
  --lr 2e-4 `
  --lambda_l1 100 `
  --mixed_precision bf16 `
  --device cuda `
  --save_every 10
```

根据 10 epoch 探针速度估计，当前 GPU 上 100 epoch 大约需要 3.5 小时。
