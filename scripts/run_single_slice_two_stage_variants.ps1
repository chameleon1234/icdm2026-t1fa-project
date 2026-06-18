$ErrorActionPreference = "Stop"

# Single-slice two-stage experiment suite.
# Run from repository root:
#   conda activate dinov3test
#   powershell -ExecutionPolicy Bypass -File scripts/run_single_slice_two_stage_variants.ps1

$python = "D:\Anaconda3\envs\dinov3test\python.exe"
$stage1Run = "pmrf_t1fa_stage1_single_sharp_stripe_smoke"
$stage1Ckpt = "outputs/$stage1Run/checkpoints/best_stage1.pt"
$roiCsv = "outputs/icdm2026/roi_weights/private_disease_sensitive_roi_weights_4x4_top6.csv"

& $python scripts/build_disease_sensitive_roi_weights.py `
  --config configs/icdm2026.yaml `
  --fa_dir data/processed/train/fa_slices `
  --split train `
  --tasks cn_vs_ad,cn_vs_mci,mci_vs_ad,cn_scd_vs_mci_ad `
  --roi_rows 4 `
  --roi_cols 4 `
  --top_k 6 `
  --output_csv $roiCsv

& $python -m pmrf_t1fa.train_pmrf_t1fa_stage1 `
  --context_slices 1 `
  --run_name $stage1Run `
  --stage1_training_preset sharp_adversarial `
  --stage1_model_variant detail `
  --stage1_prediction_mode residual `
  --stage1_detail_scale 0.45 `
  --epochs 3 `
  --batch_size 4 `
  --train_limit 1024 `
  --val_limit 512 `
  --lr 8e-5 `
  --lpips_max_weight 0.08 `
  --adv_weight 0.006 `
  --wm_l1_weight 0.12 `
  --roi_consistency_weight 0.06 `
  --grad_weight 0.08 `
  --hf_weight 0.05 `
  --detail_hf_weight 0.08 `
  --detail_lap_weight 0.04 `
  --stripe_weight 1.0 `
  --best_metric detail_paired `
  --fid_eval_every 1 `
  --no_auto_resume `
  --mixed_precision bf16

$variants = @("metric", "ds_roi", "uncertainty", "atlas", "hybrid")
foreach ($variant in $variants) {
  $runName = "pmrf_t1fa_stage2_single_ds_${variant}_smoke"
  & $python -m pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector `
    --stage1_ckpt $stage1Ckpt `
    --run_name $runName `
    --variant $variant `
    --epochs 3 `
    --batch_size 2 `
    --train_limit 1024 `
    --val_limit 512 `
    --lr 8e-5 `
    --width 48 `
    --num_blocks 8 `
    --disease_roi_csv $roiCsv `
    --disease_roi_weight 1.5 `
    --roi_weight 0.5 `
    --wm_l1_weight 1.0 `
    --stripe_weight 1.0 `
    --hf_preserve_weight 1.0 `
    --fid_eval_every 1 `
    --mixed_precision bf16
}
