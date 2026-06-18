$ErrorActionPreference = "Stop"

# Focused single-slice Stage 2 rerun after the first variant sweep.
# Run from repository root:
#   conda activate dinov3test
#   powershell -ExecutionPolicy Bypass -File scripts/run_single_slice_stage2_winners.ps1

$python = "D:\Anaconda3\envs\dinov3test\python.exe"
$stage1Ckpt = "outputs/pmrf_t1fa_stage1_single_sharp_stripe_smoke/checkpoints/best_stage1.pt"
$roiCsv = "outputs/icdm2026/roi_weights/private_disease_sensitive_roi_weights_4x4_top6.csv"

if (!(Test-Path $stage1Ckpt)) {
  throw "Missing Stage1 checkpoint: $stage1Ckpt"
}

if (!(Test-Path $roiCsv)) {
  & $python scripts/build_disease_sensitive_roi_weights.py `
    --config configs/icdm2026.yaml `
    --fa_dir data/processed/train/fa_slices `
    --split train `
    --tasks cn_vs_ad,cn_vs_mci,mci_vs_ad,cn_scd_vs_mci_ad `
    --roi_rows 4 `
    --roi_cols 4 `
    --top_k 6 `
    --output_csv $roiCsv
}

$variants = @("atlas", "hybrid")
foreach ($variant in $variants) {
  $runName = "pmrf_t1fa_stage2_single_ds_${variant}_2048_e5_stripe2"
  & $python -m pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector `
    --stage1_ckpt $stage1Ckpt `
    --run_name $runName `
    --variant $variant `
    --epochs 5 `
    --batch_size 2 `
    --train_limit 2048 `
    --val_limit 512 `
    --lr 8e-5 `
    --width 48 `
    --num_blocks 8 `
    --disease_roi_csv $roiCsv `
    --disease_roi_weight 1.5 `
    --roi_weight 0.5 `
    --wm_l1_weight 1.0 `
    --stripe_weight 2.0 `
    --hf_preserve_weight 1.0 `
    --best_max_delta_stripe 0.005 `
    --fid_eval_every 1 `
    --mixed_precision bf16
}
