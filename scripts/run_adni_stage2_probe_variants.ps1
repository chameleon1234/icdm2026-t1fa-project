$ErrorActionPreference = "Continue"

$Python = "D:\Anaconda3\envs\dinov3test\python.exe"
$LogDir = "outputs\icdm2026\logs\adni_stage2_probe_variants"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stage1Ckpt = "outputs\adni_stage1_single_sharp_stripe_smoke_4096_e5\checkpoints\best_stage1.pt"
$TrainT1 = "data/adni_processed/train/t1_slices"
$TrainFA = "data/adni_processed/train/fa_slices"
$ValT1 = "data/adni_processed/val/t1_slices"
$ValFA = "data/adni_processed/val/fa_slices"

function Run-Probe {
  param(
    [string]$RunName,
    [int]$Width,
    [int]$Blocks,
    [double]$CorrectionScale,
    [double]$FinalL1,
    [double]$FinalMSE,
    [double]$FinalSSIM,
    [double]$WM,
    [double]$ROI,
    [double]$Stripe,
    [double]$HF,
    [double]$Bounded,
    [double]$SharpRetention
  )

  & $Python -m pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector `
    --train_t1_dir $TrainT1 `
    --train_fa_dir $TrainFA `
    --val_t1_dir $ValT1 `
    --val_fa_dir $ValFA `
    --stage1_ckpt $Stage1Ckpt `
    --run_name $RunName `
    --variant hybrid `
    --epochs 3 `
    --batch_size 2 `
    --train_limit 2048 `
    --val_limit 512 `
    --lr 8e-5 `
    --width $Width `
    --num_blocks $Blocks `
    --correction_scale $CorrectionScale `
    --final_l1_weight $FinalL1 `
    --final_mse_weight $FinalMSE `
    --final_ssim_weight $FinalSSIM `
    --wm_l1_weight $WM `
    --roi_weight $ROI `
    --disease_roi_weight 0.0 `
    --correction_l1_weight 0.35 `
    --bounded_weight $Bounded `
    --sharp_retention_weight $SharpRetention `
    --stripe_weight $Stripe `
    --hf_preserve_weight $HF `
    --uncertainty_weight 0.25 `
    --atlas_smooth_weight 0.15 `
    --best_min_sharp_retention 0.95 `
    --best_min_delta_disease_roi -1.0 `
    --best_max_delta_stripe 0.01 `
    --fid_eval_every 999 `
    --save_every 3 `
    --mixed_precision bf16 `
    > "$LogDir\$RunName.stdout.log" `
    2> "$LogDir\$RunName.stderr.log"
}

Run-Probe `
  -RunName "adni_stage2_probe_capacity_w64_b12_e3" `
  -Width 64 -Blocks 12 -CorrectionScale 0.18 `
  -FinalL1 0.25 -FinalMSE 0.20 -FinalSSIM 0.15 `
  -WM 1.0 -ROI 0.50 -Stripe 2.0 -HF 1.0 -Bounded 0.04 -SharpRetention 0.50

Run-Probe `
  -RunName "adni_stage2_probe_strongcorr_s030_e3" `
  -Width 48 -Blocks 8 -CorrectionScale 0.30 `
  -FinalL1 0.45 -FinalMSE 0.35 -FinalSSIM 0.25 `
  -WM 1.5 -ROI 0.80 -Stripe 4.0 -HF 0.80 -Bounded 0.02 -SharpRetention 0.75

Run-Probe `
  -RunName "adni_stage2_probe_metricheavy_s026_e3" `
  -Width 48 -Blocks 10 -CorrectionScale 0.26 `
  -FinalL1 0.55 -FinalMSE 0.45 -FinalSSIM 0.30 `
  -WM 1.8 -ROI 1.00 -Stripe 5.0 -HF 0.60 -Bounded 0.02 -SharpRetention 0.85

$Rows = New-Object System.Collections.Generic.List[string]
$Rows.Add("run`tdelta_psnr`tdelta_ssim`tdelta_wm`tdelta_roi`tdelta_stripe`tsharp_retention`tcorrection_l1")
foreach ($Run in @(
  "adni_stage2_probe_capacity_w64_b12_e3",
  "adni_stage2_probe_strongcorr_s030_e3",
  "adni_stage2_probe_metricheavy_s026_e3"
)) {
  $Path = "outputs\$Run\best_metrics.json"
  if (!(Test-Path $Path)) {
    $Rows.Add("$Run`tmissing`t`t`t`t`t`t")
    continue
  }
  $Data = Get-Content $Path -Raw | ConvertFrom-Json
  $Metrics = $Data.best_gated_metrics
  if ($null -eq $Metrics -or $Metrics.PSObject.Properties.Count -eq 0) {
    $Metrics = $Data.best_score_metrics
  }
  $Rows.Add((
    "$Run`t{0:N4}`t{1:N4}`t{2:N6}`t{3:N6}`t{4:N6}`t{5:N4}`t{6:N6}" -f `
      [double]$Metrics.delta_psnr,
      [double]$Metrics.delta_ssim,
      [double]$Metrics.delta_wm_l1,
      [double]$Metrics.delta_roi,
      [double]$Metrics.delta_stripe,
      [double]$Metrics.sharp_retention,
      [double]$Metrics.correction_l1
  ))
}
$Out = "outputs\icdm2026\tables\adni_stage2_probe_variants.tsv"
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
$Rows | Set-Content -Path $Out -Encoding UTF8
Get-Content $Out
