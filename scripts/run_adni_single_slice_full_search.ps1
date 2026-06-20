$ErrorActionPreference = "Stop"

# Full single-slice ADNI search for the final two-stage route.
# Run from repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/run_adni_single_slice_full_search.ps1

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PythonGpu = "D:\Anaconda3\envs\dinov3test\python.exe"
$PythonCpu = "D:\Anaconda3\envs\dinov3test\python.exe"

$Stage1Run = "adni_stage1_single_sharp_full_e30"
$Stage1Ckpt = "outputs/$Stage1Run/checkpoints/best_stage1.pt"
$Stage1Pred = "outputs/icdm2026/predictions/ADNI_STAGE1_SINGLE_SHARP_FULL_E30"

$FlowRun = "adni_single_fidelity_flow_full_e20"
$FlowCkpt = "outputs/$FlowRun/checkpoints/best_fidelity_corrector.pt"
$FlowPred = "outputs/icdm2026/predictions/ADNI_SINGLE_FIDELITY_FLOW_FULL_E20"

$MultiRun = "adni_single_ds_multihead_full_e20"
$MultiCkpt = "outputs/$MultiRun/checkpoints/best_ds_corrector.pt"
$MultiPred = "outputs/icdm2026/predictions/ADNI_SINGLE_DS_MULTIHEAD_FULL_E20"

$LogRoot = "outputs/icdm2026/logs/adni_single_slice_full_search"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Run-Step {
    param(
        [string]$Name,
        [string]$Command
    )
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Stamp] START $Name" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
    $StepScript = Join-Path $LogRoot "$Name.ps1"
    $NormalizedCommand = (($Command -split "\r?\n") | ForEach-Object { $_.Trim() } | Where-Object { $_ }) -join " "
    Set-Content -Path $StepScript -Value $NormalizedCommand -Encoding UTF8
    powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $StepScript 2>&1 |
        Tee-Object -FilePath "$LogRoot/$Name.log"
    if ($LASTEXITCODE -ne 0) {
        $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$Stamp] FAIL $Name exit=$LASTEXITCODE" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
        exit $LASTEXITCODE
    }
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Stamp] DONE $Name" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
}

function Eval-Method {
    param(
        [string]$Method,
        [string]$PredDir,
        [int]$VisualizeCount = 24
    )
    Run-Step "eval_$Method" @"
& '$PythonCpu' scripts/evaluate_method_folder.py `
  --pred_dir $PredDir `
  --method $Method `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --visualize_count $VisualizeCount `
  --reset_visualize_manifest
"@
}

Run-Step "01_stage1_single_sharp_full_e30" @"
& '$PythonGpu' -m pmrf_t1fa.train_pmrf_t1fa_stage1 `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --context_slices 1 `
  --run_name $Stage1Run `
  --stage1_training_preset sharp_adversarial `
  --stage1_model_variant detail `
  --stage1_prediction_mode residual `
  --stage1_detail_scale 0.58 `
  --epochs 30 `
  --batch_size 4 `
  --lr 8e-5 `
  --lpips_max_weight 0.10 `
  --adv_weight 0.010 `
  --wm_l1_weight 0.10 `
  --roi_consistency_weight 0.05 `
  --grad_weight 0.08 `
  --hf_weight 0.06 `
  --detail_hf_weight 0.10 `
  --detail_lap_weight 0.05 `
  --stripe_weight 0.75 `
  --best_metric detail_paired `
  --paired_sharp_weight 14.0 `
  --detail_target_sharp_ratio 0.92 `
  --detail_max_sharp_ratio 1.18 `
  --detail_oversharp_penalty_weight 8.0 `
  --fid_eval_every 999 `
  --save_every 10 `
  --preview_every 1024 `
  --no_auto_resume `
  --mixed_precision bf16
"@

if (-not (Test-Path $Stage1Ckpt)) {
    throw "Missing Stage1 checkpoint: $Stage1Ckpt"
}

Run-Step "02_export_stage1_best" @"
& '$PythonGpu' scripts/export_pm_dirf_predictions.py `
  --stage stage1 `
  --stage1_ckpt $Stage1Ckpt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir $Stage1Pred `
  --device cuda `
  --batch_size 8
"@

Eval-Method "ADNI_STAGE1_SINGLE_SHARP_FULL_E30" $Stage1Pred 24

Run-Step "03_stage2_single_fidelity_flow_e20" @"
& '$PythonGpu' -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --stage1_ckpt $Stage1Ckpt `
  --run_name $FlowRun `
  --corrector_mode flow `
  --epochs 20 `
  --batch_size 2 `
  --lr 8e-5 `
  --width 48 `
  --num_blocks 10 `
  --eval_steps 6 `
  --source_noise_scale 0.03 `
  --final_l1_weight 0.60 `
  --final_mse_weight 0.55 `
  --final_ssim_weight 0.30 `
  --wm_l1_weight 1.20 `
  --roi_weight 0.45 `
  --hf_preserve_weight 2.50 `
  --best_min_sharp_retention 0.97 `
  --best_min_delta_psnr 0.0 `
  --best_min_delta_ssim 0.0 `
  --save_every 5 `
  --mixed_precision bf16
"@

if (-not (Test-Path $FlowCkpt)) {
    throw "Missing Fidelity Flow checkpoint: $FlowCkpt"
}

Run-Step "04_export_single_fidelity_flow" @"
& '$PythonGpu' scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt $Stage1Ckpt `
  --fidelity_corrector_ckpt $FlowCkpt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir $FlowPred `
  --device cuda `
  --batch_size 8
"@

Eval-Method "ADNI_SINGLE_FIDELITY_FLOW_FULL_E20" $FlowPred 24

Run-Step "05_effect_single_fidelity_flow" @"
& '$PythonCpu' scripts/evaluate_stage2_correction_effect.py `
  --coarse_dir $Stage1Pred `
  --refined_dir $FlowPred `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --method ADNI_SINGLE_FIDELITY_FLOW_FULL_E20
"@

Run-Step "06_stage2_single_multihead_e20" @"
& '$PythonGpu' -m pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --stage1_ckpt $Stage1Ckpt `
  --run_name $MultiRun `
  --variant multihead `
  --epochs 20 `
  --batch_size 2 `
  --lr 8e-5 `
  --width 56 `
  --num_blocks 12 `
  --correction_scale 0.24 `
  --final_l1_weight 0.45 `
  --final_mse_weight 0.35 `
  --final_ssim_weight 0.28 `
  --wm_l1_weight 1.50 `
  --roi_weight 0.90 `
  --disease_roi_weight 0.0 `
  --correction_l1_weight 0.30 `
  --bounded_weight 0.02 `
  --sharp_retention_weight 1.00 `
  --stripe_weight 5.0 `
  --hf_preserve_weight 1.00 `
  --uncertainty_weight 0.25 `
  --atlas_smooth_weight 0.15 `
  --best_min_sharp_retention 0.97 `
  --best_min_delta_disease_roi -1.0 `
  --best_max_delta_stripe 0.01 `
  --fid_eval_every 999 `
  --save_every 5 `
  --mixed_precision bf16
"@

if (-not (Test-Path $MultiCkpt)) {
    $MultiCkpt = "outputs/$MultiRun/checkpoints/best_score_ds_corrector.pt"
}
if (-not (Test-Path $MultiCkpt)) {
    throw "Missing Multihead checkpoint: $MultiCkpt"
}

Run-Step "07_export_single_multihead" @"
& '$PythonGpu' scripts/export_ds_corrector_predictions.py `
  --stage1_ckpt $Stage1Ckpt `
  --ds_corrector_ckpt $MultiCkpt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir $MultiPred `
  --device cuda `
  --batch_size 8
"@

Eval-Method "ADNI_SINGLE_DS_MULTIHEAD_FULL_E20" $MultiPred 24

Run-Step "08_effect_single_multihead" @"
& '$PythonCpu' scripts/evaluate_stage2_correction_effect.py `
  --coarse_dir $Stage1Pred `
  --refined_dir $MultiPred `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --method ADNI_SINGLE_DS_MULTIHEAD_FULL_E20
"@

Run-Step "09_summary_table" @"
& '$PythonCpu' -c "import json, pathlib; names=['ADNI_STAGE1_SINGLE_SHARP_FULL_E30','ADNI_SINGLE_FIDELITY_FLOW_FULL_E20','ADNI_SINGLE_DS_MULTIHEAD_FULL_E20','ADNI_PM_DIRF_FIDELITY_FLOW_FULL','ADNI_PM_STAGE1_LPIPS_GAN_FULL','ADNI_UNet_E50','ADNI_Pix2Pix_E50']; print('Method`tPSNR`tSSIM`tSharp`tWM_MAE`tROI_CCC'); [print(f'{n}`t{json.loads((pathlib.Path('outputs/icdm2026/metrics')/(n+'_summary.json')).read_text()).get('PSNR_mean',0):.3f}`t{json.loads((pathlib.Path('outputs/icdm2026/metrics')/(n+'_summary.json')).read_text()).get('SSIM_mean',0):.4f}`t{json.loads((pathlib.Path('outputs/icdm2026/metrics')/(n+'_summary.json')).read_text()).get('Sharpness_Ratio_mean',0):.3f}`t{json.loads((pathlib.Path('outputs/icdm2026/metrics')/(n+'_summary.json')).read_text()).get('WM_Masked_MAE_mean',0):.4f}`t{json.loads((pathlib.Path('outputs/icdm2026/metrics')/(n+'_summary.json')).read_text()).get('ROI_CCC',0):.4f}') for n in names if (pathlib.Path('outputs/icdm2026/metrics')/(n+'_summary.json')).exists()]"
"@

$Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$Stamp] ALL_DONE" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
