$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PythonGpu = "D:\Anaconda3\envs\dinov3test\python.exe"
$PythonCpu = "D:\Anaconda3\python.exe"

$Stage1Run = "adni_pmrf_stage1_lpips_gan_5slice_full_e80"
$Stage2Run = "adni_pmrf_stage2_fidelity_flow_full_e40"
$Stage1Ckpt = "outputs/$Stage1Run/checkpoints/best_stage1.pt"
$Stage2Ckpt = "outputs/$Stage2Run/checkpoints/best_fidelity_corrector.pt"
$LogRoot = "outputs/icdm2026/logs/adni_full_training"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Run-Step {
    param(
        [string]$Name,
        [string]$Command
    )
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Stamp] START $Name" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
    powershell -NoProfile -ExecutionPolicy Bypass -Command $Command 2>&1 |
        Tee-Object -FilePath "$LogRoot/$Name.log"
    if ($LASTEXITCODE -ne 0) {
        $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "[$Stamp] FAIL $Name exit=$LASTEXITCODE" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
        exit $LASTEXITCODE
    }
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$Stamp] DONE $Name" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
}

Run-Step "01_stage1_full" @"
& '$PythonGpu' -m pmrf_t1fa.train_pmrf_t1fa_stage1 `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --run_name $Stage1Run `
  --context_slices 5 `
  --stage1_training_preset sharp_adversarial `
  --epochs 80 `
  --batch_size 4 `
  --lr 1e-4 `
  --fid_eval_every 999 `
  --preview_every 1024 `
  --mixed_precision bf16 `
  --no_auto_resume
"@

if (-not (Test-Path $Stage1Ckpt)) {
    throw "Missing Stage1 checkpoint: $Stage1Ckpt"
}

Run-Step "02_stage2_fidelity_flow_full" @"
& '$PythonGpu' -m pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --stage1_ckpt $Stage1Ckpt `
  --run_name $Stage2Run `
  --corrector_mode flow `
  --epochs 40 `
  --batch_size 2 `
  --lr 8e-5 `
  --width 48 `
  --num_blocks 8 `
  --eval_steps 4 `
  --mixed_precision bf16 `
  --source_noise_scale 0.03
"@

if (-not (Test-Path $Stage2Ckpt)) {
    throw "Missing Stage2 checkpoint: $Stage2Ckpt"
}

Run-Step "03_export_stage1_full" @"
& '$PythonGpu' scripts/export_pm_dirf_predictions.py `
  --stage stage1 `
  --stage1_ckpt $Stage1Ckpt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL `
  --device cuda
"@

Run-Step "04_export_fidelity_flow_full" @"
& '$PythonGpu' scripts/export_fidelity_corrector_predictions.py `
  --stage1_ckpt $Stage1Ckpt `
  --fidelity_corrector_ckpt $Stage2Ckpt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL `
  --device cuda `
  --batch_size 8
"@

foreach ($Method in @(
    "ADNI_PM_STAGE1_LPIPS_GAN_FULL",
    "ADNI_PM_DIRF_FIDELITY_FLOW_FULL"
)) {
    Run-Step "05_eval_$Method" @"
& '$PythonCpu' scripts/evaluate_method_folder.py `
  --pred_dir outputs/icdm2026/predictions/$Method `
  --method $Method `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --visualize_count 8
"@
}

Run-Step "06_downstream_two_stage_full" @"
& '$PythonCpu' scripts/evaluate_downstream_classification.py `
  --config configs/icdm2026.yaml `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --split test `
  --method ADNI_PM_STAGE1_LPIPS_GAN_FULL=outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL `
  --method ADNI_PM_DIRF_FIDELITY_FLOW_FULL=outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL `
  --include_t1 `
  --include_fa_gt `
  --tasks cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad `
  --repeat_seeds 0,1,2,3,4,5,6,7,8,9 `
  --n_splits 5 `
  --max_features 12 `
  --output_root outputs/icdm2026/downstream_adni_two_stage_full
"@

$Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$Stamp] ALL_DONE" | Tee-Object -FilePath "$LogRoot/pipeline.log" -Append
