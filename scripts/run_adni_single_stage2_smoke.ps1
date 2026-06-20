$ErrorActionPreference = "Continue"

$Python = "D:\Anaconda3\envs\dinov3test\python.exe"
$LogDir = "outputs\icdm2026\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stage1Run = "adni_stage1_single_sharp_stripe_smoke_4096_e5"
$Stage1Ckpt = "outputs\$Stage1Run\checkpoints\best_stage1.pt"
$Stage1Pred = "outputs\icdm2026\predictions\ADNI_STAGE1_SINGLE_SHARP_STRIPE_SMOKE_4096_E5"
$Stage2Run = "adni_stage2_single_ds_hybrid_smoke_4096_e5"
$Stage2Ckpt = "outputs\$Stage2Run\checkpoints\best_ds_corrector.pt"
$Stage2Pred = "outputs\icdm2026\predictions\ADNI_DS_HYBRID_SINGLE_SMOKE_4096_E5"

& $Python -m pmrf_t1fa.train_pmrf_t1fa_stage1 `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --context_slices 1 `
  --run_name $Stage1Run `
  --stage1_training_preset sharp_adversarial `
  --stage1_model_variant detail `
  --stage1_prediction_mode residual `
  --stage1_detail_scale 0.45 `
  --epochs 5 `
  --batch_size 4 `
  --train_limit 4096 `
  --val_limit 1024 `
  --lr 8e-5 `
  --lpips_max_weight 0.10 `
  --adv_weight 0.01 `
  --wm_l1_weight 0.12 `
  --roi_consistency_weight 0.06 `
  --grad_weight 0.08 `
  --hf_weight 0.05 `
  --detail_hf_weight 0.08 `
  --detail_lap_weight 0.04 `
  --stripe_weight 2.0 `
  --best_metric detail_paired `
  --paired_sharp_weight 12.0 `
  --detail_target_sharp_ratio 0.85 `
  --detail_max_sharp_ratio 1.15 `
  --detail_oversharp_penalty_weight 8.0 `
  --fid_eval_every 999 `
  --save_every 5 `
  --no_auto_resume `
  --mixed_precision bf16 `
  > "$LogDir\$Stage1Run.stdout.log" `
  2> "$LogDir\$Stage1Run.stderr.log"

& $Python scripts/export_pm_dirf_predictions.py `
  --stage stage1 `
  --stage1_ckpt $Stage1Ckpt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir $Stage1Pred `
  --device cuda `
  --batch_size 8 `
  > "$LogDir\$Stage1Run.export.stdout.log" `
  2> "$LogDir\$Stage1Run.export.stderr.log"

& $Python scripts/evaluate_method_folder.py `
  --pred_dir $Stage1Pred `
  --method ADNI_STAGE1_SINGLE_SHARP_STRIPE_SMOKE_4096_E5 `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --visualize_count 24 `
  --reset_visualize_manifest `
  > "$LogDir\$Stage1Run.eval.stdout.log" `
  2> "$LogDir\$Stage1Run.eval.stderr.log"

& $Python -m pmrf_t1fa.train_pmrf_t1fa_stage2_ds_corrector `
  --train_t1_dir data/adni_processed/train/t1_slices `
  --train_fa_dir data/adni_processed/train/fa_slices `
  --val_t1_dir data/adni_processed/val/t1_slices `
  --val_fa_dir data/adni_processed/val/fa_slices `
  --stage1_ckpt $Stage1Ckpt `
  --run_name $Stage2Run `
  --variant hybrid `
  --epochs 5 `
  --batch_size 2 `
  --train_limit 4096 `
  --val_limit 1024 `
  --lr 8e-5 `
  --width 48 `
  --num_blocks 8 `
  --disease_roi_weight 0.0 `
  --roi_weight 0.5 `
  --wm_l1_weight 1.0 `
  --stripe_weight 2.0 `
  --hf_preserve_weight 1.0 `
  --uncertainty_weight 0.25 `
  --atlas_smooth_weight 0.15 `
  --best_min_sharp_retention 0.95 `
  --best_min_delta_disease_roi -1.0 `
  --best_max_delta_stripe 0.005 `
  --fid_eval_every 999 `
  --save_every 5 `
  --mixed_precision bf16 `
  > "$LogDir\$Stage2Run.stdout.log" `
  2> "$LogDir\$Stage2Run.stderr.log"

if (!(Test-Path $Stage2Ckpt)) {
  $Stage2Ckpt = "outputs\$Stage2Run\checkpoints\best_score_ds_corrector.pt"
}

& $Python scripts/export_ds_corrector_predictions.py `
  --stage1_ckpt $Stage1Ckpt `
  --ds_corrector_ckpt $Stage2Ckpt `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --output_dir $Stage2Pred `
  --device cuda `
  --batch_size 8 `
  > "$LogDir\$Stage2Run.export.stdout.log" `
  2> "$LogDir\$Stage2Run.export.stderr.log"

& $Python scripts/evaluate_method_folder.py `
  --pred_dir $Stage2Pred `
  --method ADNI_DS_HYBRID_SINGLE_SMOKE_4096_E5 `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --adni_slice_manifest data/adni_processed/adni_slice_manifest.csv `
  --visualize_count 24 `
  --reset_visualize_manifest `
  > "$LogDir\$Stage2Run.eval.stdout.log" `
  2> "$LogDir\$Stage2Run.eval.stderr.log"

& $Python scripts/evaluate_stage2_correction_effect.py `
  --coarse_dir $Stage1Pred `
  --refined_dir $Stage2Pred `
  --test_t1_dir data/adni_processed/test/t1_slices `
  --test_fa_dir data/adni_processed/test/fa_slices `
  --method ADNI_DS_HYBRID_SINGLE_SMOKE_4096_E5 `
  > "$LogDir\$Stage2Run.correction.stdout.log" `
  2> "$LogDir\$Stage2Run.correction.stderr.log"

& $Python scripts/build_adni_method_slice_panels.py `
  --method_preset adni_compact `
  --slice_source_dir outputs/icdm2026/figures/method_slices/ADNI_DS_HYBRID_SINGLE_SMOKE_4096_E5 `
  --output_dir outputs/icdm2026/figures/adni_single_stage2_smoke_panels `
  --gt_dir data/adni_processed/test/fa_slices `
  --dpi 220 `
  > "$LogDir\$Stage2Run.panels.stdout.log" `
  2> "$LogDir\$Stage2Run.panels.stderr.log"
