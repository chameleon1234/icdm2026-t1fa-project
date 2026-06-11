from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_GPU = Path(r"D:\Anaconda3\envs\dinov3test\python.exe")
PYTHON_CPU = Path(r"D:\Anaconda3\python.exe")

STAGE1_RUN = "adni_pmrf_stage1_lpips_gan_5slice_full_e80"
STAGE2_RUN = "adni_pmrf_stage2_fidelity_flow_full_e40"
STAGE1_CKPT = ROOT / "outputs" / STAGE1_RUN / "checkpoints" / "best_stage1.pt"
STAGE2_CKPT = ROOT / "outputs" / STAGE2_RUN / "checkpoints" / "best_fidelity_corrector.pt"
LOG_ROOT = ROOT / "outputs" / "icdm2026" / "logs" / "adni_full_training"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_pipeline(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp()}] {message}\n")


def run_step(name: str, command: list[str]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    write_pipeline(f"START {name}")
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{stamp()}] COMMAND {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        write_pipeline(f"FAIL {name} exit={code}")
        raise SystemExit(code)
    write_pipeline(f"DONE {name}")


def gpu(*args: str) -> list[str]:
    return [str(PYTHON_GPU), *args]


def cpu(*args: str) -> list[str]:
    return [str(PYTHON_CPU), *args]


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    run_step(
        "01_stage1_full",
        gpu(
            "-m",
            "pmrf_t1fa.train_pmrf_t1fa_stage1",
            "--train_t1_dir",
            "data/adni_processed/train/t1_slices",
            "--train_fa_dir",
            "data/adni_processed/train/fa_slices",
            "--val_t1_dir",
            "data/adni_processed/val/t1_slices",
            "--val_fa_dir",
            "data/adni_processed/val/fa_slices",
            "--run_name",
            STAGE1_RUN,
            "--context_slices",
            "5",
            "--stage1_training_preset",
            "sharp_adversarial",
            "--epochs",
            "80",
            "--batch_size",
            "4",
            "--lr",
            "1e-4",
            "--fid_eval_every",
            "999",
            "--preview_every",
            "1024",
            "--mixed_precision",
            "bf16",
            "--no_auto_resume",
        ),
    )
    if not STAGE1_CKPT.exists():
        raise FileNotFoundError(f"Missing Stage1 checkpoint: {STAGE1_CKPT}")

    run_step(
        "02_stage2_fidelity_flow_full",
        gpu(
            "-m",
            "pmrf_t1fa.train_pmrf_t1fa_stage2_fidelity_corrector",
            "--train_t1_dir",
            "data/adni_processed/train/t1_slices",
            "--train_fa_dir",
            "data/adni_processed/train/fa_slices",
            "--val_t1_dir",
            "data/adni_processed/val/t1_slices",
            "--val_fa_dir",
            "data/adni_processed/val/fa_slices",
            "--stage1_ckpt",
            str(STAGE1_CKPT.relative_to(ROOT)),
            "--run_name",
            STAGE2_RUN,
            "--corrector_mode",
            "flow",
            "--epochs",
            "40",
            "--batch_size",
            "2",
            "--lr",
            "8e-5",
            "--width",
            "48",
            "--num_blocks",
            "8",
            "--eval_steps",
            "4",
            "--mixed_precision",
            "bf16",
            "--source_noise_scale",
            "0.03",
        ),
    )
    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(f"Missing Stage2 checkpoint: {STAGE2_CKPT}")

    run_step(
        "03_export_stage1_full",
        gpu(
            "scripts/export_pm_dirf_predictions.py",
            "--stage",
            "stage1",
            "--stage1_ckpt",
            str(STAGE1_CKPT.relative_to(ROOT)),
            "--test_t1_dir",
            "data/adni_processed/test/t1_slices",
            "--test_fa_dir",
            "data/adni_processed/test/fa_slices",
            "--output_dir",
            "outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL",
            "--device",
            "cuda",
        ),
    )

    run_step(
        "04_export_fidelity_flow_full",
        gpu(
            "scripts/export_fidelity_corrector_predictions.py",
            "--stage1_ckpt",
            str(STAGE1_CKPT.relative_to(ROOT)),
            "--fidelity_corrector_ckpt",
            str(STAGE2_CKPT.relative_to(ROOT)),
            "--test_t1_dir",
            "data/adni_processed/test/t1_slices",
            "--test_fa_dir",
            "data/adni_processed/test/fa_slices",
            "--output_dir",
            "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL",
            "--device",
            "cuda",
            "--batch_size",
            "8",
        ),
    )

    for method in [
        "ADNI_PM_STAGE1_LPIPS_GAN_FULL",
        "ADNI_PM_DIRF_FIDELITY_FLOW_FULL",
    ]:
        run_step(
            f"05_eval_{method}",
            cpu(
                "scripts/evaluate_method_folder.py",
                "--pred_dir",
                f"outputs/icdm2026/predictions/{method}",
                "--method",
                method,
                "--adni_slice_manifest",
                "data/adni_processed/adni_slice_manifest.csv",
                "--visualize_count",
                "8",
            ),
        )

    run_step(
        "06_downstream_two_stage_full",
        cpu(
            "scripts/evaluate_downstream_classification.py",
            "--config",
            "configs/icdm2026.yaml",
            "--adni_slice_manifest",
            "data/adni_processed/adni_slice_manifest.csv",
            "--split",
            "test",
            "--method",
            "ADNI_PM_STAGE1_LPIPS_GAN_FULL=outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL",
            "--method",
            "ADNI_PM_DIRF_FIDELITY_FLOW_FULL=outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL",
            "--include_t1",
            "--include_fa_gt",
            "--tasks",
            "cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad",
            "--repeat_seeds",
            "0,1,2,3,4,5,6,7,8,9",
            "--n_splits",
            "5",
            "--max_features",
            "12",
            "--output_root",
            "outputs/icdm2026/downstream_adni_two_stage_full",
        ),
    )

    write_pipeline("ALL_DONE")


if __name__ == "__main__":
    main()
