import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_GPU = Path(r"D:\Anaconda3\envs\dinov3test\python.exe")
PYTHON_CPU = Path(r"D:\Anaconda3\python.exe")
LOG_ROOT = PROJECT_ROOT / "outputs" / "icdm2026" / "logs" / "full_comparison_suite"
PRED_ROOT = PROJECT_ROOT / "outputs" / "icdm2026" / "predictions"
METRICS_ROOT = PROJECT_ROOT / "outputs" / "icdm2026" / "metrics"


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    train_t1: str
    train_fa: str
    val_t1: str
    val_fa: str
    test_t1: str
    test_fa: str
    manifest: str
    expected_pngs: int
    downstream_tasks: str


ADNI = DatasetPreset(
    name="adni",
    train_t1="data/adni_processed/train/t1_slices",
    train_fa="data/adni_processed/train/fa_slices",
    val_t1="data/adni_processed/val/t1_slices",
    val_fa="data/adni_processed/val/fa_slices",
    test_t1="data/adni_processed/test/t1_slices",
    test_fa="data/adni_processed/test/fa_slices",
    manifest="data/adni_processed/adni_slice_manifest.csv",
    expected_pngs=5616,
    downstream_tasks="cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad",
)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def png_count(pred_dir: str | Path) -> int:
    path = PROJECT_ROOT / pred_dir
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def has_full_predictions(pred_dir: str | Path, expected_pngs: int) -> bool:
    return png_count(pred_dir) >= expected_pngs


def has_file(path: str | Path) -> bool:
    return (PROJECT_ROOT / path).exists()


def metric_exists(method: str) -> bool:
    return (METRICS_ROOT / f"{method}_summary.json").exists()


def command_to_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def run_step(
    step_name: str,
    command: list[str],
    *,
    dry_run: bool,
    skip_reason: str = "",
) -> None:
    pipeline_log = LOG_ROOT / "pipeline.log"
    step_log = LOG_ROOT / f"{step_name}.log"
    if skip_reason:
        message = f"[{now()}] SKIP {step_name}: {skip_reason}"
        print(message)
        write_line(pipeline_log, message)
        return

    start_message = f"[{now()}] START {step_name}"
    command_message = f"[{now()}] COMMAND {command_to_text(command)}"
    print(start_message)
    print(command_message)
    write_line(pipeline_log, start_message)
    write_line(step_log, command_message)
    if dry_run:
        write_line(pipeline_log, f"[{now()}] DRY_RUN {step_name}")
        return

    with open(step_log, "a", encoding="utf-8", errors="replace") as log_handle:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        fail_message = f"[{now()}] FAIL {step_name} returncode={completed.returncode}"
        write_line(pipeline_log, fail_message)
        raise RuntimeError(fail_message)
    done_message = f"[{now()}] DONE {step_name}"
    print(done_message)
    write_line(pipeline_log, done_message)


def train_stack_unet7_command(ds: DatasetPreset) -> list[str]:
    return [
        str(PYTHON_GPU),
        "scripts/train_stack_unet_current_split.py",
        "--train_t1_dir",
        ds.train_t1,
        "--train_fa_dir",
        ds.train_fa,
        "--val_t1_dir",
        ds.val_t1,
        "--val_fa_dir",
        ds.val_fa,
        "--test_t1_dir",
        ds.test_t1,
        "--run_name",
        "adni_stack_unet7_e50",
        "--epochs",
        "50",
        "--batch_size",
        "12",
        "--lr",
        "1e-4",
        "--context_slices",
        "7",
        "--width",
        "32",
        "--mixed_precision",
        "bf16",
    ]


def export_stack_unet7_command(ds: DatasetPreset) -> list[str]:
    return [
        str(PYTHON_GPU),
        "scripts/export_stack_unet_predictions.py",
        "--ckpt",
        "outputs/adni_stack_unet7_e50/checkpoints/best_stack_unet.pt",
        "--output_dir",
        "outputs/icdm2026/predictions/ADNI_StackUNet7_E50",
        "--method",
        "ADNI_StackUNet7_E50",
        "--test_t1_dir",
        ds.test_t1,
        "--test_fa_dir",
        ds.test_fa,
        "--context_slices",
        "7",
        "--width",
        "32",
        "--device",
        "cuda",
        "--batch_size",
        "8",
    ]


def image_eval_command(ds: DatasetPreset, method: str, pred_dir: str, visualize_count: int) -> list[str]:
    return [
        str(PYTHON_CPU),
        "scripts/evaluate_method_folder.py",
        "--pred_dir",
        pred_dir,
        "--method",
        method,
        "--test_t1_dir",
        ds.test_t1,
        "--test_fa_dir",
        ds.test_fa,
        "--adni_slice_manifest",
        ds.manifest,
        "--visualize_count",
        str(visualize_count),
    ]


def downstream_command(ds: DatasetPreset, methods: list[tuple[str, str]], output_root: str) -> list[str]:
    command = [
        str(PYTHON_CPU),
        "scripts/evaluate_downstream_classification.py",
        "--adni_slice_manifest",
        ds.manifest,
        "--split",
        "test",
        "--include_t1",
        "--include_fa_gt",
        "--tasks",
        ds.downstream_tasks,
        "--repeat_seeds",
        "0,1,2,3,4,5,6,7,8,9",
        "--n_splits",
        "5",
        "--max_features",
        "12",
        "--output_root",
        output_root,
    ]
    for method, pred_dir in methods:
        command.extend(["--method", f"{method}={pred_dir}"])
    return command


def build_adni_methods() -> list[tuple[str, str]]:
    return [
        ("ADNI_UNet_E50", "outputs/icdm2026/predictions/ADNI_UNet_E50"),
        ("ADNI_StackUNet5_E12", "outputs/icdm2026/predictions/ADNI_StackUNet5_E12"),
        ("ADNI_StackUNet7_E50", "outputs/icdm2026/predictions/ADNI_StackUNet7_E50"),
        ("ADNI_Pix2Pix_E50", "outputs/icdm2026/predictions/ADNI_Pix2Pix_E50"),
        ("ADNI_CycleGAN_E50", "outputs/icdm2026/predictions/ADNI_CycleGAN_E50"),
        ("ADNI_PM_STAGE1_LPIPS_GAN_FULL", "outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL"),
        ("ADNI_PM_DIRF_FIDELITY_FLOW_FULL", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ]


def run_adni(args: argparse.Namespace) -> None:
    ds = ADNI
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    write_line(LOG_ROOT / "pipeline.log", f"[{now()}] SUITE_START dataset={ds.name}")

    run_step(
        "01_train_stack_unet7_e50",
        train_stack_unet7_command(ds),
        dry_run=args.dry_run,
        skip_reason=(
            "checkpoint exists"
            if has_file("outputs/adni_stack_unet7_e50/checkpoints/best_stack_unet.pt") and not args.force_train
            else ""
        ),
    )
    run_step(
        "02_export_stack_unet7_e50",
        export_stack_unet7_command(ds),
        dry_run=args.dry_run,
        skip_reason=(
            "full predictions exist"
            if has_full_predictions("outputs/icdm2026/predictions/ADNI_StackUNet7_E50", ds.expected_pngs)
            and not args.force_export
            else ""
        ),
    )

    methods = build_adni_methods()
    for method, pred_dir in methods:
        if not has_full_predictions(pred_dir, ds.expected_pngs):
            write_line(LOG_ROOT / "pipeline.log", f"[{now()}] WARN missing_or_incomplete {method} {pred_dir}")
            continue
        run_step(
            f"03_eval_{method}",
            image_eval_command(ds, method, pred_dir, args.visualize_count),
            dry_run=args.dry_run,
            skip_reason=("metric exists" if metric_exists(method) and not args.force_metrics else ""),
        )

    available_methods = [(method, pred_dir) for method, pred_dir in methods if has_full_predictions(pred_dir, ds.expected_pngs)]
    run_step(
        "04_downstream_adni_full_comparison",
        downstream_command(ds, available_methods, "outputs/icdm2026/downstream_adni_full_comparison_v2"),
        dry_run=args.dry_run,
        skip_reason=(
            "downstream repeated summary exists"
            if has_file("outputs/icdm2026/downstream_adni_full_comparison_v2/classification_repeated_summary.csv")
            and not args.force_downstream
            else ""
        ),
    )

    status = {
        "dataset": ds.name,
        "methods": [
            {
                "method": method,
                "prediction_dir": pred_dir,
                "png_count": png_count(pred_dir),
                "metric_exists": metric_exists(method),
            }
            for method, pred_dir in methods
        ],
        "updated_at": now(),
    }
    status_path = PROJECT_ROOT / "outputs" / "icdm2026" / "tables" / "adni_full_comparison_v2_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_line(LOG_ROOT / "pipeline.log", f"[{now()}] ALL_DONE status={status_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recoverable full comparison experiments.")
    parser.add_argument("--dataset", choices=["adni"], default="adni")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--force_metrics", action="store_true")
    parser.add_argument("--force_downstream", action="store_true")
    parser.add_argument("--visualize_count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "adni":
        run_adni(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
