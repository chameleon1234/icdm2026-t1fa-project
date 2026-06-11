import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_GPU = Path(r"D:\Anaconda3\envs\dinov3test\python.exe")
PY_CPU = Path(r"D:\Anaconda3\python.exe")
LOG_ROOT = PROJECT_ROOT / "outputs" / "icdm2026" / "logs" / "heavy_baseline_suite"
PRED_ROOT = PROJECT_ROOT / "outputs" / "icdm2026" / "predictions"
METRIC_ROOT = PROJECT_ROOT / "outputs" / "icdm2026" / "metrics"
DOWNSTREAM_ROOT = PROJECT_ROOT / "outputs" / "icdm2026" / "downstream_adni_heavy_baselines"


def _python(path: Path) -> str:
    return str(path if path.exists() else sys.executable)


def _run(name: str, cmd: list[str], skip_path: Path | None = None) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    if skip_path is not None and skip_path.exists():
        message = f"[SKIP] {name}: {skip_path} exists"
        print(message, flush=True)
        with (LOG_ROOT / "pipeline.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        return
    message = f"[RUN] {name}: {' '.join(cmd)}"
    print(message, flush=True)
    with (LOG_ROOT / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"{name} failed with code {process.returncode}; see {log_path}")


def _png_count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def _prediction_done(path: Path, expected: int) -> bool:
    return path.exists() and (path / "export_manifest.csv").exists() and _png_count(path) >= expected


def _eval_done(method: str) -> bool:
    return (METRIC_ROOT / f"{method}_summary.json").exists()


def _run_export_and_eval(method: dict[str, object], args: argparse.Namespace) -> None:
    name = str(method["name"])
    pred_dir = PRED_ROOT / name
    expected = int(args.expected_slices)
    export_cmd = [str(method["python"]), *[str(item) for item in method["export_args"]]]
    if not _prediction_done(pred_dir, expected):
        _run(f"export_{name}", export_cmd)
    else:
        _run(f"export_{name}", export_cmd, skip_path=pred_dir / "export_manifest.csv")

    summary_path = METRIC_ROOT / f"{name}_summary.json"
    eval_cmd = [
        _python(PY_CPU),
        "scripts/evaluate_method_folder.py",
        "--pred_dir",
        str(pred_dir),
        "--method",
        name,
        "--adni_slice_manifest",
        args.adni_slice_manifest,
        "--visualize_count",
        str(args.visualize_count),
    ]
    _run(f"eval_{name}", eval_cmd, skip_path=summary_path if _eval_done(name) else None)


def _build_methods(args: argparse.Namespace) -> list[dict[str, object]]:
    test_args = [
        "--test_t1_dir",
        args.test_t1_dir,
        "--test_fa_dir",
        args.test_fa_dir,
    ]
    gpu = _python(PY_GPU)
    return [
        {
            "name": "ADNI_DDIM_E100_K50_PRETRAINED",
            "python": gpu,
            "export_args": [
                "scripts/export_legacy_baseline_predictions.py",
                "--model_type",
                "ddim",
                "--ckpt",
                "checkpoints_baseline_ddim_steps50/epoch_100.pt",
                "--output_dir",
                str(PRED_ROOT / "ADNI_DDIM_E100_K50_PRETRAINED"),
                "--method",
                "ADNI_DDIM_E100_K50_PRETRAINED",
                *test_args,
                "--sample_steps",
                str(args.ddim_steps),
                "--batch_size",
                str(args.ddim_batch_size),
                "--device",
                args.device,
            ],
        },
        {
            "name": "ADNI_DBM_E100_K40_PRETRAINED",
            "python": gpu,
            "export_args": [
                "scripts/export_dbm_predictions.py",
                "--ckpt",
                "checkpoints_baseline_dbm/epoch_100.pt",
                "--output_dir",
                str(PRED_ROOT / "ADNI_DBM_E100_K40_PRETRAINED"),
                "--method",
                "ADNI_DBM_E100_K40_PRETRAINED",
                *test_args,
                "--sample_steps",
                str(args.dbm_steps),
                "--batch_size",
                str(args.dbm_batch_size),
                "--device",
                args.device,
            ],
        },
        {
            "name": "ADNI_MOTFM_I2I_K10_PRETRAINED",
            "python": gpu,
            "export_args": [
                "scripts/export_motfm_predictions.py",
                "--ckpt",
                "MOTFM-main/checkpoints_t1_fa_i2i/t1_fa_config/version_1/checkpoints/last.ckpt",
                "--config_path",
                "MOTFM-main/checkpoints_t1_fa_i2i/t1_fa_config/version_1/hparams.yaml",
                "--output_dir",
                str(PRED_ROOT / "ADNI_MOTFM_I2I_K10_PRETRAINED"),
                "--method",
                "ADNI_MOTFM_I2I_K10_PRETRAINED",
                *test_args,
                "--sample_steps",
                str(args.motfm_steps),
                "--batch_size",
                str(args.motfm_batch_size),
                "--device",
                args.device,
            ],
        },
    ]


def _run_downstream(method_names: list[str], args: argparse.Namespace) -> None:
    summary_path = DOWNSTREAM_ROOT / "classification_repeated_summary.csv"
    method_args: list[str] = []
    for name in method_names:
        method_args.extend(["--method", f"{name}={PRED_ROOT / name}"])
    for name in args.include_method:
        method_args.extend(["--method", name])
    cmd = [
        _python(PY_CPU),
        "scripts/evaluate_downstream_classification.py",
        "--adni_slice_manifest",
        args.adni_slice_manifest,
        "--split",
        "test",
        "--include_t1",
        "--include_fa_gt",
        *method_args,
        "--tasks",
        args.tasks,
        "--repeat_seeds",
        args.repeat_seeds,
        "--n_splits",
        str(args.n_splits),
        "--max_features",
        str(args.max_features),
        "--output_root",
        str(DOWNSTREAM_ROOT),
    ]
    _run("downstream_heavy_adni", cmd, skip_path=summary_path)


def _write_manifest(method_names: list[str], args: argparse.Namespace) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "methods": method_names,
        "test_t1_dir": args.test_t1_dir,
        "test_fa_dir": args.test_fa_dir,
        "adni_slice_manifest": args.adni_slice_manifest,
        "ddim_steps": args.ddim_steps,
        "dbm_steps": args.dbm_steps,
        "motfm_steps": args.motfm_steps,
    }
    (LOG_ROOT / "heavy_baseline_suite_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heavy ADNI baseline exports, image metrics, and downstream utility.")
    parser.add_argument("--test_t1_dir", default="data/adni_processed/test/t1_slices")
    parser.add_argument("--test_fa_dir", default="data/adni_processed/test/fa_slices")
    parser.add_argument("--adni_slice_manifest", default="data/adni_processed/adni_slice_manifest.csv")
    parser.add_argument("--expected_slices", type=int, default=5616)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--dbm_steps", type=int, default=40)
    parser.add_argument("--motfm_steps", type=int, default=10)
    parser.add_argument("--ddim_batch_size", type=int, default=8)
    parser.add_argument("--dbm_batch_size", type=int, default=8)
    parser.add_argument("--motfm_batch_size", type=int, default=4)
    parser.add_argument("--visualize_count", type=int, default=8)
    parser.add_argument("--skip_downstream", action="store_true")
    parser.add_argument(
        "--tasks",
        default="cn_vs_mci_spectrum,cn_vs_ad,mci_spectrum_vs_ad",
    )
    parser.add_argument("--repeat_seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=12)
    parser.add_argument(
        "--include_method",
        action="append",
        default=[
            "ADNI_FREQ_STACKLOW_OURSHIGH_B035_FULL=outputs/icdm2026/predictions/ADNI_FREQ_STACKLOW_OURSHIGH_B035_FULL",
            "ADNI_FREQ_FLOWBASE_PMLOW_B035_FULL=outputs/icdm2026/predictions/ADNI_FREQ_FLOWBASE_PMLOW_B035_FULL",
        ],
        help="Extra NAME=DIR downstream method specs, repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = _build_methods(args)
    method_names = [str(method["name"]) for method in methods]
    _write_manifest(method_names, args)
    for method in methods:
        _run_export_and_eval(method, args)
    if not args.skip_downstream:
        _run_downstream(method_names, args)
    print("Heavy baseline suite finished.", flush=True)


if __name__ == "__main__":
    main()
