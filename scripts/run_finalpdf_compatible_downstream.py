import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_REPEAT_SEEDS = ",".join(str(seed) for seed in range(20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final.pdf-compatible downstream classification: ROI mean features, "
            "linear SVM, and early T1+synthesized-FA fusion."
        )
    )
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--output_root", default="outputs/icdm2026")
    parser.add_argument("--repeat_seeds", default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--private_only", action="store_true")
    parser.add_argument("--adni_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _project_path(project_root: Path, relative: str) -> str:
    return str(project_root / relative)


def _base_command(project_root: Path, python_exe: str, output_root: Path, repeat_seeds: str) -> list[str]:
    return [
        python_exe,
        _project_path(project_root, "scripts/evaluate_downstream_classification.py"),
        "--config",
        _project_path(project_root, "configs/icdm2026.yaml"),
        "--split",
        "test",
        "--include_t1",
        "--include_fa_gt",
        "--n_splits",
        "5",
        "--repeat_seeds",
        repeat_seeds,
        "--max_features",
        "0",
        "--feature_set",
        "roi_mean",
        "--classifier",
        "linear_svm",
        "--output_root",
        str(output_root),
    ]


def build_private_command(project_root: Path, python_exe: str, output_root: Path, repeat_seeds: str = DEFAULT_REPEAT_SEEDS) -> list[str]:
    command = _base_command(project_root, python_exe, output_root, repeat_seeds)
    methods = [
        ("UNet_CurrentSplit_E100", "outputs/icdm2026/predictions/UNet_CurrentSplit_E100"),
        ("Pix2Pix_CurrentSplit_E100", "outputs/icdm2026/predictions/Pix2Pix_CurrentSplit_E100"),
        ("CycleGAN_CurrentSplit_E100", "outputs/icdm2026/predictions/CycleGAN_CurrentSplit_E100"),
        ("Stage1_LPIPS_GAN", "outputs/icdm2026/predictions/PM_STAGE1_LPIPS_GAN_5SLICE_FINAL"),
        ("Fidelity_Flow", "outputs/icdm2026/predictions/PM_DIRF_FIDELITY_FLOW_FULL"),
    ]
    for name, rel_path in methods:
        command.extend(["--method", f"{name}={_project_path(project_root, rel_path)}"])
    fusions = [
        "T1_PLUS_UNet=T1_ONLY+UNet_CurrentSplit_E100",
        "T1_PLUS_Pix2Pix=T1_ONLY+Pix2Pix_CurrentSplit_E100",
        "T1_PLUS_CycleGAN=T1_ONLY+CycleGAN_CurrentSplit_E100",
        "T1_PLUS_Stage1=T1_ONLY+Stage1_LPIPS_GAN",
        "T1_PLUS_Ours=T1_ONLY+Fidelity_Flow",
        "T1_PLUS_GT=T1_ONLY+FA_GT",
    ]
    for fusion in fusions:
        command.extend(["--fusion", fusion])
    command.extend(["--tasks", "cn_vs_ad,cn_vs_mci,mci_vs_ad"])
    return command


def build_adni_command(project_root: Path, python_exe: str, output_root: Path, repeat_seeds: str = DEFAULT_REPEAT_SEEDS) -> list[str]:
    command = _base_command(project_root, python_exe, output_root, repeat_seeds)
    command.extend(["--adni_slice_manifest", _project_path(project_root, "data/adni_processed/adni_slice_manifest.csv")])
    methods = [
        ("ADNI_UNet_E50", "outputs/icdm2026/predictions/ADNI_UNet_E50"),
        ("ADNI_Pix2Pix_E50", "outputs/icdm2026/predictions/ADNI_Pix2Pix_E50"),
        ("ADNI_CycleGAN_E50", "outputs/icdm2026/predictions/ADNI_CycleGAN_E50"),
        ("ADNI_PM_STAGE1_LPIPS_GAN_FULL", "outputs/icdm2026/predictions/ADNI_PM_STAGE1_LPIPS_GAN_FULL"),
        ("ADNI_PM_DIRF_FIDELITY_FLOW_FULL", "outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL"),
    ]
    for name, rel_path in methods:
        command.extend(["--method", f"{name}={_project_path(project_root, rel_path)}"])
    fusions = [
        "T1_PLUS_UNet=T1_ONLY+ADNI_UNet_E50",
        "T1_PLUS_Pix2Pix=T1_ONLY+ADNI_Pix2Pix_E50",
        "T1_PLUS_CycleGAN=T1_ONLY+ADNI_CycleGAN_E50",
        "T1_PLUS_Stage1=T1_ONLY+ADNI_PM_STAGE1_LPIPS_GAN_FULL",
        "T1_PLUS_Ours=T1_ONLY+ADNI_PM_DIRF_FIDELITY_FLOW_FULL",
        "T1_PLUS_GT=T1_ONLY+FA_GT",
    ]
    for fusion in fusions:
        command.extend(["--fusion", fusion])
    command.extend(["--tasks", "cn_vs_ad,cn_vs_mci_spectrum,mci_spectrum_vs_ad"])
    return command


def run_command(command: list[str], dry_run: bool = False) -> None:
    print("\n" + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / args.output_root
    run_private = not args.adni_only
    run_adni = not args.private_only

    if run_private:
        run_command(
            build_private_command(
                project_root,
                args.python_exe,
                output_root / "downstream_finalpdf_compatible_private",
                args.repeat_seeds,
            ),
            dry_run=args.dry_run,
        )
    if run_adni:
        run_command(
            build_adni_command(
                project_root,
                args.python_exe,
                output_root / "downstream_finalpdf_compatible_adni",
                args.repeat_seeds,
            ),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
