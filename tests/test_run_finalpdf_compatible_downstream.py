from pathlib import Path


def test_finalpdf_compatible_private_command_contains_roi_svm_and_fusions():
    from scripts.run_finalpdf_compatible_downstream import build_private_command

    command = build_private_command(
        project_root=Path("project"),
        python_exe="python",
        output_root=Path("out/private"),
        repeat_seeds="0,1",
    )
    text = " ".join(str(part).replace("\\", "/") for part in command)

    assert "scripts/evaluate_downstream_classification.py" in text
    assert "--feature_set roi_mean" in text
    assert "--classifier linear_svm" in text
    assert "--include_t1" in command
    assert "--include_fa_gt" in command
    assert "T1_PLUS_Ours=T1_ONLY+Fidelity_Flow" in command
    assert "T1_PLUS_Stage1=T1_ONLY+Stage1_LPIPS_GAN" in command
    assert "cn_vs_ad,cn_vs_mci,mci_vs_ad" in command


def test_finalpdf_compatible_adni_command_uses_manifest_and_ours():
    from scripts.run_finalpdf_compatible_downstream import build_adni_command

    command = build_adni_command(
        project_root=Path("project"),
        python_exe="python",
        output_root=Path("out/adni"),
        repeat_seeds="0,1",
    )

    assert "--adni_slice_manifest" in command
    normalized = [str(part).replace("\\", "/") for part in command]
    assert any(part.endswith("data/adni_processed/adni_slice_manifest.csv") for part in normalized)
    assert any(
        part.endswith("ADNI_PM_DIRF_FIDELITY_FLOW_FULL=project/outputs/icdm2026/predictions/ADNI_PM_DIRF_FIDELITY_FLOW_FULL")
        for part in normalized
    )
    assert "T1_PLUS_Ours=T1_ONLY+ADNI_PM_DIRF_FIDELITY_FLOW_FULL" in command
    assert "cn_vs_ad,cn_vs_mci_spectrum,mci_spectrum_vs_ad" in command
