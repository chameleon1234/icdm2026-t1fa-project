from pathlib import Path


def test_build_disease_sensitive_roi_weights_command_defaults_to_train_fa():
    from scripts.build_disease_sensitive_roi_weights import build_command_args

    args = build_command_args(project_root=Path("project"), output_csv=Path("out/weights.csv"))
    normalized = [str(item).replace("\\", "/") for item in args]

    assert "project/data/processed/train/fa_slices" in normalized
    assert "--split" in args
    assert "train" in args
    assert "cn_vs_ad,cn_vs_mci,mci_vs_ad,cn_scd_vs_mci_ad" in args
