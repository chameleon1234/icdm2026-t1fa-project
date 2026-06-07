import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image.astype(np.uint8))
    assert ok
    encoded.tofile(str(path))


def test_evaluate_downstream_classification_cli_writes_summary_outputs(tmp_path):
    image_dir = tmp_path / "method"
    rows = []
    for class_id, group_name in enumerate(["CN", "SCD", "MCI", "AD"], start=1):
        for subject_idx in range(3):
            subject_id = f"sub-{class_id}{subject_idx:02d}"
            rows.append(
                {
                    "subject_id": subject_id,
                    "group_id": class_id,
                    "group_name": group_name,
                    "gender": 0,
                    "age": 70,
                    "edu": 12,
                    "MMSE": 25,
                    "split": "test",
                }
            )
            for z in [20, 21, 22]:
                image = np.full((16, 16), class_id * 45 + subject_idx + z % 3, dtype=np.uint8)
                _write_png(image_dir / f"{subject_id}_z{z:03d}.png", image)

    subject_index_csv = tmp_path / "subject_index.csv"
    pd.DataFrame(rows).to_csv(subject_index_csv, index=False)

    config_path = tmp_path / "config.yaml"
    config = {
        "data": {"processed_root": str(tmp_path / "processed")},
        "evaluation": {"classification_tasks": ["four_class"]},
        "outputs": {"root": str(tmp_path / "outputs")},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "downstream"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_downstream_classification.py",
            "--config",
            str(config_path),
            "--subject_index_csv",
            str(subject_index_csv),
            "--method",
            f"Toy={image_dir}",
            "--fusion",
            "ToySelf=Toy+Toy",
            "--tasks",
            "four_class",
            "--regression_targets",
            "MMSE",
            "--repeat_seeds",
            "1,2",
            "--n_splits",
            "3",
            "--output_root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    summary = pd.read_csv(output_root / "classification_summary.csv")
    assert "Saved classification summary" in result.stdout
    assert (output_root / "subject_features.csv").exists()
    assert (output_root / "classification_predictions.csv").exists()
    assert (output_root / "classification_summary.json").exists()
    assert (output_root / "regression_summary.csv").exists()
    assert (output_root / "regression_predictions.csv").exists()
    assert (output_root / "classification_repeated_summary.csv").exists()
    assert (output_root / "confusion_matrices" / "Toy_four_class_confusion.png").exists()
    assert (output_root / "confusion_matrices" / "ToySelf_four_class_confusion.png").exists()
    assert summary.loc[0, "method"] == "Toy"
    assert summary.loc[0, "task"] == "four_class"
    assert int(summary.loc[0, "n_subjects"]) == 12
    assert "ToySelf" in set(summary["method"])
    repeated = pd.read_csv(output_root / "classification_repeated_summary.csv")
    assert int(repeated.loc[0, "n_repeats"]) == 2


def test_evaluate_downstream_classification_cli_accepts_adni_slice_manifest(tmp_path):
    image_dir = tmp_path / "adni_method"
    manifest_rows = []
    subjects = [
        ("002_S_0413", "CN", 40),
        ("002_S_1155", "CN", 55),
        ("003_S_0907", "MCI_spectrum", 120),
        ("003_S_1122", "MCI_spectrum", 135),
        ("005_S_0221", "AD", 190),
        ("005_S_0814", "AD", 205),
    ]
    for subject, group_name, base_intensity in subjects:
        for z in [20, 21, 22]:
            filename = f"sub-{subject}_z{z:03d}.png"
            image = np.full((16, 16), base_intensity + z % 3, dtype=np.uint8)
            _write_png(image_dir / filename, image)
            manifest_rows.append(
                {
                    "subject": subject,
                    "split": "test",
                    "filename": filename,
                    "normalized_group": group_name,
                    "raw_group": group_name,
                }
            )
    adni_manifest = tmp_path / "adni_slice_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(adni_manifest, index=False)

    config_path = tmp_path / "config.yaml"
    config = {
        "data": {"processed_root": str(tmp_path / "processed")},
        "evaluation": {"classification_tasks": ["cn_vs_mci_spectrum_ad"]},
        "outputs": {"root": str(tmp_path / "outputs")},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_root = tmp_path / "downstream_adni"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_downstream_classification.py",
            "--config",
            str(config_path),
            "--adni_slice_manifest",
            str(adni_manifest),
            "--method",
            f"ADNI_Toy={image_dir}",
            "--tasks",
            "cn_vs_mci_spectrum_ad",
            "--n_splits",
            "2",
            "--output_root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    features = pd.read_csv(output_root / "subject_features.csv")
    summary = pd.read_csv(output_root / "classification_summary.csv")
    assert "Saved classification summary" in result.stdout
    assert set(features["subject_id"]) == {f"sub-{subject}" for subject, _, _ in subjects}
    assert int(summary.loc[0, "n_subjects"]) == 6
    assert summary.loc[0, "task"] == "cn_vs_mci_spectrum_ad"
