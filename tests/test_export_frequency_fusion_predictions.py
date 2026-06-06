import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image.astype(np.uint8))
    assert ok
    encoded.tofile(str(path))


def _read_png(path: Path) -> np.ndarray:
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    assert image is not None
    return image


def test_export_frequency_fusion_combines_low_source_and_high_source(tmp_path):
    low_dir = tmp_path / "low"
    high_dir = tmp_path / "high"
    out_dir = tmp_path / "out"
    base = np.tile(np.linspace(40, 180, 32, dtype=np.uint8), (32, 1))
    sharp = base.copy()
    sharp[:, 15:17] = 255
    _write_png(low_dir / "sub-001_z020.png", base)
    _write_png(high_dir / "sub-001_z020.png", sharp)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_frequency_fusion_predictions.py",
            "--low_dir",
            str(low_dir),
            "--high_dir",
            str(high_dir),
            "--output_dir",
            str(out_dir),
            "--sigma",
            "1.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    fused = _read_png(out_dir / "sub-001_z020.png")
    assert fused.shape == base.shape
    assert fused[:, 15:17].mean() > base[:, 15:17].mean()
    assert (out_dir / "export_manifest.csv").exists()


def test_export_frequency_fusion_sharp_base_mode_preserves_high_source_edges(tmp_path):
    low_dir = tmp_path / "low"
    high_dir = tmp_path / "high"
    out_dir = tmp_path / "out"
    smooth_low = np.full((32, 32), 90, dtype=np.uint8)
    sharp = np.full((32, 32), 120, dtype=np.uint8)
    sharp[:, 15:17] = 255
    _write_png(low_dir / "sub-001_z020.png", smooth_low)
    _write_png(high_dir / "sub-001_z020.png", sharp)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_frequency_fusion_predictions.py",
            "--low_dir",
            str(low_dir),
            "--high_dir",
            str(high_dir),
            "--output_dir",
            str(out_dir),
            "--mode",
            "sharp_base_low_residual",
            "--low_residual_gain",
            "0.5",
            "--sigma",
            "1.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    fused = _read_png(out_dir / "sub-001_z020.png")
    assert fused[:, 15:17].mean() > smooth_low[:, 15:17].mean()
    assert fused[:, 15:17].mean() > fused[:, :8].mean()
