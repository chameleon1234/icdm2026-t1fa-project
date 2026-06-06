import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse low-frequency FA from one method with high-frequency detail from another method.")
    parser.add_argument("--low_dir", required=True, help="PNG folder providing low-frequency structure, e.g. PM_STAGE1.")
    parser.add_argument("--high_dir", required=True, help="PNG folder providing high-frequency texture, e.g. Stage1_LPIPS_GAN.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["low_base_high_detail", "sharp_base_low_residual"],
        default="low_base_high_detail",
        help="low_base_high_detail uses LP(low)+HP(high); sharp_base_low_residual keeps high image as base and injects LP(low)-LP(high).",
    )
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--high_gain", type=float, default=1.0)
    parser.add_argument("--low_residual_gain", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def _read_png01(path: Path) -> np.ndarray:
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def _write_png01(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image_u8)
    if not ok:
        raise ValueError(f"Failed to encode PNG: {path}")
    encoded.tofile(str(path))


def _lowpass(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def fuse_frequency(
    low_image: np.ndarray,
    high_image: np.ndarray,
    sigma: float,
    high_gain: float = 1.0,
    mode: str = "low_base_high_detail",
    low_residual_gain: float = 0.35,
) -> np.ndarray:
    if high_image.shape != low_image.shape:
        high_image = cv2.resize(high_image, (low_image.shape[1], low_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    low_base = _lowpass(low_image, sigma)
    high_base = _lowpass(high_image, sigma)
    high_detail = high_image - high_base
    if mode == "low_base_high_detail":
        return np.clip(low_base + high_gain * high_detail, 0.0, 1.0)
    if mode == "sharp_base_low_residual":
        low_residual = low_base - high_base
        return np.clip(high_image + low_residual_gain * low_residual, 0.0, 1.0)
    raise ValueError(f"Unknown fusion mode: {mode}")


def main() -> None:
    args = parse_args()
    low_dir = Path(args.low_dir)
    high_dir = Path(args.high_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    low_files = sorted(low_dir.glob("*.png"))
    if args.limit > 0:
        low_files = low_files[: args.limit]
    if not low_files:
        raise FileNotFoundError(f"No PNG files found in {low_dir}")

    rows = []
    missing = []
    for low_path in tqdm(low_files, desc="Frequency fusion"):
        high_path = high_dir / low_path.name
        if not high_path.exists():
            missing.append(low_path.name)
            if args.allow_missing:
                continue
            continue
        low_image = _read_png01(low_path)
        high_image = _read_png01(high_path)
        fused = fuse_frequency(
            low_image,
            high_image,
            sigma=args.sigma,
            high_gain=args.high_gain,
            mode=args.mode,
            low_residual_gain=args.low_residual_gain,
        )
        out_path = output_dir / low_path.name
        _write_png01(out_path, fused)
        rows.append(
            {
                "fname": low_path.name,
                "low_path": str(low_path),
                "high_path": str(high_path),
                "output_path": str(out_path),
                "sigma": args.sigma,
                "mode": args.mode,
                "high_gain": args.high_gain,
                "low_residual_gain": args.low_residual_gain,
            }
        )
    if missing and not args.allow_missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} high-frequency files in {high_dir}: {preview}")
    if not rows:
        raise RuntimeError("No fused PNG files were exported.")
    pd.DataFrame(rows).to_csv(output_dir / "export_manifest.csv", index=False, encoding="utf-8")
    print(f"FREQUENCY_FUSION: exported={len(rows)} output_dir={output_dir}")
    print(f"Saved manifest to: {output_dir / 'export_manifest.csv'}")


if __name__ == "__main__":
    main()
