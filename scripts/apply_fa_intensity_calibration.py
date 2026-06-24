from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate predicted FA intensity with train-set FA slice statistics. "
            "Only low-frequency brightness/contrast is adjusted; high-frequency detail is preserved."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Prediction PNG folder to calibrate.")
    parser.add_argument("--t1_dir", required=True, help="T1 PNG folder matching input filenames.")
    parser.add_argument("--train_t1_dir", required=True, help="Training T1 PNG folder for brain masks.")
    parser.add_argument("--train_fa_dir", required=True, help="Training FA GT PNG folder for target statistics.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--brain_threshold", type=float, default=0.05)
    parser.add_argument("--lowpass_sigma", type=float, default=4.0)
    parser.add_argument("--mean_alpha", type=float, default=0.75, help="Blend strength for per-slice low-frequency mean matching.")
    parser.add_argument("--std_alpha", type=float, default=0.35, help="Blend strength for per-slice low-frequency contrast matching.")
    parser.add_argument("--min_scale", type=float, default=0.85)
    parser.add_argument("--max_scale", type=float, default=1.15)
    parser.add_argument("--max_low_delta", type=float, default=0.04, help="Clamp low-frequency correction magnitude.")
    parser.add_argument(
        "--upper_quantile",
        type=float,
        default=0.995,
        help="Train FA per-slice upper quantile used as a soft anti-overbright ceiling. Set >=1 to disable.",
    )
    parser.add_argument("--clip_alpha", type=float, default=0.60, help="Blend strength when applying the upper quantile ceiling.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _read_png01(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def _write_png01(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.clip(image * 255.0, 0, 255).round().astype(np.uint8))
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def _parse_z(name: str) -> int:
    match = re.search(r"_z(\d+)\.png$", name)
    if not match:
        raise ValueError(f"Cannot parse z-index from filename: {name}")
    return int(match.group(1))


def _brain_mask_from_t1(t1: np.ndarray, threshold: float) -> np.ndarray:
    return (t1 > threshold).astype(np.float32)


def _lowpass(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32)
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)


def _masked_stats(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    active = mask > 0.5
    if not np.any(active):
        return float(np.mean(image)), float(np.std(image))
    values = image[active]
    return float(np.mean(values)), float(np.std(values))


def _masked_quantile(image: np.ndarray, mask: np.ndarray, quantile: float) -> float:
    active = mask > 0.5
    if not np.any(active):
        return float(np.quantile(image, quantile))
    return float(np.quantile(image[active], quantile))


def fit_slice_stats(
    train_t1_dir: Path,
    train_fa_dir: Path,
    brain_threshold: float,
    lowpass_sigma: float,
    upper_quantile: float,
) -> dict[str, dict[str, float]]:
    by_z: dict[int, dict[str, list[float]]] = {}
    all_values: dict[str, list[float]] = {"mean": [], "std": [], "upper": []}
    for fa_path in tqdm(sorted(train_fa_dir.glob("*.png")), desc="Fit FA intensity stats"):
        t1_path = train_t1_dir / fa_path.name
        if not t1_path.exists():
            continue
        z = _parse_z(fa_path.name)
        t1 = _read_png01(t1_path)
        fa = _read_png01(fa_path)
        if fa.shape != t1.shape:
            fa = cv2.resize(fa, (t1.shape[1], t1.shape[0]), interpolation=cv2.INTER_CUBIC)
        brain = _brain_mask_from_t1(t1, brain_threshold)
        fa_low = _lowpass(fa, lowpass_sigma)
        mean, std = _masked_stats(fa_low, brain)
        upper = _masked_quantile(fa, brain, upper_quantile) if upper_quantile < 1.0 else 1.0
        bucket = by_z.setdefault(z, {"mean": [], "std": [], "upper": []})
        bucket["mean"].append(mean)
        bucket["std"].append(std)
        bucket["upper"].append(upper)
        all_values["mean"].append(mean)
        all_values["std"].append(std)
        all_values["upper"].append(upper)

    if not by_z:
        raise FileNotFoundError(f"No matched train T1/FA PNGs found in {train_t1_dir} and {train_fa_dir}")

    stats: dict[str, dict[str, float]] = {}
    for z, values in by_z.items():
        stats[str(z)] = {
            "mean": float(np.mean(values["mean"])),
            "std": float(np.mean(values["std"])),
            "upper": float(np.mean(values["upper"])),
            "n": float(len(values["mean"])),
        }
    stats["global"] = {
        "mean": float(np.mean(all_values["mean"])),
        "std": float(np.mean(all_values["std"])),
        "upper": float(np.mean(all_values["upper"])),
        "n": float(len(all_values["mean"])),
    }
    return stats


def calibrate_image(
    pred: np.ndarray,
    t1: np.ndarray,
    target_stats: dict[str, float],
    brain_threshold: float,
    lowpass_sigma: float,
    mean_alpha: float,
    std_alpha: float,
    min_scale: float,
    max_scale: float,
    max_low_delta: float,
    upper_quantile_enabled: bool,
    clip_alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if t1.shape != pred.shape:
        t1 = cv2.resize(t1, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
    brain = _brain_mask_from_t1(t1, brain_threshold)
    pred_low = _lowpass(pred, lowpass_sigma)
    pred_high = pred - pred_low
    current_mean, current_std = _masked_stats(pred_low, brain)
    target_mean = float(target_stats["mean"])
    target_std = max(float(target_stats["std"]), 1e-6)
    raw_scale = target_std / max(current_std, 1e-6)
    scale = float(np.clip(1.0 + std_alpha * (raw_scale - 1.0), min_scale, max_scale))
    low_matched = target_mean + scale * (pred_low - current_mean)
    low_delta = np.clip(low_matched - pred_low, -max_low_delta, max_low_delta)
    calibrated = pred.copy()
    brain_bool = brain > 0.5
    calibrated_low = pred_low + mean_alpha * low_delta
    calibrated[brain_bool] = (calibrated_low + pred_high)[brain_bool]

    upper = float(target_stats.get("upper", 1.0))
    clipped_pixels = 0
    if upper_quantile_enabled and upper < 1.0:
        over = brain_bool & (calibrated > upper)
        clipped_pixels = int(over.sum())
        calibrated[over] = (1.0 - clip_alpha) * calibrated[over] + clip_alpha * upper

    calibrated = np.clip(calibrated, 0.0, 1.0)
    after_mean, after_std = _masked_stats(_lowpass(calibrated, lowpass_sigma), brain)
    return calibrated, {
        "target_mean": target_mean,
        "target_std": target_std,
        "target_upper": upper,
        "before_low_mean": current_mean,
        "before_low_std": current_std,
        "after_low_mean": after_mean,
        "after_low_std": after_std,
        "low_mean_shift": after_mean - current_mean,
        "scale": scale,
        "clipped_pixels": float(clipped_pixels),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    t1_dir = Path(args.t1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_stats = fit_slice_stats(
        train_t1_dir=Path(args.train_t1_dir),
        train_fa_dir=Path(args.train_fa_dir),
        brain_threshold=float(args.brain_threshold),
        lowpass_sigma=float(args.lowpass_sigma),
        upper_quantile=float(args.upper_quantile),
    )

    input_files = sorted(input_dir.glob("*.png"))
    if args.limit > 0:
        input_files = input_files[: args.limit]
    rows = []
    for pred_path in tqdm(input_files, desc="Apply FA intensity calibration"):
        t1_path = t1_dir / pred_path.name
        if not t1_path.exists():
            raise FileNotFoundError(f"Missing T1 for {pred_path.name}: {t1_path}")
        z = _parse_z(pred_path.name)
        target_stats = slice_stats.get(str(z), slice_stats["global"])
        pred = _read_png01(pred_path)
        t1 = _read_png01(t1_path)
        calibrated, stats = calibrate_image(
            pred=pred,
            t1=t1,
            target_stats=target_stats,
            brain_threshold=float(args.brain_threshold),
            lowpass_sigma=float(args.lowpass_sigma),
            mean_alpha=float(args.mean_alpha),
            std_alpha=float(args.std_alpha),
            min_scale=float(args.min_scale),
            max_scale=float(args.max_scale),
            max_low_delta=float(args.max_low_delta),
            upper_quantile_enabled=float(args.upper_quantile) < 1.0,
            clip_alpha=float(args.clip_alpha),
        )
        out_path = output_dir / pred_path.name
        _write_png01(out_path, calibrated)
        rows.append({"fname": pred_path.name, "z": z, "output_path": str(out_path), **stats})

    pd.DataFrame(rows).to_csv(output_dir / "export_manifest.csv", index=False, encoding="utf-8")
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "t1_dir": str(t1_dir),
        "train_t1_dir": str(args.train_t1_dir),
        "train_fa_dir": str(args.train_fa_dir),
        "brain_threshold": args.brain_threshold,
        "lowpass_sigma": args.lowpass_sigma,
        "mean_alpha": args.mean_alpha,
        "std_alpha": args.std_alpha,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "max_low_delta": args.max_low_delta,
        "upper_quantile": args.upper_quantile,
        "clip_alpha": args.clip_alpha,
        "slice_stats": slice_stats,
    }
    with open(output_dir / "fa_intensity_calibration.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"FA_INTENSITY_CALIBRATION: exported={len(rows)} output_dir={output_dir}")
    print(f"Saved manifest to: {output_dir / 'export_manifest.csv'}")


if __name__ == "__main__":
    main()
