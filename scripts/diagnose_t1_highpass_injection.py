import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject T1 high-pass detail into a prediction folder for fast visual diagnostics."
    )
    parser.add_argument("--pred_dir", required=True, help="Base predicted FA PNG folder, e.g. PM_STAGE1 or PM_DIRF.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--test_t1_dir", default="", help="Override test T1 slice directory from config.")
    parser.add_argument("--test_fa_dir", default="", help="Override test FA slice directory from config.")
    parser.add_argument("--output_root", default="", help="Root directory for generated prediction folders.")
    parser.add_argument("--method_prefix", default="T1HP_INJECT", help="Prefix for generated method folders.")
    parser.add_argument("--alphas", default="0.2,0.4,0.6")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma used for low-pass subtraction.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    return parser.parse_args()


def parse_alpha_list(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise ValueError("--alphas must contain at least one value")
    return alphas


def alpha_to_tag(alpha: float) -> str:
    text = f"{alpha:g}".replace("-", "m").replace(".", "p")
    return text


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_grayscale_tensor(path: str | Path, target_shape: tuple[int, int] | None = None) -> torch.Tensor:
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if target_shape is not None and image.shape != target_shape:
        image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.clamp(0.0, 1.0)


def write_grayscale_png(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (tensor.detach().cpu().squeeze().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def gaussian_lowpass(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    array = image.detach().cpu().squeeze().numpy().astype(np.float32)
    blurred = cv2.GaussianBlur(array, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return torch.from_numpy(blurred).to(dtype=image.dtype).view_as(image)


def build_brain_mask(t1: torch.Tensor, fa: torch.Tensor | None, t1_threshold: float, fa_threshold: float) -> torch.Tensor:
    mask = t1 > float(t1_threshold)
    if fa is not None:
        mask = mask | (fa > float(fa_threshold))
    return mask.float()


def inject_t1_highpass(
    prediction: torch.Tensor,
    t1: torch.Tensor,
    alpha: float,
    sigma: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lowpass = gaussian_lowpass(t1, sigma=sigma)
    highpass = t1 - lowpass
    if mask is not None:
        highpass = highpass * mask
    injected = torch.clamp(prediction + float(alpha) * highpass, 0.0, 1.0)
    return injected, highpass


def run_injection(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = _read_yaml(args.config)
    processed_root = Path(config["data"]["processed_root"])
    pred_dir = Path(args.pred_dir)
    test_t1_dir = Path(args.test_t1_dir) if args.test_t1_dir else processed_root / "test" / "t1_slices"
    test_fa_dir = Path(args.test_fa_dir) if args.test_fa_dir else processed_root / "test" / "fa_slices"
    output_root = Path(args.output_root) if args.output_root else Path(config["outputs"]["predictions_root"])
    alphas = parse_alpha_list(args.alphas)

    pred_files = sorted(path for path in pred_dir.glob("*.png") if path.name != "export_manifest.csv")
    if args.limit > 0:
        pred_files = pred_files[: args.limit]
    if not pred_files:
        raise FileNotFoundError(f"No prediction PNG files found in {pred_dir}")

    summaries: list[dict[str, Any]] = []
    for alpha in alphas:
        method = f"{args.method_prefix}_A{alpha_to_tag(alpha)}"
        output_dir = output_root / method
        output_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        for pred_path in pred_files:
            t1_path = test_t1_dir / pred_path.name
            fa_path = test_fa_dir / pred_path.name
            if not t1_path.exists():
                raise FileNotFoundError(f"Missing paired T1 slice: {t1_path}")
            prediction = read_grayscale_tensor(pred_path)
            t1 = read_grayscale_tensor(t1_path, target_shape=tuple(prediction.shape[-2:]))
            fa = read_grayscale_tensor(fa_path, target_shape=tuple(prediction.shape[-2:])) if fa_path.exists() else None
            mask = build_brain_mask(t1, fa, args.brain_t1_threshold, args.brain_fa_threshold)
            injected, _ = inject_t1_highpass(prediction, t1, alpha=alpha, sigma=args.sigma, mask=mask)
            write_grayscale_png(output_dir / pred_path.name, injected)
            exported += 1
        summary = {
            "method": method,
            "base_pred_dir": str(pred_dir),
            "output_dir": str(output_dir),
            "alpha": alpha,
            "sigma": args.sigma,
            "exported": exported,
            "diagnostic_only": True,
            "note": "T1 high-pass injection is for feasibility diagnosis, not a final synthesis method.",
        }
        with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        summaries.append(summary)
        print(f"{method}: exported={exported} output_dir={output_dir}")
    return summaries


def main() -> None:
    run_injection(parse_args())


if __name__ == "__main__":
    main()
