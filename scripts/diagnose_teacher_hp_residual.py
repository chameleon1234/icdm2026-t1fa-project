import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.models.pmrf_t1fa import prepare_stage1_input, reduce_rgb_to_single_channel
from pmrf_t1fa.train_pmrf_t1fa_stage2 import laplacian_variance, load_stage1_model, make_slice_dataset, predict_stage1_batch
from pmrf_t1fa.train_pmrf_t1fa_stage2_hp_refiner import highpass, load_teacher_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether a teacher folder has enough high-pass residual over Stage 1 coarse.")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt")
    parser.add_argument("--teacher_pred_dir", required=True)
    parser.add_argument("--t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--hp_kernel_size", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


@torch.no_grad()
def main() -> dict[str, Any]:
    args = parse_args()
    device = torch.device(args.device)
    stage1, prediction_mode, stage1_channels, detail_scale = load_stage1_model(args.stage1_ckpt, device)
    dataset = make_slice_dataset(args.t1_dir, args.fa_dir, stage1_channels)
    if args.limit > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    hp_std = []
    hp_abs_mean = []
    hp_abs_p95 = []
    teacher_sharp_ratio = []
    coarse_sharp_ratio = []
    teacher_minus_coarse_sharp_ratio = []
    teacher_target_l1 = []
    missing = 0
    total = 0

    for batch in tqdm(loader, desc="Diagnosing teacher HP residual"):
        t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
        target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
        coarse = predict_stage1_batch(stage1, t1_img, device, prediction_mode, detail_scale).to(device)
        try:
            teacher = load_teacher_batch(args.teacher_pred_dir, batch["fname"], device, target.shape)
        except FileNotFoundError:
            missing += len(batch["fname"])
            continue
        residual = highpass(teacher, args.hp_kernel_size) - highpass(coarse, args.hp_kernel_size)
        flat = residual.flatten(start_dim=1)
        abs_flat = flat.abs()
        hp_std.extend(flat.std(dim=1, unbiased=False).detach().cpu().tolist())
        hp_abs_mean.extend(abs_flat.mean(dim=1).detach().cpu().tolist())
        hp_abs_p95.extend(torch.quantile(abs_flat, 0.95, dim=1).detach().cpu().tolist())
        target_sharp = laplacian_variance(target).item()
        teacher_sharp_ratio.append(laplacian_variance(teacher).item() / max(target_sharp, 1e-8))
        coarse_sharp_ratio.append(laplacian_variance(coarse).item() / max(target_sharp, 1e-8))
        teacher_minus_coarse_sharp_ratio.append(
            (laplacian_variance(teacher).item() - laplacian_variance(coarse).item()) / max(target_sharp, 1e-8)
        )
        teacher_target_l1.append(torch.nn.functional.l1_loss(teacher, target).item())
        total += len(batch["fname"])

    summary = {
        "teacher_pred_dir": str(args.teacher_pred_dir),
        "stage1_ckpt": str(args.stage1_ckpt),
        "t1_dir": str(args.t1_dir),
        "fa_dir": str(args.fa_dir),
        "evaluated": int(total),
        "missing": int(missing),
        "hp_kernel_size": int(args.hp_kernel_size),
        "hp_residual_std_mean": _mean(hp_std),
        "hp_residual_abs_mean": _mean(hp_abs_mean),
        "hp_residual_abs_p95_mean": _mean(hp_abs_p95),
        "teacher_sharp_ratio_mean": _mean(teacher_sharp_ratio),
        "coarse_sharp_ratio_mean": _mean(coarse_sharp_ratio),
        "teacher_minus_coarse_sharp_ratio_mean": _mean(teacher_minus_coarse_sharp_ratio),
        "teacher_target_l1_mean": _mean(teacher_target_l1),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    main()
