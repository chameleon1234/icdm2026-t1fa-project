import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pmrf_t1fa.models.pmrf_t1fa import center_channel, prepare_stage1_input, reduce_rgb_to_single_channel, single_channel_laplacian
from pmrf_t1fa.train_pmrf_t1fa_stage2 import (
    SSIMLoss,
    build_training_masks,
    clamp_to_image_range,
    load_stage1_model,
    make_slice_dataset,
    maybe_limit_dataset,
    predict_stage1_batch,
)
from pmrf_t1fa.train_pmrf_t1fa_stage2_hp_refiner import (
    HighPassRefinerNet,
    _autocast_context,
    _average_metrics,
    _metric_dict,
    apply_hp_residual_cap,
    build_hp_refiner_loss,
    build_residual_gate,
    hp_refiner_best_key,
    hp_refiner_legacy_score,
    parse_kernel_list,
    save_preview,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-train a Stage 2 refiner with adjacent Stage1 coarse context.")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_wmroi_detail_5slice/checkpoints/best_stage1.pt")
    parser.add_argument("--stage1_device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run_name", default="pmrf_t1fa_stage2_3d_context_refiner_smoke")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=512)
    parser.add_argument("--val_limit", type=int, default=256)
    parser.add_argument("--preview_batch_size", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=128)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--hp_kernel_size", type=int, default=5)
    parser.add_argument("--lp_kernel_size", type=int, default=13)
    parser.add_argument("--hp_residual_weight", type=float, default=0.6)
    parser.add_argument("--hp_image_weight", type=float, default=0.6)
    parser.add_argument("--wm_hp_weight", type=float, default=1.2)
    parser.add_argument("--multiscale_hp_weight", type=float, default=0.0)
    parser.add_argument("--multiscale_wm_hp_weight", type=float, default=0.0)
    parser.add_argument("--multiscale_hp_kernels", default="3,5,9")
    parser.add_argument("--wm_final_l1_weight", type=float, default=2.0)
    parser.add_argument("--wm_guard_weight", type=float, default=20.0)
    parser.add_argument("--wm_guard_margin", type=float, default=0.0)
    parser.add_argument("--lowpass_weight", type=float, default=2.0)
    parser.add_argument("--sharpness_floor_weight", type=float, default=8.0)
    parser.add_argument("--sharpness_floor_margin", type=float, default=0.05)
    parser.add_argument("--best_min_delta_sharp", type=float, default=0.08)
    parser.add_argument("--best_min_delta_wm_l1", type=float, default=0.0)
    parser.add_argument("--final_l1_weight", type=float, default=0.20)
    parser.add_argument("--final_ssim_weight", type=float, default=0.05)
    parser.add_argument("--brain_t1_threshold", type=float, default=0.05)
    parser.add_argument("--brain_fa_threshold", type=float, default=0.02)
    parser.add_argument("--wm_quantile", type=float, default=0.65)
    parser.add_argument("--wm_min_threshold", type=float, default=0.20)
    parser.add_argument("--residual_scale", type=float, default=0.55)
    parser.add_argument("--residual_gate_mode", default="edge", choices=["none", "edge"])
    parser.add_argument("--residual_gate_min", type=float, default=0.50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--keep_existing_best", action="store_true")
    return parser.parse_args()


def ensure_dirs(run_name: str) -> tuple[Path, Path, Path]:
    root = Path("outputs") / run_name
    ckpt = root / "checkpoints"
    preview = root / "previews"
    ckpt.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    return root, ckpt, preview


def make_neighbor_stage1_inputs(t1_stack: torch.Tensor) -> torch.Tensor:
    """Approximate z-1/z/z+1 Stage1 windows from an existing centered T1 stack."""
    if t1_stack.dim() != 4:
        raise ValueError(f"Expected BCHW T1 stack, got {tuple(t1_stack.shape)}")
    if t1_stack.shape[1] < 3 or t1_stack.shape[1] % 2 == 0:
        raise ValueError(f"Expected odd context channels >=3, got {t1_stack.shape[1]}")
    left = torch.cat([t1_stack[:, :1], t1_stack[:, :-1]], dim=1)
    center = t1_stack
    right = torch.cat([t1_stack[:, 1:], t1_stack[:, -1:]], dim=1)
    return torch.stack([left, center, right], dim=1)


@torch.no_grad()
def predict_stage1_neighbor_coarse(
    stage1: torch.nn.Module,
    t1_stack: torch.Tensor,
    stage1_device: torch.device,
    prediction_mode: str,
    detail_scale: float,
) -> torch.Tensor:
    windows = make_neighbor_stage1_inputs(t1_stack)
    batch, neighbors, channels, height, width = windows.shape
    flat = windows.reshape(batch * neighbors, channels, height, width)
    coarse = predict_stage1_batch(stage1, flat, stage1_device, prediction_mode, detail_scale)
    return coarse.reshape(batch, neighbors, height, width)


def build_3d_context_refiner_input(t1_stack: torch.Tensor, coarse_triplet: torch.Tensor) -> torch.Tensor:
    if coarse_triplet.dim() != 4 or coarse_triplet.shape[1] != 3:
        raise ValueError(f"Expected Bx3xHxW coarse triplet, got {tuple(coarse_triplet.shape)}")
    coarse_center = coarse_triplet[:, 1:2]
    t1_edge = single_channel_laplacian(center_channel(t1_stack))
    coarse_edge = single_channel_laplacian(coarse_center)
    return torch.cat([t1_stack, coarse_triplet, t1_edge, coarse_edge], dim=1)


def main() -> None:
    args = parse_args()
    root_dir, ckpt_dir, preview_dir = ensure_dirs(args.run_name)
    log_path = root_dir / "train.log"
    multiscale_hp_kernels = parse_kernel_list(args.multiscale_hp_kernels)
    if not args.keep_existing_best:
        for stale_best in (ckpt_dir / "best_3d_context_refiner.pt", root_dir / "best_3d_context_refiner.pt"):
            if stale_best.exists():
                stale_best.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage1_device == "cpu":
        stage1_device = torch.device("cpu")
    elif args.stage1_device == "cuda":
        stage1_device = torch.device("cuda")
    else:
        stage1_device = device

    stage1, stage1_prediction_mode, stage1_channels, stage1_detail_scale = load_stage1_model(args.stage1_ckpt, stage1_device)
    if stage1_channels < 3:
        raise ValueError("3D context refiner requires a multi-slice Stage1 checkpoint, preferably 5-slice.")

    train_dataset = maybe_limit_dataset(make_slice_dataset(args.train_t1_dir, args.train_fa_dir, stage1_channels), args.train_limit)
    val_dataset = maybe_limit_dataset(make_slice_dataset(args.val_t1_dir, args.val_fa_dir, stage1_channels), args.val_limit)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    model = HighPassRefinerNet(in_channels=stage1_channels + 5, width=args.width, num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    ssim_loss_fn = SSIMLoss().to(device)
    best_key: tuple[float, float, float] | None = None
    global_step = 0

    print(
        f"3D context refiner on {len(train_dataset)} train / {len(val_dataset)} val slices | "
        f"stage1_channels={stage1_channels} | input_channels={stage1_channels + 5} | device={device} | "
        f"width={args.width} blocks={args.num_blocks} | residual_scale={args.residual_scale} "
        f"gate={args.residual_gate_mode}@{args.residual_gate_min} | "
        f"best_gate=delta_sharp>={args.best_min_delta_sharp},delta_wm_l1>={args.best_min_delta_wm_l1}"
    )

    for epoch in range(args.epochs):
        model.train()
        train_totals: Dict[str, float] = {}
        for batch in tqdm(train_loader, desc=f"3DContext Epoch {epoch + 1}/{args.epochs}"):
            t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
            target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
            with torch.no_grad(), _autocast_context(device, args.mixed_precision):
                coarse_triplet = predict_stage1_neighbor_coarse(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale).to(device)
            coarse_center = coarse_triplet[:, 1:2]
            brain_mask, wm_mask = build_training_masks(
                t1_img,
                target,
                args.brain_t1_threshold,
                args.brain_fa_threshold,
                args.wm_quantile,
                args.wm_min_threshold,
            )
            model_input = build_3d_context_refiner_input(t1_img, coarse_triplet)
            residual_gate = build_residual_gate(t1_img, coarse_center, args.residual_gate_mode, args.residual_gate_min)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                hp_pred = apply_hp_residual_cap(model(model_input), args.residual_scale, residual_gate)
                losses = build_hp_refiner_loss(
                    hp_pred,
                    coarse_center,
                    target,
                    None,
                    brain_mask,
                    wm_mask,
                    args.hp_kernel_size,
                    args.lp_kernel_size,
                    args.hp_residual_weight,
                    args.hp_image_weight,
                    args.wm_hp_weight,
                    args.multiscale_hp_weight,
                    args.multiscale_wm_hp_weight,
                    multiscale_hp_kernels,
                    args.wm_final_l1_weight,
                    args.wm_guard_weight,
                    args.wm_guard_margin,
                    args.lowpass_weight,
                    args.sharpness_floor_weight,
                    args.sharpness_floor_margin,
                    0.0,
                    args.final_l1_weight,
                    args.final_ssim_weight,
                    ssim_loss_fn,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key, value in losses.items():
                if key == "final":
                    continue
                train_totals[key] = train_totals.get(key, 0.0) + float(value.detach().cpu())
            if args.preview_every > 0 and global_step % args.preview_every == 0:
                save_preview(
                    preview_dir,
                    global_step,
                    t1_img[: args.preview_batch_size],
                    coarse_center[: args.preview_batch_size],
                    losses["final"][: args.preview_batch_size],
                    target[: args.preview_batch_size],
                )

        model.eval()
        rows: list[Dict[str, float]] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"3DContext Val {epoch + 1}/{args.epochs}"):
                t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
                target = reduce_rgb_to_single_channel(batch["fa_slice"].to(device))
                with _autocast_context(device, args.mixed_precision):
                    coarse_triplet = predict_stage1_neighbor_coarse(stage1, t1_img, stage1_device, stage1_prediction_mode, stage1_detail_scale).to(device)
                    coarse_center = coarse_triplet[:, 1:2]
                    residual_gate = build_residual_gate(t1_img, coarse_center, args.residual_gate_mode, args.residual_gate_min)
                    hp_pred = apply_hp_residual_cap(model(build_3d_context_refiner_input(t1_img, coarse_triplet)), args.residual_scale, residual_gate)
                    refined = clamp_to_image_range(coarse_center + hp_pred)
                rows.append(_metric_dict(refined.float(), coarse_center.float(), target.float(), t1_img, ssim_loss_fn, args))

        metrics = _average_metrics(rows)
        train_batches = max(len(train_loader), 1)
        train_summary = {key: value / train_batches for key, value in train_totals.items()}
        score = hp_refiner_legacy_score(metrics)
        candidate_key = hp_refiner_best_key(metrics, args.best_min_delta_sharp, args.best_min_delta_wm_l1)
        passed_best_gate = candidate_key is not None
        is_best = passed_best_gate and (best_key is None or candidate_key > best_key)
        if is_best:
            best_key = candidate_key
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "score": score,
                    "best_key": list(candidate_key),
                    "metrics": metrics,
                    "stage1_channels": stage1_channels,
                    "input_channels": stage1_channels + 5,
                },
                ckpt_dir / "best_3d_context_refiner.pt",
            )
            torch.save(model.state_dict(), root_dir / "best_3d_context_refiner.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "score": score,
                "best_key": list(candidate_key) if candidate_key is not None else None,
                "metrics": metrics,
                "stage1_channels": stage1_channels,
                "input_channels": stage1_channels + 5,
            },
            ckpt_dir / "latest_3d_context_refiner.pt",
        )
        log_row = {
            "epoch": epoch + 1,
            "score": score,
            "passed_best_gate": passed_best_gate,
            "is_best": bool(is_best),
            "train": train_summary,
            "val": metrics,
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        print(
            f"[3DContext][Epoch {epoch + 1}] PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"L1={metrics['l1']:.6f} HP_L1={metrics['hp_l1']:.6f} WM_L1={metrics['wm_l1']:.6f} "
            f"SharpRatio={metrics['sharp_ratio']:.4f} CoarseSharpRatio={metrics['coarse_sharp_ratio']:.4f} "
            f"DeltaSharp={metrics['delta_sharp']:.4f} DeltaWM_L1={metrics['delta_wm_l1']:.6f} "
            f"DeltaPSNR={metrics['delta_psnr']:.4f} Score={score:.4f} Gate={int(passed_best_gate)} Best={int(is_best)}"
        )


if __name__ == "__main__":
    main()
