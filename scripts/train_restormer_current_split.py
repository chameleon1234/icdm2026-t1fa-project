import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_stack_unet_current_split import build_dataset
from scripts.train_unet_current_split import (
    _autocast_context,
    _collate,
    _limit_dataset,
    _psnr_from_mse,
    _to01,
    assert_disjoint_subject_splits,
    collect_subjects_from_slice_dir,
)
from train_baseline_restormer import RestormerBaseline, SSIMLoss


class OutputActivatedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, activation: str):
        super().__init__()
        self.model = model
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.activation == "tanh":
            return torch.tanh(out)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fair single-slice Restormer baseline on the current split.")
    parser.add_argument("--train_t1_dir", default="data/processed/train/t1_slices")
    parser.add_argument("--train_fa_dir", default="data/processed/train/fa_slices")
    parser.add_argument("--val_t1_dir", default="data/processed/val/t1_slices")
    parser.add_argument("--val_fa_dir", default="data/processed/val/fa_slices")
    parser.add_argument("--test_t1_dir", default="data/processed/test/t1_slices")
    parser.add_argument("--run_name", default="restormer_current_split_e12")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--num_blocks", default="2,2,2,2")
    parser.add_argument("--num_heads", default="1,2,4,8")
    parser.add_argument("--expansion_factor", type=float, default=2.66)
    parser.add_argument("--output_activation", choices=["tanh", "linear"], default="tanh")
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed_precision", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected comma-separated integers, got {text!r}")
    return values


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    n_pixels = 0
    n_items = 0
    for batch in tqdm(loader, desc="Restormer Val", leave=False):
        if batch is None:
            continue
        t1 = batch["t1_slice"].to(device, non_blocking=True)
        fa = batch["fa_slice"].to(device, non_blocking=True)
        pred = model(t1)
        pred01 = _to01(pred)
        fa01 = _to01(fa)
        diff = pred01 - fa01
        total_l1 += diff.abs().sum().item()
        total_mse += (diff * diff).sum().item()
        n_pixels += diff.numel()
        n_items += t1.shape[0]
    mse = total_mse / max(n_pixels, 1)
    return {
        "psnr": _psnr_from_mse(mse),
        "mse": mse,
        "mae": total_l1 / max(n_pixels, 1),
        "n_items": float(n_items),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_subjects = collect_subjects_from_slice_dir(args.train_t1_dir)
    val_subjects = collect_subjects_from_slice_dir(args.val_t1_dir)
    test_subjects = collect_subjects_from_slice_dir(args.test_t1_dir)
    assert_disjoint_subject_splits(train_subjects, val_subjects, test_subjects)

    run_dir = Path(args.output_root) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _limit_dataset(build_dataset(args.train_t1_dir, args.train_fa_dir, 1), args.train_limit)
    val_dataset = _limit_dataset(build_dataset(args.val_t1_dir, args.val_fa_dir, 1), args.val_limit)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    num_blocks = _parse_int_tuple(args.num_blocks)
    num_heads = _parse_int_tuple(args.num_heads)
    if len(num_blocks) != 4 or len(num_heads) != 4:
        raise ValueError("--num_blocks and --num_heads must each contain four integers.")

    base_model = RestormerBaseline(
        in_channels=1,
        out_channels=1,
        dim=args.dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        expansion_factor=args.expansion_factor,
    )
    model = OutputActivatedModel(base_model, args.output_activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ssim_loss = SSIMLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.mixed_precision == "fp16")

    split_summary = {
        "train_subjects": len(train_subjects),
        "val_subjects": len(val_subjects),
        "test_subjects": len(test_subjects),
        "train_slices": len(train_dataset),
        "val_slices": len(val_dataset),
        "context_slices": 1,
        "test_split_checked": True,
    }
    (run_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_psnr = float("-inf")
    best_epoch = -1
    started = time.time()
    print(
        f"Restormer current split training | train={len(train_dataset)} val={len(val_dataset)} "
        f"| subjects={len(train_subjects)}/{len(val_subjects)}/{len(test_subjects)} "
        f"| dim={args.dim} blocks={num_blocks} activation={args.output_activation} "
        f"| device={device} | mp={args.mixed_precision}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"Restormer Epoch {epoch}/{args.epochs}")
        for batch in progress:
            if batch is None:
                continue
            t1 = batch["t1_slice"].to(device, non_blocking=True)
            fa = batch["fa_slice"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, args.mixed_precision):
                pred = model(t1)
                loss_l1 = F.l1_loss(pred, fa)
                loss_mse = F.mse_loss(pred, fa)
                loss_ssim = ssim_loss(pred, fa)
                loss = args.l1_weight * loss_l1 + args.mse_weight * loss_mse + args.ssim_weight * loss_ssim
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{loss.item():.5f}")

        val_metrics = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(steps, 1)
        is_best = val_metrics["psnr"] > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_restormer.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"restormer_epoch_{epoch:04d}.pt")
        row = {"epoch": epoch, "train_loss": avg_loss, **val_metrics, "best": int(is_best)}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
        print(
            f"[Restormer][Epoch {epoch}] train_loss={avg_loss:.6f} "
            f"val_PSNR={val_metrics['psnr']:.4f} val_MAE={val_metrics['mae']:.6f} Best={int(is_best)}"
        )

    summary = {
        "run_name": args.run_name,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "elapsed_sec": time.time() - started,
        "args": vars(args),
        "split_summary": split_summary,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved best checkpoint: {ckpt_dir / 'best_restormer.pt'}")
    return summary


if __name__ == "__main__":
    train(parse_args())
