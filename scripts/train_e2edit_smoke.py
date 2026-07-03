from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import T1FADataset
from pmrf_t1fa.e2edit import (
    E2EDiTCorrector,
    E2EDiTStage1,
    SingleBranchNet,
    branch_diagnostics,
    compute_batch_metrics,
    e2edit_m0_loss,
    e2edit_m1_loss,
    e2edit_m2_loss,
    e2edit_m3_loss,
    highpass,
    make_wm_proxy,
)


METHODS = {
    "M0_single_branch": "M0",
    "M1_dual_branch_naive": "M1",
    "M2_dual_branch_antimean": "M2",
    "M3_dual_branch_antimean_corrector": "M3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2EDiT ADNI smoke training with M0/M1/M2/M3 ablations.")
    p.add_argument("--train_t1_dir", default="data/adni_processed/train/t1_slices")
    p.add_argument("--train_fa_dir", default="data/adni_processed/train/fa_slices")
    p.add_argument("--val_t1_dir", default="data/adni_processed/val/t1_slices")
    p.add_argument("--val_fa_dir", default="data/adni_processed/val/fa_slices")
    p.add_argument("--output_root", default="outputs/icdm2026/e2edit/smoke")
    p.add_argument("--train_limit", type=int, default=4096)
    p.add_argument("--val_limit", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--preview_count", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--auto_full", action="store_true", help="Reserved hook; this smoke script only writes GO/NO-GO.")
    return p.parse_args()


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "no":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def _limit_dataset(dataset: T1FADataset, limit: int) -> torch.utils.data.Dataset:
    if limit and limit > 0:
        return Subset(dataset, list(range(min(limit, len(dataset)))))
    return dataset


def _to_single(batch: Dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    t1 = batch["t1_slice"].to(device, non_blocking=True).float()
    fa = batch["fa_slice"].to(device, non_blocking=True).float()
    if t1.shape[1] > 1:
        t1 = t1[:, :1]
    if fa.shape[1] > 1:
        fa = fa[:, :1]
    return t1, fa, list(batch["fname"])


def _score(metrics: Dict[str, float]) -> float:
    return (
        metrics["psnr"]
        + 8.0 * metrics["ssim"]
        + 2.0 * metrics["roi_ccc_proxy"]
        + 0.8 * metrics["hf_corr"]
        - 8.0 * metrics["wm_mae"]
        - 2.0 * abs(metrics["sharp_ratio"] - 1.0)
        - 10.0 * metrics["overbright"]
    )


def _mean_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for row in rows for k in row})
    return {k: float(np.nanmean([row.get(k, np.nan) for row in rows])) for k in keys}


@torch.no_grad()
def evaluate_model(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
) -> Dict[str, float]:
    if method == "M3":
        model[0].eval()
        model[1].eval()
    else:
        model.eval()
    rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []
    for batch in tqdm(loader, desc=f"Val {method}", leave=False):
        t1, target, _ = _to_single(batch, device)
        wm = make_wm_proxy(t1)
        with _autocast(device, mixed_precision):
            if method == "M0":
                pred = model(t1)
                row = compute_batch_metrics(pred, target, wm)
            elif method in {"M1", "M2"}:
                out = model(t1)
                pred = out["prior"]
                row = compute_batch_metrics(pred, target, wm)
                diag_rows.append(branch_diagnostics(out["base"], out["detail"], out["prior"], target))
            else:
                s1, corrector = model
                out1 = s1(t1)
                out2 = corrector(t1, out1["prior"], wm)
                pred = out2["final"]
                row = compute_batch_metrics(pred, target, wm)
                diag_rows.append(branch_diagnostics(out1["base"], out1["detail"], out1["prior"], target, out2["correction"]))
        rows.append(row)
    summary = _mean_rows(rows)
    if diag_rows:
        summary.update(_mean_rows(diag_rows))
    summary["score"] = _score(summary)
    return summary


def train_one(
    method_dir: Path,
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    m2_ckpt: Path | None = None,
) -> Dict[str, float]:
    method_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = method_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    if method == "M0":
        model: Any = SingleBranchNet(args.width, args.num_blocks).to(device)
        params = list(model.parameters())
    elif method in {"M1", "M2"}:
        model = E2EDiTStage1(args.width, args.num_blocks).to(device)
        params = list(model.parameters())
    else:
        if m2_ckpt is None or not m2_ckpt.exists():
            raise FileNotFoundError(f"M3 requires M2 checkpoint, got {m2_ckpt}")
        stage1 = E2EDiTStage1(args.width, args.num_blocks).to(device)
        checkpoint = torch.load(m2_ckpt, map_location=device)
        stage1.load_state_dict(checkpoint["model"])
        for p in stage1.parameters():
            p.requires_grad_(False)
        corrector = E2EDiTCorrector(args.width, args.num_blocks).to(device)
        model = (stage1, corrector)
        params = list(corrector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    best: Dict[str, float] | None = None
    best_epoch = 0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        if method == "M3":
            model[0].eval()
            model[1].train()
        else:
            model.train()
        total_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"{method} Epoch {epoch}/{args.epochs}", leave=False):
            t1, target, _ = _to_single(batch, device)
            wm = make_wm_proxy(t1)
            opt.zero_grad(set_to_none=True)
            with _autocast(device, args.mixed_precision):
                if method == "M0":
                    pred = model(t1)
                    loss, _ = e2edit_m0_loss(pred, target, wm)
                elif method == "M1":
                    out = model(t1)
                    loss, _ = e2edit_m1_loss(out["base"], out["detail"], target, wm)
                elif method == "M2":
                    out = model(t1)
                    loss, _ = e2edit_m2_loss(out["base"], out["detail"], target, wm)
                else:
                    with torch.no_grad():
                        out1 = model[0](t1)
                    out2 = model[1](t1, out1["prior"], wm)
                    loss, _ = e2edit_m3_loss(out2["final"], out1["prior"], out2["correction"], target, wm)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            bs = int(t1.shape[0])
            total_loss += float(loss.detach().item()) * bs
            seen += bs
        metrics = evaluate_model(method, model, val_loader, device, args.mixed_precision)
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = total_loss / max(1, seen)
        is_best = best is None or metrics["score"] > best["score"]
        if is_best:
            best = dict(metrics)
            best_epoch = epoch
            if method == "M3":
                torch.save({"stage1": model[0].state_dict(), "corrector": model[1].state_dict(), "metrics": best, "args": vars(args)}, ckpt_dir / "best.pt")
            else:
                torch.save({"model": model.state_dict(), "metrics": best, "args": vars(args)}, ckpt_dir / "best.pt")
        log_rows.append({"method": method, **metrics, "best": float(is_best)})
        print(
            f"[E2EDiT][{method}][Epoch {epoch}] PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"WM={metrics['wm_mae']:.4f} ROI={metrics['roi_ccc_proxy']:.4f} Sharp={metrics['sharp_ratio']:.4f} "
            f"HF={metrics['hf_corr']:.4f} Score={metrics['score']:.4f} Best={int(is_best)}"
        )
    _write_csv(method_dir / "train_log.csv", log_rows)
    assert best is not None
    best["best_epoch"] = float(best_epoch)
    with (method_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    return best


def _img(x: torch.Tensor) -> np.ndarray:
    a = x.detach().float().cpu().squeeze().numpy()
    a = np.clip((a + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return a


@torch.no_grad()
def write_preview_panels(
    m0: SingleBranchNet,
    m2: E2EDiTStage1,
    corrector: E2EDiTCorrector,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    m0.eval()
    m2.eval()
    corrector.eval()
    saved = 0
    for batch in loader:
        t1, target, names = _to_single(batch, device)
        wm = make_wm_proxy(t1)
        pred0 = m0(t1)
        out2 = m2(t1)
        out3 = corrector(t1, out2["prior"], wm)
        for i, name in enumerate(names):
            cols = [
                ("T1", t1[i]),
                ("FA_GT", target[i]),
                ("M0", pred0[i]),
                ("B", out2["base"][i]),
                ("H", out2["detail"][i] * 4.0),
                ("P_M2", out2["prior"][i]),
                ("Delta", out3["correction"][i] * 8.0),
                ("Final_M3", out3["final"][i]),
                ("Error", (out3["final"][i] - target[i]).abs() * 4.0 - 1.0),
                ("HP_GT", highpass(target[i : i + 1], 5)[0] * 4.0),
                ("HP_M3", highpass(out3["final"][i : i + 1], 5)[0] * 4.0),
            ]
            tile_imgs = []
            for label, tensor in cols:
                im = cv2.cvtColor(_img(tensor), cv2.COLOR_GRAY2BGR)
                cv2.putText(im, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                tile_imgs.append(im)
            panel = np.concatenate(tile_imgs, axis=1)
            cv2.imwrite(str(out_dir / f"{Path(name).stem}_e2edit_panel.png"), panel)
            saved += 1
            if saved >= count:
                return


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["method", "best_epoch", "psnr", "ssim", "mae", "wm_mae", "roi_ccc_proxy", "sharp_ratio", "tenengrad_ratio", "hf_corr", "overbright", "score"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decide(rows: Dict[str, Dict[str, float]]) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    m0, m2, m3 = rows["M0"], rows["M2"], rows["M3"]
    freq_ok = True
    if m2["psnr"] < m0["psnr"] - 0.1:
        freq_ok = False
        reasons.append(f"M2 PSNR lower than M0 by >0.1 ({m2['psnr']:.4f} vs {m0['psnr']:.4f}).")
    if m2["ssim"] < m0["ssim"] - 0.003:
        freq_ok = False
        reasons.append(f"M2 SSIM lower than M0 by >0.003 ({m2['ssim']:.4f} vs {m0['ssim']:.4f}).")
    if abs(m2["sharp_ratio"] - 1.0) > abs(m0["sharp_ratio"] - 1.0):
        freq_ok = False
        reasons.append("M2 SharpRatio is not closer to 1 than M0.")
    if m2["wm_mae"] > m0["wm_mae"] + 0.003:
        freq_ok = False
        reasons.append("M2 WM-MAE clearly worse than M0.")
    if m2.get("base_hp_abs", 1.0) >= m2.get("detail_hp_abs", 0.0) * 1.3 and m2.get("base_hp_abs", 0.0) > 0.01:
        freq_ok = False
        reasons.append("B branch still carries too much high-frequency energy relative to H.")
    corr_ok = True
    if m3["psnr"] <= m2["psnr"]:
        corr_ok = False
        reasons.append("M3 did not improve PSNR over M2.")
    if m3["ssim"] < m2["ssim"] - 0.002:
        corr_ok = False
        reasons.append("M3 SSIM dropped relative to M2.")
    if m3["wm_mae"] >= m2["wm_mae"]:
        corr_ok = False
        reasons.append("M3 did not lower WM-MAE over M2.")
    if m3["roi_ccc_proxy"] <= m2["roi_ccc_proxy"]:
        corr_ok = False
        reasons.append("M3 did not improve ROI proxy over M2.")
    if not (0.90 <= m3["sharp_ratio"] <= 1.10):
        corr_ok = False
        reasons.append("M3 SharpRatio outside [0.90, 1.10].")
    threshold_ok = True
    checks = [
        (m3["psnr"] >= 27.8, "M3 PSNR < 27.8."),
        (m3["ssim"] >= 0.895, "M3 SSIM < 0.895."),
        (m3["wm_mae"] <= 0.065, "M3 WM-MAE > 0.065."),
        (m3["roi_ccc_proxy"] >= 0.78, "M3 ROI proxy < 0.78."),
        (0.90 <= m3["sharp_ratio"] <= 1.10, "M3 SharpRatio outside [0.90, 1.10]."),
    ]
    for ok, msg in checks:
        if not ok:
            threshold_ok = False
            reasons.append(msg)
    go = freq_ok and corr_ok and threshold_ok
    if go:
        reasons.append("GO: M2 supports frequency decoupling and M3 supports bounded correction under smoke criteria.")
    return go, reasons


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = _limit_dataset(T1FADataset(args.train_t1_dir, args.train_fa_dir, preload_ram=False), args.train_limit)
    val_set = _limit_dataset(T1FADataset(args.val_t1_dir, args.val_fa_dir, preload_ram=False), args.val_limit)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    print(
        f"E2EDiT smoke on {len(train_set)} train / {len(val_set)} val | device={device} "
        f"width={args.width} blocks={args.num_blocks} range=[-1,1]"
    )
    all_metrics: Dict[str, Dict[str, float]] = {}
    m2_ckpt: Path | None = None
    for folder, method in METHODS.items():
        method_dir = output_root / folder
        metrics_path = method_dir / "best_metrics.json"
        ckpt_path = method_dir / "checkpoints" / "best.pt"
        if metrics_path.exists() and ckpt_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"[E2EDiT][{method}] skip existing best metrics at {metrics_path}")
        else:
            metrics = train_one(method_dir, method, train_loader, val_loader, args, device, m2_ckpt=m2_ckpt)
        metrics["method"] = method
        metrics["folder"] = folder
        all_metrics[method] = metrics
        if method == "M2":
            m2_ckpt = output_root / folder / "checkpoints" / "best.pt"
    rows = [all_metrics[METHODS[k]] for k in METHODS]
    table_dir = output_root / "tables"
    _write_csv(table_dir / "smoke_metrics_summary.csv", rows)
    _write_md_table(table_dir / "smoke_metrics_summary.md", rows)

    m0 = SingleBranchNet(args.width, args.num_blocks).to(device)
    m0.load_state_dict(torch.load(output_root / "M0_single_branch" / "checkpoints" / "best.pt", map_location=device)["model"])
    m2 = E2EDiTStage1(args.width, args.num_blocks).to(device)
    m2.load_state_dict(torch.load(output_root / "M2_dual_branch_antimean" / "checkpoints" / "best.pt", map_location=device)["model"])
    corrector = E2EDiTCorrector(args.width, args.num_blocks).to(device)
    corrector.load_state_dict(torch.load(output_root / "M3_dual_branch_antimean_corrector" / "checkpoints" / "best.pt", map_location=device)["corrector"])
    write_preview_panels(m0, m2, corrector, val_loader, output_root / "figures" / "preview_panels", device, args.preview_count)

    go, reasons = _decide(all_metrics)
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# E2EDiT Smoke Report",
        "",
        "This smoke test uses ADNI train/val only and does not use A080, PriorFlow, template sources, offline blend folders, GAN, or LPIPS.",
        "",
        (table_dir / "smoke_metrics_summary.md").read_text(encoding="utf-8"),
    ]
    (report_dir / "smoke_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    cn_lines = [
        "# E2EDiT smoke 测试报告",
        "",
        "本 smoke test 只使用 ADNI train/val，不使用 A080、PriorFlow、template source、离线 blend folder、GAN 或 LPIPS。",
        "",
        (table_dir / "smoke_metrics_summary.md").read_text(encoding="utf-8"),
    ]
    (report_dir / "smoke_report_cn.md").write_text("\n".join(cn_lines), encoding="utf-8")
    decision = ["# E2EDiT Go/No-Go Decision", "", f"Decision: {'GO' if go else 'NO-GO'}", ""]
    decision += [f"- {r}" for r in reasons]
    (report_dir / "go_no_go_decision.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    decision_cn = ["# E2EDiT Go/No-Go 判定", "", f"判定：{'GO' if go else 'NO-GO'}", ""]
    decision_cn += [f"- {r}" for r in reasons]
    if go and args.auto_full:
        decision_cn.append("")
        decision_cn.append("注意：本脚本仅完成 smoke 判定；full training 请使用 scripts/train_e2edit_full.py。")
    (report_dir / "go_no_go_decision_cn.md").write_text("\n".join(decision_cn) + "\n", encoding="utf-8")
    print(f"E2EDiT smoke decision: {'GO' if go else 'NO-GO'}")
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
