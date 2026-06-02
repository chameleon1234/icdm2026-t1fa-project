import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pmrf_t1fa.models.pmrf_t1fa import (
    DetailRefinementFlowUNet,
    DetailStage1Net,
    RefinementFlowUNet,
    Stage1Net,
    build_stage2_condition,
    euler_refine,
    infer_stage1_detail_scale,
    infer_stage1_in_channels,
    infer_stage1_model_variant,
    infer_stage1_prediction_mode,
    infer_stage2_condition_mode,
    prepare_stage1_input,
    predict_stage1_fa,
    reduce_rgb_to_single_channel,
    stage2_condition_channels,
)
from src.data.t1fa_stack_dataset import T1FAStackDataset
from src.datasets import T1FADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PMRF/PM-DIRF predictions as test-set-aligned PNG files.")
    parser.add_argument("--stage", default="stage1", choices=["stage1", "stage2"], help="stage1 exports PM_STAGE1; stage2 exports PM_DIRF.")
    parser.add_argument("--config", default="configs/icdm2026.yaml")
    parser.add_argument("--stage1_ckpt", default="outputs/pmrf_t1fa_stage1_paired_residual/checkpoints/best_stage1.pt")
    parser.add_argument("--stage2_ckpt", default="outputs/pm_dirf_default/checkpoints/best_stage2.pt")
    parser.add_argument("--output_dir", default="", help="Override default prediction output directory.")
    parser.add_argument("--test_t1_dir", default="", help="Override test T1 slice directory from config.")
    parser.add_argument("--test_fa_dir", default="", help="Override test FA slice directory from config.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_steps", type=int, default=-1, help="Use checkpoint eval_steps when -1.")
    parser.add_argument("--detail_boost_override", type=float, default=-1.0, help="Override checkpoint detail_boost when >= 0.")
    parser.add_argument("--condition_on_coarse", action="store_true", help="Force Stage 2 coarse conditioning on.")
    parser.add_argument("--disable_condition_on_coarse", action="store_false", dest="condition_on_coarse")
    parser.add_argument(
        "--condition_mode",
        default="auto",
        choices=["auto", "none", "coarse", "t1", "coarse_t1", "coarse_t1_edge"],
        help="Stage 2 conditioning. auto reads the checkpoint and keeps old coarse-only checkpoints compatible.",
    )
    parser.add_argument("--auto_condition_from_ckpt", action="store_true")
    parser.add_argument("--dynamic_condition_rollout", action="store_true", help="Rebuild the Stage 2 condition from the current rollout image at every step.")
    parser.add_argument("--force_static_condition_rollout", action="store_true", help="Ignore a checkpoint dynamic rollout setting.")
    parser.set_defaults(condition_on_coarse=True)
    return parser.parse_args()


def stage_to_method(stage: str) -> str:
    if stage == "stage1":
        return "PM_STAGE1"
    if stage == "stage2":
        return "PM_DIRF"
    raise ValueError(f"Unsupported stage: {stage}")


def resolve_output_dir(stage: str, predictions_root: str | Path, output_dir: str | Path) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(predictions_root) / stage_to_method(stage)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def checkpoint_args(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        return checkpoint.get("args", {})
    return {}


def load_stage1(stage1_ckpt: str | Path, device: torch.device) -> tuple[torch.nn.Module, str, int, float]:
    checkpoint = torch.load(stage1_ckpt, map_location="cpu")
    stage1_channels = infer_stage1_in_channels(checkpoint)
    ckpt_args = checkpoint_args(checkpoint)
    if infer_stage1_model_variant(ckpt_args) == "detail":
        model = DetailStage1Net(in_channels=stage1_channels, out_channels=1)
    else:
        model = Stage1Net(in_channels=stage1_channels, out_channels=1)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return (
        model,
        infer_stage1_prediction_mode(ckpt_args),
        stage1_channels,
        infer_stage1_detail_scale(ckpt_args),
    )


def make_slice_dataset(t1_dir: Path, fa_dir: Path, stage1_channels: int):
    if stage1_channels > 1:
        return T1FAStackDataset(t1_dir, fa_dir, context_slices=stage1_channels, target_size=(224, 224))
    return T1FADataset(str(t1_dir), str(fa_dir), preload_ram=False)


def resolve_stage2_settings(args: argparse.Namespace) -> tuple[str, bool, int, float, bool]:
    if args.stage != "stage2":
        return "none", False, 0, 0.0, False
    checkpoint = torch.load(args.stage2_ckpt, map_location="cpu")
    ckpt_args = checkpoint_args(checkpoint)
    condition_on_coarse = bool(args.condition_on_coarse)
    if args.auto_condition_from_ckpt or args.condition_mode == "auto":
        condition_on_coarse = bool(ckpt_args.get("condition_on_coarse", condition_on_coarse))
    if args.condition_mode == "auto":
        condition_mode = infer_stage2_condition_mode(ckpt_args)
    else:
        condition_mode = infer_stage2_condition_mode(
            {"condition_mode": args.condition_mode, "condition_on_coarse": condition_on_coarse}
        )
    condition_on_coarse = condition_mode in {"coarse", "coarse_t1", "coarse_t1_edge"}
    eval_steps = args.eval_steps if args.eval_steps > 0 else int(ckpt_args.get("eval_steps", 1))
    detail_boost = (
        float(args.detail_boost_override)
        if float(args.detail_boost_override) >= 0.0
        else float(ckpt_args.get("detail_boost", 0.0))
    )
    dynamic_condition = bool(ckpt_args.get("dynamic_condition_rollout", False))
    if args.dynamic_condition_rollout:
        dynamic_condition = True
    if args.force_static_condition_rollout:
        dynamic_condition = False
    return condition_mode, condition_on_coarse, eval_steps, detail_boost, dynamic_condition


def load_stage2(
    stage2_ckpt: str | Path,
    device: torch.device,
    condition_mode: str,
    stage1_channels: int,
) -> RefinementFlowUNet:
    checkpoint = torch.load(stage2_ckpt, map_location="cpu")
    state_dict = checkpoint_state_dict(checkpoint)
    inferred_condition_channels = int(state_dict["inc.weight"].shape[1]) - 1 if "inc.weight" in state_dict else None
    expected_with_delta = stage2_condition_channels(condition_mode, stage1_channels, include_delta_from_initial=True)
    expected_without_delta = stage2_condition_channels(condition_mode, stage1_channels, include_delta_from_initial=False)
    if inferred_condition_channels in {expected_with_delta, expected_without_delta}:
        condition_channels = inferred_condition_channels
    else:
        condition_channels = expected_with_delta
    include_delta_from_initial = condition_channels == expected_with_delta
    ckpt_args = checkpoint_args(checkpoint)
    if ckpt_args.get("stage2_model_variant", "single") == "detail":
        model = DetailRefinementFlowUNet(input_channels=1, condition_channels=condition_channels)
    else:
        model = RefinementFlowUNet(input_channels=1, condition_channels=condition_channels)
    model.load_state_dict(state_dict)
    model.include_delta_from_initial = include_delta_from_initial
    model.to(device)
    model.eval()
    return model


def save_prediction_png(prediction: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = prediction.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3:
        image = image[0]
    image_01 = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)
    image_u8 = (image_01.numpy() * 255.0).round().astype(np.uint8)
    ok, encoded = cv2.imencode(".png", image_u8)
    if not ok:
        raise ValueError(f"Failed to encode PNG: {output_path}")
    encoded.tofile(str(output_path))


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_export_summary(
    method: str,
    args: argparse.Namespace,
    output_dir: Path,
    exported: int,
    manifest_path: Path,
    stage1_prediction_mode: str,
    stage1_detail_scale: float,
    stage1_channels: int,
    condition_mode: str,
    condition_on_coarse: bool,
    eval_steps: int,
    detail_boost: float,
    dynamic_condition: bool,
    include_delta_from_initial: bool,
    velocity_schedule: str,
) -> dict[str, Any]:
    is_stage2 = args.stage == "stage2"
    return {
        "method": method,
        "stage": args.stage,
        "output_dir": str(output_dir),
        "exported": exported,
        "stage1_ckpt": str(args.stage1_ckpt),
        "stage1_prediction_mode": stage1_prediction_mode,
        "stage1_detail_scale": stage1_detail_scale,
        "stage1_channels": stage1_channels,
        "stage2_ckpt": str(args.stage2_ckpt) if is_stage2 else "",
        "condition_mode": condition_mode if is_stage2 else "none",
        "condition_on_coarse": condition_on_coarse if is_stage2 else False,
        "eval_steps": eval_steps if is_stage2 else 0,
        "detail_boost": detail_boost if is_stage2 else 0.0,
        "dynamic_condition_rollout": dynamic_condition if is_stage2 else False,
        "include_delta_from_initial": include_delta_from_initial if is_stage2 else False,
        "velocity_schedule": velocity_schedule if is_stage2 else "none",
        "manifest": str(manifest_path),
    }


@torch.no_grad()
def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_yaml(args.config)
    processed_root = Path(config["data"]["processed_root"])
    predictions_root = Path(config["outputs"]["predictions_root"])
    test_t1_dir = Path(args.test_t1_dir) if args.test_t1_dir else processed_root / "test" / "t1_slices"
    test_fa_dir = Path(args.test_fa_dir) if args.test_fa_dir else processed_root / "test" / "fa_slices"
    output_dir = resolve_output_dir(args.stage, predictions_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.stage1_ckpt).exists():
        raise FileNotFoundError(f"Missing Stage 1 checkpoint: {args.stage1_ckpt}")
    if args.stage == "stage2" and not Path(args.stage2_ckpt).exists():
        raise FileNotFoundError(f"Missing Stage 2 checkpoint: {args.stage2_ckpt}")

    device = torch.device(args.device)
    stage1, stage1_prediction_mode, stage1_channels, stage1_detail_scale = load_stage1(args.stage1_ckpt, device)
    dataset = make_slice_dataset(test_t1_dir, test_fa_dir, stage1_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    condition_mode, condition_on_coarse, eval_steps, detail_boost, dynamic_condition = resolve_stage2_settings(args)
    stage2 = None
    velocity_schedule = "none"
    if args.stage == "stage2":
        stage2_checkpoint = torch.load(args.stage2_ckpt, map_location="cpu")
        velocity_schedule = str(checkpoint_args(stage2_checkpoint).get("velocity_schedule", "constant"))
        stage2 = load_stage2(
            args.stage2_ckpt,
            device,
            condition_mode=condition_mode,
            stage1_channels=stage1_channels,
        )

    exported_rows: list[dict[str, Any]] = []
    exported = 0
    method = stage_to_method(args.stage)
    for batch in tqdm(loader, desc=f"Exporting {method}", leave=False):
        t1_img = prepare_stage1_input(batch["t1_slice"].to(device), stage1_channels)
        coarse = predict_stage1_fa(
            stage1,
            t1_img,
            clamp=True,
            prediction_mode=stage1_prediction_mode,
            detail_scale=stage1_detail_scale,
        )
        if args.stage == "stage1":
            prediction = coarse
        else:
            assert stage2 is not None
            include_delta = bool(getattr(stage2, "include_delta_from_initial", True))
            condition = build_stage2_condition(
                coarse,
                t1_img,
                condition_mode,
                initial_coarse=coarse,
                include_delta_from_initial=include_delta,
            )
            prediction = euler_refine(
                stage2,
                coarse,
                num_steps=eval_steps,
                condition=condition,
                clamp=True,
                detail_boost=detail_boost,
                dynamic_condition=dynamic_condition,
                condition_mode=condition_mode,
                t1_img=t1_img,
                initial_coarse=coarse,
                include_delta_from_initial=include_delta,
            )

        batch_filenames = batch["fname"]
        for i, filename in enumerate(batch_filenames):
            if args.limit > 0 and exported >= args.limit:
                break
            output_path = output_dir / str(filename)
            save_prediction_png(prediction[i : i + 1], output_path)
            exported_rows.append(
                {
                    "method": method,
                    "stage": args.stage,
                    "fname": str(filename),
                    "output_path": str(output_path),
                }
            )
            exported += 1
        if args.limit > 0 and exported >= args.limit:
            break

    manifest_path = output_dir / "export_manifest.csv"
    _write_manifest(manifest_path, exported_rows)
    include_delta = bool(getattr(stage2, "include_delta_from_initial", False)) if args.stage == "stage2" else False
    summary = build_export_summary(
        method=method,
        args=args,
        output_dir=output_dir,
        exported=exported,
        manifest_path=manifest_path,
        stage1_prediction_mode=stage1_prediction_mode,
        stage1_detail_scale=stage1_detail_scale,
        stage1_channels=stage1_channels,
        condition_mode=condition_mode,
        condition_on_coarse=condition_on_coarse,
        eval_steps=eval_steps,
        detail_boost=detail_boost,
        dynamic_condition=dynamic_condition,
        include_delta_from_initial=include_delta,
        velocity_schedule=velocity_schedule,
    )
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"{method}: exported={exported} output_dir={output_dir}")
    print(f"Saved manifest to: {manifest_path}")
    return summary


if __name__ == "__main__":
    export_predictions(parse_args())
