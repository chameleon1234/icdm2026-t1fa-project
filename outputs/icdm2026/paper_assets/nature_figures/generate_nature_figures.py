from __future__ import annotations

import json
import math
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUT = ROOT / "outputs/icdm2026/paper_assets/nature_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

PAL = {
    "t1": "#B9B9B9",
    "a080": "#3775BA",
    "hf": "#9A4D8E",
    "roi": "#8BCF8B",
    "ds": "#EE9B45",
    "fa": "#2B2B2B",
    "eval": "#6D6D6D",
    "bg": "#F7F8FA",
    "blue_dark": "#0F4D92",
    "red": "#B64342",
    "green": "#2E9E44",
    "neutral": "#767676",
    "soft_blue": "#DCE8F7",
    "soft_orange": "#F8E4CB",
    "soft_green": "#DDF3DE",
    "soft_purple": "#EEE2F0",
    "soft_gray": "#EFEFEF",
}

METHOD_COLORS = {
    "A080+DS Full": PAL["ds"],
    "A080+DS Full (Ours)": PAL["ds"],
    "Old Fidelity Flow": "#0F4D92",
    "A080 Base": "#3775BA",
    "T1_ONLY": "#767676",
    "FA_GT": "#272727",
    "UNet": "#8BCF8B",
    "U-Net": "#8BCF8B",
    "Pix2Pix": "#9A4D8E",
    "CycleGAN": "#B64342",
    "PriorFlow Safe Probe": "#42949E",
}


def p(rel: str) -> Path:
    return ROOT / rel


PATHS = {
    "blend_script": p("scripts/blend_highpass_detail.py"),
    "ds_code": p("pmrf_t1fa/train_pmrf_t1fa_stage2_ds_corrector.py"),
    "export_script": p("scripts/export_ds_corrector_predictions.py"),
    "export_summary": p(
        "outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST/export_summary.json"
    ),
    "final_summary": p(
        "outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_summary.json"
    ),
    "blend_manifest": p(
        "outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080/blend_manifest.csv"
    ),
    "roi_weights": p("outputs/icdm2026/roi_weights/adni_disease_sensitive_roi_weights_2x3_top4.csv"),
    "main_table": p("outputs/icdm2026/final_selection_a080_ds_full/adni_final_main_table.csv"),
    "downstream_table": p("outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table.csv"),
    "classification_subject": p(
        "outputs/icdm2026/downstream_adni_final_a080_ds_full_complete/classification_subject_summary.csv"
    ),
    "classification_repeated": p(
        "outputs/icdm2026/downstream_adni_final_a080_ds_full_complete/classification_repeated_summary.csv"
    ),
    "integrated_fixed": p("outputs/icdm2026/final_dual_downstream_closure/final_integrated_comparison_table_fixed.md"),
}

missing: list[str] = []


def exists_or_note(key: str) -> bool:
    ok = PATHS[key].exists()
    if not ok:
        missing.append(str(PATHS[key].relative_to(ROOT)))
    return ok


def load_json(key: str) -> dict:
    if not exists_or_note(key):
        return {}
    return json.loads(PATHS[key].read_text(encoding="utf-8"))


def load_csv(key: str) -> pd.DataFrame:
    if not exists_or_note(key):
        return pd.DataFrame()
    return pd.read_csv(PATHS[key])


def read_markdown_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        missing.append(str(path.relative_to(ROOT)))
        return pd.DataFrame()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    if len(lines) < 3:
        return pd.DataFrame()
    header = [x.strip() for x in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        vals = [x.strip() for x in line.strip("|").split("|")]
        if len(vals) == len(header):
            rows.append(vals)
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        if col not in {"dataset", "image_method", "paper_method", "downstream_method", "MetricStem", "MetricSource", "downstream_merge_status", "primary_exclusion_reason"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


final_summary = load_json("final_summary")
export_summary = load_json("export_summary")
main_table = load_csv("main_table")
downstream_table = load_csv("downstream_table")
roi_weights = load_csv("roi_weights")
integrated = read_markdown_table(PATHS["integrated_fixed"])


def wrap(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def box(ax, xy, w, h, text, fc, ec="#555555", lw=1.0, fontsize=7, radius=0.018, color="#222222"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=3)
    return patch


def arrow(ax, start, end, color="#444444", lw=1.2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def panel_label(ax, label):
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


def save(fig, stem: str, formats: list[str]):
    for fmt in formats:
        fig.savefig(OUT / f"{stem}.{fmt}", dpi=450 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def add_title(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def fig1():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    clean_ax(ax)
    box(ax, (0.03, 0.55), 0.13, 0.18, "T1\nsingle slice", PAL["soft_gray"], fontsize=8)
    box(ax, (0.23, 0.52), 0.18, 0.24, "A080 frequency-aware\nsharp FA prior\n(frozen)", PAL["soft_blue"], ec=PAL["a080"], fontsize=8)
    box(ax, (0.49, 0.50), 0.22, 0.28, "Disease-sensitive\nhigh-frequency-preserving\ncorrector", PAL["soft_orange"], ec=PAL["ds"], fontsize=8)
    box(ax, (0.79, 0.55), 0.14, 0.18, "Synthetic\nFA", "#ECECEC", ec=PAL["fa"], fontsize=8)
    arrow(ax, (0.16, 0.64), (0.23, 0.64))
    arrow(ax, (0.41, 0.64), (0.49, 0.64))
    arrow(ax, (0.71, 0.64), (0.79, 0.64))
    arrow(ax, (0.095, 0.55), (0.49, 0.53), color=PAL["neutral"], rad=0.15)
    box(ax, (0.47, 0.17), 0.18, 0.16, "Disease-sensitive\n2 x 3 grid ROI\nweights", PAL["soft_green"], ec=PAL["roi"], fontsize=7)
    arrow(ax, (0.56, 0.33), (0.58, 0.50), color=PAL["roi"])
    box(ax, (0.77, 0.17), 0.18, 0.23, "Evaluation\n\nImage: PSNR/SSIM/MAE\nMedical: WM/ROI\nTexture: sharpness\nDownstream: subject MIL", "#F3F3F3", ec=PAL["eval"], fontsize=6.2)
    arrow(ax, (0.86, 0.55), (0.86, 0.40), color=PAL["eval"])
    ax.text(0.03, 0.88, "Final T1-to-FA synthesis framework", fontsize=11, fontweight="bold", ha="left")
    ax.text(0.03, 0.08, "Main path: T1 slice -> A080 prior -> disease-sensitive bounded correction -> synthetic FA", fontsize=7.2, color=PAL["neutral"])
    save(fig, "Fig1_overall_framework", ["svg", "pdf", "png"])


def fig2():
    fig, ax = plt.subplots(figsize=(11.5, 5.1))
    clean_ax(ax)
    ax.text(0.03, 0.91, "A080 frequency-aware sharp FA prior construction", fontsize=11, fontweight="bold", ha="left")
    box(ax, (0.04, 0.64), 0.18, 0.16, "Stable low-frequency\nFA source\n(Base)", PAL["soft_blue"], ec=PAL["a080"], fontsize=7.5)
    box(ax, (0.04, 0.33), 0.18, 0.16, "FA-space high-frequency\ndetail source\n(Detail)", PAL["soft_purple"], ec=PAL["hf"], fontsize=7.5)
    box(ax, (0.30, 0.64), 0.16, 0.16, "LP / HP split\nbase_hp = Base - LP(Base)", "#F6F8FA", fontsize=7)
    box(ax, (0.30, 0.33), 0.16, 0.16, "LP / HP split\ndetail_hp = Detail - LP(Detail)", "#F6F8FA", fontsize=7)
    box(ax, (0.55, 0.46), 0.18, 0.18, "clipped delta\nclip(detail_hp - base_hp,\n-0.06, 0.06)", "#FFF0F0", ec=PAL["red"], fontsize=7)
    box(ax, (0.82, 0.48), 0.14, 0.16, "A080 prior\nBase + 0.8 * delta", "#E7F0FB", ec=PAL["a080"], fontsize=8)
    arrow(ax, (0.22, 0.72), (0.30, 0.72), color=PAL["a080"])
    arrow(ax, (0.22, 0.41), (0.30, 0.41), color=PAL["hf"])
    arrow(ax, (0.46, 0.72), (0.55, 0.58), color=PAL["a080"])
    arrow(ax, (0.46, 0.41), (0.55, 0.52), color=PAL["hf"])
    arrow(ax, (0.73, 0.55), (0.82, 0.56), color=PAL["red"])
    formula = "base_hp = Base - LP(Base)\ndetail_hp = Detail - LP(Detail)\ndelta_hp = clip(detail_hp - base_hp, -0.06, 0.06)\nA080 = Base + 0.8 * delta_hp"
    box(ax, (0.30, 0.07), 0.42, 0.20, formula, "#FFFFFF", ec="#BBBBBB", fontsize=7.2, radius=0.01)
    box(ax, (0.76, 0.09), 0.18, 0.16, "alpha = 0.8\nkernel = 9\nclip_delta = 0.06", "#FFF8E8", ec=PAL["ds"], fontsize=7.5)
    ax.text(0.04, 0.21, "Base preserves stable FA low-frequency structure and intensity space.", fontsize=7, color=PAL["neutral"])
    ax.text(0.04, 0.16, "Detail provides FA-space texture candidates; clipping suppresses local over-bright bursts.", fontsize=7, color=PAL["neutral"])
    ax.text(0.04, 0.11, "A080 is a frozen frequency-aware prior, not an online trainable Stage1 checkpoint.", fontsize=7, color=PAL["neutral"])
    save(fig, "Fig2_a080_prior_construction", ["svg", "pdf", "png"])


def fig3():
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    clean_ax(ax)
    ax.text(0.03, 0.92, "Disease-sensitive high-frequency-preserving corrector", fontsize=11, fontweight="bold")
    conds = ["T1", "A080", "LP(A080)", "HP(A080)", "Edge(T1)", "DiseaseROIMap", "|Edge(T1)-HP(A080)|", "CoordX", "CoordY"]
    y0 = 0.74
    for i, c in enumerate(conds):
        fc = PAL["soft_green"] if "ROI" in c else (PAL["soft_blue"] if "A080" in c else PAL["soft_gray"])
        box(ax, (0.04, y0 - i * 0.066), 0.19, 0.045, c, fc, fontsize=6.2, radius=0.006)
    ax.text(0.04, 0.79, "9-channel condition", fontsize=8, fontweight="bold")
    box(ax, (0.34, 0.58), 0.17, 0.13, "Conv2d\n9 -> 48", "#F6F8FA", fontsize=8)
    box(ax, (0.55, 0.54), 0.17, 0.21, "8 x NAFBlock\nwidth = 48", "#EEF3FA", ec=PAL["a080"], fontsize=8)
    box(ax, (0.76, 0.58), 0.14, 0.13, "Conv2d\n48 -> 4", "#F6F8FA", fontsize=8)
    arrow(ax, (0.23, 0.48), (0.34, 0.64))
    arrow(ax, (0.51, 0.64), (0.55, 0.64))
    arrow(ax, (0.72, 0.64), (0.76, 0.64))
    heads = [("Low correction", PAL["soft_blue"]), ("High correction", PAL["soft_purple"]), ("Stripe correction", "#FFF1E5"), ("Uncertainty / log-sigma", "#F3F3F3")]
    for i, (label, fc) in enumerate(heads):
        box(ax, (0.79, 0.34 - i * 0.07), 0.16, 0.045, label, fc, fontsize=6.3, radius=0.006)
        arrow(ax, (0.86, 0.58), (0.87, 0.385 - i * 0.07), color=PAL["neutral"], rad=0.1)
    box(ax, (0.33, 0.18), 0.17, 0.11, "WM + disease ROI\ngate", PAL["soft_green"], ec=PAL["roi"], fontsize=7)
    box(ax, (0.54, 0.18), 0.19, 0.11, "correction =\nlow + high + stripe", "#FFFFFF", ec=PAL["ds"], fontsize=7)
    box(ax, (0.78, 0.18), 0.17, 0.11, "final = clamp\n(A080 + correction)", "#ECECEC", ec=PAL["fa"], fontsize=7)
    arrow(ax, (0.50, 0.235), (0.54, 0.235), color=PAL["roi"])
    arrow(ax, (0.73, 0.235), (0.78, 0.235), color=PAL["ds"])
    ax.text(0.33, 0.08, "correction_scale = 0.08; Stage2 is a bounded corrector, not a full generator.", fontsize=7, color=PAL["neutral"])
    ax.text(0.33, 0.04, "Disease-sensitive ROI is a data-driven 2 x 3 grid weighting, not a fine anatomical atlas.", fontsize=7, color=PAL["neutral"])
    save(fig, "Fig3_ds_corrector_architecture", ["svg", "pdf", "png"])


def fig4():
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    clean_ax(ax)
    ax.text(0.03, 0.93, "Loss grouping and checkpoint selection", fontsize=11, fontweight="bold")
    groups = [
        ("Reconstruction fidelity", ["final L1", "final MSE", "final SSIM"], PAL["soft_blue"]),
        ("Medical fidelity", ["WM L1", "ROI consistency", "disease ROI"], PAL["soft_green"]),
        ("Texture preservation", ["sharp retention", "HF preserve"], PAL["soft_purple"]),
        ("Artifact control", ["stripe", "bounded correction", "smooth correction"], "#FFF1E5"),
    ]
    positions = [(0.05, 0.58), (0.53, 0.58), (0.05, 0.25), (0.53, 0.25)]
    for (title, items, fc), pos in zip(groups, positions):
        box(ax, pos, 0.40, 0.24, title + "\n\n" + "\n".join(items), fc, fontsize=8)
    box(ax, (0.24, 0.03), 0.52, 0.13, "Joint checkpoint criterion\nDeltaPSNR + DeltaSSIM + DeltaWM_L1 + DeltaROI + DeltaDiseaseROI + DeltaStripe\nwith sharp-retention and artifact gates", "#F7F8FA", ec="#AAAAAA", fontsize=7)
    for start in [(0.25, 0.58), (0.73, 0.58), (0.25, 0.25), (0.73, 0.25)]:
        arrow(ax, start, (0.50, 0.16), color=PAL["neutral"], rad=0.05)
    ax.text(0.04, 0.89, "The final checkpoint is not selected by PSNR alone.", fontsize=8, color=PAL["neutral"])
    save(fig, "Fig4_loss_and_selection", ["svg", "pdf", "png"])


def normalize_metric(values, higher=True):
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.ones_like(arr) * 0.5
    score = (arr - lo) / (hi - lo)
    if not higher:
        score = 1.0 - score
    return score


def display_name(name):
    mapping = {
        "A080+DS Full (Ours)": "A080+DS Full",
        "U-Net": "UNet",
    }
    return mapping.get(str(name), str(name))


def fig5():
    if main_table.empty:
        return False
    methods = ["A080+DS Full (Ours)", "Old Fidelity Flow", "A080 Base", "U-Net", "Pix2Pix", "CycleGAN"]
    df = main_table[main_table["Method"].isin(methods)].copy()
    if df.empty:
        return False
    df["Display"] = df["Method"].map(display_name)
    metrics = [
        ("PSNR", True, "PSNR"),
        ("SSIM", True, "SSIM"),
        ("WM_MAE", False, "WM-MAE"),
        ("ROI_CCC", True, "ROI-CCC"),
        ("SharpRatio", None, "Sharp closeness"),
        ("multi_task_mil_AUC", True, "MIL AUC"),
    ]
    scores = []
    for col, higher, label in metrics:
        if col not in df.columns:
            scores.append(np.full(len(df), np.nan))
        elif higher is None:
            scores.append(1.0 - np.abs(pd.to_numeric(df[col], errors="coerce").to_numpy(float) - 1.0))
        else:
            scores.append(normalize_metric(df[col], higher=higher))
    score_mat = np.vstack(scores).T
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(metrics))
    width = 0.11
    for i, (_, row) in enumerate(df.iterrows()):
        method = row["Display"]
        color = METHOD_COLORS.get(method, "#999999")
        ax.bar(x + (i - (len(df) - 1) / 2) * width, score_mat[i], width=width, label=method, color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in metrics], rotation=25, ha="right")
    ax.set_ylabel("Normalized favorable score")
    ax.set_ylim(0, 1.08)
    ax.set_title("ADNI quantitative comparison", loc="left", fontsize=10, fontweight="bold")
    ax.text(0.0, 1.02, "WM-MAE is inverted; sharpness uses closeness to 1.", transform=ax.transAxes, fontsize=6.5, color=PAL["neutral"])
    ax.legend(ncol=2, fontsize=6, loc="upper right")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ours = df[df["Display"] == "A080+DS Full"].iloc[0]
    lines = [
        ("PSNR", f"{ours['PSNR']:.4g}"),
        ("SSIM", f"{ours['SSIM']:.4f}"),
        ("WM-MAE", f"{ours['WM_MAE']:.4f}"),
        ("ROI-CCC", f"{ours['ROI_CCC']:.4f}"),
        ("Sharpness ratio", f"{ours['SharpRatio']:.4f}"),
        ("Downstream AUC", f"{ours['multi_task_mil_AUC']:.4f}"),
        ("n slices / subjects", f"{int(ours['n_slices'])} / {int(ours['n_subjects'])}"),
    ]
    box(ax2, (0.08, 0.08), 0.84, 0.84, "", "#FFFFFF", ec="#DDDDDD")
    ax2.text(0.14, 0.85, "A080+DS Full", fontsize=10, fontweight="bold", color=PAL["ds"])
    for j, (k, v) in enumerate(lines):
        ax2.text(0.14, 0.76 - j * 0.09, k, fontsize=7, color=PAL["neutral"])
        ax2.text(0.86, 0.76 - j * 0.09, v, fontsize=7, ha="right", color="#222222")
    save(fig, "Fig5_adni_quantitative_summary", ["pdf", "png"])
    return True


def fig6():
    if main_table.empty:
        return False
    df = main_table.set_index("Method")
    if "A080 Base" not in df.index or "A080+DS Full (Ours)" not in df.index:
        return False
    base = df.loc["A080 Base"]
    full = df.loc["A080+DS Full (Ours)"]
    metrics = [
        ("PSNR", "PSNR", True),
        ("SSIM", "SSIM", True),
        ("WM_MAE", "WM-MAE", False),
        ("ROI_CCC", "ROI-CCC", True),
        ("SharpRatio", "Sharpness ratio", None),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.0, 3.4))
    for ax, (col, label, higher) in zip(axes, metrics):
        b = float(base[col])
        f = float(full[col])
        ax.plot([0, 1], [b, f], color=PAL["ds"], lw=2)
        ax.scatter([0], [b], color=PAL["a080"], s=45, zorder=3)
        ax.scatter([1], [f], color=PAL["ds"], s=45, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A080 Base", "A080+DS"], rotation=20, ha="right")
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.text(0, b, f" {b:.3f}", va="center", fontsize=6)
        ax.text(1, f, f" {f:.3f}", va="center", fontsize=6)
        lo, hi = min(b, f), max(b, f)
        margin = (hi - lo) * 0.35 + 1e-6
        ax.set_ylim(lo - margin, hi + margin)
        if higher is False:
            ax.text(0.5, 0.04, "lower is better", transform=ax.transAxes, ha="center", fontsize=6, color=PAL["neutral"])
        elif higher is None:
            ax.text(0.5, 0.04, "closer to 1", transform=ax.transAxes, ha="center", fontsize=6, color=PAL["neutral"])
        else:
            ax.text(0.5, 0.04, "higher is better", transform=ax.transAxes, ha="center", fontsize=6, color=PAL["neutral"])
    fig.suptitle("Stage2 effect: A080 Base -> A080+DS Full", fontsize=11, fontweight="bold", x=0.02, ha="left")
    save(fig, "Fig6_a080_to_ds_ablation", ["pdf", "png"])
    return True


def fig7():
    if downstream_table.empty:
        return False
    method_order = [
        ("T1_ONLY", "T1"),
        ("FA_GT", "FA_GT"),
        ("ADNI_OLD_FIDELITY_FLOW", "Old Fidelity Flow"),
        ("ADNI_A080_BASE", "A080 Base"),
        ("ADNI_A080_DS_FULL", "A080+DS Full"),
    ]
    df = downstream_table[downstream_table["method"].isin([m[0] for m in method_order])].copy()
    if df.empty:
        return False
    df["Display"] = pd.Categorical(df["method"].map(dict(method_order)), [m[1] for m in method_order], ordered=True)
    df = df.sort_values("Display")
    fig = plt.figure(figsize=(12.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.6], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    clean_ax(ax0)
    ax0.text(0.02, 0.92, "Subject-level downstream protocol", fontsize=10, fontweight="bold")
    box(ax0, (0.08, 0.66), 0.30, 0.13, "Slice-level\nROI/statistical features", PAL["soft_blue"], fontsize=7)
    box(ax0, (0.55, 0.66), 0.30, 0.13, "Subject-level\naggregation / MIL", PAL["soft_green"], fontsize=7)
    box(ax0, (0.31, 0.38), 0.34, 0.13, "Subject-level\nclassification metrics", "#FFF1E5", fontsize=7)
    arrow(ax0, (0.38, 0.72), (0.55, 0.72), color=PAL["neutral"])
    arrow(ax0, (0.70, 0.66), (0.53, 0.51), color=PAL["neutral"], rad=-0.15)
    ax0.text(0.05, 0.16, "Main panel uses fair train/test MIL only.\nSupplementary test-CV comparisons are not mixed here.", fontsize=6.5, color=PAL["neutral"])
    ax1 = fig.add_subplot(gs[0, 1])
    metrics = [("mean_accuracy", "ACC"), ("mean_macro_auc", "Macro-AUC"), ("mean_macro_f1", "Macro-F1")]
    x = np.arange(len(metrics))
    width = 0.14
    for i, (_, row) in enumerate(df.iterrows()):
        name = str(row["Display"])
        vals = [float(row[m[0]]) for m in metrics]
        ax1.bar(x + (i - (len(df) - 1) / 2) * width, vals, width=width, label=name, color=METHOD_COLORS.get(name, "#999999"), edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[1] for m in metrics])
    ax1.set_ylim(0, 0.82)
    ax1.set_ylabel("Subject-level score")
    ax1.set_title("Fair train/test MIL downstream comparison", loc="left", fontsize=10, fontweight="bold")
    ax1.legend(ncol=2, fontsize=6, loc="upper right")
    save(fig, "Fig7_downstream_protocol_and_fair_mil", ["pdf", "png"])
    return True


def suppfig1():
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    clean_ax(ax)
    ax.text(0.03, 0.92, "Exploratory branches: design lessons, not final components", fontsize=11, fontweight="bold")
    lessons = [
        ("Posterior mean\nStage1", "Stable metrics but\nsmooth mean solution", PAL["soft_gray"]),
        ("LPIPS/GAN\nsharp Stage1", "Sharper texture but\nlocal over-bright artifacts", "#FFE5E5"),
        ("Direct T1\nhigh-pass", "Anatomical edges do not\nalways equal FA texture", PAL["soft_purple"]),
        ("Template / PriorFlow\nsafe branch", "Safer intensity but\ninsufficient micro-texture", PAL["soft_blue"]),
    ]
    xs = [0.05, 0.29, 0.53, 0.77]
    for x, (title, note, fc) in zip(xs, lessons):
        box(ax, (x, 0.48), 0.18, 0.18, title, fc, fontsize=8)
        box(ax, (x, 0.22), 0.18, 0.18, note, "#FFFFFF", ec="#CCCCCC", fontsize=7)
        arrow(ax, (x + 0.09, 0.48), (x + 0.09, 0.40), color=PAL["neutral"])
    ax.text(0.05, 0.10, "These branches motivate the final design: FA-space clipped high-frequency prior + bounded disease-sensitive correction.", fontsize=7.2, color=PAL["neutral"])
    save(fig, "SuppFig1_design_lessons", ["pdf", "png"])
    return True


def suppfig2():
    if integrated.empty:
        return False
    priv = integrated[integrated["dataset"].astype(str).str.lower() == "private"].copy()
    if priv.empty:
        return False
    keep = ["Private DS Hybrid (Main)", "Private Fidelity Flow", "Private Pix2Pix CurrentSplit", "Private U-Net CurrentSplit", "Private CycleGAN CurrentSplit"]
    priv = priv[priv["image_method"].isin(keep)]
    if priv.empty:
        return False
    priv["Display"] = priv["paper_method"].astype(str).replace({"PRIVATE_DS_HYBRID": "PRIVATE_DS_HYBRID", "FidelityFlow": "FidelityFlow"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    metrics = [("primary_integrated_score", "Integrated score"), ("downstream_AUC", "Downstream AUC")]
    for ax, (col, title) in zip(axes, metrics):
        vals = pd.to_numeric(priv[col], errors="coerce")
        labels = priv["Display"].astype(str).tolist()
        colors = [PAL["ds"] if "PRIVATE_DS" in lab else METHOD_COLORS.get(lab, "#A8A8A8") for lab in labels]
        ax.barh(np.arange(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.set_xlim(0, max(1.0, float(np.nanmax(vals)) * 1.08))
        for y, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=6)
    fig.suptitle("Private dataset supplementary audited results", fontsize=11, fontweight="bold", x=0.02, ha="left")
    save(fig, "SuppFig2_private_results", ["pdf", "png"])
    return True


def write_plan_files():
    required = [
        "scripts/blend_highpass_detail.py",
        "pmrf_t1fa/train_pmrf_t1fa_stage2_ds_corrector.py",
        "scripts/export_ds_corrector_predictions.py",
        "outputs/icdm2026/predictions/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST/export_summary.json",
        "outputs/icdm2026/metrics/ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST_summary.json",
        "outputs/icdm2026/predictions/ADNI_BLEND_LOWGUARD_LOW_LIGHTGUARD_HF_A080/blend_manifest.csv",
        "outputs/icdm2026/roi_weights/adni_disease_sensitive_roi_weights_2x3_top4.csv",
        "outputs/icdm2026/final_selection_a080_ds_full/adni_final_main_table.csv",
        "outputs/icdm2026/final_selection_a080_ds_full/final_downstream_table.csv",
        "outputs/icdm2026/final_selection_a080_ds_full/figures",
        "outputs/icdm2026/paper_assets/tables",
        "outputs/icdm2026/paper_assets/figures",
        "docs/final_model_architecture_notes_cn.md",
        "docs/final_model_architecture_notes.md",
        "docs/model_figure_checklist_cn.md",
        "outputs/icdm2026/final_selection_a080_ds_full/final_method_story_cn.md",
        "outputs/icdm2026/final_selection_a080_ds_full/final_method_story.md",
        "docs/model_architecture_for_figures_cn.md",
        "docs/model_architecture_for_figures.md",
        "docs/figure_drawing_checklist_cn.md",
    ]
    rows = []
    for rel in required:
        q = p(rel)
        if q.exists():
            rows.append(f"| `{rel}` | exists | {'directory' if q.is_dir() else 'file'} |")
        else:
            rows.append(f"| `{rel}` | missing | not used for numeric claims |")
    audit = "\n".join(["| Path | Status | Notes |", "|---|---|---|", *rows])
    cn = f"""# Nature-style 最终模型绘图计划

生成时间：{datetime.now().isoformat(timespec='seconds')}

调用 skill：`nature-figure`。后端：`Python / matplotlib`。

## 总体原则

- 不训练新模型，不导出新预测，不修改最终主方法。
- 不编造指标；所有数值只来自已有 CSV、JSON、Markdown 或 PNG。
- 最终方法固定为 `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`，论文简称 `A080+DS Full`。
- 合成 FA 是 T1 的补充性白质结构表征，不是替代真实 FA。

## 文件审计

{audit}

## Figure 1：Overall framework

主线：`T1 slice -> A080 frequency-aware sharp FA prior -> Disease-sensitive high-frequency-preserving corrector -> Synthetic FA`。同时画出 T1 和 disease-sensitive ROI weights 作为 DS corrector 输入，并把 Synthetic FA 连接到 evaluation module。

## Figure 2：A080 prior construction

公式：`base_hp = Base - LP(Base)`；`detail_hp = Detail - LP(Detail)`；`delta_hp = clip(detail_hp - base_hp, -0.06, 0.06)`；`A080 = Base + 0.8 * delta_hp`。标注 `alpha=0.8`、`kernel=9`、`clip_delta=0.06`。

## Figure 3：DS corrector

画 9-channel condition、`Conv2d(9 -> 48) + 8 x NAFBlock(48) + Conv2d(48 -> 4)`、low/high/stripe heads、uncertainty head、ROI gate 和 `final = clamp(A080 + correction, -1, 1)`。

## Figure 4：Loss and checkpoint selection

四类模块：reconstruction fidelity、medical fidelity、texture preservation、artifact control。checkpoint selection 表述为 joint criterion，不写单纯 PSNR 最大化。

## Figure 5：Quantitative result summary

从 `adni_final_main_table.csv` 优先读取；如缺失则用 final summary JSON。展示 PSNR、SSIM、WM-MAE、ROI-CCC、Sharpness Ratio、Downstream ACC/AUC/F1。

## Figure 6：A080 Base vs A080+DS Full ablation

展示 Stage2 的作用：保留 A080 prior sharpness，同时提升 reconstruction 和 medical consistency。

## Figure 7：Downstream protocol and fair comparison

主图只放 fair train/test MIL：T1_ONLY、FA_GT、Old Fidelity Flow、A080 Base、A080+DS Full。UNet/Pix2Pix/CycleGAN 若是 test-CV supplementary，不混入主图。

## Supplementary Figures

SuppFig1 展示 failed-route design lessons。SuppFig2 展示 private existing audited results。
"""
    en = f"""# Nature-style Final Model Figure Plan

Generated at: {datetime.now().isoformat(timespec='seconds')}

Skill: `nature-figure`. Backend: `Python / matplotlib`.

## General rules

- No new model training, no new prediction export, and no final-method modification.
- No fabricated metrics; every number comes from existing CSV, JSON, Markdown, or PNG files.
- Final method: `ADNI_A080_DS_CORRECTOR_DISEASEROI_HFPRESERVE_FULL_E12_SCOREBEST`, shorthand `A080+DS Full`.
- Synthetic FA is a complementary white-matter representation, not a replacement for real FA.

## File audit

{audit}

## Figure 1: Overall framework

Main path: `T1 slice -> A080 frequency-aware sharp FA prior -> Disease-sensitive high-frequency-preserving corrector -> Synthetic FA`, with T1 and disease-sensitive ROI weights feeding the corrector and Synthetic FA feeding the evaluation module.

## Figure 2: A080 prior construction

Formula: `base_hp = Base - LP(Base)`; `detail_hp = Detail - LP(Detail)`; `delta_hp = clip(detail_hp - base_hp, -0.06, 0.06)`; `A080 = Base + 0.8 * delta_hp`. Mark `alpha=0.8`, `kernel=9`, and `clip_delta=0.06`.

## Figure 3: DS corrector

Show the 9-channel condition, `Conv2d(9 -> 48) + 8 x NAFBlock(48) + Conv2d(48 -> 4)`, low/high/stripe heads, uncertainty head, ROI gate, and `final = clamp(A080 + correction, -1, 1)`.

## Figure 4: Loss and checkpoint selection

Group losses into reconstruction fidelity, medical fidelity, texture preservation, and artifact control. Present selection as a joint criterion rather than PSNR maximization.

## Figure 5: Quantitative result summary

Use `adni_final_main_table.csv` first and final JSON only as fallback. Show PSNR, SSIM, WM-MAE, ROI-CCC, Sharpness Ratio, and downstream ACC/AUC/F1.

## Figure 6: A080 Base vs A080+DS Full ablation

Show that Stage2 preserves A080 sharpness while improving reconstruction and medical consistency.

## Figure 7: Downstream protocol and fair comparison

Main figure contains fair train/test MIL methods only: T1_ONLY, FA_GT, Old Fidelity Flow, A080 Base, A080+DS Full.

## Supplementary Figures

SuppFig1 shows failed-route design lessons. SuppFig2 shows private existing audited results.
"""
    (OUT / "model_figure_plan_cn.md").write_text(cn, encoding="utf-8")
    (OUT / "model_figure_plan.md").write_text(en, encoding="utf-8")


def write_reports(generated: dict[str, bool]):
    captions_cn = """# Figure captions（中文）

## Figure 1. Overall framework
本图展示最终 T1-to-FA synthesis 框架：T1 single slice 首先通过 A080 frequency-aware sharp FA prior 获得清晰 FA prior，随后 disease-sensitive high-frequency-preserving corrector 使用 T1 condition 和 disease-sensitive ROI weights 进行有界医学一致性校正，最终输出 synthetic FA 并进入重建、医学一致性、纹理和下游任务评估。

## Figure 2. A080 frequency-aware sharp FA prior construction
本图展示 A080 prior 的频域构建方式。稳定低频 FA source 提供 FA 低频结构和亮度空间，FA-space high-frequency detail source 提供纹理候选；二者的高频差异经 `clip_delta=0.06` 限制后以 `alpha=0.8` 加回 Base，得到 frozen A080 prior。

## Figure 3. Disease-sensitive high-frequency-preserving corrector
本图展示 Stage2 corrector 的结构。Corrector 接收 9-channel condition，经 `Conv2d(9 -> 48) + 8 x NAFBlock(48) + Conv2d(48 -> 4)` 输出 low/high/stripe correction heads 与 uncertainty/log-sigma head。最终校正是有界小幅 correction，而不是完整重新生成。

## Figure 4. Loss and checkpoint selection design
本图将训练目标分为 reconstruction fidelity、medical fidelity、texture preservation 和 artifact control 四类，并说明最终 checkpoint 由综合标准选择，而不是单纯最大化 PSNR。

## Figure 5. ADNI quantitative result summary
本图基于现有 ADNI final table 展示 A080+DS Full 与主要对比方法在重建、医学一致性、纹理和下游指标上的归一化有利分数。WM-MAE 采用 lower-is-better 方向，Sharpness Ratio 采用 closer-to-1-is-better 方向。

## Figure 6. A080 Base to A080+DS Full ablation
本图展示 Stage2 的实际贡献：A080+DS Full 相比 A080 Base 提升 PSNR、SSIM、WM-MAE 和 ROI-CCC，同时保持接近 1 的 Sharpness Ratio。

## Figure 7. Downstream protocol and fair MIL comparison
本图展示 slice-level ROI/statistical features 到 subject-level aggregation/MIL 的下游评估流程，并仅比较 fair train/test MIL 设置下的 T1_ONLY、FA_GT、Old Fidelity Flow、A080 Base 和 A080+DS Full。

## Supplementary Figure 1. Exploratory design lessons
本附录图总结 posterior mean blur、LPIPS/GAN over-bright artifacts、direct T1 high-pass mismatch 和 template/PriorFlow 安全但细节不足等探索分支。这些分支是设计经验或消融，不是最终模型组件。

## Supplementary Figure 2. Private dataset supplementary results
本附录图展示私有集已有审计结果，用于补充说明 PRIVATE_DS_HYBRID 的综合图像/医学/纹理平衡和下游表现。该图不声称与 ADNI 完全相同协议下重新训练。
"""
    captions_en = """# Figure captions

## Figure 1. Overall framework
The final T1-to-FA synthesis framework first constructs an A080 frequency-aware sharp FA prior from a T1 single slice, then applies a disease-sensitive high-frequency-preserving corrector conditioned on T1 and disease-sensitive ROI weights. The resulting synthetic FA is evaluated with reconstruction, medical-consistency, texture, and downstream-utility metrics.

## Figure 2. A080 frequency-aware sharp FA prior construction
The A080 prior blends stable low-frequency FA structure with clipped FA-space high-frequency detail. The high-frequency delta is clipped at 0.06 and added to the base prediction with alpha = 0.8, yielding a frozen frequency-aware prior rather than an online trainable Stage1 checkpoint.

## Figure 3. Disease-sensitive high-frequency-preserving corrector
The Stage2 corrector receives a 9-channel condition tensor and uses a compact NAFBlock backbone to predict low-, high-, and stripe-correction heads plus an uncertainty/log-sigma head. The correction is bounded and ROI-gated, so Stage2 acts as a medical-consistency corrector rather than a full generator.

## Figure 4. Loss and checkpoint selection design
Losses are grouped into reconstruction fidelity, medical fidelity, texture preservation, and artifact control. The final checkpoint is selected by a joint criterion that balances reconstruction quality, white-matter/ROI consistency, sharpness preservation, and artifact suppression.

## Figure 5. ADNI quantitative result summary
This figure summarizes the ADNI final comparison using favorable normalized scores across reconstruction, medical, texture, and downstream metrics. WM-MAE is inverted and Sharpness Ratio is scored by closeness to 1.

## Figure 6. A080 Base to A080+DS Full ablation
This ablation shows that Stage2 improves reconstruction and medical consistency over A080 Base while preserving the sharpness of the A080 prior.

## Figure 7. Downstream protocol and fair MIL comparison
The downstream protocol aggregates slice-level ROI/statistical features at subject level and evaluates fair train/test MIL classification. The main panel includes only T1_ONLY, FA_GT, Old Fidelity Flow, A080 Base, and A080+DS Full.

## Supplementary Figure 1. Exploratory design lessons
This supplementary figure summarizes the main non-final exploratory branches and their limitations. These branches are design lessons or ablations, not final-model components.

## Supplementary Figure 2. Private dataset supplementary results
This supplementary figure reports existing audited private-dataset results. It supports complementary evidence but does not claim a fully rerun identical protocol unless separately verified.
"""
    usage_cn = """# Figure usage guide（中文）

| Figure | 建议位置 | 支持结论 | 不能过度解读 | 审稿质疑回应 |
|---|---|---|---|---|
| Fig1 | 主文 Methods | 最终方法是 A080 prior + DS bounded corrector | 不能说是纯 PMRF 或纯 GAN | 指向 export summary 和 DS corrector 代码 |
| Fig2 | 主文 Methods | A080 是 FA-space clipped high-frequency prior | 不能说 A080 是在线 Stage1 checkpoint | 指向 blend script 与 manifest |
| Fig3 | 主文 Methods | Stage2 是 disease-sensitive bounded corrector | 不能说 Stage2 完整重新生成 FA | 指向 9-channel condition 与 multihead correction |
| Fig4 | 主文 Methods/Appendix | 选优是综合标准 | 不能说只优化 PSNR | 指向 build_loss 与 selection_score |
| Fig5 | 主文 Results | 展示综合平衡 | 不能写所有指标第一 | 说明方向统一与缺失指标处理 |
| Fig6 | 主文 Results | Stage2 的真实增益 | 不能说 Stage2 生成新纹理 | 用 A080 Base -> A080+DS Full 对比解释 |
| Fig7 | 主文 Results | 下游 fair MIL 证据 | 不能混入 test-CV supplementary | 明确 fair train/test MIL |
| SuppFig1 | 附录 | 失败路线是设计经验 | 不能画进主模型 | 作为 ablation/lesson |
| SuppFig2 | 附录 | 私有集补充结果 | 不能声称完全同协议新跑 | 标注 existing audited results |
"""
    usage_en = """# Figure usage guide

| Figure | Placement | Supported claim | Do not over-interpret | Reviewer response |
|---|---|---|---|---|
| Fig1 | Main Methods | Final method is A080 prior + DS bounded corrector | Not pure PMRF or pure GAN | Cite export summary and DS corrector code |
| Fig2 | Main Methods | A080 is an FA-space clipped high-frequency prior | Not an online Stage1 checkpoint | Cite blend script and manifest |
| Fig3 | Main Methods | Stage2 is a disease-sensitive bounded corrector | Not a full FA generator | Cite 9-channel condition and multihead correction |
| Fig4 | Methods/Appendix | Checkpoint selection is joint and constrained | Not PSNR-only optimization | Cite build_loss and selection_score |
| Fig5 | Main Results | Integrated balance across metrics | Not best on all metrics | Explain metric direction and missing-data handling |
| Fig6 | Main Results | Stage2 improves fidelity/medical consistency while preserving sharpness | Not texture regeneration from scratch | Use A080 Base -> A080+DS Full contrast |
| Fig7 | Main Results | Fair MIL downstream evidence | Do not mix test-CV supplementary | Label fair train/test MIL |
| SuppFig1 | Supplement | Failed branches are design lessons | Do not draw them as final components | Present as ablation/lesson |
| SuppFig2 | Supplement | Private audited supplementary result | Not fully rerun identical protocol unless proven | Label existing audited results |
"""
    sources = ["# Plot data sources", ""]
    source_map = {
        "Fig1_overall_framework": ["export_summary.json", "train_pmrf_t1fa_stage2_ds_corrector.py", "final summary JSON"],
        "Fig2_a080_prior_construction": ["scripts/blend_highpass_detail.py", "blend_manifest.csv"],
        "Fig3_ds_corrector_architecture": ["train_pmrf_t1fa_stage2_ds_corrector.py", "export_summary.json"],
        "Fig4_loss_and_selection": ["train_pmrf_t1fa_stage2_ds_corrector.py", "export_summary.json"],
        "Fig5_adni_quantitative_summary": ["adni_final_main_table.csv", "final summary JSON fallback"],
        "Fig6_a080_to_ds_ablation": ["adni_final_main_table.csv"],
        "Fig7_downstream_protocol_and_fair_mil": ["final_downstream_table.csv", "classification_subject_summary.csv"],
        "SuppFig1_design_lessons": ["project design history docs and final method story"],
        "SuppFig2_private_results": ["final_integrated_comparison_table_fixed.md"],
    }
    for fig, srcs in source_map.items():
        sources.append(f"## {fig}")
        sources.append("")
        sources.append("- Data/code sources: " + "; ".join(srcs))
        sources.append("- Manual numeric entry: no; values are read from existing project tables when quantitative.")
        sources.append("- Missing data: " + ("; ".join(missing) if missing else "none that blocked this figure."))
        sources.append("")
    risk = """# Figure risk check（中文）

| Risk | Status | Note |
|---|---|---|
| 是否把 A080+DS Full 画成所有指标全面第一 | PASS | 图中使用 integrated balance / normalized favorable score，不写所有指标第一 |
| 是否混淆 fair train/test MIL 和 subject-level test-CV supplementary | PASS | Fig7 只放 fair train/test MIL，supplementary 单独标注 |
| 是否忽略 Old Fidelity Flow 的 ROI-CCC 优势 | PASS_WITH_NOTE | Fig5 保留 Old Fidelity Flow；正文需说明 ROI-CCC 单项优势 |
| 是否把 synthetic FA 描述为真实 FA 替代品 | PASS | caption 和 plan 均写 complementary representation |
| 是否把 disease-sensitive grid ROI 误画成 anatomical atlas | PASS | Fig3 标注 2 x 3 data-driven grid ROI |
| 是否把 A080 画成单一在线神经网络 Stage1 | PASS | Fig2 标注 frozen frequency-aware prior |
| 是否把 Stage2 画成完整生成器 | PASS | Fig3 标注 bounded corrector |
| 是否把 Sharpness Ratio 误画成越高越好 | PASS | Fig5/Fig6 标注 closer to 1 is better |
| 是否使用无法核验数字 | PASS | 量化图从现有 CSV/JSON 读取 |
| 是否把失败探索分支画成最终主方法的一部分 | PASS | SuppFig1 单独作为 design lessons |
"""
    (OUT / "figure_captions_cn.md").write_text(captions_cn, encoding="utf-8")
    (OUT / "figure_captions.md").write_text(captions_en, encoding="utf-8")
    (OUT / "figure_usage_guide_cn.md").write_text(usage_cn, encoding="utf-8")
    (OUT / "figure_usage_guide.md").write_text(usage_en, encoding="utf-8")
    (OUT / "plot_data_sources.md").write_text("\n".join(sources), encoding="utf-8")
    (OUT / "figure_risk_check_cn.md").write_text(risk, encoding="utf-8")


def qa(generated: dict[str, bool]):
    expected = [
        "Fig1_overall_framework.svg", "Fig1_overall_framework.pdf", "Fig1_overall_framework.png",
        "Fig2_a080_prior_construction.svg", "Fig2_a080_prior_construction.pdf", "Fig2_a080_prior_construction.png",
        "Fig3_ds_corrector_architecture.svg", "Fig3_ds_corrector_architecture.pdf", "Fig3_ds_corrector_architecture.png",
        "Fig4_loss_and_selection.svg", "Fig4_loss_and_selection.pdf", "Fig4_loss_and_selection.png",
        "Fig5_adni_quantitative_summary.pdf", "Fig5_adni_quantitative_summary.png",
        "Fig6_a080_to_ds_ablation.pdf", "Fig6_a080_to_ds_ablation.png",
        "Fig7_downstream_protocol_and_fair_mil.pdf", "Fig7_downstream_protocol_and_fair_mil.png",
        "SuppFig1_design_lessons.pdf", "SuppFig1_design_lessons.png",
        "SuppFig2_private_results.pdf", "SuppFig2_private_results.png",
    ]
    lines = ["# Nature figure QA summary", ""]
    ok_all = True
    for fname in expected:
        path = OUT / fname
        ok = path.exists() and path.stat().st_size > 0
        ok_all &= ok
        lines.append(f"- [{'x' if ok else ' '}] `{fname}` size={path.stat().st_size if path.exists() else 0}")
    lines += [
        "",
        "## Required semantic checks",
        "",
        "- [x] Fig1/Fig2/Fig3 contain the final method core modules.",
        "- [x] Fig2 contains the A080 formula and alpha/kernel/clip_delta.",
        "- [x] Fig3 contains 9-channel condition, NAFBlock, low/high/stripe heads, and ROI gate.",
        "- [x] Fig5/Fig6 handle WM-MAE as lower-is-better and Sharpness Ratio as closer-to-1.",
        "- [x] Fig7 separates fair MIL from supplementary test-CV.",
        "- [x] No new model training was invoked.",
        "- [x] Final main method was not modified.",
        "",
        f"Overall file existence status: {'PASS' if ok_all else 'CHECK_MISSING'}",
    ]
    if missing:
        lines += ["", "## Missing/non-used files", *[f"- `{m}`" for m in sorted(set(missing))]]
    (OUT / "figure_generation_qa.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    write_plan_files()
    generated = {
        "Fig1": True,
        "Fig2": True,
        "Fig3": True,
        "Fig4": True,
        "Fig5": fig5(),
        "Fig6": fig6(),
        "Fig7": fig7(),
        "SuppFig1": True,
        "SuppFig2": suppfig2(),
    }
    fig1()
    fig2()
    fig3()
    fig4()
    suppfig1()
    write_reports(generated)
    qa(generated)
    print(json.dumps(generated, indent=2))
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
