#!/usr/bin/env python
"""生成 §67 五张图（只读 outputs/tables 的 CSV，不碰模型与缓存）。

配色取自经过验证的默认调色板（明暗两种模式均通过 CVD / 对比度检查，
本次为浅色 PNG）：系列 1 蓝 #2a78d6、系列 2 橙 #eb6834、负向极 #e34948；
网格 #e1e0d9、基线 #c3c2b7、弱化轴字 #898781、正文 #0b0b0b / #52514e。
图内文字用英文（方案 §67 的图名为英文）。输出 outputs/figures/：
PNG（300dpi）+ 合并 PDF。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_config, load_paths

# 调色板（dataviz 默认参考实例，浅色模式）
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
GRAY, MUTED, GRID, BASE = "#898781", "#898781", "#e1e0d9", "#c3c2b7"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def err(ax, xs, ys, los, his, color, lw=1.4, cap=2.5):
    ax.errorbar(xs, ys, yerr=[np.asarray(ys) - np.asarray(los),
                              np.asarray(his) - np.asarray(ys)],
                fmt="none", ecolor=color, elinewidth=lw, capsize=cap, zorder=3)


# ---------------------------------------------------------------- Figure 1/2
def fig_e1(tables, metric_kind, path, pdf, model_label, seed):
    """metric_kind='acc' → Figure 1；'margin' → Figure 2。按 dataset 分面板，
    factor 按值降序；bar + 图片聚类 bootstrap CI；随机线 0.5（acc）。
    标题标注实际模型与种子（R7，不再写死）。"""
    df = pd.read_csv(tables / "e1_results.csv")
    df = df[df.protocol == "2afc_self_built"]
    datasets = ["clevr", "pug_spar"]
    titles = {"clevr": "CLEVR val (self-built 2AFC)", "pug_spar": "PUG:SPAR (self-built 2AFC)"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), constrained_layout=True)
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds].sort_values(
            "accuracy" if metric_kind == "acc" else "mean_margin", ascending=False)
        xs = np.arange(len(sub))
        col, val = ("accuracy", "acc") if metric_kind == "acc" else ("mean_margin", "margin")
        lo = sub[f"{val}_ci_low"].to_numpy(); hi = sub[f"{val}_ci_high"].to_numpy()
        yv = sub[col].to_numpy()
        ax.bar(xs, yv - (0 if metric_kind == "margin" else 0), width=0.62,
               color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=2)
        err(ax, xs, yv, lo, hi, INK2)
        if metric_kind == "acc":
            ax.axhline(0.5, color=BASE, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.text(0.99, 0.505, "chance", ha="right", va="bottom",
                    fontsize=7.5, color=MUTED, transform=ax.get_yaxis_transform())
            ax.set_ylim(0.25, 1.0)
        else:
            ax.axhline(0.0, color=BASE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(xs, sub.factor, rotation=28, ha="right")
        ax.set_title(titles[ds])
        style_axes(ax)
    ylab = "2AFC accuracy" if metric_kind == "acc" else "Mean matching margin"
    fig.suptitle(f"{ylab} by structural factor ({model_label}, seed {seed})",
                 fontsize=11, y=1.06)
    axes[0].set_ylabel(ylab)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig_delta_magnitude(tables, path, pdf):
    df = pd.read_csv(tables / "e2_delta_stats.csv").set_index("factor")
    order = ["environment", "camera", "texture", "size"]
    nu_raw = float(df["nuisance_norm_visual_raw"].iloc[0])
    nu_z = float(df["nuisance_norm_visual_z"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    for ax, col, nu, unit, name in [
            (axes[0], "mean_delta_norm_raw", nu_raw, "‖Δ‖ (raw)", "Raw space"),
            (axes[1], "mean_delta_norm_z", nu_z, "‖Δ‖ (normalized)", "Normalized space")]:
        xs = np.arange(len(order))
        vals = [float(df.loc[f, col]) for f in order]
        ax.bar(xs, vals, width=0.62, color=BLUE, edgecolor=SURFACE,
               linewidth=1, zorder=2)
        ax.axhline(nu, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.text(0.02, nu, " visual nuisance", ha="left", va="bottom",
                fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
        ax.set_xticks(xs, order, rotation=20, ha="right")
        ax.set_title(name)
        ax.set_ylabel(unit)
        style_axes(ax)
    fig.suptitle("Delta magnitude by PUG: Animals factor (mean factor-change step)",
                 fontsize=11, y=1.06)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
LADDER = [
    ("same_transition_sim_diff_context", "same\ntransition\n(diff ctx)"),
    ("same_factor_dt_sc_sim", "same factor\nΔt\n(same ctx)"),
    ("same_factor_dt_dc_sim", "same factor\nΔt\n(diff ctx)"),
    ("diff_factor_sc_sim", "diff factor\n(same ctx)"),
    ("diff_factor_dc_sim", "diff factor\n(diff ctx)"),
    ("random_similarity", "random\ndeltas"),
]


def fig_ladder(tables, path, pdf):
    df = pd.read_csv(tables / "e2_delta_stats.csv").set_index("factor")
    order = ["environment", "camera", "texture", "size"]
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.0), constrained_layout=True)
    for ax, fac in zip(axes, order):
        xs = np.arange(len(LADDER))
        vals = [float(df.loc[fac, c]) for c, _ in LADDER]
        ax.plot(xs, vals, color=BLUE, lw=1.6, zorder=2)
        ax.scatter(xs, vals, s=26, color=BLUE, zorder=3)
        # same-transition 是唯一带聚类 bootstrap CI 的层
        lo = float(df.loc[fac, "same_transition_ci_low"])
        hi = float(df.loc[fac, "same_transition_ci_high"])
        ax.errorbar([0], [vals[0]], yerr=[[vals[0] - lo], [hi - vals[0]]],
                    fmt="none", ecolor=INK2, elinewidth=1.2, capsize=2.5, zorder=4)
        ax.text(0, vals[0], f" {vals[0]:.3f}", ha="center", va="bottom",
                fontsize=7, color=INK2)
        ax.set_xticks(xs, [lab for _, lab in LADDER], fontsize=6.2)
        ax.set_title(fac)
        ax.set_ylim(bottom=min(min(vals), 0) - 0.03)
        style_axes(ax)
    axes[0].set_ylabel("Δ–Δ cosine similarity (normalized)")
    fig.suptitle("Similarity ladder by PUG: Animals factor", fontsize=11, y=1.06)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
def fig_cogent(tables, path, pdf):
    cog = pd.read_csv(tables / "e3_cogent.csv")
    gap = pd.read_csv(tables / "e3_difficulty_gap.csv").set_index("shape_group")
    units = pd.read_csv(tables / "e3_unit_gaps.csv")
    units = units[units.negative_type == "pooled_equal_type"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6),
                                   width_ratios=[1, 1.25],
                                   constrained_layout=True)
    # (a) A vs B：题型明细 + CI
    sub = cog[cog.negative_type != "pooled"]
    cats = [("cube_cylinder", "color"), ("cube_cylinder", "shape"),
            ("sphere", "color"), ("sphere", "shape")]
    xs = np.arange(len(cats))
    w = 0.36
    for off, cond, color in [(-w / 2, "A", BLUE), (w / 2, "B", ORANGE)]:
        ys, los, his = [], [], []
        for grp, nt in cats:
            r = sub[(sub.condition == cond) & (sub.shape_group == grp)
                    & (sub.negative_type == nt)].iloc[0]
            ys.append(r.accuracy); los.append(r.acc_ci_low); his.append(r.acc_ci_high)
        ax1.bar(xs + off, ys, width=w, color=color, edgecolor=SURFACE,
                linewidth=1, zorder=2, label=f"Condition {cond}")
        err(ax1, xs + off, ys, los, his, INK2)
    ax1.axhline(0.5, color=BASE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax1.set_xticks(xs, [f"{g}\n{t}-neg" for g, t in cats], fontsize=8)
    ax1.set_ylim(0.4, 0.95)
    ax1.set_ylabel("2AFC accuracy")
    ax1.legend(loc="upper left", ncols=2)
    gc = gap.loc["cube_cylinder"]; gs = gap.loc["sphere"]
    ax1.set_title("CoGenT valA vs valB (pooled)\n"
                  f"equal-type gap A−B: cube/cyl {gc.difficulty_gap:+.3f} "
                  f"[{gc.gap_ci_low:+.3f},{gc.gap_ci_high:+.3f}] · "
                  f"sphere {gs.difficulty_gap:+.3f} "
                  f"[{gs.gap_ci_low:+.3f},{gs.gap_ci_high:+.3f}]", fontsize=8.5)
    style_axes(ax1)

    # (b) 逐 (shape_group, color) A−B gap + CI：颜色 = CI 方向（发散对）
    ys_all, labels = [], []
    for grp in ["cube_cylinder", "sphere"]:
        for color in ["gray", "blue", "brown", "yellow",
                      "red", "green", "purple", "cyan"]:
            r = units[(units.shape_group == grp) & (units.color == color)].iloc[0]
            ys_all.append(r); labels.append(f"{grp} / {color}")
    ypos = np.arange(len(ys_all))[::-1]
    for y, r in zip(ypos, ys_all):
        c = BLUE if r.ci_low > 0 else (RED if r.ci_high < 0 else GRAY)
        ax2.plot([r.ci_low, r.ci_high], [y, y], color=c, lw=1.4, zorder=2)
        ax2.scatter([r.gap], [y], s=22, color=c, zorder=3)
    ax2.axvline(0.0, color=BASE, lw=1.0, zorder=1)
    ax2.set_yticks(ypos, labels, fontsize=6.8)
    ax2.set_xlabel("A − B equal-type accuracy gap")
    ax2.grid(axis="y", visible=False); ax2.grid(axis="x", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_title("Per-color unit gaps (16 units, 95% cluster-bootstrap CI)\n"
                  "blue: A easier (CI>0) · red: B easier (CI<0) · gray: ns",
                  fontsize=8.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14",
                    help="结果所属模型：定位 tables 目录与图标题（R2/R7）")
    ap.add_argument("--seed", type=int, default=0,
                    help="结果所属种子：定位 tables 目录与图标题")
    ap.add_argument("--tables-dir", default=None,
                    help="直接指定 tables 目录（默认 outputs/<model>/seed_<seed>/tables，"
                         "不存在时回退旧布局 outputs/tables）")
    args = ap.parse_args()
    paths = load_paths()
    out = paths["dirs"]["outputs"]
    # 以所需的全量结果文件（而非目录存在）判定：只有 _smoke 文件的新目录不误选
    new_tables = out / args.model / f"seed_{args.seed}" / "tables"
    if args.tables_dir:
        tables = Path(args.tables_dir)
    elif (new_tables / "e1_results.csv").exists():
        tables = new_tables
    elif (out / "tables" / "e1_results.csv").exists():
        tables = out / "tables"      # 旧布局（R2 隔离修复前的历史结果）
    else:
        raise SystemExit(f"找不到全量结果：{new_tables / 'e1_results.csv'} 与旧布局均缺失，"
                         f"请先跑全量或用 --tables-dir 指定")
    model_label = load_config(args.model).get("label", args.model)
    # 默认组合（clip_vitl14, seed 0）沿用 outputs/figures 保持文件名兼容；
    # 其他模型/种子写入独立子目录，互不覆盖（R2/R7）
    if args.model == "clip_vitl14" and args.seed == 0 and not args.tables_dir:
        figdir = out / "figures"
    else:
        figdir = out / "figures" / f"{args.model}_seed{args.seed}"
    figdir.mkdir(parents=True, exist_ok=True)
    print(f"tables={tables} | model={model_label} | seed={args.seed} | figures={figdir}")

    with PdfPages(figdir / "GraphDelta_stage1_figures.pdf") as pdf:
        fig_e1(tables, "acc", figdir / "fig1_2afc_accuracy.png", pdf, model_label, args.seed)
        fig_e1(tables, "margin", figdir / "fig2_matching_margin.png", pdf, model_label, args.seed)
        fig_delta_magnitude(tables, figdir / "fig3_delta_magnitude.png", pdf)
        fig_ladder(tables, figdir / "fig4_delta_similarity_ladder.png", pdf)
        fig_cogent(tables, figdir / "fig5_cogent_ab.png", pdf)
    for f in sorted(figdir.iterdir()):
        print(f"written {f}")


if __name__ == "__main__":
    main()
