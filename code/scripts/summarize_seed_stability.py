#!/usr/bin/env python
"""多种子稳定性汇总（方案 §82 seeds 0/1/2）。

每个种子优先读新布局 outputs/<model>/seed_{k}/tables（R2 隔离后的规范目录），
不存在时回退旧布局 outputs/tables（seed 0）/ tables_seed{k}（历史结果），
检验三种子下关键判定是否翻转：
- E1 自建 2AFC：accuracy CI 与 0.5 的关系（含 / 上 / 下）是否一致；
- E1 factor gaps：gap CI 符号（正 / 负 / 含 0）是否一致；
- SPAR 官方：各任务 top1/top5/hardneg 的三种子最大绝对差（描述，无阈值判定）；
- E2：Norm Gap 与 Transition Gap（z 空间）的 CI 符号是否一致；
- E3：difficulty gap 数值与 CI 符号；逐 (shape_group,color) 单元 gap 点符号
  三种子是否同号。

输出 outputs/tables/seed_stability.csv + 控制台摘要；判定翻转的行 stable=False。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_paths
from src.utils.logging import get_logger


def ci_vs_ref(lo, hi, ref=0.5):
    """CI 与参考值的关系：above / below / contains。"""
    if lo > ref:
        return "above"
    if hi < ref:
        return "below"
    return "contains"


def ci_sign(lo, hi):
    if lo > 0:
        return "pos"
    if hi < 0:
        return "neg"
    return "ns"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14",
                    help="汇总的模型（新布局 outputs/<model>/seed_k/tables；R2）")
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["dirs"]["outputs"]
    tabs = {}
    for s in [0, 1, 2]:
        new = out_dir / args.model / f"seed_{s}" / "tables"
        legacy = out_dir / ("tables" if s == 0 else f"tables_seed{s}")
        # 以全量结果文件判定（只有 _smoke 文件的新目录不误选）
        if (new / "e1_results.csv").exists():
            tabs[s] = new
        elif (legacy / "e1_results.csv").exists():
            tabs[s] = legacy       # R2 隔离修复前跑出的历史结果
        else:
            raise SystemExit(f"missing full results in both {new} and {legacy}")
    logger = get_logger("seed_stability", out_dir / "logs")
    for s, d in tabs.items():
        logger.info(f"[seeds] seed {s} tables: {d}")

    def read(seed, name):
        return pd.read_csv(tabs[seed] / name)

    rows = []

    def add(section, key, metric, vals, stable, note=""):
        rows.append({"section": section, "key": key, "metric": metric,
                     "seed0": vals[0], "seed1": vals[1], "seed2": vals[2],
                     "stable": stable, "note": note})

    # E1 自建 2AFC：accuracy CI vs 0.5
    for (ds, fac), grp in read(0, "e1_results.csv").groupby(["dataset", "factor"]):
        cls, accs = [], []
        for s in [0, 1, 2]:
            r = read(s, "e1_results.csv")
            r = r[(r.dataset == ds) & (r.factor == fac)].iloc[0]
            cls.append(ci_vs_ref(r.acc_ci_low, r.acc_ci_high))
            accs.append(round(float(r.accuracy), 4))
        stable = len(set(cls)) == 1
        add("e1_acc", f"{ds}/{fac}", "accuracy", accs, stable,
            f"CI-vs-0.5: {'/'.join(cls)}")

    # E1 factor gaps：gap CI 符号
    for _, g0 in read(0, "e1_factor_gaps.csv").iterrows():
        key = f"{g0.dataset}/{g0.factor_a}-{g0.factor_b}"
        cls, gaps = [], []
        for s in [0, 1, 2]:
            g = read(s, "e1_factor_gaps.csv")
            g = g[(g.dataset == g0.dataset) & (g.factor_a == g0.factor_a)
                  & (g.factor_b == g0.factor_b)].iloc[0]
            cls.append(ci_sign(g.gap_ci_low, g.gap_ci_high))
            gaps.append(round(float(g.gap), 4))
        stable = len(set(cls)) == 1
        add("e1_gap", key, "gap", gaps, stable, f"CI 符号: {'/'.join(cls)}")

    # SPAR 官方：top1/top5/hardneg 三种子最大绝对差（描述）
    def read_so(seed):
        df = read(seed, "e1_spar_official.csv")
        return df.fillna({"texture": "", "pos_pair": ""})

    for _, o0 in read_so(0).iterrows():
        if o0.n_images == 0:
            continue
        key = f"{o0.task}|bg={o0.add_background}|tex={o0.texture}|pair={o0.pos_pair}|hn={o0.hardneg}"
        for metric in ["top1", "top5"]:
            vals = []
            for s in [0, 1, 2]:
                o = read_so(s)
                m = (o.task == o0.task) & (o.add_background == o0.add_background) \
                    & (o.texture == o0.texture) & (o.pos_pair == o0.pos_pair) \
                    & (o.hardneg == o0.hardneg)
                v = o[m].iloc[0][metric]
                vals.append(None if pd.isna(v) else round(float(v), 4))
            vv = [v for v in vals if v is not None]
            maxdiff = None if len(vv) < 2 else round(max(vv) - min(vv), 4)
            add("spar_official", key, metric, vals, True,
                f"max|diff|={maxdiff}")

    # E2：Norm Gap 与 Transition Gap 的 CI 符号
    for _, d0 in read(0, "e2_delta_stats.csv").iterrows():
        fac = d0.factor
        for metric, lo_c, hi_c in [
                ("norm_gap_visual_raw", "norm_gap_visual_raw_ci_low",
                 "norm_gap_visual_raw_ci_high"),
                ("transition_gap_z", "gap_ci_low", "gap_ci_high")]:
            cls, vals = [], []
            for s in [0, 1, 2]:
                d = read(s, "e2_delta_stats.csv")
                d = d[d.factor == fac].iloc[0]
                if metric == "transition_gap_z":
                    cls.append(ci_sign(d[lo_c], d[hi_c]))
                    vals.append(round(float(d["transition_similarity_gap"]), 4))
                else:
                    cls.append(ci_sign(d[lo_c], d[hi_c]))
                    vals.append(round(float(d[metric]), 4))
            stable = len(set(cls)) == 1
            add("e2", f"{fac}/{metric}", metric, vals, stable,
                f"CI 符号: {'/'.join(cls)}")

    # E3：difficulty gap + 逐单元符号一致性
    for grp_name in ["cube_cylinder", "sphere"]:
        cls, vals = [], []
        for s in [0, 1, 2]:
            d = read(s, "e3_difficulty_gap.csv")
            d = d[d.shape_group == grp_name].iloc[0]
            cls.append(ci_sign(d.gap_ci_low, d.gap_ci_high))
            vals.append(round(float(d.difficulty_gap), 4))
        stable = len(set(cls)) == 1
        add("e3_gap", grp_name, "difficulty_gap", vals, stable,
            f"CI 符号: {'/'.join(cls)}")

    unit = {s: read(s, "e3_unit_gaps.csv") for s in [0, 1, 2]}
    u0 = unit[0][unit[0].negative_type == "pooled_equal_type"]
    same_sign = 0
    for _, r0 in u0.iterrows():
        key = f"{r0.shape_group}/{r0.color}"
        vals = []
        for s in [0, 1, 2]:
            u = unit[s]
            r = u[(u.negative_type == "pooled_equal_type") & (u.shape_group == r0.shape_group)
                  & (u.color == r0.color)].iloc[0]
            vals.append(round(float(r.gap), 4))
        signs = {np.sign(v) for v in vals}
        if len(signs) == 1:
            same_sign += 1
        add("e3_unit", key, "gap", vals, len(signs) == 1,
            f"point-gap 符号{'一致' if len(signs) == 1 else '翻转'}")
    logger.info(f"[seeds] E3 单元 gap 三种子同号: {same_sign}/{len(u0)}")

    out = pd.DataFrame(rows)
    path_out = tabs[0] / "seed_stability.csv"
    out.to_csv(path_out, index=False)
    unstable = out[~out.stable]
    logger.info(f"[seeds] written {path_out} ({len(out)} rows; "
                f"stable={int(out.stable.sum())}, UNSTABLE={len(unstable)})")
    for _, r in unstable.iterrows():
        logger.info(f"[seeds] FLIP {r.section} {r.key} {r.metric}: "
                    f"{r.seed0}/{r.seed1}/{r.seed2} ({r.note})")
    return out


if __name__ == "__main__":
    main()
