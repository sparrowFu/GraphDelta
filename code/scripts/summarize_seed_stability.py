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

审查 A4：CI 或点估计缺失（NaN）时分类为 invalid，不产生"稳定"结论。
审查 A6：三种子混用新旧布局（修复前/后口径）时默认停止，需
--allow-mixed-legacy 才输出临时对照，CSV 记录各种子来源布局。
复核 B3：SPAR 描述行 stable 记 NA（不计入稳定总数），三种子均未定义的
指标（如 hardneg 的 top5）跳过；复核 B8：per_world 输出时 world 纳入 key
与匹配，每种子每 key 必须恰好一行。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_paths, verify_e3_consistency, verify_run_meta
from src.utils.logging import get_logger


def ci_vs_ref(lo, hi, ref=0.5):
    """CI 与参考值的关系：above / below / contains / invalid。

    invalid：CI 缺失（NaN/Inf，上游 bootstrap 有效重采样不足等）——
    不能与任何参考值比较（审查 A4）。
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "invalid"
    if lo > ref:
        return "above"
    if hi < ref:
        return "below"
    return "contains"


def ci_sign(lo, hi):
    """CI 符号：pos / neg / ns / invalid（CI 缺失时不能判为不显著，审查 A4）。"""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "invalid"
    if lo > 0:
        return "pos"
    if hi < 0:
        return "neg"
    return "ns"


def stable_classes(cls):
    """三种子分类一致且无 invalid 才判 stable（审查 A4：缺失 ≠ 稳定）。"""
    return len(set(cls)) == 1 and "invalid" not in cls


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14",
                    help="汇总的模型（新布局 outputs/<model>/seed_k/tables；R2）")
    ap.add_argument("--allow-mixed-legacy", action="store_true",
                    help="三种子混用新旧布局时仍输出（仅作版本临时对照，审查 A6）")
    ap.add_argument("--allow-exploratory-e3", action="store_true",
                    help="E3 派生单元表状态异常（failed/stale/指纹不匹配）时仍汇总，"
                         "但输出写入独立的 seed_stability_exploratory.csv，"
                         "不冒充正式稳定性结果（四轮问题 4）")
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["dirs"]["outputs"]
    tabs, layouts = {}, {}
    LEGACY_MODEL = "clip_vitl14"   # 旧布局 outputs/tables* 的唯一来源模型（R2）
    for s in [0, 1, 2]:
        new = out_dir / args.model / f"seed_{s}" / "tables"
        legacy = out_dir / ("tables" if s == 0 else f"tables_seed{s}")
        # 以全量结果文件判定（只有 _smoke 文件的新目录不误选）
        if (new / "e1_results.csv").exists():
            verify_run_meta(new, args.model, s)
            tabs[s], layouts[s] = new, "new"
        elif args.model == LEGACY_MODEL and (legacy / "e1_results.csv").exists():
            tabs[s], layouts[s] = legacy, "legacy"   # 仅历史属主模型可回退
        else:
            raise SystemExit(
                f"model={args.model} seed {s} 缺少全量结果：{new} 不存在；"
                f"旧布局 outputs/tables* 属于 {LEGACY_MODEL}，不能作为 "
                f"{args.model} 的数据源。请先跑全量（run_e1/e2/e3/spar_official "
                f"--model {args.model} --seed {s}）。")
    logger = get_logger("seed_stability", out_dir / "logs")
    for s, d in tabs.items():
        logger.info(f"[seeds] seed {s} tables: {d} ({layouts[s]})")
    if len(set(layouts.values())) > 1:
        # A6：新旧布局分别是修复前/后的口径，混用的"稳定/翻转"不能解释为
        # 固定协议下的抽样稳定性——默认停止，只允许显式要求的临时对照。
        if not args.allow_mixed_legacy:
            raise SystemExit(
                "[seeds] 三种子混用新旧布局（新=修复后、旧=修复前数据，口径"
                "不可比），已停止正式稳定性汇总（审查 A6）。请补齐同版本三种子"
                "后重跑；确要输出临时对照请加 --allow-mixed-legacy（CSV 会记录"
                "各种子来源布局）。")
        logger.warning("[seeds] --allow-mixed-legacy：三种子混用新旧布局，"
                       "结果只能作为版本对照，不是固定协议下的抽样稳定性")

    # 四轮问题 4：绘图入口已校验 E3 一致性 sidecar，正式稳定性汇总此前没有
    # 同样检查——明确 failed/stale/指纹不匹配的探索单元表会静默进入正式结论。
    # 默认停止；显式 --allow-exploratory-e3 时输出独立文件，不写正式 CSV。
    e3_exploratory = False
    for s, d in tabs.items():
        try:
            verify_e3_consistency(d)
        except SystemExit as e:
            if not args.allow_exploratory_e3:
                raise SystemExit(f"seed {s} 的 E3 派生表不可消费：{e}")
            e3_exploratory = True
            logger.warning(f"[seeds] seed {s}: {e}")
    if e3_exploratory:
        logger.warning("[seeds] --allow-exploratory-e3：E3 派生表状态异常，"
                       "本次输出为探索对照，写入 seed_stability_exploratory.csv，"
                       "不构成正式稳定性结论")

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
        stable = stable_classes(cls)
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
        stable = stable_classes(cls)
        add("e1_gap", key, "gap", gaps, stable, f"CI 符号: {'/'.join(cls)}")

    # SPAR 官方：top1/top5/hardneg 三种子最大绝对差（描述，无阈值判定）
    # 复核 B3：不适用指标（三种子均未定义，如 hardneg 的 top5）跳过，不计行；
    # 描述性行 stable 记 NA（空），不计入"稳定"总数。
    # 复核 B8：per_world 输出时 world 是 key 的一部分，纳入 key 与匹配；
    # 每种子每个 key 必须恰好一行，不允许 .iloc[0] 掩盖重复。
    def read_so(seed):
        df = read(seed, "e1_spar_official.csv")
        return df.fillna({"texture": "", "pos_pair": ""})

    has_world = "world" in read_so(0).columns

    def so_key(row):
        k = (f"{row.task}|bg={row.add_background}|tex={row.texture}"
             f"|pair={row.pos_pair}|hn={row.hardneg}")
        return f"{k}|w={row.world}" if has_world else k

    def so_match(df, row):
        m = (df.task == row.task) & (df.add_background == row.add_background) \
            & (df.texture == row.texture) & (df.pos_pair == row.pos_pair) \
            & (df.hardneg == row.hardneg)
        if has_world:
            m &= df.world == row.world
        return m

    for _, o0 in read_so(0).iterrows():
        if o0.n_images == 0:
            continue
        key = so_key(o0)
        for metric in ["top1", "top5"]:
            vals = []
            for s in [0, 1, 2]:
                o = read_so(s)
                m = so_match(o, o0)
                if int(m.sum()) != 1:
                    raise SystemExit(
                        f"[seeds] e1_spar_official seed {s} 中 key 命中 {int(m.sum())} 行"
                        f"（应为 1）：{key}——结果表存在重复或缺失行（复核 B8）。")
                v = o[m].iloc[0][metric]
                vals.append(None if pd.isna(v) else round(float(v), 4))
            if all(v is None for v in vals):
                continue    # 三种子均未定义（如 hardneg 的 top5）→ 不适用
            vv = [v for v in vals if v is not None]
            maxdiff = None if len(vv) < 2 else round(max(vv) - min(vv), 4)
            miss = "含缺失值; " if len(vv) < 3 else ""
            add("spar_official", key, metric, vals, None,   # 描述行：stable=NA
                f"{miss}max|diff|={maxdiff}")

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
            stable = stable_classes(cls)
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
        stable = stable_classes(cls)
        add("e3_gap", grp_name, "difficulty_gap", vals, stable,
            f"CI 符号: {'/'.join(cls)}")

    unit = {s: read(s, "e3_unit_gaps.csv") for s in [0, 1, 2]}
    u0 = unit[0][unit[0].negative_type == "pooled_equal_type"]
    same_sign = 0
    for _, r0 in u0.iterrows():
        key = f"{r0.shape_group}/{r0.color}"
        raw = []
        for s in [0, 1, 2]:
            u = unit[s]
            r = u[(u.negative_type == "pooled_equal_type") & (u.shape_group == r0.shape_group)
                  & (u.color == r0.color)].iloc[0]
            raw.append(float(r.gap))
        # A4：点估计缺失（NaN）时不能凭 sign 集合判稳定
        # B4：符号用原始值判定，round 只用于显示（避免先舍入把 ±1e-5 变成 0）
        vals = [round(v, 4) for v in raw]
        ok = all(np.isfinite(v) for v in raw)
        signs = {np.sign(v) for v in raw} if ok else {"invalid"}
        stable = ok and len(signs) == 1
        if stable:
            same_sign += 1
        add("e3_unit", key, "gap", vals, stable,
            ("gap 缺失（invalid）" if not ok else
             f"point-gap 符号{'一致' if stable else '翻转'}"))
    logger.info(f"[seeds] E3 单元 gap 三种子同号: {same_sign}/{len(u0)}")

    out = pd.DataFrame(rows)
    for s in (0, 1, 2):   # A6：记录各种子数据来源布局（new/legacy）
        out[f"layout_seed{s}"] = layouts[s]
    # 四轮问题 4：E3 探索模式输出独立文件，不与正式 seed_stability.csv 混写
    path_out = tabs[0] / ("seed_stability_exploratory.csv" if e3_exploratory
                          else "seed_stability.csv")
    out.to_csv(path_out, index=False)
    # B3：stable 列含 NA（SPAR 描述行），计数与筛选需 NA 安全
    n_stable = int((out.stable == True).sum())
    n_na = int(out.stable.isna().sum())
    unstable = out[out.stable == False]
    logger.info(f"[seeds] written {path_out} ({len(out)} rows; "
                f"stable={n_stable}, NA(描述)={n_na}, UNSTABLE={len(unstable)})")
    for _, r in unstable.iterrows():
        logger.info(f"[seeds] FLIP {r.section} {r.key} {r.metric}: "
                    f"{r.seed0}/{r.seed1}/{r.seed2} ({r.note})")
    return out


if __name__ == "__main__":
    main()
