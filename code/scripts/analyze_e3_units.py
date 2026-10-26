#!/usr/bin/env python
"""E3 逐单元分析（方案 §71 情况 C 排查）。

用 e3_items_cogent.jsonl 的 item 级结果，按 (shape_group, color) 单元计算
A−B 难度差 + 图片聚类 bootstrap CI，检验"逐组合效应方向不一致、被等权
平均抵消"的怀疑。口径与主指标一致：题型等权（cluster_gap_ci_equal_types）；
明细行（含题型拆分）用 cluster_gap_ci。

- 16 个主单元各报 95% CI，属多重比较，只做描述、不做联合检验。
- 先用全部 items 复算一次总 gap，与 e3_difficulty_gap.csv 主指标对照
  （同一函数、同一数据，应一致到数值精度）；不一致时默认停止输出单元表，
  检查结果写 e3_unit_consistency.json（复核 B6；--no-strict 为探索模式）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.bootstrap import cluster_gap_ci, cluster_gap_ci_equal_types
from src.utils.io import load_config, load_paths, verify_run_meta
from src.utils.logging import get_logger


def load_items(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records)


def split_condition(sub, cond):
    s = sub[sub["condition"] == cond]
    return (s["item_acc"].to_numpy(dtype=np.float64),
            s["image_id"].to_numpy(), s["negative_type"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default=None,
                    help="items 所在目录（默认 outputs/clip_vitl14/seed_0/tables，"
                         "不存在时回退旧布局 outputs/tables；多种子校验时显式指向对应目录）")
    ap.add_argument("--no-strict", action="store_true",
                    help="一致性检查不过时仍输出单元表（探索模式；检查结果始终写 "
                         "e3_unit_consistency.json，复核 B6）")
    ap.add_argument("--check-atol", type=float, default=1e-8,
                    help="总 gap 复算与主表比较的绝对容差（默认 1e-8）")
    args = ap.parse_args()

    paths = load_paths()
    cfg = load_config("e3_cogent")
    bs = cfg["bootstrap"]
    types = cfg["sampling"]["negative_types"]
    out = paths["dirs"]["outputs"]
    default_new = out / "clip_vitl14" / "seed_0" / "tables"
    if args.tables_dir:
        tables = Path(args.tables_dir)
    elif (default_new / "e3_items_cogent.jsonl").exists():
        verify_run_meta(default_new, "clip_vitl14", 0)
        tables = default_new
    else:
        tables = out / "tables"      # 旧布局（R2 隔离修复前的历史结果）
    logger = get_logger("analyze_e3_units", paths["dirs"]["outputs"] / "logs")
    logger.info(f"[E3-units] tables dir: {tables}")

    df = load_items(tables / "e3_items_cogent.jsonl")
    logger.info(f"[E3-units] loaded {len(df)} items")

    # 总 gap 复算（强校验：应与 e3_difficulty_gap.csv 主指标一致到数值精度）。
    # 复核 B6：此前只打日志、不拦截——items 与主表冲突时仍输出单元表，下游
    # Figure 5 可能展示冲突来源。现在：检查结果写 e3_unit_consistency.json；
    # 不一致时默认停止（--no-strict 为探索模式，仍输出单元表）。
    ref_path = tables / "e3_difficulty_gap.csv"
    checks, check_fail = [], False
    if ref_path.exists():
        ref = pd.read_csv(ref_path).set_index("shape_group")
        for grp in ["cube_cylinder", "sphere"]:
            sub = df[df["shape_group"] == grp]
            acc_a, cl_a, ty_a = split_condition(sub, "A")
            acc_b, cl_b, ty_b = split_condition(sub, "B")
            gap, lo, hi = cluster_gap_ci_equal_types(
                acc_a, cl_a, ty_a, acc_b, cl_b, ty_b, types,
                n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
            r_gap = float(ref.loc[grp, "difficulty_gap"])
            both_nan = np.isnan(gap) and np.isnan(r_gap)
            diff = (abs(gap - r_gap)
                    if np.isfinite(gap) and np.isfinite(r_gap) else None)
            ok = bool(both_nan or (diff is not None and diff <= args.check_atol))
            checks.append({"shape_group": grp, "recomputed": gap, "report": r_gap,
                           "abs_diff": diff, "ok": ok})
            check_fail |= not ok
            logger.info(f"[E3-units] total gap {grp}: recomputed={gap:.6f} "
                        f"report={r_gap:.6f} |diff|={diff if diff is not None else 'NA'}"
                        f" ok={ok}")
    sidecar = tables / "e3_unit_consistency.json"
    sidecar.write_text(json.dumps(
        {"all_ok": not check_fail, "atol": args.check_atol, "checks": checks,
         "tables_dir": str(tables)}, ensure_ascii=False, indent=2))
    logger.info(f"[E3-units] consistency: all_ok={not check_fail} -> {sidecar}")
    if check_fail and not args.no_strict:
        raise SystemExit(
            "[E3-units] items 复算的总 gap 与主表不一致（见 "
            f"{sidecar}）——数据来源冲突，已停止输出单元表；"
            "探索模式（仍输出，供排查）用 --no-strict。")

    rows = []
    for grp in ["cube_cylinder", "sphere"]:
        for color in cfg["colors"]:
            sub = df[(df["shape_group"] == grp) & (df["color"] == color)]
            acc_a, cl_a, ty_a = split_condition(sub, "A")
            acc_b, cl_b, ty_b = split_condition(sub, "B")
            # 主单元：题型等权（与主指标同口径）；缺整类题型 → NaN
            gap, lo, hi = cluster_gap_ci_equal_types(
                acc_a, cl_a, ty_a, acc_b, cl_b, ty_b, types,
                n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
            rows.append({"shape_group": grp, "color": color,
                         "negative_type": "pooled_equal_type",
                         "n_A": len(acc_a), "n_B": len(acc_b),
                         "acc_A": float(acc_a.mean()), "acc_B": float(acc_b.mean()),
                         "gap": gap, "ci_low": lo, "ci_high": hi})
            # 明细：逐题型（A/B 图片不相交，普通两样本聚类差值）
            for t in types:
                sa = sub[(sub["condition"] == "A") & (sub["negative_type"] == t)]
                sb = sub[(sub["condition"] == "B") & (sub["negative_type"] == t)]
                va, ca, _ = split_condition(sa, "A")
                vb, cb, _ = split_condition(sb, "B")
                gap, lo, hi = cluster_gap_ci(
                    va, ca, vb, cb, n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
                rows.append({"shape_group": grp, "color": color, "negative_type": t,
                             "n_A": len(va), "n_B": len(vb),
                             "acc_A": float(va.mean()), "acc_B": float(vb.mean()),
                             "gap": gap, "ci_low": lo, "ci_high": hi})

    out = pd.DataFrame(rows)
    path_out = tables / "e3_unit_gaps.csv"
    out.to_csv(path_out, index=False)
    logger.info(f"[E3-units] written {path_out} ({len(out)} rows)")

    # 摘要：主单元 16 个（CI 不含 0 的按方向计数 + 离散度）
    pooled = out[out["negative_type"] == "pooled_equal_type"]
    sig_pos = pooled[pooled["ci_low"] > 0]
    sig_neg = pooled[pooled["ci_high"] < 0]
    logger.info(
        f"[E3-units] summary pooled_equal_type: n_units={len(pooled)} "
        f"sig_pos={len(sig_pos)} sig_neg={len(sig_neg)} "
        f"gap_std={float(pooled['gap'].std(ddof=1)):.4f} "
        f"gap_min={float(pooled['gap'].min()):+.4f} "
        f"gap_max={float(pooled['gap'].max()):+.4f}")
    for _, r in pooled.iterrows():
        flag = ("+" if r["ci_low"] > 0 else "-" if r["ci_high"] < 0 else " ")
        logger.info(f"[E3-units] {flag} {r['shape_group']:13s} {r['color']:6s} "
                    f"gap={r['gap']:+.4f} CI=[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
                    f"acc_A={r['acc_A']:.3f} acc_B={r['acc_B']:.3f}")
    logger.info("[E3-units] 注意：16 个主单元各报 95% CI，属多重比较，"
                "只做描述、不做联合检验。")


if __name__ == "__main__":
    main()
