#!/usr/bin/env python
"""导出 E1 低分题型的人工审查清单（方案第 2 节，开篇待办 1）。

对象：
- SPAR 两类稳定低于 50% 的题型：attribute（B/32 约 39.9%）、
  relation（L/14 约 45.6%），与报告数字一一对应；
- CLEVR binding 题作为对照（约 49.5%–51.5%，随机水平）。

分层维度：模型一致错 / 模型分歧 / 模型一致对（对照），层内按正例取值
（attribute 的纹理组合、relation 的方位轴）轮转抽样，保证取值覆盖。
种子固定，可复现；导出图片与逐题成绩，供逐题人工核验。

输出：
- outputs/audit/e1_audit_checklist.csv（含待填审查列）
- outputs/audit/e1_audit_summary.json（分层与覆盖统计）
- outputs/audit/images/spar/、outputs/audit/images/clevr/

用法（任意目录）：python code/scripts/export_audit_items.py [--spar-per-class N] [--clevr-n N]
"""
import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

# common/ 提供 src.*（共用包），本阶段 src/ 提供阶段包
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from src.data.pug_common import iter_image_batches
from src.data.pug_spar import META_COLUMNS as SPAR_META
from src.utils.io import load_paths
from src.utils.logging import get_logger

MODELS = ["clip_vitl14", "clip_vitb32", "siglip2_so400m"]
SEEDS = [0, 1, 2]
SAMPLE_SEED = 20261103
# SPAR 两类各 100 题（一致错 40 / 分歧 40 / 一致对 20），CLEVR 对照 50 题
SPAR_QUOTA = {"一致错": 40, "分歧": 40, "一致对": 20}
CLEVR_QUOTA = 50
AUDIT_COLS = ["ck1_pos_true", "ck2_neg_false", "ck3_visible",
              "ck4_clear", "issue_note"]


def load_items(outputs_dir: Path, model: str, dataset: str):
    """读取一个模型全部种子的逐题结果，按种子聚合到列表。"""
    per_seed = []
    for seed in SEEDS:
        path = outputs_dir / model / f"seed_{seed}" / "tables" / f"e1_items_{dataset}.jsonl"
        items = []
        with open(path) as f:
            for line in f:
                items.append(json.loads(line))
        per_seed.append(items)
    # 各种子按相同顺序写入；逐题合并
    merged = []
    for i in range(len(per_seed[0])):
        rows = [per_seed[s][i] for s in range(len(SEEDS))]
        accs = [r["item_acc"] for r in rows]
        merged.append({
            **{k: rows[0][k] for k in ("image_id", "factor", "content_pos",
                                       "content_neg", "kind")},
            "acc_seeds": accs,
            "margin_mean": sum(r["item_margin"] for r in rows) / len(rows),
        })
    return merged


def strat_tag(model_means):
    below = [a for a in model_means if a < 0.5]
    if len(below) == len(model_means):
        return "一致错"
    if not below:
        return "一致对"
    return "分歧"


def round_robin_pick(groups: dict, n: int, rng: random.Random):
    """按 value_key 分组轮转抽样：先保证取值覆盖，组内随机。"""
    for key in groups:
        rng.shuffle(groups[key])
    picked, keys = [], list(groups)
    while len(picked) < n and keys:
        nxt = []
        for key in keys:
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
            if groups[key]:
                nxt.append(key)
        keys = nxt
    return picked


def value_key_of(factor: str, gt) -> str:
    """分层抽样的取值键：attribute 用纹理组合，relation 用方位轴。"""
    if factor == "attribute":
        return f"{gt.character_texture}|{gt.character2_texture}"
    return "left-right" if gt.character1_pos == "left" else "below-above"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spar-per-class", type=int, default=100,
                    help="SPAR 每类低分题型的抽查题数（层内配额按 40/40/20 等比缩放）")
    ap.add_argument("--clevr-n", type=int, default=CLEVR_QUOTA,
                    help="CLEVR binding 对照题数（一致错 / 分歧 / 一致对三层均分）")
    args = ap.parse_args()

    paths = load_paths()
    outputs_dir = paths["dirs"]["outputs"]
    logger = get_logger("audit_export", outputs_dir / "logs")
    rng = random.Random(SAMPLE_SEED)
    out_dir = outputs_dir / "audit"
    img_spar = out_dir / "images" / "spar"
    img_clevr = out_dir / "images" / "clevr"
    for d in (img_spar, img_clevr):
        d.mkdir(parents=True, exist_ok=True)

    # ---- SPAR 两类低分题型 ----
    spar_meta = pd.read_parquet(paths["dirs"]["manifests"] / "pug_spar_meta.parquet")
    meta_by_id = {f"pug_spar_{int(r.row_id):06d}": r for r in spar_meta.itertuples()}

    # 按比例缩放层内配额（默认 40/40/20）
    scale = args.spar_per_class / 100.0
    quota = {k: max(1, round(v * scale)) for k, v in SPAR_QUOTA.items()}

    # 三模型逐题成绩（跨种子均值），按题合并；两类题型共用，只读一次
    by_model = {m: load_items(outputs_dir, m, "spar") for m in MODELS}
    checklist, picked_row_ids = [], set()
    for factor in ("attribute", "relation"):
        rows = {}
        for m in MODELS:
            for it in by_model[m]:
                if it["factor"] != factor:
                    continue
                key = (it["image_id"], it["factor"], it["content_pos"])
                rows.setdefault(key, {"item": it, "acc": {}})
                rows[key]["acc"][m] = sum(it["acc_seeds"]) / len(it["acc_seeds"])
        # 分层 + 取值键
        strata = collections.defaultdict(lambda: collections.defaultdict(list))
        for key, r in rows.items():
            means = [r["acc"][m] for m in MODELS]
            tag = strat_tag(means)
            gt = meta_by_id.get(r["item"]["image_id"])
            if gt is None:
                continue
            strata[tag][value_key_of(factor, gt)].append({**r, "gt": gt})
        for tag, groups in strata.items():
            n = quota.get(tag, 0)
            if not n:
                continue
            for r in round_robin_pick(groups, n, rng):
                it, gt = r["item"], r["gt"]
                acc_str = ";".join(
                    f"{m}={r['acc'][m]:.3f}" for m in MODELS)
                checklist.append({
                    "dataset": "spar", "image_id": it["image_id"],
                    "factor": factor, "kind": it["kind"],
                    "content_pos": it["content_pos"], "content_neg": it["content_neg"],
                    "gt_world": gt.world_name,
                    "gt_char1": gt.character_name, "gt_char2": gt.character2_name,
                    "gt_tex1": gt.character_texture, "gt_tex2": gt.character2_texture,
                    "gt_pos1": gt.character1_pos, "gt_pos2": gt.character2_pos,
                    "acc_models": acc_str, "margin_mean": round(it["margin_mean"], 4),
                    "strat": tag, "value_key": value_key_of(factor, gt),
                })
                picked_row_ids.add(int(it["image_id"].split("_")[-1]))
        logger.info(f"[audit] spar/{factor}: 抽取 {sum(1 for c in checklist if c['factor']==factor)} 题")

    # ---- CLEVR binding 对照 ----
    clevr_items = {}
    for m in MODELS:
        for it in load_items(outputs_dir, m, "clevr"):
            if it["factor"] != "binding":
                continue
            key = (it["image_id"], it["content_pos"])
            clevr_items.setdefault(key, {"item": it, "acc": {}})
            clevr_items[key]["acc"][m] = sum(it["acc_seeds"]) / len(it["acc_seeds"])
    clevr_strata = collections.defaultdict(list)
    for key, r in clevr_items.items():
        clevr_strata[strat_tag([r["acc"][m] for m in MODELS])].append(r)
    clevr_picked = []
    for tag in ("一致错", "分歧", "一致对"):
        pool = clevr_strata.get(tag, [])
        rng.shuffle(pool)
        take = min(len(pool), max(1, round(args.clevr_n / 3)))
        clevr_picked.extend((tag, r) for r in pool[:take])
    for tag, r in clevr_picked:
        it = r["item"]
        checklist.append({
            "dataset": "clevr", "image_id": it["image_id"],
            "factor": "binding", "kind": it["kind"],
            "content_pos": it["content_pos"], "content_neg": it["content_neg"],
            "gt_world": "", "gt_char1": "", "gt_char2": "",
            "gt_tex1": "", "gt_tex2": "", "gt_pos1": "", "gt_pos2": "",
            "acc_models": ";".join(f"{m}={r['acc'][m]:.3f}" for m in MODELS),
            "margin_mean": round(it["margin_mean"], 4),
            "strat": tag, "value_key": "binding",
        })

    # ---- 导出清单 ----
    df = pd.DataFrame(checklist)
    for col in AUDIT_COLS:
        df[col] = ""
    csv_path = out_dir / "e1_audit_checklist.csv"
    df.to_csv(csv_path, index=False)

    # ---- 抽取被抽中题目的图片 ----
    parquet_dir = str(paths["datasets"]["pug_spar"])
    n_img = 0
    for row_ids, images, _meta in iter_image_batches(
            parquet_dir, SPAR_META, only_row_ids=picked_row_ids):
        for rid, img in zip(row_ids, images):
            img.save(img_spar / f"pug_spar_{rid:06d}.png")
            n_img += 1
    clevr_img_dir = Path(paths["datasets"]["clevr"]) / "images" / "val"
    n_clevr = 0
    for c in df[df.dataset == "clevr"]["image_id"]:
        src = clevr_img_dir / c
        if src.exists():
            shutil.copy(src, img_clevr / c)
            n_clevr += 1
        else:
            logger.warning(f"[audit] CLEVR 图片缺失: {src}")

    # ---- 汇总 ----
    summary = {
        "sample_seed": SAMPLE_SEED,
        "models": MODELS, "seeds": SEEDS,
        "per_class_target": args.spar_per_class, "clevr_target": args.clevr_n,
        "counts": {"|".join(map(str, k)): int(v) for k, v in
                   df.groupby(["dataset", "factor", "strat"]).size().items()},
        "value_coverage": {
            f"{fac}": sorted(df[df.factor == fac]["value_key"].unique().tolist())
            for fac in ("attribute", "relation")
        },
        "images_exported": {"spar": n_img, "clevr": n_clevr},
        "audit_columns": AUDIT_COLS,
        "checklist_columns_note": {
            "ck1_pos_true": "正句为真（对照图片与 gt_* 列）",
            "ck2_neg_false": "负句为假",
            "ck3_visible": "被描述对象完整可见，无遮挡/裁剪",
            "ck4_clear": "指代无歧义（对象、视角、主客体顺序）",
            "issue_note": "问题描述；发现系统性错误记录生成环节与受影响范围",
        },
    }
    with open(out_dir / "e1_audit_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"[audit] 清单 {len(df)} 题 -> {csv_path}")
    logger.info(f"[audit] 图片 spar={n_img} clevr={n_clevr} -> {out_dir/'images'}")


if __name__ == "__main__":
    main()
