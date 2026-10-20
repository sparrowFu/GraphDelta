"""E3: Composition Difficulty Diagnosis（CoGenT valA/valB + sphere 对照，方案 §46–§51）。

- 冻结 baseline 的 A/B 难度比较，不解释为 seen/unseen（§3 问题三 / §51）。
- sphere 对照：A/B 中都取全部颜色，用于分离颜色词难度与组合难度。
- 逐 (color, shape)、逐颜色报告；Difficulty Gap 只在 cube/cylinder 上计算。
"""
import numpy as np
import pandas as pd

from ..data.captions import cogent_color_shape_contents
from ..data.cogent import condition_image_dir, load_condition_scenes
from ..features.cache import ImageFeatureCache, TextCache
from ..features.extract import extract_clevr_images
from ..metrics.bootstrap import diff_proportions_ci
from ..utils.io import load_config
from .common import EXPORT_FIELDS, eval_2afc_items, subsample


def build_cogent_items(scenes, condition, cfg, rng):
    colors, shapes = cfg["colors"], cfg["shapes"]
    cap = cfg["sampling"]["per_scene_cap"]
    items = []
    for scene in scenes:
        objs = scene["objects"]
        cs_pairs = {(o["color"], o["shape"]) for o in objs}
        order = rng.permutation(len(objs))[:cap]
        for oi in order:
            o = objs[int(oi)]
            grp = "sphere" if o["shape"] == "sphere" else "cube_cylinder"
            for ntype in cfg["sampling"]["negative_types"]:
                if ntype == "color":
                    cand = [c for c in colors
                            if c != o["color"] and (c, o["shape"]) not in cs_pairs]
                    if not cand:
                        continue
                    v = cand[int(rng.integers(len(cand)))]
                    pos, neg = cogent_color_shape_contents(o, neg_color=v)
                else:
                    cand = [s for s in shapes
                            if s != o["shape"] and (o["color"], s) not in cs_pairs]
                    if not cand:
                        continue
                    v = cand[int(rng.integers(len(cand)))]
                    pos, neg = cogent_color_shape_contents(o, neg_shape=v)
                items.append({
                    "image_id": scene["image_filename"], "condition": condition,
                    "shape_group": grp, "negative_type": ntype,
                    "color": o["color"], "shape": o["shape"],
                    "content_pos": pos, "content_neg": neg, "kind": "phrase"})
    return items


def check_palette(scenes, condition, cfg, logger):
    """核对官方调色板（只报告，不参与生成逻辑）。"""
    pal = cfg["palettes"][condition]
    bad = 0
    for s in scenes[:5000]:
        for o in s["objects"]:
            shape, color = o["shape"], o["color"]
            if shape in pal and color not in pal[shape]:
                bad += 1
    logger.info(f"[E3] palette check {condition}: {bad} violations in first 5000 scenes"
                f"（应为 0；sphere 不受限制）")


def run(model, paths, logger, smoke=False):
    cfg = load_config("e3_cogent")
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name)
    tcache = TextCache(paths["dirs"]["cache"], model.name)
    out_tables = paths["dirs"]["outputs"] / "tables"
    rng = np.random.default_rng(cfg["seeds"][0])
    bs = cfg["bootstrap"]
    target = min(300, cfg["sampling"]["target_per_condition_shape"]) if smoke \
        else cfg["sampling"]["target_per_condition_shape"]

    rows = []
    evals = {}          # (condition, shape_group) -> metrics dict
    all_items = []

    for cond in ["A", "B"]:
        ccfg = cfg["conditions"][cond]
        scenes = load_condition_scenes(paths["datasets"]["cogent"], cond)
        check_palette(scenes, cond, cfg, logger)
        if smoke:
            scenes = scenes[:300]
        key = (f"cogent_val{cond}_smoke" if smoke else f"cogent_val{cond}")
        extract_clevr_images(model, cache, key, scenes,
                             condition_image_dir(paths["datasets"]["cogent"], cond),
                             logger,
                             only_ids=[s["image_filename"] for s in scenes] if smoke else None)
        items = build_cogent_items(scenes, cond, cfg, rng)
        for grp in ["cube_cylinder", "sphere"]:
            sub = [it for it in items if it["shape_group"] == grp]
            sub = subsample(sub, target, rng)
            if not sub:
                logger.warning(f"[E3] {cond}/{grp}: no items")
                continue
            m = eval_2afc_items(model, tcache, cache, key, sub,
                                n_boot=bs["n_boot"], ci=bs["ci"])
            evals[(cond, grp)] = m
            rows.append({"condition": cond, "shape_group": grp, "negative_type": "pooled",
                         **{k: m[k] for k in EXPORT_FIELDS}})
            for ntype in cfg["sampling"]["negative_types"]:
                nsub = [it for it in sub if it["negative_type"] == ntype]
                if not nsub:
                    continue
                mm = eval_2afc_items(model, tcache, cache, key, nsub,
                                     n_boot=bs["n_boot"], ci=bs["ci"])
                rows.append({"condition": cond, "shape_group": grp, "negative_type": ntype,
                             **{k: mm[k] for k in EXPORT_FIELDS}})
            for it, a in zip(sub, m["item_acc"]):
                all_items.append({**it, "item_acc": float(a)})
            logger.info(f"[E3] {cond}/{grp}: n={m['n_samples']} acc={m['accuracy']:.4f}")

    out = pd.DataFrame(rows)
    path = out_tables / ("e3_cogent_smoke.csv" if smoke else "e3_cogent.csv")
    out.to_csv(path, index=False)
    logger.info(f"[E3] written {path}")

    # Difficulty Gap（A − B；主量在 cube/cylinder，sphere 为对照）
    gap_rows = []
    for grp in ["cube_cylinder", "sphere"]:
        if ("A", grp) in evals and ("B", grp) in evals:
            a, b = evals[("A", grp)], evals[("B", grp)]
            gap, lo, hi = diff_proportions_ci(a["item_acc"], b["item_acc"],
                                              n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
            gap_rows.append({
                "shape_group": grp,
                "acc_A": a["accuracy"], "acc_B": b["accuracy"],
                "difficulty_gap": gap, "gap_ci_low": lo, "gap_ci_high": hi,
                "note": "主指标" if grp == "cube_cylinder" else "sphere 对照",
            })
    gap_df = pd.DataFrame(gap_rows)
    path_g = out_tables / ("e3_difficulty_gap_smoke.csv" if smoke else "e3_difficulty_gap.csv")
    gap_df.to_csv(path_g, index=False)
    logger.info(f"[E3] written {path_g}")

    # 逐 (color, shape) 与逐颜色
    recs = [{"color": it["color"], "shape": it["shape"],
             "condition": it["condition"], "acc": it["item_acc"]} for it in all_items]
    df = pd.DataFrame(recs)
    combo = df.groupby(["color", "shape"]).agg(
        n=("acc", "size"), accuracy=("acc", "mean")).reset_index()
    path_c = out_tables / ("e3_cogent_by_combo_smoke.csv" if smoke else "e3_cogent_by_combo.csv")
    combo.to_csv(path_c, index=False)
    by_color = df.groupby(["color", "condition"]).agg(
        n=("acc", "size"), accuracy=("acc", "mean")).reset_index()
    path_b = out_tables / ("e3_cogent_by_color_smoke.csv" if smoke else "e3_cogent_by_color.csv")
    by_color.to_csv(path_b, index=False)
    logger.info(f"[E3] written {path_c}, {path_b}")
    return out
