"""E3: Composition Difficulty Diagnosis（CoGenT valA/valB + sphere 对照，方案 §46–§51）。

- 冻结 baseline 的 A/B 难度比较，不解释为 seen/unseen（§3 问题三 / §51）。
- Difficulty Gap：A/B 各自按 image_id 聚类 bootstrap，跨 negative_type
  等权平均，避免题型配比差异混入 A−B（审查报告 P2-9）。
- sphere 对照：A/B 中都取全部颜色，用于分离颜色词难度与组合难度。
- 逐 (color, shape)、逐颜色报告保留 condition / shape_group / negative_type /
  mean_margin（审查报告 P2-10）；Difficulty Gap 只在 cube/cylinder 上计算。
"""
import numpy as np
import pandas as pd

from ..data.captions import cogent_color_shape_contents
from ..data.cogent import condition_image_dir, load_condition_scenes
from ..features.cache import ImageFeatureCache, TextCache
from ..features.extract import extract_clevr_images
from ..metrics.bootstrap import cluster_gap_ci, cluster_gap_ci_equal_types
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


def run(model, paths, logger, smoke=False, seed=None):
    cfg = load_config("e3_cogent")
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name)
    tcache = TextCache(paths["dirs"]["cache"], model.name)
    out_tables = paths["dirs"]["outputs"] / "tables"
    rng = np.random.default_rng(cfg["seeds"][0] if seed is None else seed)
    bs = cfg["bootstrap"]
    target = min(300, cfg["sampling"]["target_per_condition_shape"]) if smoke \
        else cfg["sampling"]["target_per_condition_shape"]

    rows = []
    evals = {}          # (condition, shape_group) -> metrics dict
    per_item = {}       # (condition, shape_group) -> (acc, clusters, types) 数组
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
            clusters = [it["image_id"] for it in sub]
            m = eval_2afc_items(model, tcache, cache, key, sub,
                                n_boot=bs["n_boot"], ci=bs["ci"], clusters=clusters)
            evals[(cond, grp)] = m
            per_item[(cond, grp)] = (m["item_acc"], clusters,
                                     [it["negative_type"] for it in sub])
            rows.append({"condition": cond, "shape_group": grp, "negative_type": "pooled",
                         **{k: m[k] for k in EXPORT_FIELDS}})
            for ntype in cfg["sampling"]["negative_types"]:
                nidx = [i for i, it in enumerate(sub) if it["negative_type"] == ntype]
                if not nidx:
                    continue
                nsub = [sub[i] for i in nidx]
                mm = eval_2afc_items(model, tcache, cache, key, nsub,
                                     n_boot=bs["n_boot"], ci=bs["ci"],
                                     clusters=[sub[i]["image_id"] for i in nidx])
                rows.append({"condition": cond, "shape_group": grp, "negative_type": ntype,
                             **{k: mm[k] for k in EXPORT_FIELDS}})
            for it, a, mg in zip(sub, m["item_acc"], m["item_margin"]):
                all_items.append({**{k: it[k] for k in
                                     ("image_id", "condition", "shape_group", "negative_type",
                                      "color", "shape")},
                                  "item_acc": float(a), "item_margin": float(mg)})
            logger.info(f"[E3] {cond}/{grp}: n={m['n_samples']} acc={m['accuracy']:.4f}")

    out = pd.DataFrame(rows)
    path = out_tables / ("e3_cogent_smoke.csv" if smoke else "e3_cogent.csv")
    out.to_csv(path, index=False)
    logger.info(f"[E3] written {path}")

    # Difficulty Gap（A − B）：图片聚类 bootstrap + 跨题型等权平均（P2-9）
    gap_rows = []
    for grp in ["cube_cylinder", "sphere"]:
        if ("A", grp) in per_item and ("B", grp) in per_item:
            acc_a, cl_a, ty_a = per_item[("A", grp)]
            acc_b, cl_b, ty_b = per_item[("B", grp)]
            types = cfg["sampling"]["negative_types"]

            def equal_type_acc(acc, ty):
                return float(np.mean([np.mean(np.asarray(acc)[np.asarray(ty) == t])
                                      for t in types]))

            eq_a, eq_b = equal_type_acc(acc_a, ty_a), equal_type_acc(acc_b, ty_b)
            gap, lo, hi = cluster_gap_ci_equal_types(
                acc_a, cl_a, ty_a, acc_b, cl_b, ty_b, types,
                n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
            # 汇总口径（题型配比不等权）作敏感性对照；CI 同样按图片聚类
            # （二轮审查问题 5：pooled CI 不再用独立 item bootstrap）
            pooled_gap, plo, phi = cluster_gap_ci(acc_a, cl_a, acc_b, cl_b,
                                                  n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
            gap_rows.append({
                "shape_group": grp,
                "acc_A": float(np.mean(acc_a)), "acc_B": float(np.mean(acc_b)),
                "acc_A_equal_type": eq_a, "acc_B_equal_type": eq_b,
                "difficulty_gap": gap, "gap_ci_low": lo, "gap_ci_high": hi,
                "pooled_gap_mixed_types": pooled_gap,
                "pooled_gap_ci_low": plo, "pooled_gap_ci_high": phi,
                "note": "主指标（题型等权）" if grp == "cube_cylinder" else "sphere 对照（题型等权）",
            })
    gap_df = pd.DataFrame(gap_rows)
    path_g = out_tables / ("e3_difficulty_gap_smoke.csv" if smoke else "e3_difficulty_gap.csv")
    gap_df.to_csv(path_g, index=False)
    logger.info(f"[E3] written {path_g}")

    # 逐 (color, shape)（含题型明细 + pooled 行）与逐颜色（P2-10）
    df = pd.DataFrame(all_items)
    detail = df.groupby(["condition", "shape_group", "color", "shape", "negative_type"]).agg(
        n=("item_acc", "size"), accuracy=("item_acc", "mean"),
        mean_margin=("item_margin", "mean")).reset_index()
    pooled = df.groupby(["condition", "shape_group", "color", "shape"]).agg(
        n=("item_acc", "size"), accuracy=("item_acc", "mean"),
        mean_margin=("item_margin", "mean")).reset_index()
    pooled.insert(4, "negative_type", "pooled")
    combo = pd.concat([detail, pooled], ignore_index=True).sort_values(
        ["condition", "shape_group", "color", "shape", "negative_type"])
    path_c = out_tables / ("e3_cogent_by_combo_smoke.csv" if smoke else "e3_cogent_by_combo.csv")
    combo.to_csv(path_c, index=False)
    by_color = df.groupby(["condition", "shape_group", "color"]).agg(
        n=("item_acc", "size"), accuracy=("item_acc", "mean"),
        mean_margin=("item_margin", "mean")).reset_index()
    path_b = out_tables / ("e3_cogent_by_color_smoke.csv" if smoke else "e3_cogent_by_color.csv")
    by_color.to_csv(path_b, index=False)
    logger.info(f"[E3] written {path_c} ({len(combo)} rows), {path_b} ({len(by_color)} rows)")
    return out
