"""E1: State-level Structural Diagnosis（PUG: SPAR + CLEVR val，方案 §27–§33）。

- CLEVR：六类文本 hard negative（object/color/size/material/relation/binding-style），
  全部通过"正例为真、负例为假"检查（20a）；relation 使用官方 relationships 字段，
  语义为 [i,j] ∈ relationships[rel] ⟺ objs[j] 在 objs[i] 的 rel 方向
  （审查报告 P1-1：主语取 objs[j]，参考对象取 objs[i]）。
- SPAR：scene/position/attribute/relation 四类；过滤双对象同种动物场景
  （正负文本可能相同或同时为真，审查报告 P1-2）；position/attribute/relation
  另过滤 character2_name == "blank"。协议为自建 2AFC，官方检索协议见
  e1_spar_official.py（审查报告 P2-11）。
- 只做 image → (T+, T-) 方向（29a）。
- 每图每 factor 采样上限 per_scene_cap 生效；filter 原因计数写盘（P2-5）。
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from ..data import clevr
from ..data.captions import (
    clevr_attribute_contents,
    clevr_binding_contents,
    clevr_relation_contents,
    spar_attribute_contents,
    spar_position_contents,
    spar_relation_contents,
    spar_scene_contents,
)
from ..data.pug_spar import META_COLUMNS as SPAR_META_COLUMNS
from ..data.pug_spar import build_metadata as build_spar_metadata
from ..features.cache import ImageFeatureCache, TextCache
from ..features.extract import extract_clevr_images, extract_pug_images
from ..metrics.bootstrap import cluster_gap_ci
from ..utils.io import load_config, write_jsonl
from .common import EXPORT_FIELDS, cap_per_image, eval_2afc_items, subsample


# ---------------- CLEVR items ----------------

def build_clevr_items(scenes, factors, rng):
    """返回 (items, stats)。stats 记录各类过滤原因的计数（方案 §20a）。"""
    items = {f: [] for f in factors}
    stats = {"scenes": len(scenes), "objects_total": 0,
             "attribute": {}, "relation": {}, "binding": {}}
    for scene in scenes:
        objs = scene["objects"]
        counts = clevr.scene_combo_counts(scene)
        combos = set(counts)
        cs_pairs = {(o["color"], o["shape"]) for o in objs}
        stats["objects_total"] += len(objs)

        def unique(o):
            return counts[clevr.combo(o)] == 1

        def push(factor, kind, pos, neg):
            items[factor].append({
                "image_id": scene["image_filename"], "factor": factor,
                "content_pos": pos, "content_neg": neg, "kind": kind})

        # 单属性负例：负例组合必须不在图中
        for o in objs:
            if not unique(o):
                for factor in ("color", "size", "material", "object"):
                    if factor in factors:
                        stats["attribute"].setdefault(factor, {})["nonunique_object"] = \
                            stats["attribute"].get(factor, {}).get("nonunique_object", 0) + 1
                continue
            for factor, key in [("color", "color"), ("size", "size"),
                                ("material", "material"), ("object", "shape")]:
                if factor not in factors:
                    continue
                cur = o[key]
                order = rng.permutation([v for v in clevr.VALUE_SETS[key] if v != cur])
                neg_v = None
                for v in order:
                    if clevr.combo_with(o, key, v) not in combos:
                        neg_v = v
                        break
                st = stats["attribute"].setdefault(factor, {})
                if neg_v is None:
                    st["no_valid_negative"] = st.get("no_valid_negative", 0) + 1
                    continue
                pos, neg = clevr_attribute_contents(o, key, neg_v)
                push(factor, "phrase", pos, neg)
                st["kept"] = st.get("kept", 0) + 1

        # relation 负例：官方 relationships 邻接表。
        # 语义（官方生成器 compute_all_relationships，审查报告 P1-1）：
        #   j in relationships[rel][i]  ⟺  objs[j] 在 objs[i] 的 rel 方向
        #   → 正例主语 = objs[j]，参考对象 = objs[i]
        # 负例 "objs[j] 在 objs[i] 的 anti 方向" 为真 ⟺ (i,j) ∈ relationships[anti]
        if "relation" in factors:
            relmap = scene.get("relationships", {})

            def pair_set(r):
                return {(i, j) for i, js in enumerate(relmap.get(r, [])) for j in js}

            anti_sets = {r: pair_set(r) for r in clevr.REL_ANTONYM}
            rst = stats["relation"]
            for rel, adjacency in relmap.items():
                for i, jlist in enumerate(adjacency):
                    for j in jlist:
                        subj, ref = objs[j], objs[i]
                        rst["candidates"] = rst.get("candidates", 0) + 1
                        if not (unique(subj) and unique(ref)):
                            rst["nonunique_object"] = rst.get("nonunique_object", 0) + 1
                            continue
                        if (i, j) in anti_sets[clevr.REL_ANTONYM[rel]]:
                            rst["antonym_conflict"] = rst.get("antonym_conflict", 0) + 1
                            continue
                        pos, neg = clevr_relation_contents(subj, ref, rel)
                        push("relation", "sentence", pos, neg)
                        rst["kept"] = rst.get("kept", 0) + 1

        # binding-style：换绑后两个组合不得都在图中。
        # caption 只含 (color, shape)，唯一性按 (color, shape) 组合判定
        #（完整属性唯一不足以保证 caption 唯一，二轮审查解释边界项）
        if "binding" in factors:
            cs_counts = Counter((o["color"], o["shape"]) for o in objs)
            n = len(objs)
            bst = stats["binding"]
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = objs[i], objs[j]
                    bst["candidates"] = bst.get("candidates", 0) + 1
                    if a["color"] == b["color"] or a["shape"] == b["shape"]:
                        bst["same_color_or_shape"] = bst.get("same_color_or_shape", 0) + 1
                        continue
                    if cs_counts[(a["color"], a["shape"])] > 1 \
                            or cs_counts[(b["color"], b["shape"])] > 1:
                        bst["nonunique_object"] = bst.get("nonunique_object", 0) + 1
                        continue
                    if (b["color"], a["shape"]) in cs_pairs and (a["color"], b["shape"]) in cs_pairs:
                        bst["swap_in_scene"] = bst.get("swap_in_scene", 0) + 1
                        continue
                    pos, neg = clevr_binding_contents(a, b)
                    push("binding", "phrase", pos, neg)
                    bst["kept"] = bst.get("kept", 0) + 1
    return items, stats


# ---------------- SPAR items ----------------

def build_spar_items(meta, factors, rng, two_object_only, require_diff_texture):
    """返回 (items, stats)。双对象同种动物场景整体过滤（审查报告 P1-2）。"""
    animals = sorted((set(meta["character_name"]) | set(meta["character2_name"])) - {"blank"})
    items = {f: [] for f in factors}
    stats = {"rows": len(meta), "same_animal_filtered": 0, "blank_two_object_skipped": 0,
             "attribute_same_texture_filtered": 0, "scene_no_candidate": 0}

    def push(factor, kind, rid, pos, neg):
        items[factor].append({"image_id": rid, "factor": factor,
                              "content_pos": pos, "content_neg": neg, "kind": kind})

    for row in meta.to_dict("records"):
        two = row["character2_name"] != "blank"
        rid = row["sample_id"]
        if two and row["character_name"] == row["character2_name"]:
            stats["same_animal_filtered"] += 1
            continue
        if "scene" in factors:
            excl = {row["character_name"], row["character2_name"]}
            cand = [a for a in animals if a not in excl]
            if not cand:
                stats["scene_no_candidate"] += 1
            else:
                other = cand[int(rng.integers(len(cand)))]
                pos, neg = spar_scene_contents(row, other)
                push("scene", "phrase", rid, pos, neg)
        if not two:
            stats["blank_two_object_skipped"] += 1
            continue
        if "position" in factors:
            pos, neg = spar_position_contents(row)
            push("position", "sentence", rid, pos, neg)
        if "attribute" in factors:
            if require_diff_texture and row["character_texture"] == row["character2_texture"]:
                stats["attribute_same_texture_filtered"] += 1
            else:
                pos, neg = spar_attribute_contents(row)
                push("attribute", "phrase", rid, pos, neg)
        if "relation" in factors:
            pos, neg = spar_relation_contents(row)
            push("relation", "sentence", rid, pos, neg)
    return items, stats


# ---------------- run ----------------

def run(model, paths, logger, smoke=False, seed=None):
    cfg = load_config("e1_state")
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name)
    tcache = TextCache(paths["dirs"]["cache"], model.name)
    out_tables = paths["dirs"]["outputs"] / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seeds"][0] if seed is None else seed)
    bs = cfg["bootstrap"]
    cap = cfg["sampling"].get("per_scene_cap", 1)
    target = min(200, cfg["sampling"]["target_per_cell"]) if smoke \
        else cfg["sampling"]["target_per_cell"]

    rows = []
    acc_store = {}       # (dataset, factor) -> (item_acc, image_id)，用于 factor gap CI

    # ---- CLEVR val ----
    cl = cfg["clevr"]
    scenes = clevr.load_scenes(f"{paths['datasets']['clevr']}/{cl['scene_file']}")
    if smoke:
        scenes = scenes[:300]
    key = "clevr_val_smoke" if smoke else "clevr_val"
    extract_clevr_images(model, cache, key, scenes,
                         f"{paths['datasets']['clevr']}/{cl['image_dir']}", logger,
                         only_ids=[s["image_filename"] for s in scenes] if smoke else None)
    items_by_factor, clevr_stats = build_clevr_items(scenes, cl["factors"], rng)
    clevr_all_items = []
    for f in cl["factors"]:
        its = cap_per_image(items_by_factor[f], cap, rng)
        its = subsample(its, target, rng)
        if not its:
            logger.warning(f"[E1] clevr factor={f}: no valid items")
            continue
        m = eval_2afc_items(model, tcache, cache, key, its, n_boot=bs["n_boot"], ci=bs["ci"],
                            clusters=[it["image_id"] for it in its])
        rows.append({"dataset": "clevr", "factor": f, "protocol": "2afc_self_built",
                     **{k: m[k] for k in EXPORT_FIELDS}})
        acc_store[("clevr", f)] = (m["item_acc"], [it["image_id"] for it in its])
        clevr_all_items.extend(its)
        logger.info(f"[E1] clevr/{f}: n={m['n_samples']} acc={m['accuracy']:.4f}")
    write_jsonl(out_tables / ("e1_items_clevr_smoke.jsonl" if smoke else "e1_items_clevr.jsonl"),
                clevr_all_items)
    stats_path = out_tables / ("e1_clevr_filter_stats_smoke.json" if smoke
                               else "e1_clevr_filter_stats.json")
    stats_path.write_text(json.dumps(clevr_stats, indent=2, ensure_ascii=False))
    logger.info(f"[E1] written {stats_path}")

    # ---- PUG: SPAR（自建 2AFC）----
    sp = cfg["spar"]
    meta = build_spar_metadata(paths["datasets"]["pug_spar"])
    if smoke:
        meta = meta.head(1500).reset_index(drop=True)
    key = "pug_spar_smoke" if smoke else "pug_spar"
    extract_pug_images(model, cache, key, paths["datasets"]["pug_spar"], SPAR_META_COLUMNS,
                       logger, only_row_ids=set(meta["row_id"]) if smoke else None,
                       id_prefix="pug_spar")
    spar_items, spar_stats = build_spar_items(meta, sp["factors"], rng,
                                              set(sp["two_object_only"]),
                                              sp["attribute_require_diff_texture"])
    spar_all_items = []
    for f in sp["factors"]:
        its = cap_per_image(spar_items[f], cap, rng)
        its = subsample(its, target, rng)
        if not its:
            logger.warning(f"[E1] spar factor={f}: no valid items")
            continue
        m = eval_2afc_items(model, tcache, cache, key, its, n_boot=bs["n_boot"], ci=bs["ci"],
                            clusters=[it["image_id"] for it in its])
        rows.append({"dataset": "pug_spar", "factor": f, "protocol": "2afc_self_built",
                     **{k: m[k] for k in EXPORT_FIELDS}})
        acc_store[("pug_spar", f)] = (m["item_acc"], [it["image_id"] for it in its])
        spar_all_items.extend(its)
        logger.info(f"[E1] spar/{f}: n={m['n_samples']} acc={m['accuracy']:.4f}")
    write_jsonl(out_tables / ("e1_items_spar_smoke.jsonl" if smoke else "e1_items_spar.jsonl"),
                spar_all_items)
    stats_path = out_tables / ("e1_spar_filter_stats_smoke.json" if smoke
                               else "e1_spar_filter_stats.json")
    stats_path.write_text(json.dumps(spar_stats, indent=2, ensure_ascii=False))
    logger.info(f"[E1] written {stats_path}")

    out = pd.DataFrame(rows)
    path = out_tables / ("e1_results_smoke.csv" if smoke else "e1_results.csv")
    out.to_csv(path, index=False)
    logger.info(f"[E1] written {path} ({len(out)} rows)")

    # ---- factor 间 Accuracy Gap 的 bootstrap CI（方案 §66）----
    # 同一图片可同时出现在两个 factor 的样本里：按图片并集同步重采样
    # （cluster_gap_ci），不把共享图片当独立样本（二轮审查问题 2）
    gap_rows = []
    for dataset in ("clevr", "pug_spar"):
        fs = [f for (d, f) in acc_store if d == dataset]
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                (a, cla), (b, clb) = acc_store[(dataset, fs[i])], acc_store[(dataset, fs[j])]
                gap, lo, hi = cluster_gap_ci(a, cla, b, clb,
                                             n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
                gap_rows.append({"dataset": dataset, "factor_a": fs[i], "factor_b": fs[j],
                                 "acc_a": float(np.mean(a)), "acc_b": float(np.mean(b)),
                                 "gap": gap, "gap_ci_low": lo, "gap_ci_high": hi})
    gap_df = pd.DataFrame(gap_rows)
    path_g = out_tables / ("e1_factor_gaps_smoke.csv" if smoke else "e1_factor_gaps.csv")
    gap_df.to_csv(path_g, index=False)
    logger.info(f"[E1] written {path_g} ({len(gap_df)} pairs)")
    return out
