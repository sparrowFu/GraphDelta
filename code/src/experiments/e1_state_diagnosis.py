"""E1: State-level Structural Diagnosis（PUG: SPAR + CLEVR val，方案 §27–§33）。

- CLEVR：六类文本 hard negative（object/color/size/material/relation/binding-style），
  全部通过"正例为真、负例为假"检查（20a）；relation 使用官方 relationships 字段。
- SPAR：scene/position/attribute/relation 四类；position/attribute/relation 先过滤双对象样本。
- 只做 image → (T+, T-) 方向（29a）。
"""
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
from ..utils.io import load_config, write_jsonl
from .common import EXPORT_FIELDS, eval_2afc_items, subsample


# ---------------- CLEVR items ----------------

def build_clevr_items(scenes, factors, rng):
    items = {f: [] for f in factors}
    for scene in scenes:
        objs = scene["objects"]
        counts = clevr.scene_combo_counts(scene)
        combos = set(counts)
        cs_pairs = {(o["color"], o["shape"]) for o in objs}

        def unique(o):
            return counts[clevr.combo(o)] == 1

        def push(factor, kind, pos, neg):
            items[factor].append({
                "image_id": scene["image_filename"], "factor": factor,
                "content_pos": pos, "content_neg": neg, "kind": kind})

        # 单属性负例：负例组合必须不在图中
        for o in objs:
            if not unique(o):
                continue
            for factor, key in [("color", "color"), ("size", "size"),
                                ("material", "material"), ("object", "shape")]:
                if factor not in factors:
                    continue
                cur = o[key]
                order = rng.permutation([v for v in clevr.VALUE_SETS[key] if v != cur])
                for v in order:
                    if clevr.combo_with(o, key, v) not in combos:
                        pos, neg = clevr_attribute_contents(o, key, v)
                        push(factor, "phrase", pos, neg)
                        break

        # relation 负例：官方 relationships（邻接表：rel[i] = 所有 j 使 i rel-of j）；
        # 反义关系对 (i,j) 不得同时成立
        if "relation" in factors:
            relmap = scene.get("relationships", {})

            def pair_set(r):
                return {(i, j) for i, js in enumerate(relmap.get(r, [])) for j in js}

            anti_sets = {r: pair_set(r) for r in clevr.REL_ANTONYM}
            for rel, adjacency in relmap.items():
                for i, jlist in enumerate(adjacency):
                    for j in jlist:
                        a, b = objs[i], objs[j]
                        if not (unique(a) and unique(b)):
                            continue
                        if (i, j) in anti_sets[clevr.REL_ANTONYM[rel]]:
                            continue
                        pos, neg = clevr_relation_contents(a, b, rel)
                        push("relation", "sentence", pos, neg)

        # binding-style：换绑后两个组合不得都在图中
        if "binding" in factors:
            n = len(objs)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = objs[i], objs[j]
                    if a["color"] == b["color"] or a["shape"] == b["shape"]:
                        continue
                    if not (unique(a) and unique(b)):
                        continue
                    if (b["color"], a["shape"]) in cs_pairs and (a["color"], b["shape"]) in cs_pairs:
                        continue
                    pos, neg = clevr_binding_contents(a, b)
                    push("binding", "phrase", pos, neg)
    return items


# ---------------- SPAR items ----------------

def build_spar_items(meta, factors, rng, two_object_only, require_diff_texture):
    animals = sorted((set(meta["character_name"]) | set(meta["character2_name"])) - {"blank"})
    items = {f: [] for f in factors}

    def push(factor, kind, rid, pos, neg):
        items[factor].append({"image_id": rid, "factor": factor,
                              "content_pos": pos, "content_neg": neg, "kind": kind})

    for row in meta.to_dict("records"):
        two = row["character2_name"] != "blank"
        rid = row["sample_id"]
        if "scene" in factors:
            excl = {row["character_name"], row["character2_name"]}
            cand = [a for a in animals if a not in excl]
            if cand:
                other = cand[int(rng.integers(len(cand)))]
                pos, neg = spar_scene_contents(row, other)
                push("scene", "phrase", rid, pos, neg)
        if not two:
            continue
        if "position" in factors:
            pos, neg = spar_position_contents(row)
            push("position", "sentence", rid, pos, neg)
        if "attribute" in factors and (
                not require_diff_texture or row["character_texture"] != row["character2_texture"]):
            pos, neg = spar_attribute_contents(row)
            push("attribute", "phrase", rid, pos, neg)
        if "relation" in factors:
            pos, neg = spar_relation_contents(row)
            push("relation", "sentence", rid, pos, neg)
    return items


# ---------------- run ----------------

def run(model, paths, logger, smoke=False):
    cfg = load_config("e1_state")
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name)
    tcache = TextCache(paths["dirs"]["cache"], model.name)
    out_tables = paths["dirs"]["outputs"] / "tables"
    rng = np.random.default_rng(cfg["seeds"][0])
    bs = cfg["bootstrap"]
    target = min(200, cfg["sampling"]["target_per_cell"]) if smoke \
        else cfg["sampling"]["target_per_cell"]

    rows = []

    # ---- CLEVR val ----
    cl = cfg["clevr"]
    scenes = clevr.load_scenes(f"{paths['datasets']['clevr']}/{cl['scene_file']}")
    if smoke:
        scenes = scenes[:300]
    key = "clevr_val_smoke" if smoke else "clevr_val"
    extract_clevr_images(model, cache, key, scenes,
                         f"{paths['datasets']['clevr']}/{cl['image_dir']}", logger,
                         only_ids=[s["image_filename"] for s in scenes] if smoke else None)
    items_by_factor = build_clevr_items(scenes, cl["factors"], rng)
    clevr_all_items = []
    for f in cl["factors"]:
        its = subsample(items_by_factor[f], target, rng)
        if not its:
            logger.warning(f"[E1] clevr factor={f}: no valid items")
            continue
        m = eval_2afc_items(model, tcache, cache, key, its, n_boot=bs["n_boot"], ci=bs["ci"])
        rows.append({"dataset": "clevr", "factor": f, **{k: m[k] for k in EXPORT_FIELDS}})
        clevr_all_items.extend(its)
        logger.info(f"[E1] clevr/{f}: n={m['n_samples']} acc={m['accuracy']:.4f}")
    write_jsonl(out_tables / ("e1_items_clevr_smoke.jsonl" if smoke else "e1_items_clevr.jsonl"),
                clevr_all_items)

    # ---- PUG: SPAR ----
    sp = cfg["spar"]
    meta = build_spar_metadata(paths["datasets"]["pug_spar"])
    if smoke:
        meta = meta.head(1500).reset_index(drop=True)
    key = "pug_spar_smoke" if smoke else "pug_spar"
    extract_pug_images(model, cache, key, paths["datasets"]["pug_spar"], SPAR_META_COLUMNS,
                       logger, only_row_ids=set(meta["row_id"]) if smoke else None,
                       id_prefix="pug_spar")
    spar_items = build_spar_items(meta, sp["factors"], rng,
                                  set(sp["two_object_only"]),
                                  sp["attribute_require_diff_texture"])
    spar_all_items = []
    for f in sp["factors"]:
        its = subsample(spar_items[f], target, rng)
        if not its:
            logger.warning(f"[E1] spar factor={f}: no valid items")
            continue
        m = eval_2afc_items(model, tcache, cache, key, its, n_boot=bs["n_boot"], ci=bs["ci"])
        rows.append({"dataset": "pug_spar", "factor": f, **{k: m[k] for k in EXPORT_FIELDS}})
        spar_all_items.extend(its)
        logger.info(f"[E1] spar/{f}: n={m['n_samples']} acc={m['accuracy']:.4f}")
    write_jsonl(out_tables / ("e1_items_spar_smoke.jsonl" if smoke else "e1_items_spar.jsonl"),
                spar_all_items)

    out = pd.DataFrame(rows)
    path = out_tables / ("e1_results_smoke.csv" if smoke else "e1_results.csv")
    out.to_csv(path, index=False)
    logger.info(f"[E1] written {path} ({len(out)} rows)")
    return out
