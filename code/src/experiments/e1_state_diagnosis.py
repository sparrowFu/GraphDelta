"""E1: State-level Structural Diagnosis（PUG: SPAR + CLEVR val，方案 §27–§33）。

- CLEVR：六类文本 hard negative（object/color/size/material/relation/binding-style），
  全部通过"正例为真、负例为假"检查（20a）；relation 使用官方 relationships 字段，
  语义为 [i,j] ∈ relationships[rel] ⟺ objs[j] 在 objs[i] 的 rel 方向
  （审查报告 P1-1：主语取 objs[j]，参考对象取 objs[i]）。
- SPAR：scene/position/attribute/relation 四类；过滤双对象同种动物场景
  （正负文本可能相同或同时为真，审查报告 P1-2）；双对象判定要求两侧动物
  均非 blank（R3：character_name 也可能为 blank，不能只看 character2）；
  双侧均 blank 的纯背景行不进入动物描述任务。协议为自建 2AFC，官方检索
  协议见 e1_spar_official.py（审查报告 P2-11）。
- 只做 image → (T+, T-) 方向（29a）。
- 每图每 factor 采样上限 per_scene_cap 生效；filter 原因计数写盘（P2-5）。
- 采样平衡（codex 审查 2026-10-28）：target 子样本按 pos_value（正例一侧
  的取值）分层等额抽取，"恒选某个词"的策略期望准确率被压到 50%/1/k；
  SPAR position 以 50% 概率从 c2 视角出题，正例方位词覆盖 left/right/
  below/above。逐题分数（item_acc/item_margin）写入 items jsonl，逐侧
  准确率写 e1_acc_by_pos_value.csv。
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
from ..metrics.bootstrap import cluster_gap_ci, cluster_mean_ci
from ..utils.io import commit_experiment, load_config, output_tables_dir, write_jsonl
from .common import EXPORT_FIELDS, balanced_subsample, cap_per_image, eval_2afc_items


def _validate_spar_protocol(sp):
    """四轮补充 B：same_animal_filter / two_object_only 此前是"写了但不生效"的
    配置——过滤逻辑固定执行，改配置不会产生任何消融效果。

    协议固定为：
    - 双对象同种动物场景整体过滤（P1-2：放开会产生相同或同时为真的正负
      文本，不能用配置关闭）；
    - position/attribute/relation 固定为双对象任务（R3：单动物行不产生这
      三类 item，scene 对单/双动物均适用）。

    与固定协议不一致的取值直接拒绝（显式报错），不再静默忽略。
    """
    if not sp.get("same_animal_filter", True):
        raise SystemExit(
            "[E1] spar.same_animal_filter=false 不受支持：双对象同种动物场景的"
            "正负文本可能相同或同时为真（P1-2），协议要求整体过滤，不能用配置"
            "关闭。如需其他协议请先设计仍保证正负文本有效性的替代方案。")
    expected = sorted(f for f in sp.get("factors", [])
                      if f in ("position", "attribute", "relation"))
    got = sorted(set(sp.get("two_object_only", expected)))
    if got != expected:
        raise SystemExit(
            f"[E1] spar.two_object_only 不受支持：position/attribute/relation "
            f"固定为双对象任务（R3），当前协议要求 two_object_only={expected}，"
            f"配置为 {got}——修改该列表不会改变过滤行为，已停止。")


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

        def push(factor, kind, pos, neg, pos_value=None):
            # pos_value：正例一侧的取值（codex 审查 2026-10-28）——用于分层
            # 等额抽样与逐侧准确率报告；binding 正负例词集相同，天然免疫
            # 词偏好，不标。
            items[factor].append({
                "image_id": scene["image_filename"], "factor": factor,
                "content_pos": pos, "content_neg": neg, "kind": kind,
                **({"pos_value": pos_value} if pos_value is not None else {})})

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
                push(factor, "phrase", pos, neg, pos_value=cur)
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
                        push("relation", "sentence", pos, neg, pos_value=rel)
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

def build_spar_items(meta, factors, rng, require_diff_texture):
    """返回 (items, stats)。双对象同种动物场景整体过滤（审查报告 P1-2）；
    双对象判定要求两侧动物均非 blank，纯背景行整体跳过（审查报告 R3）；
    position/attribute/relation 固定为双对象任务（R3，配置层面由
    _validate_spar_protocol 校验，不作为可关闭开关）。"""
    animals = sorted((set(meta["character_name"]) | set(meta["character2_name"])) - {"blank"})
    items = {f: [] for f in factors}
    stats = {"rows": len(meta), "background_skipped": 0, "same_animal_filtered": 0,
             "single_animal_rows": 0, "attribute_same_texture_filtered": 0,
             "scene_no_candidate": 0}

    def push(factor, kind, rid, pos, neg, pos_value=None):
        items[factor].append({"image_id": rid, "factor": factor,
                              "content_pos": pos, "content_neg": neg, "kind": kind,
                              **({"pos_value": pos_value} if pos_value is not None else {})})

    for row in meta.to_dict("records"):
        c1, c2 = row["character_name"], row["character2_name"]
        has1, has2 = c1 != "blank", c2 != "blank"
        rid = row["sample_id"]
        if not (has1 or has2):
            stats["background_skipped"] += 1       # 纯背景：动物描述任务不适用（R3）
            continue
        two = has1 and has2
        if two and c1 == c2:
            stats["same_animal_filtered"] += 1
            continue
        if "scene" in factors:
            excl = {c for c in (c1, c2) if c != "blank"}
            cand = [a for a in animals if a not in excl]
            if not cand:
                stats["scene_no_candidate"] += 1
            else:
                other = cand[int(rng.integers(len(cand)))]
                pos, neg = spar_scene_contents(row, other)
                push("scene", "phrase", rid, pos, neg)
        if not two:
            stats["single_animal_rows"] += 1
            continue
        if "position" in factors:
            # 视角随机翻转（codex 审查 2026-10-28）：官方 character1_pos 只有
            # left/below，不翻转则正例方位词永远取不到 right/above，词偏好
            # 会污染准确率；翻转后正例方位词覆盖全部四个词，抽样再等额。
            flip = bool(rng.integers(2))
            pos, neg, pos_word = spar_position_contents(row, flip=flip)
            push("position", "sentence", rid, pos, neg, pos_value=pos_word)
        if "attribute" in factors:
            if require_diff_texture and row["character_texture"] == row["character2_texture"]:
                stats["attribute_same_texture_filtered"] += 1
            else:
                pos, neg = spar_attribute_contents(row)
                # 正负例含同一对纹理词（交换配对），词偏好天然无差别；
                # pos_value 取正例首个纹理，仅供逐侧报告。
                push("attribute", "phrase", rid, pos, neg,
                     pos_value=str(row["character_texture"]).lower())
        if "relation" in factors:
            pos, neg = spar_relation_contents(row)
            # 主客互换题两句共用方位词，词偏好天然无差别；pos_value 仅供报告
            push("relation", "sentence", rid, pos, neg,
                 pos_value=row["character1_pos"])
    return items, stats


# ---------------- run ----------------

def run(model, paths, logger, smoke=False, seed=None):
    cfg = load_config("e1_state")
    _validate_spar_protocol(cfg.get("spar", {}))   # 补充 B：不支持的选项显式拒绝
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name, protocol=model.cache_protocol)
    tcache = TextCache(paths["dirs"]["cache"], model.name, protocol=model.cache_protocol)
    eff_seed = cfg["seeds"][0] if seed is None else seed
    # 四轮问题 2：smoke 与全量输出是不同文件，来源键分开登记，smoke 运行
    # 不再更新全量结果的当前来源（E2 的 e2_<dataset>_smoke 约定一致）。
    experiment = "e1_state" + ("_smoke" if smoke else "")
    out_tables = output_tables_dir(paths, model.name, eff_seed,   # R2：按模型/种子隔离
                                   extra={"cache_protocol": model.cache_protocol},
                                   experiment=experiment)
    rng = np.random.default_rng(eff_seed)
    bs = cfg["bootstrap"]
    cap = cfg["sampling"].get("per_scene_cap", 1)
    target = min(200, cfg["sampling"]["target_per_cell"]) if smoke \
        else cfg["sampling"]["target_per_cell"]

    rows = []
    acc_store = {}       # (dataset, factor) -> (item_acc, image_id)，用于 factor gap CI
    pv_rows = []         # 逐正例取值的准确率（codex 审查 2026-10-28：暴露词偏好）

    def eval_factor_items(dataset, f, its):
        """共用流程：分层抽样 → 打分 → 逐题分数写回 item → 汇总行 + 逐侧行。"""
        m = eval_2afc_items(model, tcache, cache, key, its,
                            n_boot=bs["n_boot"], ci=bs["ci"],
                            clusters=[it["image_id"] for it in its])
        for it, ia, im in zip(its, m["item_acc"], m["item_margin"]):
            it["item_acc"] = float(ia)
            it["item_margin"] = float(im)
        rows.append({"dataset": dataset, "factor": f, "protocol": "2afc_self_built",
                     **{k: m[k] for k in EXPORT_FIELDS}})
        acc_store[(dataset, f)] = (m["item_acc"], [it["image_id"] for it in its])
        if its and its[0].get("pos_value") is not None:
            groups = {}
            for it, ia, im in zip(its, m["item_acc"], m["item_margin"]):
                g = groups.setdefault(it["pos_value"], ([], [], []))
                g[0].append(ia); g[1].append(im); g[2].append(it["image_id"])
            for v in sorted(groups):
                accs, mgs, clus = groups[v]
                acc, lo, hi = cluster_mean_ci(np.asarray(accs), clus,
                                              n_boot=bs["n_boot"], ci=bs["ci"], seed=0)
                mm, mlo, mhi = cluster_mean_ci(np.asarray(mgs), clus,
                                               n_boot=bs["n_boot"], ci=bs["ci"], seed=1)
                pv_rows.append({"dataset": dataset, "factor": f, "pos_value": v,
                                "n": len(accs), "accuracy": acc,
                                "acc_ci_low": lo, "acc_ci_high": hi,
                                "mean_margin": mm, "margin_ci_low": mlo,
                                "margin_ci_high": mhi})
        return m

    # ---- CLEVR val ----
    cl = cfg["clevr"]
    scenes = clevr.load_scenes(f"{paths['datasets']['clevr']}/{cl['scene_file']}")
    if smoke:
        scenes = scenes[:300]
        logger.info(f"[E1] clevr smoke subset: {len(scenes)} scenes")
    key = "clevr_val_smoke" if smoke else "clevr_val"
    extract_clevr_images(model, cache, key, scenes,
                         f"{paths['datasets']['clevr']}/{cl['image_dir']}", logger,
                         only_ids=[s["image_filename"] for s in scenes] if smoke else None)
    items_by_factor, clevr_stats = build_clevr_items(scenes, cl["factors"], rng)
    clevr_all_items = []
    for f in cl["factors"]:
        its = cap_per_image(items_by_factor[f], cap, rng)
        its = balanced_subsample(its, target, rng)
        if not its:
            logger.warning(f"[E1] clevr/{f}: no valid items")
            continue
        m = eval_factor_items("clevr", f, its)
        clevr_all_items.extend(its)
        dist = Counter(it.get("pos_value") for it in its if it.get("pos_value"))
        logger.info(f"[E1] clevr/{f}: n={m['n_samples']} acc={m['accuracy']:.4f} "
                    f"pos_value_dist={dict(sorted(dist.items()))}")
    items_path = out_tables / ("e1_items_clevr_smoke.jsonl" if smoke else "e1_items_clevr.jsonl")
    write_jsonl(items_path, clevr_all_items)
    logger.info(f"[E1] written {items_path} ({len(clevr_all_items)} items)")
    stats_path = out_tables / ("e1_clevr_filter_stats_smoke.json" if smoke
                               else "e1_clevr_filter_stats.json")
    stats_path.write_text(json.dumps(clevr_stats, indent=2, ensure_ascii=False))
    logger.info(f"[E1] written {stats_path} ({len(clevr_stats)} factors)")

    # ---- PUG: SPAR（自建 2AFC）----
    sp = cfg["spar"]
    meta = build_spar_metadata(paths["datasets"]["pug_spar"])
    if smoke:
        meta = meta.head(1500).reset_index(drop=True)
        logger.info(f"[E1] spar smoke subset: {len(meta)} rows")
    key = "pug_spar_smoke" if smoke else "pug_spar"
    extract_pug_images(model, cache, key, paths["datasets"]["pug_spar"], SPAR_META_COLUMNS,
                       logger, only_row_ids=set(meta["row_id"]) if smoke else None,
                       id_prefix="pug_spar")
    spar_items, spar_stats = build_spar_items(meta, sp["factors"], rng,
                                              sp["attribute_require_diff_texture"])
    spar_all_items = []
    for f in sp["factors"]:
        its = cap_per_image(spar_items[f], cap, rng)
        its = balanced_subsample(its, target, rng)
        if not its:
            logger.warning(f"[E1] spar/{f}: no valid items")
            continue
        m = eval_factor_items("pug_spar", f, its)
        spar_all_items.extend(its)
        dist = Counter(it.get("pos_value") for it in its if it.get("pos_value"))
        logger.info(f"[E1] spar/{f}: n={m['n_samples']} acc={m['accuracy']:.4f} "
                    f"pos_value_dist={dict(sorted(dist.items()))}")
    items_path = out_tables / ("e1_items_spar_smoke.jsonl" if smoke else "e1_items_spar.jsonl")
    write_jsonl(items_path, spar_all_items)
    logger.info(f"[E1] written {items_path} ({len(spar_all_items)} items)")
    stats_path = out_tables / ("e1_spar_filter_stats_smoke.json" if smoke
                               else "e1_spar_filter_stats.json")
    stats_path.write_text(json.dumps(spar_stats, indent=2, ensure_ascii=False))
    logger.info(f"[E1] written {stats_path} ({len(spar_stats)} factors)")

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
    # 逐正例取值准确率（codex 审查 2026-10-28）：词偏好会表现为一侧趋近
    # 100%、另一侧趋近 0%；有辨别无偏好的模型两侧接近总体值。
    pv_df = pd.DataFrame(pv_rows)
    path_pv = out_tables / ("e1_acc_by_pos_value_smoke.csv" if smoke
                            else "e1_acc_by_pos_value.csv")
    pv_df.to_csv(path_pv, index=False)
    logger.info(f"[E1] written {path_pv} ({len(pv_df)} rows)")
    # 四轮问题 1：全部输出成功写出后才提交成功来源（含产物清单）；
    # 中途失败时 last_success 保持旧值，本次尝试只留在 status=running 里。
    suffix = "_smoke" if smoke else ""
    commit_experiment(out_tables, experiment, [
        f"e1_items_clevr{suffix}.jsonl", f"e1_clevr_filter_stats{suffix}.json",
        f"e1_items_spar{suffix}.jsonl", f"e1_spar_filter_stats{suffix}.json",
        f"e1_results{suffix}.csv", f"e1_factor_gaps{suffix}.csv",
        f"e1_acc_by_pos_value{suffix}.csv"])
    return out
