"""E2: Single-Factor Representation Response Diagnosis（PUG: Animals，方案 §34–§45）。

- 配对：固定其余因素、只变一个因素；Δ 方向按 canonical 顺序（38）。
- 主分析单位 transition_key = (factor, source, target)（38/40）。
- 相似度分层：same_transition / same_factor_dt / diff_factor / random × context（41）。
- nuisance 基线：同图两个增强视图 + 同内容不同模板（39a）。
- 统计：Transition Similarity Gap 的聚类 bootstrap（66）。
"""
import numpy as np
import pandas as pd

from ..data.captions import pug_animals_caption, templates, wrap
from ..data.pair_matcher import build_factor_pairs
from ..data.pug_animals import META_COLUMNS as ANIMALS_META_COLUMNS
from ..data.pug_animals import build_metadata as build_animals_metadata
from ..data.pug_common import iter_image_batches
from ..features.cache import ImageFeatureCache, TextCache
from ..features.delta import augment_view, compute_deltas
from ..features.extract import extract_pug_images
from ..metrics.bootstrap import cluster_gap_ci, cluster_mean_ci
from ..metrics.delta_stats import build_delta_frame, group_similarities, transition_table
from ..utils.io import load_config


def _nuisance_visual(model, paths, meta, cfg, logger, smoke):
    n_cfg = cfg["nuisance"]
    n_img = min(200 if smoke else n_cfg["n_images"], len(meta))
    sel = meta.sample(n=n_img, random_state=0)
    sel_rows = set(sel["row_id"].tolist())
    smin, smax = n_cfg["crop_scale"]
    q1, q2 = n_cfg["jpeg_quality"]
    v1s, v2s = [], []
    for row_ids, images, _m in iter_image_batches(paths["datasets"]["pug_animals"],
                                                  ANIMALS_META_COLUMNS,
                                                  only_row_ids=sel_rows):
        for rid, img in zip(row_ids, images):
            v1s.append(augment_view(img, 100000 + rid, smin, smax, q1))
            v2s.append(augment_view(img, 200000 + rid, smin, smax, q2))
    logger.info(f"[E2] nuisance visual: {len(v1s)} images x 2 views")
    o1 = model.encode_images(v1s)
    o2 = model.encode_images(v2s)
    d = np.linalg.norm(o2["raw"] - o1["raw"], axis=1)
    return float(np.mean(d)), float(np.median(d))


def _nuisance_text(model, tcache, meta, cfg, logger, smoke):
    n_cfg = cfg["nuisance"]
    n_cap = min(200 if smoke else n_cfg["n_captions"], len(meta))
    sel = meta.sample(n=n_cap, random_state=1)
    caps = [pug_animals_caption(r) for r in sel.to_dict("records")]
    ph = templates()["phrase"]
    i0, i1 = n_cfg["text_template_indices"]
    e0, _ = tcache.encode(model, [wrap(c, ph[i0]) for c in caps])
    e1, _ = tcache.encode(model, [wrap(c, ph[i1]) for c in caps])
    d = np.linalg.norm(e0 - e1, axis=1)
    logger.info(f"[E2] nuisance text: {len(caps)} captions")
    return float(np.mean(d)), float(np.median(d))


def run(model, paths, logger, smoke=False):
    cfg = load_config("e2_change")
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name)
    tcache = TextCache(paths["dirs"]["cache"], model.name)
    out_tables = paths["dirs"]["outputs"] / "tables"

    meta = build_animals_metadata(paths["datasets"]["pug_animals"])
    if smoke:
        envs = sorted(meta["environment"].unique())[:6]
        anims = sorted(meta["animal"].unique())[:6]
        meta = meta[meta["environment"].isin(envs) & meta["animal"].isin(anims)].reset_index(drop=True)
        logger.info(f"[E2] smoke subset: {len(meta)} images")

    orders = cfg["canonical_orders"]
    samp = cfg["sampling"]
    max_total = 2000 if smoke else samp["max_total_per_factor"]
    pairs_by_factor = {}
    for f in cfg["factors"]:
        p = build_factor_pairs(meta, f, orders, samp["max_pairs_per_context"], max_total, seed=0)
        pairs_by_factor[f] = p
        logger.info(f"[E2] factor={f}: {len(p)} pairs, {p['context_key'].nunique()} contexts")

    delta_df = build_delta_frame(pairs_by_factor)
    needed_rows = sorted(set(delta_df["row_id_a"]) | set(delta_df["row_id_b"]))
    key = "pug_animals_smoke" if smoke else "pug_animals"
    extract_pug_images(model, cache, key, paths["datasets"]["pug_animals"],
                       ANIMALS_META_COLUMNS, logger,
                       only_row_ids=set(needed_rows) if smoke else None,
                       id_prefix="pug_animals")

    _ids, index, raw, norm = cache.arrays(key)
    id_of_row = dict(zip(meta["row_id"].astype(int), meta["sample_id"]))
    row_index = {}
    for r in needed_rows:
        sid = id_of_row[int(r)]
        if sid not in index:
            raise KeyError(f"{sid} missing in cache {key}")
        row_index[int(r)] = index[sid]
    delta_raw, delta_norm = compute_deltas(delta_df, row_index, raw, norm)

    nu_v_mean, nu_v_med = _nuisance_visual(model, paths, meta, cfg, logger, smoke)
    nu_t_mean, nu_t_med = _nuisance_text(model, tcache, meta, cfg, logger, smoke)
    logger.info(f"[E2] nuisance: visual={nu_v_mean:.4f}/{nu_v_med:.4f} "
                f"text={nu_t_mean:.4f}/{nu_t_med:.4f} (mean/median)")

    rng = np.random.default_rng(1)
    mc = cfg["similarity"]["max_comparisons_per_group"]
    n_boot, ci = cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["ci"]

    rows = []
    for f in cfg["factors"]:
        groups = group_similarities(delta_df, f, delta_norm, mc, rng)
        fmask = delta_df["factor"].to_numpy() == f
        d_raw_f = delta_raw[fmask]
        norms_f = np.linalg.norm(d_raw_f, axis=1)

        st = groups["same_transition_dc"]
        dfc = groups["diff_factor_dc"]
        st_mean, st_lo, st_hi = cluster_mean_ci(st["values"], st["clusters"],
                                                n_boot=n_boot, ci=ci, seed=0)
        gap, glo, ghi = cluster_gap_ci(st["values"], st["clusters"],
                                       dfc["values"], dfc["clusters"],
                                       n_boot=n_boot, ci=ci, seed=0)

        def gmean(name):
            v = groups[name]["values"]
            return float(np.mean(v)) if len(v) else np.nan

        rows.append({
            "dataset": "pug_animals",
            "factor": f,
            "n_pairs": int(fmask.sum()),
            "n_contexts": int(delta_df.loc[fmask, "context_key"].nunique()),
            "mean_delta_norm": float(np.mean(norms_f)),
            "median_delta_norm": float(np.median(norms_f)),
            "nuisance_norm_visual": nu_v_mean,
            "nuisance_norm_text": nu_t_mean,
            "same_transition_sim_diff_context": st_mean,
            "same_transition_ci_low": st_lo,
            "same_transition_ci_high": st_hi,
            "same_factor_dt_sc_sim": gmean("same_factor_dt_sc"),
            "same_factor_dt_dc_sim": gmean("same_factor_dt_dc"),
            "diff_factor_sc_sim": gmean("diff_factor_sc"),
            "diff_factor_dc_sim": gmean("diff_factor_dc"),
            "random_similarity": gmean("random"),
            "transition_similarity_gap": gap,
            "gap_ci_low": glo,
            "gap_ci_high": ghi,
            "factor_over_nuisance_ratio": float(np.mean(norms_f)) / max(nu_v_mean, 1e-8),
        })
        logger.info(
            f"[E2] {f}: |Δ|={np.mean(norms_f):.3f} "
            f"same_tr={st_mean:.4f} diff_fac={gmean('diff_factor_dc'):.4f} "
            f"gap={gap:.4f} [{glo:.4f}, {ghi:.4f}]")

    out = pd.DataFrame(rows)
    path = out_tables / ("e2_delta_stats_smoke.csv" if smoke else "e2_delta_stats.csv")
    out.to_csv(path, index=False)
    logger.info(f"[E2] written {path}")

    # 逐 transition 表（含平均 Delta 范数）
    tt = transition_table(delta_df, delta_norm,
                          max_pairs_per_transition=2000, rng=np.random.default_rng(2))
    raw_norms = np.linalg.norm(delta_raw, axis=1)
    tmp = delta_df.assign(delta_norm=raw_norms).groupby(
        ["factor", "source_value", "target_value"])["delta_norm"].mean().reset_index()
    tt = tt.merge(tmp, on=["factor", "source_value", "target_value"], how="left")
    tt = tt.rename(columns={"delta_norm": "mean_delta_norm"})
    path_t = out_tables / ("e2_transition_stats_smoke.csv" if smoke else "e2_transition_stats.csv")
    tt.to_csv(path_t, index=False)
    logger.info(f"[E2] written {path_t} ({len(tt)} transitions)")
    return out
