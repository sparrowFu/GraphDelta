"""E2: Single-Factor Representation Response Diagnosis（PUG: Animals / ImageNet，方案 §34–§45）。

- 数据集：run(dataset=...) 选择 pug_animals（主）或 pug_imagenet（辅助，方案 §14、
  审查 A1）；配对、特征抽取、nuisance、输出标签全部按数据集分发。
- 配对：固定其余因素、只变一个因素；Δ 方向按 canonical 顺序（38）。
- 主分析单位 transition_key = (factor, source, target)（38/40）。
- 相似度分层：same_transition / same_factor_dt / diff_factor / random × context（41）。
- 指标同时输出 raw（projection 后、归一化前）与 normalized 两个特征空间，
  列名后缀 _raw / _z（审查报告 P2-8；方案 §64 的无后缀字段对应 _z 列）。
- nuisance 基线：同图两个增强视图 + 同内容不同模板（39a）；除范数外，
  增加 nuisance delta 两两相似度与 Factor/Nuisance Norm Gap 及其 CI。
- 统计：Transition Similarity Gap 的同步聚类 bootstrap（66）。
- 各相似度分组的实际比较数写入 n_cmp_* 列，不足预算时告警（审查 §3）。
"""
import numpy as np
import pandas as pd

from ..data.captions import pug_animals_caption, pug_imagenet_caption, templates, wrap
from ..data.pair_matcher import META_COLS as ANIMALS_PAIR_COLS
from ..data.pair_matcher import build_factor_pairs
from ..data.pug_animals import META_COLUMNS as ANIMALS_META_COLUMNS
from ..data.pug_animals import build_metadata as build_animals_metadata
from ..data.pug_imagenet import META_COLS as IMAGENET_PAIR_COLS
from ..data.pug_imagenet import META_COLUMNS as IMAGENET_META_COLUMNS
from ..data.pug_imagenet import build_metadata as build_imagenet_metadata
from ..data.pug_common import iter_image_batches
from ..features.cache import ImageFeatureCache, TextCache
from ..features.delta import augment_view, compute_deltas
from ..features.extract import extract_pug_images
from ..metrics.bootstrap import cluster_gap_ci, pair_cluster_gap_ci, pair_cluster_mean_ci
from ..metrics.delta_stats import build_delta_frame, group_similarities, transition_table
from ..utils.io import load_config, output_tables_dir

# 数据集注册表（审查 A1）：E2 除 Animals 外接入 PUG:ImageNet。
# raw_meta_columns 为 parquet 原始列（迭代图像用），meta_cols 为配对语义列。
# smoke_cols：smoke 子集的筛选字段（显式声明，不依赖 meta_cols 位置——
# 三轮审查问题 1：ImageNet 前两列（实例/类别）独立取前 6 再求交可能筛成空集）。
DATASETS = {
    "pug_animals": {
        "path_key": "pug_animals",
        "build_meta": build_animals_metadata,
        "raw_meta_columns": ANIMALS_META_COLUMNS,
        "meta_cols": ANIMALS_PAIR_COLS,
        "smoke_cols": ["animal", "environment"],
        "caption": pug_animals_caption,
    },
    "pug_imagenet": {
        "path_key": "pug_imagenet",
        "build_meta": build_imagenet_metadata,
        "raw_meta_columns": IMAGENET_META_COLUMNS,
        "meta_cols": IMAGENET_PAIR_COLS,
        "smoke_cols": ["character_name", "environment"],   # 先选实例，再在其子集内选环境
        "caption": pug_imagenet_caption,
    },
}


def _rowwise_cos(A, B):
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    return (A * B).sum(axis=1) / np.maximum(na * nb, 1e-8)


def _delta_pair_sim(D, rng, k=2000):
    """nuisance delta 之间的两两余弦相似度均值（直接抽不相等的索引对）。"""
    n = len(D)
    if n < 2:
        return np.nan
    kk = min(k, n * (n - 1) // 2)
    ia = rng.integers(0, n, size=kk)
    off = rng.integers(1, n, size=kk)
    ib = (ia + off) % n
    return float(np.mean(_rowwise_cos(D[ia], D[ib])))


def _nuisance_visual(model, paths, meta, cfg, logger, smoke, seed, spec):
    """同图两个增强视图的 Δ，返回 raw/z 空间的 delta 矩阵与每条的图像 id。

    ids 按解码顺序追加（iter_image_batches 按全局 row_id 顺序产出），与 delta
    行一一对应（五轮 P3）；不能改用 sel 的抽样顺序。
    分批生成/编码/释放（复核 R2）：不同时保留全部增强图像，峰值内存与批大小
    成比例；每图增强种子固定，分批不改变任何数值结果。
    """
    n_cfg = cfg["nuisance"]
    n_img = min(200 if smoke else n_cfg["n_images"], len(meta))
    sel = meta.sample(n=n_img, random_state=seed)
    row_to_sid = dict(zip(sel["row_id"].astype(int), sel["sample_id"]))
    sel_rows = set(sel["row_id"].tolist())
    smin, smax = n_cfg["crop_scale"]
    q1, q2 = n_cfg["jpeg_quality"]
    chunk = min(128, max(1, int(getattr(model, "bs_img", 128))))
    v1s, v2s, ids = [], [], []
    d_raw, d_z = [], []

    def flush():
        nonlocal v1s, v2s
        if not v1s:
            return
        o1 = model.encode_images(v1s)
        o2 = model.encode_images(v2s)
        d_raw.append(o2["raw"] - o1["raw"])
        d_z.append(o2["normalized"] - o1["normalized"])
        v1s, v2s = [], []

    for row_ids, images, _m in iter_image_batches(paths["datasets"][spec["path_key"]],
                                                  spec["raw_meta_columns"],
                                                  only_row_ids=sel_rows):
        for rid, img in zip(row_ids, images):
            v1s.append(augment_view(img, 100000 + rid, smin, smax, q1))
            v2s.append(augment_view(img, 200000 + rid, smin, smax, q2))
            ids.append(row_to_sid[int(rid)])
            if len(v1s) >= chunk:
                flush()
    flush()
    logger.info(f"[E2] nuisance visual: {len(ids)} images x 2 views (chunk={chunk})")
    return {"raw": np.concatenate(d_raw, axis=0), "z": np.concatenate(d_z, axis=0),
            "ids": ids}


def _nuisance_text(model, tcache, meta, cfg, logger, smoke, seed, spec):
    """同内容不同模板的文本 Δ，返回 raw 与 normalized 两个空间的 delta 矩阵。"""
    n_cfg = cfg["nuisance"]
    n_cap = min(200 if smoke else n_cfg["n_captions"], len(meta))
    sel = meta.sample(n=n_cap, random_state=seed + 1)
    caps = [spec["caption"](r) for r in sel.to_dict("records")]
    ph = templates()["phrase"]
    i0, i1 = n_cfg["text_template_indices"]
    e0r, e0z = tcache.encode(model, [wrap(c, ph[i0]) for c in caps])
    e1r, e1z = tcache.encode(model, [wrap(c, ph[i1]) for c in caps])
    logger.info(f"[E2] nuisance text: {len(caps)} captions")
    return {"raw": e0r - e1r, "z": e0z - e1z}


def run(model, paths, logger, smoke=False, seed=None, dataset="pug_animals",
        allow_empty_factors=False):
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}，可选：{sorted(DATASETS)}")
    spec = DATASETS[dataset]
    cfg = load_config("e2_change")
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name, protocol=model.cache_protocol)
    tcache = TextCache(paths["dirs"]["cache"], model.name, protocol=model.cache_protocol)
    pair_seed = cfg.get("seeds", [0])[0] if seed is None else seed
    out_tables = output_tables_dir(  # R2：按模型/种子隔离
        paths, model.name, pair_seed,
        extra={"cache_protocol": model.cache_protocol},
        experiment=f"e2_{dataset}" + ("_smoke" if smoke else ""))

    meta = spec["build_meta"](paths["datasets"][spec["path_key"]])
    if smoke:
        # 三轮审查问题 1：筛选字段按数据集显式声明（smoke_cols），并**逐层**
        # 筛选（后一列在已选子集内取前 6 个值）——独立取前 6 再求交可能
        # 把有配对的数据筛成空集；筛选后为空属于配置问题，显式报错。
        sub = meta
        for c in spec["smoke_cols"]:
            vals = sorted(sub[c].astype(str).unique())[:6]
            sub = sub[sub[c].astype(str).isin(vals)]
        meta = sub.reset_index(drop=True)
        if len(meta) == 0:
            raise SystemExit(
                f"[E2] {dataset} smoke 筛选后 0 行（smoke_cols={spec['smoke_cols']}），"
                f"请检查 smoke_cols 配置与数据取值分布。")
        logger.info(f"[E2] {dataset} smoke subset: {len(meta)} rows"
                    f" (smoke_cols={spec['smoke_cols']})")

    # canonical 顺序按数据集隔离（Animals 声明标签顺序；ImageNet 按观测值排序）
    orders = cfg["datasets"][dataset].get("canonical_orders")
    factors = cfg["datasets"][dataset]["factors"]
    samp = cfg["sampling"]
    max_total = 2000 if smoke else samp["max_total_per_factor"]
    pairs_by_factor = {}
    for f in factors:
        p = build_factor_pairs(meta, f, orders, samp["max_pairs_per_context"], max_total,
                               seed=pair_seed, meta_cols=spec["meta_cols"])
        if len(p) == 0:
            # 复核 B2：空因素此前在日志访问 p['context_key'] 时抛 KeyError；
            # 现在默认带原因停止，局部诊断可显式跳过。三轮审查问题 1：
            # smoke 模式需区分"smoke 子集筛掉了覆盖"与"源数据无配对"。
            msg = (f"[E2] {dataset}/factor={f}: 0 对配对——当前 context 定义下该"
                   f"因素没有任何单因素变化对，请检查数据覆盖与因素取值")
            if smoke:
                msg += "（注意：smoke 子集可能筛掉了该因素的覆盖，完整运行无此筛选）"
            if allow_empty_factors:
                logger.warning(msg + "；已跳过（--allow-empty-factors），"
                               "本次结果不构成完整因素集")
                continue
            raise SystemExit(msg + "；只需局部诊断时加 --allow-empty-factors")
        pairs_by_factor[f] = p
        # 复核 B1：context 已含 character_name（实例），披露实例覆盖数
        n_inst = (f", instances={p['character_name'].nunique()}"
                  if "character_name" in p.columns else "")
        logger.info(f"[E2] factor={f}: {len(p)} pairs, "
                    f"{p['context_key'].nunique()} contexts{n_inst}")
    if not pairs_by_factor:
        # 三轮审查问题 4：全部因素为空时给明确停止消息，不让空 dict 进入
        # build_delta_frame（此前抛 ValueError: No objects to concatenate）
        raise SystemExit(
            f"[E2] {dataset}: 全部因素均无有效配对，停止（--allow-empty-factors "
            f"只跳过个别空因素，不适用于全空输入；请检查数据或配对协议）。")
    factors = list(pairs_by_factor)   # B2：被跳过的因素不进入后续统计

    delta_df = build_delta_frame(pairs_by_factor)
    needed_rows = sorted(set(delta_df["row_id_a"]) | set(delta_df["row_id_b"]))
    key = f"{dataset}_smoke" if smoke else dataset
    extract_pug_images(model, cache, key, paths["datasets"][spec["path_key"]],
                       spec["raw_meta_columns"], logger,
                       only_row_ids=set(needed_rows) if smoke else None,
                       id_prefix=dataset)

    _ids, index, raw, norm = cache.arrays(key)
    id_of_row = dict(zip(meta["row_id"].astype(int), meta["sample_id"]))
    row_index = {}
    for r in needed_rows:
        sid = id_of_row[int(r)]
        if sid not in index:
            raise KeyError(f"{sid} missing in cache {key}")
        row_index[int(r)] = index[sid]
    delta_raw, delta_norm = compute_deltas(delta_df, row_index, raw, norm)

    # 端点特征的对齐视图（before/after cosine 用）
    ia = delta_df["row_id_a"].map(lambda r: row_index[int(r)]).to_numpy()
    ib = delta_df["row_id_b"].map(lambda r: row_index[int(r)]).to_numpy()
    ba_cos = {"raw": _rowwise_cos(raw[ia].astype(np.float32), raw[ib].astype(np.float32)),
              "z": _rowwise_cos(norm[ia].astype(np.float32), norm[ib].astype(np.float32))}
    ba_cos_by_factor = {f: {s: float(np.mean(ba_cos[s][
        (delta_df["factor"].to_numpy() == f)])) for s in ("raw", "z")}
        for f in factors}

    # nuisance（raw / z 两空间）
    nu_v = _nuisance_visual(model, paths, meta, cfg, logger, smoke, pair_seed, spec)
    nu_t = _nuisance_text(model, tcache, meta, cfg, logger, smoke, pair_seed, spec)
    nu_rng = np.random.default_rng(pair_seed + 100)
    nu_stats = {
        "norm_visual_raw": float(np.mean(np.linalg.norm(nu_v["raw"], axis=1))),
        "norm_visual_z": float(np.mean(np.linalg.norm(nu_v["z"], axis=1))),
        "norm_text_raw": float(np.mean(np.linalg.norm(nu_t["raw"], axis=1))),
        "norm_text_z": float(np.mean(np.linalg.norm(nu_t["z"], axis=1))),
        "sim_visual_raw": _delta_pair_sim(nu_v["raw"], nu_rng),
        "sim_visual_z": _delta_pair_sim(nu_v["z"], nu_rng),
        "sim_text_raw": _delta_pair_sim(nu_t["raw"], nu_rng),
        "sim_text_z": _delta_pair_sim(nu_t["z"], nu_rng),
    }
    nu_v_ids = nu_v["ids"]
    logger.info("[E2] nuisance: " + " ".join(f"{k}={v:.4f}" for k, v in nu_stats.items()))

    mc = cfg["similarity"]["max_comparisons_per_group"]
    n_boot, ci = cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["ci"]

    rows = []
    for f in factors:
        # 同一 rng 种子 + 相同调用序列 → raw 与 z 使用完全相同的比较对
        groups = {s: group_similarities(delta_df, f, d, mc, np.random.default_rng(1000),
                                        meta_cols=spec["meta_cols"])
                  for s, d in (("raw", delta_raw), ("z", delta_norm))}
        fmask = delta_df["factor"].to_numpy() == f
        norms_f = {"raw": np.linalg.norm(delta_raw[fmask], axis=1),
                   "z": np.linalg.norm(delta_norm[fmask], axis=1)}

        # 实际比较数披露（审查 A7/§3）：raw 与 z 的比较对相同，取 z 侧计数
        n_cmp = {g: int(len(groups["z"][g]["values"]))
                 for g in ("same_transition_dc", "same_factor_dt_sc",
                           "same_factor_dt_dc", "diff_factor_sc",
                           "diff_factor_dc", "random")}
        for g, n in n_cmp.items():
            if n < mc:
                logger.warning(f"[E2] {dataset}/{f}/{g}: 仅采样到 {n}/{mc} 条比较"
                               f"（候选不足或 same-context 匹配稀疏）")

        row = {
            "dataset": dataset,
            "factor": f,
            "n_pairs": int(fmask.sum()),
            "n_contexts": int(delta_df.loc[fmask, "context_key"].nunique()),
            "mean_delta_norm_raw": float(np.mean(norms_f["raw"])),
            "median_delta_norm_raw": float(np.median(norms_f["raw"])),
            "mean_delta_norm_z": float(np.mean(norms_f["z"])),
            "median_delta_norm_z": float(np.median(norms_f["z"])),
            "before_after_cos_raw": ba_cos_by_factor[f]["raw"],
            "before_after_cos_z": ba_cos_by_factor[f]["z"],
        }
        row.update({f"n_cmp_{g}": n for g, n in n_cmp.items()})

        for s in ("raw", "z"):
            g = groups[s]
            st, dfc = g["same_transition_dc"], g["diff_factor_dc"]
            # 双端点 context 聚类 bootstrap：比较值依赖两个端点，权重 = 两端点
            # 被抽中次数之积，与端点登记顺序无关（二轮审查问题 3）
            st_mean, st_lo, st_hi = pair_cluster_mean_ci(
                st["values"], st["ctx_a"], st["ctx_b"], n_boot=n_boot, ci=ci, seed=0)
            gap, glo, ghi = pair_cluster_gap_ci(
                st["values"], st["ctx_a"], st["ctx_b"],
                dfc["values"], dfc["ctx_a"], dfc["ctx_b"],
                n_boot=n_boot, ci=ci, seed=0)
            src_counts = dfc.get("source_counts", {})
            sc_counts = g["diff_factor_sc"].get("source_counts", {})

            def gmean(name):
                v = g[name]["values"]
                return float(np.mean(v)) if len(v) else np.nan

            suffix = "" if s == "z" else "_raw"
            row.update({
                f"same_transition_sim_diff_context{suffix}": st_mean,
                f"same_transition_ci_low{suffix}": st_lo,
                f"same_transition_ci_high{suffix}": st_hi,
                f"same_factor_dt_sc_sim{suffix}": gmean("same_factor_dt_sc"),
                f"same_factor_dt_dc_sim{suffix}": gmean("same_factor_dt_dc"),
                f"diff_factor_sc_sim{suffix}": gmean("diff_factor_sc"),
                f"diff_factor_dc_sim{suffix}": gmean("diff_factor_dc"),
                f"random_similarity{suffix}": gmean("random"),
            })
            if s == "z":
                row.update({"transition_similarity_gap": gap,
                            "gap_ci_low": glo, "gap_ci_high": ghi})
            else:
                row.update({"transition_similarity_gap_raw": gap,
                            "gap_raw_ci_low": glo, "gap_raw_ci_high": ghi})

        # Factor / Nuisance Norm Gap（visual nuisance，raw 与 z）：
        # 因素端按 context 聚类、nuisance 端按原始图像单位，同步重采样
        # （二轮审查问题 4：相关 pair 不再当独立样本）
        for s in ("raw", "z"):
            ng, nlo, nhi = cluster_gap_ci(
                norms_f[s], delta_df.loc[fmask, "context_key"].to_numpy(),
                np.linalg.norm(nu_v[s], axis=1), np.asarray(nu_v_ids),
                n_boot=n_boot, ci=ci, seed=0)
            key_gap = "norm_gap_visual" if s == "z" else "norm_gap_visual_raw"
            row[key_gap] = ng
            row[f"{key_gap}_ci_low"] = nlo
            row[f"{key_gap}_ci_high"] = nhi

        row.update({
            "nuisance_norm_visual_raw": nu_stats["norm_visual_raw"],
            "nuisance_norm_visual_z": nu_stats["norm_visual_z"],
            "nuisance_norm_text_raw": nu_stats["norm_text_raw"],
            "nuisance_norm_text_z": nu_stats["norm_text_z"],
            "nuisance_sim_visual_raw": nu_stats["sim_visual_raw"],
            "nuisance_sim_visual_z": nu_stats["sim_visual_z"],
            "nuisance_sim_text_raw": nu_stats["sim_text_raw"],
            "nuisance_sim_text_z": nu_stats["sim_text_z"],
            "factor_over_nuisance_ratio": float(np.mean(norms_f["raw"]))
            / max(nu_stats["norm_visual_raw"], 1e-8),
        })
        rows.append(row)
        logger.info(
            f"[E2] {dataset}/{f}: |Δ|raw={row['mean_delta_norm_raw']:.3f} "
            f"same_tr_z={row['same_transition_sim_diff_context']:.4f} "
            f"gap_z={row['transition_similarity_gap']:.4f} "
            f"[{row['gap_ci_low']:.4f}, {row['gap_ci_high']:.4f}] "
            f"diff_factor_src={src_counts} diff_factor_sc_src={sc_counts}")

    out = pd.DataFrame(rows)
    # Animals 沿用原文件名（下游 summarize/make_figures 兼容）；ImageNet 加数据集后缀
    ds_suffix = "" if dataset == "pug_animals" else f"_{dataset}"
    path = out_tables / (f"e2_delta_stats{ds_suffix}_smoke.csv" if smoke
                         else f"e2_delta_stats{ds_suffix}.csv")
    out.to_csv(path, index=False)
    logger.info(f"[E2] written {path} ({len(out)} rows)")

    # 逐 transition 表（两个空间的 Δ 范数 + 相似度）
    tt = transition_table(delta_df, delta_norm,
                          max_pairs_per_transition=2000, rng=np.random.default_rng(2))
    tt = tt.rename(columns={"mean_sim": "mean_sim_z"})
    tt_raw = transition_table(delta_df, delta_raw,
                              max_pairs_per_transition=2000, rng=np.random.default_rng(2))
    tt = tt.merge(tt_raw.rename(columns={"mean_sim": "mean_sim_raw"}),
                  on=["factor", "source_value", "target_value", "n_contexts"], how="left")
    raw_norms = np.linalg.norm(delta_raw, axis=1)
    z_norms = np.linalg.norm(delta_norm, axis=1)
    tmp = delta_df.assign(n_raw=raw_norms, n_z=z_norms).groupby(
        ["factor", "source_value", "target_value"])[["n_raw", "n_z"]].mean().reset_index()
    tt = tt.merge(tmp, on=["factor", "source_value", "target_value"], how="left")
    tt = tt.rename(columns={"n_raw": "mean_delta_norm_raw", "n_z": "mean_delta_norm_z"})
    path_t = out_tables / (f"e2_transition_stats{ds_suffix}_smoke.csv" if smoke
                           else f"e2_transition_stats{ds_suffix}.csv")
    tt.to_csv(path_t, index=False)
    low = tt[tt["n_contexts"] < 2]
    logger.info(f"[E2] written {path_t} ({len(tt)} transitions; "
                f"{len(low)} with n_contexts<2 → mean_sim=NaN)")
    return out
