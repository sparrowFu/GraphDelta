"""E2 分层相似度分析（方案 §40 / §41）。

组定义（对因素 f：d1 恒取自 f，d2 按组取）：
- same_transition_dc ：同 transition，不同 context（同 transition 同 context 不存在）
- same_factor_dt_sc  ：同因素、不同 transition、同 context
- same_factor_dt_dc  ：同因素、不同 transition、不同 context
- diff_factor_sc/dc  ：d2 来自其他因素，按 (animal, environment) 是否相同分桶
- random             ：d2 来自全部因素，cluster 不同

每次比较记录 (cos 值, d1 的 cluster)，用于聚类 bootstrap。
"""
import numpy as np
import pandas as pd


def build_delta_frame(pairs_by_factor: dict) -> pd.DataFrame:
    frames = []
    for factor, pdf in pairs_by_factor.items():
        p = pdf.copy()
        p["transition_key"] = p["source_value"].astype(str) + "->" + p["target_value"].astype(str)
        frames.append(p)
    out = pd.concat(frames, ignore_index=True)
    out["delta_index"] = np.arange(len(out))
    return out


def _cos(D, gi, gj):
    a, b = D[gi], D[gj]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.maximum(na * nb, 1e-8)


def group_similarities(delta_df: pd.DataFrame, factor: str, delta_norm: np.ndarray,
                       max_comparisons: int, rng) -> dict:
    sub = delta_df[delta_df["factor"] == factor].reset_index(drop=True)
    f_pos = np.arange(len(sub))
    f_gidx = sub["delta_index"].to_numpy()
    f_cluster = sub["ae_key"].to_numpy()
    f_context = sub["context_key"].to_numpy()

    all_gidx = delta_df["delta_index"].to_numpy()
    all_factor = delta_df["factor"].to_numpy()
    all_cluster = delta_df["ae_key"].to_numpy()
    other_pos = np.where(all_factor != factor)[0]

    D = delta_norm
    out = {}

    def record(name, pos_i, pos_j_other_global):
        """pos_i: sub 内位置；pos_j_other_global: 全局 delta 行号。"""
        pos_i = np.asarray(pos_i, dtype=np.int64)
        gj = np.asarray(pos_j_other_global, dtype=np.int64)
        if len(pos_i) == 0:
            out[name] = {"values": np.zeros(0), "clusters": np.zeros(0)}
            return
        vals = _cos(D, f_gidx[pos_i], gj)
        out[name] = {"values": vals, "clusters": f_cluster[pos_i]}

    n_try = max_comparisons * 3

    # --- same transition, different context（d2 也在 f 内）---
    trans_groups = sub.groupby("transition_key").indices
    multi = [t for t, pos in trans_groups.items() if len(pos) >= 2]
    per = max(32, int(np.ceil(max_comparisons / max(len(multi), 1))))
    i_list, j_list = [], []
    for t, pos in trans_groups.items():
        m = len(pos)
        if m < 2:
            continue
        k = min(per, m * (m - 1) // 2)
        a = rng.choice(pos, size=k)
        b = rng.choice(pos, size=k)
        keep = a != b
        i_list.append(a[keep])
        j_list.append(b[keep])
    if i_list:
        pi, pj = np.concatenate(i_list), np.concatenate(j_list)
    else:
        pi, pj = [], []
    record("same_transition_dc", pi, f_gidx[pj] if len(pj) else np.zeros(0, dtype=np.int64))

    # --- same factor, different transition, same context ---
    ctx_groups = sub.groupby("context_key").indices
    n_multi = sum(1 for p in ctx_groups.values() if len(p) >= 2)
    per = max(32, int(np.ceil(max_comparisons / max(n_multi, 1))))
    i_list, j_list = [], []
    for ctx, pos in ctx_groups.items():
        m = len(pos)
        if m < 2:
            continue
        k = min(per, m * (m - 1) // 2)
        a = rng.choice(pos, size=k)
        b = rng.choice(pos, size=k)
        keep = a != b
        i_list.append(a[keep])
        j_list.append(b[keep])
    if i_list:
        pi, pj = np.concatenate(i_list), np.concatenate(j_list)
    else:
        pi, pj = [], []
    record("same_factor_dt_sc", pi, f_gidx[pj] if len(pj) else np.zeros(0, dtype=np.int64))

    # --- same factor, different transition, different context ---
    a = rng.choice(f_pos, size=n_try)
    b = rng.choice(f_pos, size=n_try)
    keep = f_context[a] != f_context[b]
    record("same_factor_dt_dc", a[keep][:max_comparisons], f_gidx[b[keep][:max_comparisons]])

    # --- different factor（同 / 不同 (animal, environment)）---
    if len(other_pos):
        a = rng.choice(f_pos, size=n_try)
        bj = rng.choice(other_pos, size=n_try)
        same_ae = f_cluster[a] == all_cluster[bj]
        record("diff_factor_sc", a[same_ae][:max_comparisons], all_gidx[bj[same_ae][:max_comparisons]])
        record("diff_factor_dc", a[~same_ae][:max_comparisons], all_gidx[bj[~same_ae][:max_comparisons]])
    else:
        record("diff_factor_sc", [], [])
        record("diff_factor_dc", [], [])

    # --- random（d2 来自全部因素，cluster 不同）---
    a = rng.choice(f_pos, size=n_try)
    bj = rng.choice(np.arange(len(all_gidx)), size=n_try)
    keep = f_cluster[a] != all_cluster[bj]
    record("random", a[keep][:max_comparisons], all_gidx[bj[keep][:max_comparisons]])
    return out


def transition_table(delta_df: pd.DataFrame, delta_norm: np.ndarray,
                     max_pairs_per_transition: int, rng) -> pd.DataFrame:
    """逐 transition 汇总：n_contexts、跨 context 相似度均值、平均 Delta 范数（raw）。"""
    rows = []
    for (factor, src, tgt), g in delta_df.groupby(["factor", "source_value", "target_value"]):
        gidx = g["delta_index"].to_numpy()
        n_ctx = g["context_key"].nunique()
        sims = []
        m = len(gidx)
        if m >= 2:
            k = min(max_pairs_per_transition, m * (m - 1) // 2)
            a = rng.choice(gidx, size=k)
            b = rng.choice(gidx, size=k)
            keep = a != b
            sims = _cos(delta_norm, a[keep], b[keep])
        rows.append({
            "factor": factor,
            "source_value": src,
            "target_value": tgt,
            "n_contexts": n_ctx,
            "mean_sim": float(np.mean(sims)) if len(sims) else np.nan,
        })
    return pd.DataFrame(rows)
