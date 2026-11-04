"""E2 分层相似度分析（方案 §40 / §41 / §66）。

组定义（对因素 f：d1 恒取自 f，d2 按组取）：
- same_transition_dc ：同 transition，不同 context
- same_factor_dt_sc  ：同因素、不同 transition、同 context
- same_factor_dt_dc  ：同因素、不同 transition、不同 context
- diff_factor_sc/dc  ：d2 从**全部其他因素的联合集合**统一抽样（不按因素顺序
  拼接后截头，二轮审查问题 1）；same/different context 按 f、g 共同固定的
  metadata 字段（FIXED_COLS[f] ∩ FIXED_COLS[g]）是否全部相等判定
- random             ：d2 来自全部因素，排除同一 delta 与（同因素 & 同 context）

每条比较记录两个端点的 context（ctx_a / ctx_b），供双端点聚类 bootstrap
（二轮审查问题 3）。聚类单位：完整 metadata 组合（context_key）。
比较采样直接抽取不相等的索引对，不会把有效比较删空（一轮 P2-7）。
每组比较施加 max_comparisons 全局上限（审查 A7）；实际比较数由调用方
（e2_change_diagnosis）写入结果 CSV 并在不足时告警（审查 §3）。
"""
import numpy as np
import pandas as pd

from ..data.pair_matcher import META_COLS as PAIR_META_COLS


def _subsample_cap(rng, i, j, cap):
    """全局上限（审查 A7）：拼接后的比较数超过 cap 时无放回抽 cap 条。"""
    if len(i) > cap:
        keep = rng.choice(len(i), size=cap, replace=False)
        return i[keep], j[keep]
    return i, j


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


def _distinct_pairs(rng, m, k):
    """抽取 k 对互不相等的局部索引（a != b），要求 k <= m*(m-1)//2。"""
    ia = rng.integers(0, m, size=k)
    off = rng.integers(1, m, size=k)
    return ia, (ia + off) % m


def group_similarities(delta_df: pd.DataFrame, factor: str, delta_mat: np.ndarray,
                       max_comparisons: int, rng, meta_cols=None) -> dict:
    """meta_cols：该数据集的语义列（默认 Animals 的 META_COLS，审查 A1 参数化）。

    每组比较施加 max_comparisons **全局上限**（审查 A7：此前 same_transition_dc
    / same_factor_dt_sc 按分组数×每组下限拼接，实际比较数可远超预算）。
    """
    if meta_cols is None:
        meta_cols = PAIR_META_COLS
    sub = delta_df[delta_df["factor"] == factor].reset_index(drop=True)
    f_gidx = sub["delta_index"].to_numpy()
    f_context = sub["context_key"].to_numpy()
    f_trans = sub["transition_key"].to_numpy()

    all_gidx = delta_df["delta_index"].to_numpy()
    all_factor = delta_df["factor"].to_numpy()
    all_context = delta_df["context_key"].to_numpy()

    D = delta_mat
    out = {}

    def record(name, pos_i, pos_j_other_global, extra=None):
        """pos_i: sub 内位置；pos_j_other_global: 全局 delta 行号。

        返回 values 与两个端点的 context（ctx_a = d1，ctx_b = d2）。
        n_unique_pairs 另披露**无序**独特比较对数（四轮 §5 建议 + 复核说明 2）：
        余弦相似度对称，(d1,d2) 与 (d2,d1) 是同一条比较，按无序对计数；
        _distinct_pairs 等抽样只保证单条比较内两索引不同，比较之间有放回，
        n_cmp 是抽样次数，不等于独特比较数。
        """
        pos_i = np.asarray(pos_i, dtype=np.int64)
        gj = np.asarray(pos_j_other_global, dtype=np.int64)
        entry = {"values": np.zeros(0), "ctx_a": np.zeros(0), "ctx_b": np.zeros(0),
                 "n_unique_pairs": 0}
        if extra:
            entry.update(extra)
        if len(pos_i):
            entry["values"] = _cos(D, f_gidx[pos_i], gj)
            entry["ctx_a"] = f_context[pos_i]
            entry["ctx_b"] = all_context[gj]
            entry["n_unique_pairs"] = int(len({tuple(sorted((int(a), int(b))))
                                               for a, b in zip(f_gidx[pos_i], gj)}))
        out[name] = entry

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
        ia, ib = _distinct_pairs(rng, m, k)
        i_list.append(pos[ia])
        j_list.append(pos[ib])
    if i_list:
        pi, pj = np.concatenate(i_list), np.concatenate(j_list)
        pi, pj = _subsample_cap(rng, pi, pj, max_comparisons)   # A7 全局上限
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
        ia, ib = _distinct_pairs(rng, m, k)
        i_list.append(pos[ia])
        j_list.append(pos[ib])
    if i_list:
        pi, pj = np.concatenate(i_list), np.concatenate(j_list)
        pi, pj = _subsample_cap(rng, pi, pj, max_comparisons)   # A7 全局上限
    else:
        pi, pj = [], []
    record("same_factor_dt_sc", pi, f_gidx[pj] if len(pj) else np.zeros(0, dtype=np.int64))

    # --- same factor, different transition, different context ---
    a = rng.integers(0, len(sub), size=n_try)
    b = rng.integers(0, len(sub), size=n_try)
    keep = (f_context[a] != f_context[b]) & (f_trans[a] != f_trans[b])
    record("same_factor_dt_dc", a[keep][:max_comparisons], f_gidx[b[keep][:max_comparisons]])

    # --- different factor：从其他因素联合集合统一抽样（二轮问题 1）---
    # same-context 匹配在完整数据中很稀疏（§3），拒绝采样尝试次数提高到 20×上限
    n_try_sc = max_comparisons * 20
    f_meta = {c: sub[c].to_numpy() for c in meta_cols}
    other_meta = {c: delta_df[c].to_numpy() for c in meta_cols}
    other_pos = np.where(all_factor != factor)[0]
    if len(other_pos):
        a = rng.integers(0, len(sub), size=n_try_sc)
        b = rng.choice(other_pos, size=n_try_sc)
        gb = all_factor[b]
        same = np.zeros(n_try_sc, dtype=bool)
        for g in np.unique(gb):
            m = gb == g
            common = [c for c in meta_cols if c not in (factor, g)]
            if common:
                s = np.ones(int(m.sum()), dtype=bool)
                for c in common:
                    s &= f_meta[c][a[m]] == other_meta[c][b[m]]
                same[m] = s
        asc, bsc = a[same][:max_comparisons], b[same][:max_comparisons]
        adc, bdc = a[~same][:max_comparisons], b[~same][:max_comparisons]

        def src_counts(sel):
            if not len(sel):
                return {}
            gs, ns = np.unique(all_factor[sel], return_counts=True)
            return {str(g): int(n) for g, n in zip(gs, ns)}

        record("diff_factor_sc", asc, all_gidx[bsc],
               extra={"source_counts": src_counts(bsc)})
        record("diff_factor_dc", adc, all_gidx[bdc],
               extra={"source_counts": src_counts(bdc)})
    else:
        record("diff_factor_sc", [], [])
        record("diff_factor_dc", [], [])

    # --- random（d2 来自全部因素；排除同一 delta 与 同因素同 context）---
    a = rng.integers(0, len(sub), size=n_try)
    b = rng.integers(0, len(delta_df), size=n_try)
    keep = (all_gidx[b] != f_gidx[a]) & ~(
        (all_factor[b] == factor) & (f_context[a] == all_context[b]))
    record("random", a[keep][:max_comparisons], all_gidx[b[keep][:max_comparisons]])
    return out


def transition_table(delta_df: pd.DataFrame, delta_mat: np.ndarray,
                     max_pairs_per_transition: int, rng) -> pd.DataFrame:
    """逐 transition 汇总：n_contexts、跨 context 相似度均值（点估计，无 CI）。"""
    rows = []
    for (factor, src, tgt), g in delta_df.groupby(["factor", "source_value", "target_value"]):
        gidx = g["delta_index"].to_numpy()
        n_ctx = g["context_key"].nunique()
        sims = []
        m = len(gidx)
        if m >= 2:
            k = min(max_pairs_per_transition, m * (m - 1) // 2)
            ia, ib = _distinct_pairs(rng, m, k)
            sims = _cos(delta_mat, gidx[ia], gidx[ib])
        rows.append({
            "factor": factor,
            "source_value": src,
            "target_value": tgt,
            "n_contexts": n_ctx,
            "mean_sim": float(np.mean(sims)) if len(sims) else np.nan,
        })
    return pd.DataFrame(rows)
