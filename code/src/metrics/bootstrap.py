"""Bootstrap 工具（方案 §66）：普通 bootstrap + 按 cluster 聚类 bootstrap。

- cluster_gap_ci：两组共享 cluster 时，对 cluster 并集做一次重采样、同步应用到
  两组（一轮审查 P1-4）；某次重抽样使某一侧完全没有 cluster 时该次作废重抽，
  不把空组当成 0 均值（二轮审查问题 6）。
- pair_cluster_mean_ci / pair_cluster_gap_ci：比较值依赖两个端点（context）时的
  聚类 bootstrap——对 context 全集重采样，比较权重 = 两端点被抽中次数之积，
  结果与端点登记顺序无关（二轮审查问题 3）。
- cluster_gap_ci_equal_types：E3 主差值，每条件内按图片聚类重采样、跨
  negative_type 等权平均（一轮 P2-9）；缺整类题型时返回 NaN，重抽样中
  某题型抽空时该次作废重抽（二轮审查问题 6）。
"""
import numpy as np


def mean_ci(values, n_boot=1000, ci=0.95, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    lo = (1 - ci) / 2 * 100
    return float(values.mean()), float(np.percentile(boots, lo)), \
        float(np.percentile(boots, 100 - lo))


def _cluster_sums(values, clusters):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    return uniq, sums, counts


def _ci(boots, point, ci):
    boots = np.asarray(boots, dtype=np.float64)
    if not len(boots):
        return np.nan, np.nan, np.nan
    lo = (1 - ci) / 2 * 100
    return point, float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))


def _chunk_size(k, n_boot):
    return max(1, min(n_boot, int(3e7 // max(k, 1))))


def cluster_mean_ci(values, clusters, n_boot=1000, ci=0.95, seed=0):
    """按 cluster 重采样（有放回抽 cluster，汇总其全部值）。"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    uniq, sums, counts = _cluster_sums(values, clusters)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    chunk = _chunk_size(k, n_boot)
    boots = np.empty(n_boot)
    filled = 0
    while filled < n_boot:
        draw = rng.integers(0, k, size=(chunk, k))
        vals = sums[draw].sum(axis=1) / np.maximum(counts[draw].sum(axis=1), 1)
        take = vals[:n_boot - filled]
        boots[filled:filled + len(take)] = take
        filled += len(take)
    return _ci(boots, float(values.mean()), ci)


def cluster_gap_ci(values_a, clusters_a, values_b, clusters_b, n_boot=1000, ci=0.95, seed=0):
    """Transition Similarity Gap / 共享图片的 factor Gap：同步重采样共享 cluster。

    对 A、B 两组 cluster 的并集做一次有放回重采样；每个被抽中的 cluster
    （含重复）在两组中分别累加 sum/count，再求差。某次重抽样使某一侧
    count 总数为 0 时该次作废重抽（空组不是 0 均值，二轮审查问题 6）。

    A/B 共享图片（如 E1 的 factor 对、E3 的 pooled gap、E2 的 Norm Gap）时，
    共享 cluster 的相关性被同步重采样保留（二轮审查问题 2/4/5）。

    返回 (gap_point, gap_ci_low, gap_ci_high)。
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan, np.nan
    uniq_a, sums_a, counts_a = _cluster_sums(values_a, clusters_a)
    uniq_b, sums_b, counts_b = _cluster_sums(values_b, clusters_b)
    union = np.union1d(uniq_a, uniq_b)
    k = len(union)
    sa = np.zeros(k)
    ca = np.zeros(k, dtype=np.int64)
    sb = np.zeros(k)
    cb = np.zeros(k, dtype=np.int64)
    sa[np.searchsorted(union, uniq_a)] = sums_a
    ca[np.searchsorted(union, uniq_a)] = counts_a
    sb[np.searchsorted(union, uniq_b)] = sums_b
    cb[np.searchsorted(union, uniq_b)] = counts_b
    rng = np.random.default_rng(seed)
    chunk = _chunk_size(k, n_boot)
    boots = []
    attempts = 0
    max_attempts = n_boot * 50 + 10 * chunk
    while len(boots) < n_boot and attempts < max_attempts:
        attempts += chunk
        draw = rng.integers(0, k, size=(chunk, k))
        ca_d = ca[draw].sum(axis=1)
        cb_d = cb[draw].sum(axis=1)
        ok = (ca_d > 0) & (cb_d > 0)
        if not ok.any():
            continue
        vals = sa[draw].sum(axis=1) / np.maximum(ca_d, 1) \
            - sb[draw].sum(axis=1) / np.maximum(cb_d, 1)
        boots.extend(vals[ok][:n_boot - len(boots)].tolist())
    gap = float(values_a.mean() - values_b.mean())
    return _ci(boots, gap, ci)


def pair_cluster_mean_ci(values, ctx_a, ctx_b, n_boot=1000, ci=0.95, seed=0):
    """两端点比较（cos(Δ1,Δ2) 依赖两个 context）的均值聚类 bootstrap。

    对 context 全集有放回重采样；每条比较的权重 = 两个端点被抽中次数之积，
    均值 = Σ w·v / Σ w。权重对两端点对称 → 与端点登记顺序无关（二轮问题 3）。
    某次重抽样使 Σw = 0（所有涉及的 context 都未被抽中）时该次作废重抽。
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    ctx = np.concatenate([np.asarray(ctx_a), np.asarray(ctx_b)])
    uniq, inv = np.unique(ctx, return_inverse=True)
    ia = inv[:len(values)]
    ib = inv[len(values):]
    k = len(uniq)
    rng = np.random.default_rng(seed)
    chunk = _chunk_size(k, n_boot)
    boots = []
    attempts = 0
    max_attempts = n_boot * 50 + 10 * chunk
    while len(boots) < n_boot and attempts < max_attempts:
        attempts += chunk
        m = np.bincount(rng.integers(0, k, size=k), minlength=k)  # 每个 context 被抽中次数
        w = (m[ia] * m[ib]).astype(np.float64)
        sw = w.sum()
        if sw <= 0:
            continue
        boots.append(float((w * values).sum() / sw))
    return _ci(boots, float(values.mean()), ci)


def pair_cluster_gap_ci(values_a, ctx_a_a, ctx_b_a, values_b, ctx_a_b, ctx_b_b,
                        n_boot=1000, ci=0.95, seed=0):
    """Transition Similarity Gap：两组比较的两端点 context 并入同一全集，
    一次重采样同步应用到两组（比较内与组间共享 context 的相关性都保留）。

    返回 (gap_point, gap_ci_low, gap_ci_high)。
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan, np.nan
    ctx_all = np.concatenate([np.asarray(ctx_a_a), np.asarray(ctx_b_a),
                              np.asarray(ctx_a_b), np.asarray(ctx_b_b)])
    uniq, inv = np.unique(ctx_all, return_inverse=True)
    n = len(values_a)
    ia_a, ib_a = inv[:n], inv[n:2 * n]
    ia_b, ib_b = inv[2 * n:2 * n + len(values_b)], inv[2 * n + len(values_b):]
    k = len(uniq)
    rng = np.random.default_rng(seed)
    boots = []
    attempts = 0
    max_attempts = n_boot * 50 + 200
    while len(boots) < n_boot and attempts < max_attempts:
        attempts += 1
        m = np.bincount(rng.integers(0, k, size=k), minlength=k)
        wa = (m[ia_a] * m[ib_a]).astype(np.float64)
        wb = (m[ia_b] * m[ib_b]).astype(np.float64)
        swa, swb = wa.sum(), wb.sum()
        if swa <= 0 or swb <= 0:
            continue
        boots.append(float((wa * values_a).sum() / swa - (wb * values_b).sum() / swb))
    gap = float(values_a.mean() - values_b.mean())
    return _ci(boots, gap, ci)


def diff_proportions_ci(correct_a, correct_b, n_boot=1000, ci=0.95, seed=0):
    """两个独立样本准确率之差的 bootstrap CI（两侧确无共享单元时才可用）。"""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo = (1 - ci) / 2 * 100
    gap = float(a.mean() - b.mean())
    return gap, float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))


def mean_diff_ci(values_a, values_b, n_boot=1000, ci=0.95, seed=0):
    """两个独立样本均值之差（两侧确无共享单元时才可用）。"""
    return diff_proportions_ci(values_a, values_b, n_boot=n_boot, ci=ci, seed=seed)


def _per_type_cluster_table(acc, clusters, types_arr, types):
    acc = np.asarray(acc, dtype=np.float64)
    clusters = np.asarray(clusters)
    types_arr = np.asarray(types_arr)
    uniq, inv = np.unique(clusters, return_inverse=True)
    cols = {}
    for t in types:
        m = types_arr == t
        cols[t] = (np.bincount(inv[m], weights=acc[m], minlength=len(uniq)),
                   np.bincount(inv[m], minlength=len(uniq)))
    return len(uniq), cols


def cluster_gap_ci_equal_types(acc_a, cl_a, ty_a, acc_b, cl_b, ty_b, types,
                               n_boot=1000, ci=0.95, seed=0):
    """E3 主差值：每条件内按图片聚类重采样，跨 negative_type 等权平均后求 A−B。

    - 某条件缺整类题型（全数据 count=0）→ 返回 NaN×3，与 e3 的等权点估计
      口径一致（二轮审查问题 6）；
    - 某次重抽样使任一条件的任一题型抽空 → 该次作废重抽，不记 0 准确率。
    """
    ty_a = np.asarray(ty_a)
    ty_b = np.asarray(ty_b)
    for t in types:
        if int((ty_a == t).sum()) == 0 or int((ty_b == t).sum()) == 0:
            return np.nan, np.nan, np.nan
    na, ta = _per_type_cluster_table(acc_a, cl_a, ty_a, types)
    nb, tb = _per_type_cluster_table(acc_b, cl_b, ty_b, types)

    def pooled(cols, n, draw=None):
        vals = []
        for t in types:
            s, c = cols[t]
            num = s[draw].sum() if draw is not None else s.sum()
            den = c[draw].sum() if draw is not None else c.sum()
            vals.append(num / max(den, 1))
        return float(np.mean(vals))

    rng = np.random.default_rng(seed)
    boots = []
    attempts = 0
    max_attempts = n_boot * 50 + 200
    while len(boots) < n_boot and attempts < max_attempts:
        attempts += 1
        da = rng.integers(0, na, na)
        db = rng.integers(0, nb, nb)
        if any(ta[t][1][da].sum() == 0 or tb[t][1][db].sum() == 0 for t in types):
            continue
        boots.append(pooled(ta, na, da) - pooled(tb, nb, db))
    point = pooled(ta, na) - pooled(tb, nb)
    return _ci(boots, point, ci)
