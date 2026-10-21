"""Bootstrap 工具（方案 §66）：普通 bootstrap + 按 cluster 聚类 bootstrap。

- cluster_gap_ci：两组共享 cluster 时，对 cluster 并集做一次重采样、同步应用到
  两组（审查报告 P1-4）；
- cluster_gap_ci_equal_types：E3 主差值用，每条件内按图片聚类重采样、
  跨 negative_type 等权平均（审查报告 P2-9）。
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


def cluster_mean_ci(values, clusters, n_boot=1000, ci=0.95, seed=0):
    """按 cluster 重采样（有放回抽 cluster，汇总其全部值）。"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    uniq, sums, counts = _cluster_sums(values, clusters)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = sums[draw].sum(axis=1) / np.maximum(counts[draw].sum(axis=1), 1)
    lo = (1 - ci) / 2 * 100
    point = float(values.mean())
    return point, float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))


def cluster_gap_ci(values_a, clusters_a, values_b, clusters_b, n_boot=1000, ci=0.95, seed=0):
    """Transition Similarity Gap 的聚类 bootstrap：同步重采样共享 cluster。

    对 A、B 两组 cluster 的并集做一次有放回重采样；每个被抽中的 cluster
    （含重复）在两组中分别累加 sum/count，再求差。两组完全相同时每次
    差值都为 0（审查报告 P1-4 的复现用例）。

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
    draw = rng.integers(0, k, size=(n_boot, k))
    ma = sa[draw].sum(axis=1) / np.maximum(ca[draw].sum(axis=1), 1)
    mb = sb[draw].sum(axis=1) / np.maximum(cb[draw].sum(axis=1), 1)
    boots = ma - mb
    gap = float(values_a.mean() - values_b.mean())
    lo = (1 - ci) / 2 * 100
    return gap, float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))


def diff_proportions_ci(correct_a, correct_b, n_boot=1000, ci=0.95, seed=0):
    """两个独立样本准确率之差的 bootstrap CI（E1 / E3 用）。"""
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
    """两个独立样本均值之差（Factor/Nuisance Norm Gap 等，审查报告 P2-8）。"""
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

    pooled accuracy = mean_t( acc_t )，acc_t 为该题型全部 item 的准确率
    （重采样时对抽中 cluster 的题型求和再相除）。避免 color/shape 配比
    差异混入 A−B（审查报告 P2-9）。
    """
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
    boots = np.empty(n_boot)
    for r in range(n_boot):
        da = rng.integers(0, na, na)
        db = rng.integers(0, nb, nb)
        boots[r] = pooled(ta, na, da) - pooled(tb, nb, db)
    point = pooled(ta, na) - pooled(tb, nb)
    lo = (1 - ci) / 2 * 100
    return point, float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))
