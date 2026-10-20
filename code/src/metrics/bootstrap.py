"""Bootstrap 工具（方案 §66）：普通 bootstrap + 按 context 聚类 bootstrap。"""
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
    """Transition Similarity Gap 的聚类 bootstrap（同一 resample 应用于两组）。

    返回 (gap_point, gap_ci_low, gap_ci_high)。
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan, np.nan
    uniq_a, sums_a, counts_a = _cluster_sums(values_a, clusters_a)
    uniq_b, sums_b, counts_b = _cluster_sums(values_b, clusters_b)
    rng = np.random.default_rng(seed)
    n = max(len(uniq_a), len(uniq_b))
    boots = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(uniq_a), size=len(uniq_a))
        ib = rng.integers(0, len(uniq_b), size=len(uniq_b))
        ma = sums_a[ia].sum() / max(counts_a[ia].sum(), 1)
        mb = sums_b[ib].sum() / max(counts_b[ib].sum(), 1)
        boots.append(ma - mb)
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
