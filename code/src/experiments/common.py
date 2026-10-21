"""2AFC 评估公共函数（E1 / E3 共用）。

文本按 外层模板 × item 的 顺序批量编码；跨模板聚合到 item 级
（item acc = 各模板 correct 的均值），模板间方差单独报告。

clusters（可选）：与 items 等长的聚类标签（如图像 id）。提供时
accuracy / margin 的 CI 改用聚类 bootstrap，处理同图重复观测的
相关性（审查报告 P2-5）。
"""
import numpy as np

from ..data.captions import templates, wrap
from ..metrics.bootstrap import cluster_mean_ci, mean_ci
from ..metrics.matching import aggregate_templates, per_template_scores

EXPORT_FIELDS = [
    "n_samples", "accuracy", "acc_ci_low", "acc_ci_high",
    "mean_margin", "margin_ci_low", "margin_ci_high",
    "median_margin", "margin_std", "template_variance",
]


def eval_2afc_items(model, tcache, img_cache, dataset_key, items,
                    n_boot=1000, ci=0.95, seeds=(0, 1), clusters=None):
    """items: list[dict(image_id, content_pos, content_neg, kind)]。

    返回指标 dict（含 item_acc / item_margin 数组，供差值 CI 与逐组合分析用）。
    """
    kinds = {it["kind"] for it in items}
    assert len(kinds) == 1, f"mixed template kinds in one eval: {kinds}"
    kind = kinds.pop()
    tmpls = templates()[kind]
    T, N = len(tmpls), len(items)

    _, img_norm = img_cache.get(dataset_key, [it["image_id"] for it in items])
    texts = []
    for t in tmpls:
        for it in items:
            texts.append(wrap(it["content_pos"], t))
            texts.append(wrap(it["content_neg"], t))
    _, txt = tcache.encode(model, texts)
    D = txt.shape[1]
    both = txt.reshape(T, N, 2, D)
    pos, neg = both[:, :, 0], both[:, :, 1]
    img = np.broadcast_to(img_norm[None], pos.shape)

    correct, margin = per_template_scores(img, pos, neg)
    item_acc, item_margin, _tmpl_acc, tvar = aggregate_templates(correct, margin)

    if clusters is not None:
        acc, lo, hi = cluster_mean_ci(item_acc, clusters, n_boot=n_boot, ci=ci, seed=seeds[0])
        m_mean, m_lo, m_hi = cluster_mean_ci(item_margin, clusters,
                                             n_boot=n_boot, ci=ci, seed=seeds[1])
    else:
        acc, lo, hi = mean_ci(item_acc, n_boot=n_boot, ci=ci, seed=seeds[0])
        m_mean, m_lo, m_hi = mean_ci(item_margin, n_boot=n_boot, ci=ci, seed=seeds[1])
    return {
        "n_samples": N,
        "accuracy": acc, "acc_ci_low": lo, "acc_ci_high": hi,
        "mean_margin": m_mean, "margin_ci_low": m_lo, "margin_ci_high": m_hi,
        "median_margin": float(np.median(item_margin)),
        "margin_std": float(np.std(item_margin, ddof=1)) if N > 1 else 0.0,
        "template_variance": tvar,
        "item_acc": item_acc,
        "item_margin": item_margin,
    }


def subsample(items, target, rng):
    if len(items) > target:
        keep = rng.choice(len(items), size=target, replace=False)
        items = [items[i] for i in sorted(keep.tolist())]
    return items


def cap_per_image(items, cap, rng):
    """每张图像最多保留 cap 条 item（审查报告 P2-5；cap<=0 表示不限制）。"""
    if cap is None or cap <= 0 or not items:
        return items
    by_img = {}
    for idx, it in enumerate(items):
        by_img.setdefault(it["image_id"], []).append(idx)
    keep = []
    for idxs in by_img.values():
        if len(idxs) > cap:
            sel = rng.permutation(len(idxs))[:cap]
            idxs = [idxs[k] for k in sorted(sel.tolist())]
        keep.extend(idxs)
    return [items[i] for i in sorted(keep)]
