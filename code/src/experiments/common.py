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
    # 四轮补充 C：评分前校验。
    # - 图/文维度必须严格相等：此前 img 维度为 1 时 broadcast_to 会静默扩展，
    #   余弦差上界被破坏，指标含义改变；
    # - 有限值：NaN 的 s_pos > s_neg 恒 False，会被当成确定的答错（accuracy=0
    #   的假象）；无效观测必须停止并给出模型/缓存/样本位置，不静默计分。
    mname = getattr(model, "name", type(model).__name__)
    where = f"model={mname}, cache={img_cache.dir}, dataset_key={dataset_key}"
    if img_norm.ndim != 2 or txt.ndim != 2:
        raise SystemExit(
            f"[eval] 特征不是二维矩阵：img {img_norm.shape} / txt {txt.shape}"
            f"（{where}）——特征形状异常，拒绝评分。")
    if img_norm.shape[1] != txt.shape[1]:
        raise SystemExit(
            f"[eval] 图像特征维度 {img_norm.shape[1]} != 文本特征维度 "
            f"{txt.shape[1]}（{where}）——图文特征空间不一致，广播扩展后的"
            f"相似度没有意义，拒绝评分。")
    if not np.isfinite(img_norm).all():
        bad = np.where((~np.isfinite(img_norm)).any(axis=1))[0]
        raise SystemExit(
            f"[eval] 图像特征含 NaN/Inf（{len(bad)} 条，首条 image_id="
            f"{items[int(bad[0])]['image_id']!r}；{where}）——无效特征不得"
            f"计为答错，请检查该样本的缓存与 encoder 输出。")
    if not np.isfinite(txt).all():
        raise SystemExit(
            f"[eval] 文本特征含 NaN/Inf（{where}，首个异常文本块在 "
            f"#{int(np.where((~np.isfinite(txt)).any(axis=1))[0][0])} 行）——"
            f"无效特征不得计为答错。")
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
