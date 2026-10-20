"""2AFC Matching（方案 §29 / §31）：s+ > s-，平局记错；多模板平均 + 模板间方差。"""
import numpy as np


def per_template_scores(img_norm, pos_norm, neg_norm):
    """img/pos/neg 均已 L2 归一化，形状 (N, D) 或 (T, N, D)。

    返回 correct (T, N) bool、margin (T, N) float。correct = s+ > s-（平局记错）。
    """
    img = img_norm[None] if img_norm.ndim == 2 else img_norm
    pos = pos_norm[None] if pos_norm.ndim == 2 else pos_norm
    neg = neg_norm[None] if neg_norm.ndim == 2 else neg_norm
    s_pos = np.einsum("tnd,tnd->tn", img, pos)
    s_neg = np.einsum("tnd,tnd->tn", img, neg)
    margin = s_pos - s_neg
    return s_pos > s_neg, margin


def aggregate_templates(correct_tn, margin_tn):
    """跨模板聚合到 item 级。

    返回 item_acc (N,)、item_margin (N,)、template_mean_acc (T,)、template_variance。
    """
    correct_tn = np.atleast_2d(correct_tn)
    margin_tn = np.atleast_2d(margin_tn)
    item_acc = correct_tn.mean(axis=0)
    item_margin = margin_tn.mean(axis=0)
    template_mean_acc = correct_tn.mean(axis=1)
    var = float(np.var(template_mean_acc, ddof=1)) if len(template_mean_acc) > 1 else 0.0
    return item_acc, item_margin, template_mean_acc, var
