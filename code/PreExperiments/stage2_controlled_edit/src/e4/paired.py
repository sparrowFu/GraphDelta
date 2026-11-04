"""E4-A 配对结构判断（工作方案第 5 节）。

对每个编辑对及对应文本计算完整 2×2 相似度矩阵 S[t,i,j] = z_i · t_j
（t 为模板，i∈{0=before,1=after} 图，j∈{0,1} 文本）。口径：
- I2T / T2I 单图准确率：S[t,i,i] > S[t,i,1-i]，平局记错；
- Pair Accuracy：先在同一模板内两张图均选对，再跨模板平均（七模板不增加
  独立样本数）；
- 参考分布：场景内随机交换两文本标签（场景内匹配置换），不按 25% 处理。
"""
import json

import numpy as np

from src.data.captions import templates, wrap
from src.metrics.bootstrap import cluster_mean_ci

from .captions import pair_texts
from .stats import scene_permutation_pair_acc


def compute_S_records(pairs_df, z_by_state, tcache, model) -> list:
    """逐对逐模板的 2×2 矩阵。返回 [{pair 元数据, S:(T,2,2) ndarray}]。"""
    tpls = {}
    recs, texts, order = [], [], []
    for pr in pairs_df.to_dict("records"):
        kind = None
        t = pair_texts(json.loads(pr["before_spec"]), json.loads(pr["after_spec"]),
                       pr["task"])
        kind = t["kind"]
        if kind not in tpls:
            tpls[kind] = templates()[kind]
        order.append((pr, t))
        for tpl in tpls[kind]:
            texts.append(wrap(t["t0_content"], tpl))
            texts.append(wrap(t["t1_content"], tpl))
    _, tz = tcache.encode(model, texts)
    k = 0
    for pr, t in order:
        T = len(tpls[t["kind"]])
        Zt = tz[k:k + 2 * T].reshape(T, 2, -1)
        k += 2 * T
        zi = np.stack([z_by_state[pr["before_state_id"]],
                       z_by_state[pr["after_state_id"]]])
        S = np.einsum("tjd,id->tij", Zt, zi)
        recs.append({"pair_id": pr["pair_id"], "task": pr["task"],
                     "family": pr["family"], "edit_kind": pr["edit_kind"],
                     "base_scene_id": pr["base_scene_id"], "split": pr["split"],
                     "before_state_id": pr["before_state_id"],
                     "after_state_id": pr["after_state_id"],
                     "kind": t["kind"], "t0": t["t0_content"], "t1": t["t1_content"],
                     "S": S})
    return recs


def pair_metrics(S: np.ndarray) -> dict:
    """单对指标（输入 (T,2,2)）。平局记错（cfg pair_accuracy.tie=wrong 冻结）。

    - i2t / t2i：单图 / 单文本 2AFC 口径（t×i 或 t×j 平均）；
    - pair_acc：先在同一模板内两张图均选对（c00 & c11），再跨模板平均。
    """
    c00 = S[:, 0, 0] > S[:, 0, 1]   # before 图在 (T0,T1) 中选对
    c11 = S[:, 1, 1] > S[:, 1, 0]   # after 图选对
    t00 = S[:, 0, 0] > S[:, 1, 0]   # T0 在 (I0,I1) 中选对
    t11 = S[:, 1, 1] > S[:, 0, 1]   # T1 选对
    tie = (S[:, 0, 0] == S[:, 0, 1]) | (S[:, 1, 1] == S[:, 1, 0])
    return {"pair_acc": float((c00 & c11).mean()),
            "i2t_acc": float(np.concatenate([c00, c11]).mean()),
            "t2i_acc": float(np.concatenate([t00, t11]).mean()),
            "tie_rate": float(tie.mean()),
            "margin_before": float((S[:, 0, 0] - S[:, 0, 1]).mean()),
            "margin_after": float((S[:, 1, 1] - S[:, 1, 0]).mean())}


def run_e4a(S_records, cfg, logger) -> dict:
    """汇总 E4-A：逐任务 × 家族 × split 的 Pair Accuracy + 场景内置换参考分布。"""
    perm_cfg = cfg["permutation"]
    boot_cfg = cfg["bootstrap"]
    rows, items = [], []
    by_key = {}
    for rec in S_records:
        m = pair_metrics(rec["S"])
        items.append({"pair_id": rec["pair_id"], "task": rec["task"],
                      "family": rec["family"], "edit_kind": rec["edit_kind"],
                      "split": rec["split"], **m})
        by_key.setdefault((rec["task"], rec["family"], rec["split"]), []).append((rec, m))
    for (task, fam, split), entries in sorted(by_key.items()):
        scenes = [m["pair_acc"] for _, m in entries]
        cl = [e[0]["base_scene_id"] for e in entries]
        mean, lo, hi = cluster_mean_ci(scenes, cl, n_boot=boot_cfg["n_boot"],
                                       ci=boot_cfg["ci"], seed=0)
        rows.append({"task": task, "family": fam, "split": split,
                     "n_scenes": len(entries), "pair_acc": mean,
                     "pair_acc_ci_low": lo, "pair_acc_ci_high": hi,
                     "i2t_acc": float(np.mean([m["i2t_acc"] for _, m in entries])),
                     "t2i_acc": float(np.mean([m["t2i_acc"] for _, m in entries])),
                     "tie_rate": float(np.mean([m["tie_rate"] for _, m in entries]))})
    # 任务级 heldout 参考分布（第 5 节：场景内匹配置换，不按 25% 处理）
    for task in sorted({r["task"] for r in rows}):
        held = [rec for rec in S_records
                if rec["task"] == task and rec["split"] == "heldout"]
        if not held:
            continue
        S_mean = [rec["S"].mean(axis=0) for rec in held]
        null = scene_permutation_pair_acc(S_mean, perm_cfg["n"], seed=0)
        obs = float(np.mean([pair_metrics(r["S"])["pair_acc"] for r in held]))
        rows.append({"task": task, "family": "__overall__", "split": "heldout",
                     "n_scenes": len(held), "pair_acc": obs,
                     "pair_acc_ci_low": "", "pair_acc_ci_high": "",
                     "i2t_acc": "", "t2i_acc": "", "tie_rate": "",
                     "null_mean": null["null_mean"], "null_std": null["null_std"],
                     "perm_p": null["p"], "n_perm": null["n_perm_used"],
                     "exact": null["exact"]})
    return {"rows": rows, "items": items}
