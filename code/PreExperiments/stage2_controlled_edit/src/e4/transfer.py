"""E4-D 跨场景方向迁移（工作方案第 6 节）。

- d_r 只用训练集估计并固定符号；全局非负幅度 α 在验证集选一次，所有方法
  共用同一搜索表（α=0 是"不修改"对照，不是性能基线）；
- 主终点：预先固定候选池中检索真实 after 图片的 top1 准确率；候选含原图、
  正确编辑图、同场景错误编辑（绑定=换色，词集变化单列；关系=共同平移）、
  同指令异场景 after；
- 对照方向：α=0 不修改、错误转换方向（缩放到训练方向长度）、随机方向
  （种子与数量固定，汇总全部不挑最好）、反向方向、文本差等幅方向；
  打乱配对均值方向只作代数自检（完整保留两端样本时与原均值一致），不作
  性能对照（第 6 节）。
"""
import numpy as np

from .data import pair_state_ids
from .delta import _dv_matrices  # 同口径的 Δ 组装


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def _top1(z_hat, cand_z, true_idx) -> bool:
    sims = cand_z @ z_hat
    return int(np.argmax(sims)) == true_idx


def run_e4d(manifest, rendered, z_by_state, dt_by_family, cfg, logger) -> dict:
    """方向迁移评测。返回 {rows: [...], self_check: {...}, alpha: float}。"""
    tcfg = cfg["transfer"]
    pcfg = cfg["candidates"]
    pairs_df = manifest["pairs"]
    sp = pairs_df[pairs_df["edit_kind"].isin(["swap", "flip"])]
    tasks = sorted(sp["task"].unique())
    rows, self_checks = [], []
    alphas = {}
    for task in tasks:
        zrows = _dv_matrices(sp[sp["task"] == task], z_by_state)
        by_fam = {}
        for r in zrows:
            by_fam.setdefault(r["family"], []).append(r)
        fams = sorted(by_fam)
        if not fams:
            continue
        d_r = {f: np.mean([r["dv"] for r in by_fam[f] if r["split"] == "train"],
                          axis=0) for f in fams}
        # 打乱配对代数自检：组内重配 before/after 后的均值仍 = mean(after)−mean(before)
        for f in fams:
            rs = by_fam[f]
            if len(rs) < 2:
                continue
            rng = np.random.default_rng(tcfg["random_seed"])
            perm = rng.permutation(len(rs))
            sh = np.mean([rs[i]["after"] - rs[perm[i]]["before"] for i in range(len(rs))],
                         axis=0)
            self_checks.append({"task": task, "family": f,
                                "shuffled_mean_max_diff": float(np.abs(sh - d_r[f]).max())})
        # 全局 α：验证集 + 正确方向，top1 平均最高者（并列取最小 α）
        def _scene_entries(split):
            out = []
            for f in fams:
                for r in by_fam[f]:
                    if r["split"] != split:
                        continue
                    pool = pair_state_ids(pcfg, manifest, rendered,
                                          _pair_row_of(pairs_df, r), task)
                    if len(pool) < 2:
                        continue
                    out.append((f, r, pool))
            return out

        val_entries = _scene_entries("val")
        best_a, best_acc = tcfg["alpha_grid"][0], -1.0
        for a in tcfg["alpha_grid"]:
            accs = [_retrieval_acc(z_by_state, r, pool, d_r[f], a)[0]
                    for f, r, pool in val_entries]
            if not accs:
                break
            m = float(np.mean(accs))
            if m > best_acc + 1e-12:
                best_a, best_acc = a, m
        alphas[task] = best_a if val_entries else tcfg["alpha_grid"][0]
        held_entries = _scene_entries("heldout")
        rng = np.random.default_rng(tcfg["random_seed"])
        for f, r, pool in held_entries:
            n_rand = tcfg["n_random_directions"]
            rand_dirs = [rng.standard_normal(z_by_state[r["before_id"]].shape)
                         for _ in range(n_rand)]
            wrong_f = fams[(fams.index(f) + 1) % len(fams)]
            dt = dt_by_family.get(f"{task}|{f}", {}).get("dt_mean")
            methods = [("alpha0", None), ("correct", d_r[f]),
                       ("wrong_conversion", d_r[wrong_f]),
                       ("reverse", -d_r[f])]
            if dt is not None:
                methods.append(("text_delta", dt))
            for name, direction in methods:
                ok, rank, cos_true = _retrieval_acc(
                    z_by_state, r, pool, d_r[f], alphas[task],
                    override_direction=direction, method_name=name)
                rows.append(_row(task, f, r, name, ok, rank, cos_true,
                                 alphas[task], len(pool)))
            for k, rd in enumerate(rand_dirs):
                ok, rank, cos_true = _retrieval_acc(
                    z_by_state, r, pool, d_r[f], alphas[task],
                    override_direction=rd, method_name="random")
                rows.append(_row(task, f, r, "random", ok, rank, cos_true,
                                 alphas[task], len(pool), random_k=k))
    return {"rows": rows,
            "self_check": {"shuffled_mean_max_diff": max(
                [c["shuffled_mean_max_diff"] for c in self_checks], default=float("nan"))},
            "alpha": alphas}


def _pair_row_of(pairs_df, r):
    """从 Δ 行回查 pair 行（候选池需要 pair_id / family 等字段）。"""
    hit = pairs_df[pairs_df["base_scene_id"] == r["scene"]]
    hit = hit[hit["edit_kind"].isin(["swap", "flip"])]
    if len(hit) != 1:
        raise RuntimeError(f"场景 {r['scene']} 的语义对不唯一: {len(hit)}")
    return hit.iloc[0]


def _retrieval_acc(z_by_state, r, pool, d_r, alpha,
                   override_direction=None, method_name=""):
    """预测表示检索；返回 (top1, 真实 after 的秩, 与真实 after 的 cosine)。"""
    z_b = z_by_state[r["before_id"]]
    direction = override_direction if override_direction is not None else d_r
    if method_name == "alpha0" or (override_direction is None and alpha == 0.0):
        z_hat = z_b / max(np.linalg.norm(z_b), 1e-12)
    else:
        length = np.linalg.norm(d_r)
        u = _unit(direction)
        z_hat = z_b + alpha * length * u
        n = np.linalg.norm(z_hat)
        z_hat = z_hat / n if n > 1e-12 else z_hat
    cand_ids = [p[0] for p in pool]
    cand_z = np.stack([z_by_state[sid] for sid in cand_ids])
    true_idx = cand_ids.index(r["after_id"])
    sims = cand_z @ z_hat
    order = np.argsort(-sims)
    rank = int(np.where(order == true_idx)[0][0])
    cos_true = float(cand_z[true_idx] @ z_hat)
    return bool(rank == 0), rank, cos_true


def _row(task, f, r, name, ok, rank, cos_true, alpha, pool_size, random_k=None):
    row = {"task": task, "family": f, "scene": r["scene"], "method": name,
           "top1": float(ok), "rank_true_after": rank, "cos_to_true_after": cos_true,
           "alpha": alpha, "pool_size": pool_size}
    if random_k is not None:
        row["random_k"] = random_k
    return row
