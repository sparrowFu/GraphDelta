"""E4-B 视觉变化诊断与线性探针（工作方案第 5 节）。

- Δv = after − before（z 与 raw 双空间，口径同阶段一 compute_deltas）；
- 逐转换 d_r 只用训练集估计；留出场景报告与 d_r 的方向相似度、‖Δv‖ 与
  近零比例（近零阈值 = 训练范数分位数，描述性判据）；
- 线性探针（numpy ridge 闭式解，骨干冻结）：单图状态 before/after 二分类、
  Δv 的转换身份分类；标准化统计只用训练折；
- 参照（第 5 节要求）：颜色直方图特征、像素差范数、单端点（只用 before 特征
  的转换分类，检查泄漏，期望接近随机）；
- 不做"去转换均值后残差两两同向"一类通过条件（第 5 节明确禁止）。
"""
import numpy as np

from .stats import RidgeProbe, cluster_ci, cosine_rows


def _dv_matrices(pairs_df, feat: dict):
    """语义对（swap/flip）的 Δ；feat: {state_id: 向量}。"""
    rows = []
    for pr in pairs_df.to_dict("records"):
        a, b = feat.get(pr["before_state_id"]), feat.get(pr["after_state_id"])
        if a is None or b is None:
            continue
        rows.append({"family": pr["family"], "task": pr["task"],
                     "scene": pr["base_scene_id"], "split": pr["split"],
                     "dv": b - a, "before": a, "after": b,
                     "before_id": pr["before_state_id"],
                     "after_id": pr["after_state_id"]})
    return rows


def run_e4b_stats(pairs_df, z_by_state, raw_by_state, cfg, logger) -> dict:
    """方向一致性 / 范数 / 近零比例（z 空间主口径，raw 辅助）。"""
    q = cfg["near_zero"]["train_quantile"]
    boot_cfg = cfg["bootstrap"]
    rows, items = [], []
    for (task, fam), grp in _group_semantic(pairs_df):
        zrows = _dv_matrices(grp, z_by_state)
        if not zrows:
            continue
        train = [r for r in zrows if r["split"] == "train"]
        if not train:
            logger.warning(f"[e4b] {fam} 无训练场景，跳过方向统计")
            continue
        d_r = np.mean([r["dv"] for r in train], axis=0)
        norms = np.array([np.linalg.norm(r["dv"]) for r in train])
        thr = float(np.quantile(norms, q))
        # 训练内方向一致性：两两 cosine 均值
        if len(train) >= 2:
            A = np.stack([r["dv"] for r in train])
            An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
            M = An @ An.T
            iu = np.triu_indices(len(A), k=1)
            consist = float(M[iu].mean())
        else:
            consist = float("nan")
        raw_rows = _dv_matrices(grp, raw_by_state)
        for split in ("train", "val", "heldout"):
            entries = [r for r in zrows if r["split"] == split]
            if not entries:
                continue
            cos = cosine_rows(np.stack([r["dv"] for r in entries]),
                              np.stack([d_r] * len(entries)))
            mean, lo, hi = cluster_ci(cos, [r["scene"] for r in entries],
                                      boot_cfg["n_boot"], boot_cfg["ci"])
            raw_entries = [r for r in raw_rows if r["split"] == split]
            rows.append({
                "task": task, "family": fam, "split": split, "n": len(entries),
                "cos_to_d_r": mean, "cos_ci_low": lo, "cos_ci_high": hi,
                "dv_norm_mean": float(np.mean([np.linalg.norm(r["dv"]) for r in entries])),
                "dv_norm_raw_mean": (float(np.mean([np.linalg.norm(r["dv"]) for r in raw_entries]))
                                     if raw_entries else float("nan")),
                "near_zero_frac": float(np.mean(
                    [np.linalg.norm(r["dv"]) < thr for r in entries])),
                "train_pairwise_cos": consist if split == "heldout" else "",
                "near_zero_thr": thr,
            })
        for r in zrows:
            items.append({"task": task, "family": fam, "split": r["split"],
                          "scene": r["scene"],
                          "dv_norm": float(np.linalg.norm(r["dv"])),
                          "cos_to_d_r": float(cosine_rows(
                              r["dv"][None, :], d_r[None, :])[0])})
    return {"rows": rows, "items": items}


def _group_semantic(pairs_df):
    sp = pairs_df[pairs_df["edit_kind"].isin(["swap", "flip"])]
    return sp.groupby(["task", "family"])


def color_hist(image_path, downsize: int = 64) -> np.ndarray:
    """8 色占比特征（最近色分桶；全图像素，描述性参照）。"""
    from PIL import Image
    from .captions import CLEVR_COLOR_RGB
    img = Image.open(image_path).convert("RGB").resize((downsize, downsize))
    arr = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    names = sorted(CLEVR_COLOR_RGB)
    pal = np.array([CLEVR_COLOR_RGB[n] for n in names], dtype=np.float32)
    d = ((arr[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    idx = d.argmin(axis=1)
    return np.eye(len(names))[idx].mean(axis=0)


def pixel_diff_norm(p0, p1, downsize: int = 64) -> float:
    """两图平均绝对差（像素差参照）。"""
    from PIL import Image
    a = np.asarray(Image.open(p0).convert("RGB").resize((downsize, downsize)),
                   dtype=np.float32) / 255.0
    b = np.asarray(Image.open(p1).convert("RGB").resize((downsize, downsize)),
                   dtype=np.float32) / 255.0
    return float(np.abs(a - b).mean())


def run_e4b_probes(pairs_df, z_by_state, rendered, cfg, logger) -> list:
    """探针 + 参照。返回逐 (task, probe) 行。"""
    lambdas = cfg["probe"]["lambdas"]
    rows = []
    for task in sorted(pairs_df["task"].unique()):
        sp = pairs_df[(pairs_df["task"] == task)
                      & pairs_df["edit_kind"].isin(["swap", "flip"])]
        zrows = _dv_matrices(sp, z_by_state)
        if not zrows:
            continue
        splits = {"train": [], "val": [], "heldout": []}
        for r in zrows:
            splits[r["split"]].append(r)
        if not splits["train"] or not splits["heldout"]:
            logger.warning(f"[e4b-probe] {task} 训练或留出为空，跳过")
            continue

        def _probe(name, Xtr, ytr, Xv, yv, Xh, yh, n_classes):
            p = RidgeProbe(lambdas).fit(Xtr, ytr, Xv, yv)
            acc = float((p.predict(Xh) == np.asarray(yh)).mean())
            rows.append({"task": task, "probe": name, "n_classes": n_classes,
                         "lambda": p.lam, "heldout_acc": acc,
                         "chance": 1.0 / n_classes,
                         "n_train": len(ytr), "n_heldout": len(yh)})

        # 1) 单图状态 before/after 二分类（z 特征）
        def _state_xy(entries):
            X = np.concatenate([np.stack([r["before"] for r in entries]),
                                np.stack([r["after"] for r in entries])], axis=0)
            y = np.array([0] * len(entries) + [1] * len(entries))
            return X, y
        Xtr, ytr = _state_xy(splits["train"])
        Xv, yv = _state_xy(splits["val"]) if splits["val"] else (None, None)
        Xh, yh = _state_xy(splits["heldout"])
        _probe("state_z", Xtr, ytr, Xv, yv, Xh, yh, 2)

        # 2) Δv 的转换身份分类（z 特征）
        def _dv_xy(entries):
            X = np.stack([r["dv"] for r in entries])
            y = np.array([r["family"] for r in entries])
            return X, y
        Xtr, ytr = _dv_xy(splits["train"])
        Xv, yv = _dv_xy(splits["val"]) if splits["val"] else (None, None)
        Xh, yh = _dv_xy(splits["heldout"])
        _probe("conversion_z", Xtr, ytr, Xv, yv, Xh, yh, len(set(ytr)))

        # 3) 单端点泄漏检查：只用 before 特征预测转换（期望接近随机）
        Xb_tr = np.stack([r["before"] for r in splits["train"]])
        yb_tr = np.array([r["family"] for r in splits["train"]])
        Xb_v = (np.stack([r["before"] for r in splits["val"]])
                if splits["val"] else None)
        yb_v = np.array([r["family"] for r in splits["val"]]) if splits["val"] else None
        Xb_h = np.stack([r["before"] for r in splits["heldout"]])
        yb_h = np.array([r["family"] for r in splits["heldout"]])
        _probe("conversion_before_only_z", Xb_tr, yb_tr, Xb_v, yb_v, Xb_h, yb_h,
               len(set(yb_tr)))

        # 4) 参照：颜色直方图 / 像素差（需要图片；缺图则跳过）
        img_by_state = dict(zip(rendered["state_id"], rendered["image_path"]))
        need = {r["before_id"] for r in zrows} | {r["after_id"] for r in zrows}
        if all(s in img_by_state for s in need):
            hist = {s: color_hist(img_by_state[s]) for s in sorted(need)}
            Xtr2, ytr2 = _state_xy_hist(splits["train"], hist)
            Xv2, yv2 = (_state_xy_hist(splits["val"], hist) if splits["val"]
                        else (None, None))
            Xh2, yh2 = _state_xy_hist(splits["heldout"], hist)
            _probe("state_color_hist", Xtr2, ytr2, Xv2, yv2, Xh2, yh2, 2)
            pdtr = np.array([[pixel_diff_norm(img_by_state[r["before_id"]],
                                              img_by_state[r["after_id"]])]
                             for r in splits["train"]])
            pdv = (np.array([[pixel_diff_norm(img_by_state[r["before_id"]],
                                              img_by_state[r["after_id"]])]
                             for r in splits["val"]]) if splits["val"] else None)
            pdh = np.array([[pixel_diff_norm(img_by_state[r["before_id"]],
                                             img_by_state[r["after_id"]])]
                            for r in splits["heldout"]])
            _probe("conversion_pixel_diff", pdtr, ytr, pdv, yv, pdh, yh,
                   len(set(ytr)))
        else:
            logger.warning(f"[e4b-probe] {task} 部分图片缺失，颜色/像素参照跳过")
    return rows


def _state_xy_hist(entries, hist):
    X = np.concatenate([np.stack([hist[r["before_id"]] for r in entries]),
                        np.stack([hist[r["after_id"]] for r in entries])], axis=0)
    y = np.array([0] * len(entries) + [1] * len(entries))
    return X, y
