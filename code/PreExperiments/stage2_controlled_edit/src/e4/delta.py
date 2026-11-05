"""E4-B 视觉变化诊断与线性探针（工作方案第 5 节）。

- Δv = after − before（z 与 raw 双空间，口径同阶段一 compute_deltas）；
- 逐转换 d_r 只用训练集估计；留出场景报告与 d_r 的方向相似度、‖Δv‖ 与
  近零比例（近零阈值 = 训练范数分位数，描述性判据），另与随机方向、
  同任务错误转换方向比较（待办 14）；
- 方向可靠性（待办 14）：训练内重复对半划分子集方向比较 + 按基础场景
  重采样的波动（run_e4b_reliability，配置 direction_reliability 冻结）；
- 对照诊断（第 3、5 节）：换色正控制走 E4-A 配对评分（run.py 编排）；
  换色 / jpeg / crop / 共同平移的 Δv 长度、方向与结构编辑同口径比较
  （run_e4b_controls）；
- 线性探针（numpy ridge 闭式解，骨干冻结）：单图状态 before/after 二分类、
  Δv 的转换身份分类；标准化统计只用训练折；
- 参照（第 5 节要求）：颜色直方图特征、像素差范数、单端点（只用 before 特征
  的转换分类，检查泄漏，期望接近随机）；
- 不做"去转换均值后残差两两同向"一类通过条件（第 5 节明确禁止）。
"""
import numpy as np

from .stats import RidgeProbe, cluster_ci, cosine_matrix, cosine_rows


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
    """方向一致性 / 范数 / 近零比例（z 空间主口径，raw 辅助）。

    返回 {"rows", "items", "dirs"}：dirs 含每个家族的训练方向 d_r 与近零
    阈值，供对照诊断（run_e4b_controls）与迁移评测按同口径复用。
    留出场景另报告与随机方向、同任务错误转换方向的相似度（工作方案第 5
    节、开篇待办 14：低相似度须能与随机 / 错误方向区分才有解释意义）。
    """
    q = cfg["near_zero"]["train_quantile"]
    boot_cfg = cfg["bootstrap"]
    tcfg = cfg.get("transfer", {})
    n_rand = int(tcfg.get("n_random_directions", 32))
    rand_seed = int(tcfg.get("random_seed", 0))
    groups = list(_group_semantic(pairs_df))
    # 第一遍：训练方向 d_r、近零阈值、训练内两两一致性
    prep = {}
    for (task, fam), grp in groups:
        zrows = _dv_matrices(grp, z_by_state)
        train = [r for r in zrows if r["split"] == "train"]
        if not train:
            logger.warning(f"[e4b] {fam} 无训练场景，跳过方向统计")
            continue
        d_r = np.mean([r["dv"] for r in train], axis=0)
        norms = np.array([np.linalg.norm(r["dv"]) for r in train])
        thr = float(np.quantile(norms, q))
        if len(train) >= 2:
            A = np.stack([r["dv"] for r in train])
            An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
            M = An @ An.T
            iu = np.triu_indices(len(A), k=1)
            consist = float(M[iu].mean())
        else:
            consist = float("nan")
        prep[(task, fam)] = {"grp": grp, "zrows": zrows, "d_r": d_r,
                             "thr": thr, "consist": consist}
    dirs = {"d_r": {f"{t}|{f}": v["d_r"] for (t, f), v in prep.items()},
            "thr": {f"{t}|{f}": v["thr"] for (t, f), v in prep.items()}}
    rows, items = [], []
    rng = np.random.default_rng(rand_seed)
    for (task, fam), v in prep.items():
        zrows, d_r, thr, consist = (v["zrows"], v["d_r"], v["thr"], v["consist"])
        # 错误转换方向：同任务家族按名排序取下一个（与 E4-D wrong_conversion
        # 同规则，确定性）
        fams_task = sorted(f2 for (t2, f2) in prep if t2 == task)
        wrong_f = fams_task[(fams_task.index(fam) + 1) % len(fams_task)] \
            if len(fams_task) > 1 else None
        wrong_d = prep[(task, wrong_f)]["d_r"] if wrong_f else None
        raw_rows = _dv_matrices(v["grp"], raw_by_state)
        for split in ("train", "val", "heldout"):
            entries = [r for r in zrows if r["split"] == split]
            if not entries:
                continue
            dv_mat = np.stack([r["dv"] for r in entries])
            cos = cosine_rows(dv_mat, np.stack([d_r] * len(entries)))
            mean, lo, hi = cluster_ci(cos, [r["scene"] for r in entries],
                                      boot_cfg["n_boot"], boot_cfg["ci"])
            raw_entries = [r for r in raw_rows if r["split"] == split]
            row = {
                "task": task, "family": fam, "split": split, "n": len(entries),
                "cos_to_d_r": mean, "cos_ci_low": lo, "cos_ci_high": hi,
                "dv_norm_mean": float(np.mean([np.linalg.norm(r["dv"]) for r in entries])),
                "dv_norm_raw_mean": (float(np.mean([np.linalg.norm(r["dv"]) for r in raw_entries]))
                                     if raw_entries else float("nan")),
                "near_zero_frac": float(np.mean(
                    [np.linalg.norm(r["dv"]) < thr for r in entries])),
                "train_pairwise_cos": consist if split == "heldout" else "",
                "near_zero_thr": thr,
            }
            if split == "heldout":
                # 随机方向参照：每场景与全部固定随机方向的 cosine 取平均，
                # 再跨场景平均。留出数与随机方向数不相等，必须算全组合，
                # 不能逐行配对（cosine_rows 要求行数对齐）
                rand_mat = rng.standard_normal((n_rand, d_r.shape[0]))
                row["cos_to_random_mean"] = float(
                    cosine_matrix(dv_mat, rand_mat).mean(axis=1).mean())
                if wrong_d is not None:
                    row["cos_to_wrong_conversion"] = float(cosine_rows(
                        dv_mat, np.stack([wrong_d] * len(entries))).mean())
            rows.append(row)
        for r in zrows:
            items.append({"task": task, "family": fam, "split": r["split"],
                          "scene": r["scene"],
                          "dv_norm": float(np.linalg.norm(r["dv"])),
                          "cos_to_d_r": float(cosine_rows(
                              r["dv"][None, :], d_r[None, :])[0])})
    return {"rows": rows, "items": items, "dirs": dirs}


def merge_dirs(dirs_list) -> dict:
    """合并各任务返回的 {"d_r", "thr"}（键已带 task 前缀）。

    必须逐内层字典合并：对顶层直接 dict.update 会用后处理任务的整体
    覆盖先处理任务，绑定对照因此丢失同家族结构方向（审查问题 2）。
    """
    out = {"d_r": {}, "thr": {}}
    for d in dirs_list:
        out["d_r"].update(d["d_r"])
        out["thr"].update(d["thr"])
    return out


def _group_semantic(pairs_df):
    sp = pairs_df[pairs_df["edit_kind"].isin(["swap", "flip"])]
    return sp.groupby(["task", "family"])


CONTROL_KINDS = ("recolor", "jpeg", "crop", "translate")


def control_pairs(pairs_df) -> "pd.DataFrame":
    """对照对（换色正控制 / jpeg / crop / 共同平移）。

    E4-B/C/D 的语义分析对象仍是 swap/flip，但对照不再被过滤器排除：
    换色走 E4-A 配对评分（正控制），全部对照走 Δv 诊断（审查问题 5）。
    """
    return pairs_df[pairs_df["edit_kind"].isin(CONTROL_KINDS)].copy()


def run_e4b_controls(control_df, z_by_state, dirs: dict, cfg, logger) -> list:
    """对照编辑的 Δv 诊断（方案第 3、5 节）：变化长度、与同家族结构编辑
    方向 d_r 的 cosine、近零比例，口径与 e4b_delta_stats 一致，可与结构
    编辑直接比较（审查问题 5：控制图片不能生成了事）。
    """
    boot_cfg = cfg["bootstrap"]
    rows = []
    if control_df.empty:
        return rows
    for (task, kind, fam), grp in control_df.groupby(
            ["task", "edit_kind", "family"]):
        zrows = _dv_matrices(grp, z_by_state)
        if not zrows:
            continue
        key = f"{task}|{fam}"
        d_r, thr = dirs["d_r"].get(key), dirs["thr"].get(key)
        dv = np.stack([r["dv"] for r in zrows])
        norms = np.linalg.norm(dv, axis=1)
        scenes = [r["scene"] for r in zrows]
        mean, lo, hi = cluster_ci(list(norms), scenes,
                                  boot_cfg["n_boot"], boot_cfg["ci"])
        row = {"task": task, "edit_kind": kind, "family": fam, "n": len(zrows),
               "structural_kind": "swap" if task == "binding" else "flip",
               "dv_norm_mean": mean, "dv_norm_ci_low": lo, "dv_norm_ci_high": hi}
        if d_r is not None:
            cos = cosine_rows(dv, np.stack([d_r] * len(zrows)))
            cm, cl, ch = cluster_ci(list(cos), scenes,
                                    boot_cfg["n_boot"], boot_cfg["ci"])
            row.update({"cos_to_structural_d_r": cm,
                        "cos_ci_low": cl, "cos_ci_high": ch})
        if thr is not None:
            row["near_zero_frac"] = float((norms < thr).mean())
            row["near_zero_thr"] = thr
        rows.append(row)
    return rows


def run_e4b_reliability(pairs_df, z_by_state, cfg, logger) -> list:
    """方向可靠性（工作方案第 5 节、开篇待办 14）。

    只用训练场景：(1) 重复随机对半划分，比较两个子集的平均方向；
    (2) 按基础场景有放回重采样，报告平均方向与长度的波动。重复次数、
    种子、子集比例与近零处理冻结在配置 direction_reliability。回答的是
    "换一批训练场景，估计的平均方向是否还一样"；结果为描述统计，不设
    cosine 门槛、不预设小样本必然降低相似度。
    """
    rcfg = cfg.get("direction_reliability", {})
    n_repeats = int(rcfg.get("n_repeats", 20))
    frac = float(rcfg.get("train_fraction", 0.5))
    seed = int(rcfg.get("seed", 0))
    n_boot = int(rcfg.get("n_boot", cfg["bootstrap"]["n_boot"]))
    nz_norm = float(rcfg.get("near_zero_norm", 1e-8))
    rows = []
    for (task, fam), grp in _group_semantic(pairs_df):
        zrows = [r for r in _dv_matrices(grp, z_by_state)
                 if r["split"] == "train"]
        n = len(zrows)
        if n < 4:
            if n:
                logger.warning(f"[e4b-rel] {fam} 训练场景 {n} < 4，跳过可靠性")
            continue
        A = np.stack([r["dv"] for r in zrows])
        d_full = A.mean(axis=0)
        rng = np.random.default_rng(seed)
        # (1) 对半划分子集方向比较（近零子集单列，不计入 cosine）
        k = max(2, int(round(frac * n)))
        cos_split, nz_splits, split_norms = [], 0, []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            d1, d2 = A[perm[:k]].mean(axis=0), A[perm[k:2 * k]].mean(axis=0)
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            split_norms += [float(n1), float(n2)]
            if n1 < nz_norm or n2 < nz_norm:
                nz_splits += 1
            else:
                cos_split.append(float(d1 @ d2 / (n1 * n2)))
        # (2) 按基础场景重采样：平均方向与长度的波动
        boot_cos, boot_norm = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            d_b = A[idx].mean(axis=0)
            nb = np.linalg.norm(d_b)
            boot_norm.append(float(nb))
            if nb >= nz_norm and np.linalg.norm(d_full) >= nz_norm:
                boot_cos.append(float(d_b @ d_full
                                      / (nb * np.linalg.norm(d_full))))
        rows.append({
            "task": task, "family": fam, "n_train": n,
            "n_repeats": n_repeats, "subset_fraction": frac,
            "half_split_cos_mean": float(np.mean(cos_split)) if cos_split else float("nan"),
            "half_split_cos_p05": (float(np.quantile(cos_split, 0.05))
                                   if cos_split else float("nan")),
            "half_split_near_zero_frac": nz_splits / n_repeats,
            "half_split_norm_mean": float(np.mean(split_norms)),
            "boot_cos_to_full_mean": float(np.mean(boot_cos)) if boot_cos else float("nan"),
            "boot_cos_to_full_p05": (float(np.quantile(boot_cos, 0.05))
                                     if boot_cos else float("nan")),
            "boot_norm_mean": float(np.mean(boot_norm)),
            "boot_norm_std": float(np.std(boot_norm)),
            "full_norm": float(np.linalg.norm(d_full)),
        })
    return rows


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
