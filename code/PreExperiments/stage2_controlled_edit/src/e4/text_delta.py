"""E4-C 转换级跨模态诊断（工作方案第 5 节）。

预先指定的转换级条件 = 绑定 8 个转换 + 关系 10 个类型对；推断范围限于这些
固定条件。同一转换的文本对在每个场景相同（两对象、颜色 / 形状词固定），
因此主分析是转换级的：报告条件数，不做逐场景跨模态对应的主张。

代数恒等式（同归一化点积下）：C = S00 + S11 − S01 − S10 = Δv · Δt，
本模块对该等式做数值自检（不做两份独立证据）。
"""
import itertools
import json

import numpy as np

from src.data.captions import templates, wrap

from .captions import pair_texts
from .stats import cluster_ci, sign_flip_test


def family_text_deltas(pairs_df, tcache, model) -> dict:
    """逐家族的文本 Δt（z 空间）：每家族取一对代表（文本在家族内恒定）。

    返回 {family_key: {"dt_mean": (D,), "per_template": (T,D), "kind": str,
    "t0": str, "t1": str}}，family_key = f"{task}|{family}"。
    """
    tpls = {}
    reps, texts, order = [], [], []
    for fam, grp in pairs_df.groupby(["task", "family"]):
        pr = grp.iloc[0]
        t = pair_texts(json.loads(pr["before_spec"]), json.loads(pr["after_spec"]),
                       pr["task"])
        if t["kind"] not in tpls:
            tpls[t["kind"]] = templates()[t["kind"]]
        order.append((f"{fam[0]}|{fam[1]}", t))
        for tpl in tpls[t["kind"]]:
            texts.append(wrap(t["t0_content"], tpl))
            texts.append(wrap(t["t1_content"], tpl))
    _, tz = tcache.encode(model, texts)
    out = {}
    k = 0
    for key, t in order:
        T = len(tpls[t["kind"]])
        Zt = tz[k:k + 2 * T].reshape(T, 2, -1)
        k += 2 * T
        dt = Zt[:, 1, :] - Zt[:, 0, :]
        out[key] = {"dt_mean": dt.mean(axis=0), "per_template": dt,
                    "kind": t["kind"], "t0": t["t0_content"], "t1": t["t1_content"]}
    return out


def _color_group(family: str) -> str:
    """绑定家族的颜色组（cube–sphere / cylinder–sphere），用于跨组对照。"""
    return family.split("|")[0]


def run_e4c(S_records, z_by_state, dt_by_family, cfg, logger) -> dict:
    """转换级 C 统计 + 符号翻转检验 + 跨组对照 + 代数自检。

    逐转换行为探索性描述；任务级主统计（family="__overall__"，留出场景
    合并 C 的均值 / 区间 / 符号翻转 p / 方向）单列，供汇总按预先固定的
    规则判定（审查问题 3）。
    """
    perm_cfg = cfg["permutation"]
    boot_cfg = cfg["bootstrap"]
    # 逐场景 C（模板平均）：C = Δv · Δt 对每个模板成立，直接由 S 算
    by_fam = {}
    identity_max_diff = 0.0
    for rec in S_records:
        S = rec["S"]
        C_t = S[:, 0, 0] + S[:, 1, 1] - S[:, 0, 1] - S[:, 1, 0]
        C = float(C_t.mean())
        dv = z_by_state[rec["after_state_id"]] - z_by_state[rec["before_state_id"]]
        key = f"{rec['task']}|{rec['family']}"
        if key in dt_by_family:
            # 线性性：mean_t(Δv·Δt_t) = Δv·mean_t(Δt_t)，与模板平均的 C 严格相等
            dt = dt_by_family[key]["dt_mean"]
            identity_max_diff = max(identity_max_diff, abs(C - float(dv @ dt)))
        by_fam.setdefault((rec["task"], rec["family"], rec["split"]), []).append(
            {"C": C, "scene": rec["base_scene_id"], "dv": dv})
    rows = []
    all_fams = sorted({(k[0], k[1]) for k in by_fam})
    n_cond_by_task = {}
    for t2, _ in all_fams:
        n_cond_by_task[t2] = n_cond_by_task.get(t2, 0) + 1
    for task, fam in all_fams:
        row = {"task": task, "family": fam, "n_conditions": n_cond_by_task[task]}
        for split in ("train", "val", "heldout"):
            entries = by_fam.get((task, fam, split), [])
            if not entries:
                continue
            Cs = [e["C"] for e in entries]
            mean, lo, hi = cluster_ci(Cs, [e["scene"] for e in entries],
                                      boot_cfg["n_boot"], boot_cfg["ci"])
            row[f"n_{split}"] = len(entries)
            row[f"C_mean_{split}"] = mean
            row[f"C_ci_low_{split}"] = lo
            row[f"C_ci_high_{split}"] = hi
            if split == "heldout":
                test = sign_flip_test(Cs, perm_cfg["n"], seed=0)
                row["p_heldout"] = test["p"]
                row["n_perm"] = test["n_perm_used"]
                row["exact"] = test["exact"]
        # 跨组对照：C_wrong = mean_scene(dv · Δt_g)，g 取另一颜色组（关系任务
        # 无颜色组，取其余全部家族）；同词集角色错误与正确方向仅差符号，去重
        key = f"{task}|{fam}"
        own_group = _color_group(fam) if task == "binding" else None
        wrong_dts = []
        for t2, fam2 in all_fams:
            if (t2, fam2) == (task, fam) or t2 != task:
                continue
            if task == "binding" and _color_group(fam2) == own_group:
                continue  # 同组其它转换（共享 sphere 颜色词集），归入 same 组
            wrong_dts.append(dt_by_family[f"{t2}|{fam2}"]["dt_mean"])
        held = by_fam.get((task, fam, "heldout"), [])
        if wrong_dts and held:
            cs = [float(np.mean([e["dv"] @ dt for dt in wrong_dts])) for e in held]
            row["C_crossgroup_heldout"] = float(np.mean(cs))
        rows.append(row)
    # 任务级主统计（预先指定，工作方案第 7 节：逐转换结果只作探索性描述）：
    # 留出场景合并的 C + 符号翻转检验；判定必须同时看方向（C_mean > 0），
    # 双侧 p 小只能说明偏离 0，不能区分正向对应与反向对应（审查问题 3）
    for task in sorted(n_cond_by_task):
        held = [e for (t2, _f2, s2), es in by_fam.items()
                if t2 == task and s2 == "heldout" for e in es]
        if not held:
            continue
        Cs = [e["C"] for e in held]
        mean, lo, hi = cluster_ci(Cs, [e["scene"] for e in held],
                                  boot_cfg["n_boot"], boot_cfg["ci"])
        test = sign_flip_test(Cs, perm_cfg["n"], seed=0)
        rows.append({"task": task, "family": "__overall__",
                     "n_conditions": n_cond_by_task[task],
                     "n_heldout": len(held),
                     "C_mean_heldout": mean,
                     "C_ci_low_heldout": lo, "C_ci_high_heldout": hi,
                     "p_heldout": test["p"], "n_perm": test["n_perm_used"],
                     "exact": test["exact"],
                     "direction_positive": bool(mean > 0)})
    logger.info(f"[e4c] 代数自检 max|C − Δv·Δt| = {identity_max_diff:.3e}")
    return {"rows": rows,
            "identity_check": {"max_abs_diff": identity_max_diff}}
