"""E4 汇总与阶段判断（阶段二 Step 5，工作方案第 7 节）。

正式汇总（summarize）只消费已成功提交的正式产物：核对 run_meta.json 中
实验记录（status=success）与产物清单（last_success.outputs），禁止回退到
smoke 文件（审查问题 7）；检查预先确定的任务、转换是否齐全，缺失逐项
列入 missing。smoke 汇总走 summarize_smoke，输出独立文件名，只记录流程
是否跑通，不给出机制支持标签。

判定规则（正式 E4 运行前冻结，随产物写进 e4_stage_decision.json，避免
事后口径漂移；确认性检验族 = E4-A 每任务 1 个 + E4-C 每任务 1 个 +
E4-D 每任务 × 规定对照各 1 个，族内 Holm step-down 校正，审查问题 8）：
- 结构可辨识（E4-A）：任务级 heldout pair_acc 的场景内置换 Holm p < α
  且 pair_acc > null_mean；
- 跨模态对应（E4-C）：任务级（__overall__）heldout C 的符号翻转 Holm
  p < α 且 C_mean > 0（方向检查，审查问题 3：双侧 p 小不能区分正向与
  反向；逐转换结果仅探索性描述，不进入判定）；
- 方向迁移（E4-D）：正确方向相对每个规定对照的同场景配对差 Holm p < α
  且差均值 > 0；随机方向的多次重复先按场景平均（不当作额外独立场景），
  正确方向与全部规定对照必须覆盖同一批留出场景，否则记数据不足。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generation.scene_spec import BINDING_CONVERSIONS, RELATION_TYPE_PAIRS_TRAIN
from src.metrics.bootstrap import cluster_mean_ci
from src.utils.io import load_config

from .stats import sign_flip_test

SUPPORTED = ("有支持证据", "证据不足", "数据不足")
TASKS = ("binding", "relation")
EXPECTED_FAMILIES = {"binding": len(BINDING_CONVERSIONS),
                     "relation": len(RELATION_TYPE_PAIRS_TRAIN)}
# 各产物由哪些正式实验产生（run_meta.experiments 键；smoke 实验不在其列）
PRODUCERS = {
    "e4a_results.csv": ("e4_all", "e4_a"),
    "e4c_conversion_stats.csv": ("e4_all", "e4_c"),
    "e4d_retrieval.csv": ("e4_all", "e4_d"),
}
# E4-D 规定对照（α=0 是流程参照，不作确认性比较，方案第 6 节）
TRANSFER_CONTROLS = ("wrong_conversion", "random", "reverse", "text_delta")


def _label(cond, missing=False):
    if missing:
        return SUPPORTED[2]
    return SUPPORTED[0] if cond else SUPPORTED[1]


def _holm(pvals: dict, alpha: float) -> dict:
    """Holm step-down 校正：{key: p} → {key: {"p_raw", "p_holm"}}。

    p_holm 单调不减、不超过 1；满足 p_holm < alpha 等价于 Holm 拒绝。
    """
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, max(running, (m - i) * float(p)))
        running = a
        adj[k] = {"p_raw": float(p), "p_holm": float(a)}
    return adj


def _committed(run_meta: dict, filename: str):
    """filename 是否由某个正式实验成功提交（返回实验名，否则 None）。"""
    exps = (run_meta or {}).get("experiments", {})
    for name in PRODUCERS.get(filename, ()):
        rec = exps.get(name) or {}
        outs = (rec.get("last_success") or {}).get("outputs") or []
        if rec.get("status") == "success" and filename in outs:
            return name
    return None


def _fnum(x, default=np.nan):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return default


def summarize(model_name: str, seed: int, paths: dict, logger) -> dict:
    cfg = load_config("e4_protocol")
    scfg = cfg.get("summary", {}) or {}
    alpha = float(scfg.get("alpha", cfg["permutation"]["alpha"]))
    tasks = tuple(scfg.get("tasks", TASKS))
    controls = tuple(scfg.get("transfer_controls", TRANSFER_CONTROLS))
    tables = paths["dirs"]["outputs"] / model_name / f"seed_{seed}" / "tables"
    decision = {
        "model": model_name, "seed": seed, "tables_dir": str(tables),
        "alpha": alpha,
        "rules": {
            "confirmatory_family":
                "E4-A 每任务 1 个 + E4-C 每任务 1 个 + E4-D 每任务×规定"
                f"对照{list(controls)}各 1 个",
            "multiplicity": "Holm step-down，族内校正后 p < alpha 才判支持",
            "structure": "E4-A 任务级 heldout 置换 Holm p < α 且 pair_acc > null_mean",
            "crossmodal": "E4-C 任务级 heldout C 符号翻转 Holm p < α 且 C_mean > 0"
                          "（方向检查；逐转换结果仅探索性）",
            "transfer": "E4-D 正确方向 vs 每个规定对照：同场景配对差 Holm p < α"
                        "且差均值 > 0；覆盖场景不一致记数据不足",
        },
        "provenance": {}, "holm_family": [],
        "structure_identifiability": {}, "crossmodal_alignment": {},
        "direction_transfer": {}, "missing": [],
    }
    summary_rows = []
    run_meta = {}
    mp = tables / "run_meta.json"
    if mp.exists():
        run_meta = json.loads(mp.read_text())
    else:
        decision["missing"].append("run_meta.json（无实验来源记录）")
    confirmatory = {}   # key -> 原始 p，Holm 后回填标签

    # ---------------- E4-A 结构可辨识 ----------------
    e4a_p = tables / "e4a_results.csv"
    src = _committed(run_meta, "e4a_results.csv")
    if src is None or not e4a_p.exists():
        decision["missing"].append(
            "e4a_results.csv" + ("（无成功提交记录）" if e4a_p.exists()
                                 else "（文件缺失）"))
    else:
        decision["provenance"]["e4a_results.csv"] = src
        a = pd.read_csv(e4a_p)
        overall = a[(a["family"] == "__overall__") & (a["split"] == "heldout")]
        for task in tasks:
            rows_t = overall[overall["task"] == task]
            if rows_t.empty:
                decision["missing"].append(f"e4a:{task}:任务级 heldout 行缺失")
                decision["structure_identifiability"][task] = {"label": SUPPORTED[2]}
                continue
            r = rows_t.iloc[0]
            p = _fnum(r.get("perm_p"))
            entry = {"pair_acc": _fnum(r["pair_acc"]), "perm_p": p,
                     "null_mean": _fnum(r.get("null_mean")),
                     "n_scenes": int(r.get("n_scenes", 0))}
            if np.isnan(p):
                entry["label"] = SUPPORTED[2]
            else:
                confirmatory[f"A:{task}"] = p
                entry["cond_raw"] = bool(
                    p < alpha and entry["pair_acc"] > entry["null_mean"])
            decision["structure_identifiability"][task] = entry

    # ---------------- E4-C 跨模态对应（任务级主统计 + 方向）----------------
    e4c_p = tables / "e4c_conversion_stats.csv"
    src = _committed(run_meta, "e4c_conversion_stats.csv")
    if src is None or not e4c_p.exists():
        decision["missing"].append(
            "e4c_conversion_stats.csv" + ("（无成功提交记录）" if e4c_p.exists()
                                          else "（文件缺失）"))
    else:
        decision["provenance"]["e4c_conversion_stats.csv"] = src
        c = pd.read_csv(e4c_p)
        fam_rows = c[c["family"] != "__overall__"]
        for task in tasks:
            grp = fam_rows[fam_rows["task"] == task]
            n_fam = len(grp["family"].unique())
            if n_fam and n_fam < EXPECTED_FAMILIES[task]:
                decision["missing"].append(
                    f"e4c:{task}:转换不齐全 {n_fam}/{EXPECTED_FAMILIES[task]}")
            entry = {}
            if n_fam == 0:
                decision["missing"].append(f"e4c:{task}:任务缺失")
                decision["crossmodal_alignment"][task] = {"label": SUPPORTED[2]}
                continue
            ov = c[(c["task"] == task) & (c["family"] == "__overall__")]
            if ov.empty or np.isnan(_fnum(ov.iloc[0].get("p_heldout"))):
                decision["missing"].append(f"e4c:{task}:任务级 __overall__ 行缺失")
                decision["crossmodal_alignment"][task] = {"label": SUPPORTED[2]}
                continue
            r = ov.iloc[0]
            p = _fnum(r["p_heldout"])
            entry = {"n_conditions": int(r.get("n_conditions", n_fam)),
                     "n_heldout": int(r.get("n_heldout", 0)),
                     "C_mean_heldout": _fnum(r.get("C_mean_heldout")),
                     "C_ci_low_heldout": _fnum(r.get("C_ci_low_heldout")),
                     "C_ci_high_heldout": _fnum(r.get("C_ci_high_heldout")),
                     "p_heldout": p, "n_perm": int(r.get("n_perm", 0)),
                     "exact": bool(r.get("exact", False))}
            confirmatory[f"C:{task}"] = p
            # 方向检查：C 显著为负同样 p<α，不能算跨模态对应成立
            entry["cond_raw"] = bool(p < alpha and entry["C_mean_heldout"] > 0)
            decision["crossmodal_alignment"][task] = entry

    # ---------------- E4-D 方向迁移（同场景配对差 + Holm）----------------
    e4d_p = tables / "e4d_retrieval.csv"
    src = _committed(run_meta, "e4d_retrieval.csv")
    if src is None or not e4d_p.exists():
        decision["missing"].append(
            "e4d_retrieval.csv" + ("（无成功提交记录）" if e4d_p.exists()
                                   else "（文件缺失）"))
    else:
        decision["provenance"]["e4d_retrieval.csv"] = src
        d = pd.read_csv(e4d_p)
        held = d[~d["family"].astype(str).eq("__overall__")]
        for task in tasks:
            grp = held[held["task"] == task]
            if grp.empty:
                decision["missing"].append(f"e4d:{task}:任务缺失")
                decision["direction_transfer"][task] = {"label": SUPPORTED[2]}
                continue
            # 逐场景汇总：correct 取值；随机方向的多次重复先按场景平均。
            # correct 行缺失/重复的场景不得被静默剔除（剔除会缩小场景集合，
            # 让覆盖检查失去意义）——记录后整个任务记数据不足
            by_scene, bad_correct = {}, []
            for scene, g in grp.groupby("scene"):
                corr = g[g["method"] == "correct"]
                if len(corr) != 1:
                    decision["missing"].append(
                        f"e4d:{task}:{scene}:correct 行数 {len(corr)} != 1")
                    bad_correct.append(scene)
                    continue
                e = {"correct": _fnum(corr["top1"].iloc[0])}
                for m in controls:
                    gm = g[g["method"] == m]
                    e[m] = _fnum(gm["top1"].mean()) if not gm.empty else None
                by_scene[scene] = e
            scenes = sorted(by_scene)
            controls_present = [m for m in controls
                                if all(by_scene[s][m] is not None for s in scenes)]
            controls_missing = [m for m in controls
                                if m not in controls_present]
            entry = {"n_scenes": len(scenes),
                     "correct_top1": (float(np.mean([by_scene[s]["correct"]
                                                     for s in scenes]))
                                      if scenes else float("nan")),
                     "controls": {}, "missing_controls": controls_missing}
            if controls_missing:
                decision["missing"].append(
                    f"e4d:{task}:对照未覆盖相同留出场景: {controls_missing}")
            if bad_correct:
                decision["missing"].append(
                    f"e4d:{task}:correct 行异常的场景 {len(bad_correct)} 个")
            for m in controls_present:
                # 同场景配对差（工作方案第 7 节：差值在同一批测试场景上
                # 配对计算，按基础场景聚类 bootstrap）
                diffs = [by_scene[s]["correct"] - by_scene[s][m] for s in scenes]
                mean, lo, hi = cluster_mean_ci(
                    diffs, list(scenes), n_boot=cfg["bootstrap"]["n_boot"],
                    ci=cfg["bootstrap"]["ci"], seed=0)
                test = sign_flip_test(diffs, cfg["permutation"]["n"], seed=0)
                confirmatory[f"D:{task}:{m}"] = test["p"]
                entry["controls"][m] = {
                    "mean_diff": mean, "diff_ci_low": lo, "diff_ci_high": hi,
                    "p_signflip": test["p"], "n_scenes": len(scenes)}
            if not controls_present or not scenes or bad_correct:
                entry["label"] = SUPPORTED[2]
            decision["direction_transfer"][task] = entry

    # ---------------- Holm 校正并回填标签 ----------------
    adj = _holm(confirmatory, alpha)
    decision["holm_family"] = [{"key": k, **v} for k, v in
                               sorted(adj.items(), key=lambda kv: kv[1]["p_holm"])]
    for task in tasks:
        e = decision["structure_identifiability"].get(task)
        if e and "cond_raw" in e:
            e["p_holm"] = adj[f"A:{task}"]["p_holm"]
            e["label"] = _label(e["cond_raw"] and e["p_holm"] < alpha)
            summary_rows.append({"axis": "structure", "task": task,
                                 "value": e["pair_acc"], "p": e["p_holm"],
                                 "label": e["label"]})
        e = decision["crossmodal_alignment"].get(task)
        if e and "cond_raw" in e:
            e["p_holm"] = adj[f"C:{task}"]["p_holm"]
            e["label"] = _label(e["cond_raw"] and e["p_holm"] < alpha)
            summary_rows.append({"axis": "crossmodal", "task": task,
                                 "value": e["C_mean_heldout"], "p": e["p_holm"],
                                 "label": e["label"]})
        e = decision["direction_transfer"].get(task)
        if e and e.get("label") is None:
            passes = [m for m, v in e["controls"].items()
                      if adj[f"D:{task}:{m}"]["p_holm"] < alpha
                      and v["mean_diff"] > 0]
            for m, v in e["controls"].items():
                v["p_holm"] = adj[f"D:{task}:{m}"]["p_holm"]
            e["label"] = _label(bool(passes) and len(passes) == len(e["controls"]))
            summary_rows.append({"axis": "transfer", "task": task,
                                 "value": e["correct_top1"], "p": None,
                                 "label": e["label"]})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(tables / "e4_summary.csv", index=False)
    (tables / "e4_stage_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2))
    logger.info(f"[e4-sum] {model_name}/seed_{seed}: "
                f"{json.dumps({k: v.get('label') for k, v in decision['direction_transfer'].items()})}")
    return decision


SMOKE_FILES = {"a": "e4a_results_smoke.csv", "c": "e4c_conversion_stats_smoke.csv",
               "d": "e4d_retrieval_smoke.csv"}


def summarize_smoke(model_name: str, seed: int, paths: dict, logger) -> dict:
    """smoke 汇总：只读 *_smoke.csv，独立文件名，只记录流程是否跑通。

    不读取正式文件、不给出机制支持标签（审查问题 7）。
    """
    tables = paths["dirs"]["outputs"] / model_name / f"seed_{seed}" / "tables"
    out = {"model": model_name, "seed": seed, "smoke": True,
           "tables_dir": str(tables), "axes": {}}
    rows = []
    for axis, fname in SMOKE_FILES.items():
        p = tables / fname
        rec = {"axis": axis, "file": fname}
        if not p.exists():
            rec.update({"ran": False, "n_rows": 0})
        else:
            df = pd.read_csv(p)
            per_task = {t: int((df["task"] == t).sum()) for t in TASKS
                        if "task" in df.columns}
            rec.update({"ran": True, "n_rows": len(df),
                        "rows_per_task": per_task})
        rows.append(rec)
        out["axes"][axis] = rec
    pd.DataFrame(rows).to_csv(tables / "e4_summary_smoke.csv", index=False)
    (tables / "e4_stage_decision_smoke.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    logger.info(f"[e4-sum-smoke] {model_name}/seed_{seed}: "
                + json.dumps({a: r["ran"] for a, r in out["axes"].items()}))
    return out
