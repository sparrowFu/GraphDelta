"""E4 汇总与阶段判断（阶段二 Step 5，脚本 summarize_e4.py 调用）。

只读各 e4*.csv，按冻结口径汇总；对照工作方案第 7 节的结束条件给出
"有支持证据 / 证据不足 / 数据不足"三级标注。判定规则随产物写进
e4_stage_decision.json，避免事后口径漂移：
- 结构可辨识（E4-A）：任务级 heldout pair_acc 的场景内置换 p < α；
- 跨模态对应（E4-C）：任务内 p < α 的转换占比 ≥ 1/2；
- 方向迁移（E4-D）：correct 的 top1 高于全部对照方法（α=0、错误转换、
  随机、反向、文本差）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED = ("有支持证据", "证据不足", "数据不足")


def _label(cond, missing=False):
    if missing:
        return SUPPORTED[2]
    return SUPPORTED[0] if cond else SUPPORTED[1]


def _find(tables, name: Path):
    """优先正式产物，回退 _smoke 产物（冒烟结果只查流程，不作阶段结论）。"""
    for f in (tables / name, tables / str(name).replace(".csv", "_smoke.csv")):
        if f.exists():
            return f
    return None


def summarize(model_name: str, seed: int, paths: dict, logger) -> dict:
    tables = paths["dirs"]["outputs"] / model_name / f"seed_{seed}" / "tables"
    decision = {"model": model_name, "seed": seed, "tables_dir": str(tables),
                "smoke_inputs": [], "structure_identifiability": {},
                "crossmodal_alignment": {}, "direction_transfer": {}, "missing": []}
    summary_rows = []

    e4a_p = _find(tables, "e4a_results.csv")
    if e4a_p is not None:
        if e4a_p.stem.endswith("_smoke"):
            decision["smoke_inputs"].append(e4a_p.name)
        a = pd.read_csv(e4a_p)
        overall = a[(a["family"] == "__overall__") & (a["split"] == "heldout")]
        for _, r in overall.iterrows():
            cond = (not np.isnan(r.get("perm_p", np.nan))) and r["perm_p"] < 0.05 \
                and r["pair_acc"] > r.get("null_mean", np.nan)
            decision["structure_identifiability"][r["task"]] = {
                "pair_acc": r["pair_acc"], "perm_p": r.get("perm_p", np.nan),
                "null_mean": r.get("null_mean", np.nan), "label": _label(cond)}
            summary_rows.append({"axis": "structure", "task": r["task"],
                                 "value": r["pair_acc"], "p": r.get("perm_p"),
                                 "label": _label(cond)})
    else:
        decision["missing"].append("e4a_results.csv")

    e4c_p = _find(tables, "e4c_conversion_stats.csv")
    if e4c_p is not None:
        if e4c_p.stem.endswith("_smoke"):
            decision["smoke_inputs"].append(e4c_p.name)
        c = pd.read_csv(e4c_p)
        for task, grp in c.groupby("task"):
            ps = grp["p_heldout"].dropna()
            if len(ps) == 0:
                decision["crossmodal_alignment"][task] = {"label": SUPPORTED[2]}
                continue
            frac = float((ps < 0.05).mean())
            decision["crossmodal_alignment"][task] = {
                "n_families": len(ps), "frac_significant": frac,
                "label": _label(frac >= 0.5)}
            summary_rows.append({"axis": "crossmodal", "task": task,
                                 "value": frac, "p": None,
                                 "label": _label(frac >= 0.5)})
    else:
        decision["missing"].append("e4c_conversion_stats.csv")

    e4d_p = _find(tables, "e4d_retrieval.csv")
    if e4d_p is not None:
        if e4d_p.stem.endswith("_smoke"):
            decision["smoke_inputs"].append(e4d_p.name)
        d = pd.read_csv(e4d_p)
        held = d[d["family"] != "__overall__"]
        for task, grp in held.groupby("task"):
            m = grp.groupby("method")["top1"].mean()
            if "correct" not in m:
                decision["direction_transfer"][task] = {"label": SUPPORTED[2]}
                continue
            controls = {k: float(v) for k, v in m.items() if k != "correct"}
            cond = all(m["correct"] > v for v in controls.values())
            decision["direction_transfer"][task] = {
                "correct_top1": float(m["correct"]), "controls": controls,
                "label": _label(cond)}
            summary_rows.append({"axis": "transfer", "task": task,
                                 "value": float(m["correct"]), "p": None,
                                 "label": _label(cond)})
    else:
        decision["missing"].append("e4d_retrieval.csv")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(tables / "e4_summary.csv", index=False)
    (tables / "e4_stage_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2))
    logger.info(f"[e4-sum] {model_name}/seed_{seed}: "
                f"{json.dumps({k: v.get('label') for k, v in decision['direction_transfer'].items()})}")
    return decision
