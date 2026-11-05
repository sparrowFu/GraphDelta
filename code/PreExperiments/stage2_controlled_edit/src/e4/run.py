"""E4 编排（阶段二 Step 4 入口逻辑，脚本 run_e4.py 调用）。

装载清单与渲染产物 → float32 缓存抽取 → 按 --stage 执行 E4-A/B/C/D →
按阶段一 run_meta 机制登记与提交。smoke 只加后缀与子采样提示，不改统计
口径；某阶段数据不足时警告并跳过，不中断其余阶段。
"""
import json
from pathlib import Path

import pandas as pd

from src.utils.io import commit_experiment, load_config, output_tables_dir, write_jsonl

from . import delta as e4delta
from . import paired as e4paired
from . import text_delta as e4text
from . import transfer as e4transfer
from .data import complete_semantic_pairs, load_rendered
from .features import e4_caches, extract_state_images, state_dataset_key
from .pairs import load_manifest


def run(model, paths, logger, stage="all", smoke=False, seed=None):
    cfg = load_config("e4_protocol")
    eff_seed = seed if seed is not None else cfg["seeds"][0]
    suffix = "_smoke" if smoke else ""
    experiment = f"e4_{stage}{suffix}"
    out = output_tables_dir(paths, model.name, eff_seed,
                            extra={"cache_protocol": model.cache_protocol},
                            experiment=experiment)
    icache, tcache = e4_caches(paths, model)
    ctx = {}
    for task in ("binding", "relation"):
        try:
            m = load_manifest(paths["dirs"]["edit_pairs"], task)
        except FileNotFoundError as e:
            logger.warning(f"[e4] {task} 清单缺失，跳过该任务: {e}")
            continue
        try:
            rendered = load_rendered(paths["render"]["output_dir"], task)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"[e4] {task} 渲染产物不可用，跳过该任务: {e}")
            continue
        sp = complete_semantic_pairs(m, rendered)
        if sp.empty:
            logger.warning(f"[e4] {task} 无完整语义对（渲染不完整？），跳过")
            continue
        lacking = [f for f, g in sp.groupby("family") if g["split"].nunique() < 3]
        if lacking:
            logger.warning(f"[e4] {task} 以下家族 split 不全（小批数据正常）: "
                           f"{sorted(lacking)[:8]}")
        # 缓存键带清单与图片内容指纹：任一图片重渲染/被替换后旧特征失效
        key = state_dataset_key(task, m["meta"], rendered)
        image_dir = Path(paths["render"]["output_dir"]) / task / "images"
        extract_state_images(model, icache, key, list(rendered["state_id"]),
                             image_dir, logger)
        raw_all, z_all = icache.get(key, list(rendered["state_id"]))
        have = set(rendered["state_id"])
        ctrl = e4delta.control_pairs(m["pairs"])
        ctrl = ctrl[ctrl["before_state_id"].isin(have)
                    & ctrl["after_state_id"].isin(have)]
        ctx[task] = {"manifest": m, "rendered": rendered, "sp": sp, "ctrl": ctrl,
                     "z": dict(zip(rendered["state_id"], z_all)),
                     "raw": dict(zip(rendered["state_id"], raw_all))}
    if not ctx:
        logger.warning("[e4] 无可用任务数据，空运行提交")

    S_records, dt_by_family = [], {}
    for task, c in ctx.items():
        if stage in ("all", "a", "c"):
            S_records += e4paired.compute_S_records(c["sp"], c["z"], tcache, model)
        if stage in ("all", "c", "d"):
            dt_by_family.update(e4text.family_text_deltas(c["sp"], tcache, model))

    files = []
    if stage in ("all", "a") and S_records:
        res = e4paired.run_e4a(S_records, cfg, logger)
        f = f"e4a_results{suffix}.csv"
        pd.DataFrame(res["rows"]).to_csv(out / f, index=False)
        write_jsonl(out / f"e4a_items{suffix}.jsonl",
                    [{**{k: v for k, v in rec.items() if k != "S"},
                      "S": rec["S"].tolist()} for rec in S_records])
        files += [f, f"e4a_items{suffix}.jsonl"]
    if stage in ("all", "a"):
        # 换色正控制（方案第 3、7 节）：同一套配对评分，确认评分流程能识别
        # 较简单的单对象换色（审查问题 5：控制不能只生成不评价）
        ctrl_S = []
        for task, c in ctx.items():
            recolor_df = c["ctrl"][c["ctrl"]["edit_kind"] == "recolor"]
            if not recolor_df.empty:
                ctrl_S += e4paired.compute_S_records(recolor_df, c["z"],
                                                     tcache, model)
        if ctrl_S:
            res = e4paired.run_e4a(ctrl_S, cfg, logger)
            f = f"e4a_control_results{suffix}.csv"
            pd.DataFrame(res["rows"]).to_csv(out / f, index=False)
            files.append(f)
    if stage in ("all", "b") and ctx:
        rows, items, dirs_all = [], [], {}
        for task, c in ctx.items():
            st = e4delta.run_e4b_stats(c["sp"], c["z"], c["raw"], cfg, logger)
            rows += st["rows"]
            items += st["items"]
            dirs_all.update(st["dirs"])
        if rows:
            f = f"e4b_delta_stats{suffix}.csv"
            pd.DataFrame(rows).to_csv(out / f, index=False)
            files.append(f)
        if items:
            write_jsonl(out / f"e4b_items{suffix}.jsonl", items)
            files.append(f"e4b_items{suffix}.jsonl")
        # 对照 Δv 诊断（jpeg / crop / 平移 / 换色）：与结构编辑同口径比较
        ctrl_rows = []
        for task, c in ctx.items():
            ctrl_rows += e4delta.run_e4b_controls(c["ctrl"], c["z"], dirs_all,
                                                  cfg, logger)
        if ctrl_rows:
            f = f"e4b_control_delta{suffix}.csv"
            pd.DataFrame(ctrl_rows).to_csv(out / f, index=False)
            files.append(f)
        # 方向可靠性（开篇待办 14）：训练内子集划分 + 按场景重采样波动
        rel_rows = []
        for task, c in ctx.items():
            rel_rows += e4delta.run_e4b_reliability(c["sp"], c["z"], cfg, logger)
        if rel_rows:
            f = f"e4b_reliability{suffix}.csv"
            pd.DataFrame(rel_rows).to_csv(out / f, index=False)
            files.append(f)
        probe_rows = []
        for task, c in ctx.items():
            probe_rows += e4delta.run_e4b_probes(
                c["manifest"]["pairs"], c["z"], c["rendered"], cfg, logger)
        if probe_rows:
            f = f"e4b_probe{suffix}.csv"
            pd.DataFrame(probe_rows).to_csv(out / f, index=False)
            files.append(f)
    if stage in ("all", "c") and S_records:
        rows, identity = [], 0.0
        for task, c in ctx.items():
            recs = [r for r in S_records if r["task"] == task]
            if not recs:
                continue
            res = e4text.run_e4c(recs, c["z"], dt_by_family, cfg, logger)
            rows += res["rows"]
            identity = max(identity, res["identity_check"]["max_abs_diff"])
        if rows:
            f = f"e4c_conversion_stats{suffix}.csv"
            pd.DataFrame(rows).to_csv(out / f, index=False)
            files.append(f)
            (out / f"e4c_identity{suffix}.json").write_text(json.dumps(
                {"max_abs_diff_C_vs_dv_dot_dt": identity}, indent=2))
            files.append(f"e4c_identity{suffix}.json")
    if stage in ("all", "d") and ctx:
        rows = []
        alpha, self_check = {}, {}
        for task, c in ctx.items():
            res = e4transfer.run_e4d(c["manifest"], c["rendered"], c["z"],
                                     dt_by_family, cfg, logger, smoke=smoke)
            rows += res["rows"]
            alpha.update(res["alpha"])
            self_check = res["self_check"]
        if rows:
            f = f"e4d_retrieval{suffix}.csv"
            pd.DataFrame(rows).to_csv(out / f, index=False)
            files.append(f)
            (out / f"e4d_meta{suffix}.json").write_text(json.dumps(
                {"alpha": alpha, "self_check": self_check}, indent=2))
            files.append(f"e4d_meta{suffix}.json")
    commit_experiment(out, experiment, files)
    logger.info(f"[e4] 完成 {experiment}: {len(files)} 个产物 -> {out}")
    return {"tables_dir": str(out), "files": files}
