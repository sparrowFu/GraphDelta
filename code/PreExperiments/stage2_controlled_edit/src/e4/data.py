"""E4 数据装载与候选池构造（工作方案第 5、6 节）。"""
import json
from pathlib import Path

import pandas as pd

from .pairs import load_manifest  # noqa: F401 （run_e4 直接从 e4.data 取）


def rendered_states_path(render_root: Path, task: str) -> Path:
    return Path(render_root) / task / "rendered_states.parquet"


def load_rendered(render_root, task: str) -> pd.DataFrame:
    """渲染回填表：state_id / image_path / scene_json / validated。

    缺文件直接报错——E4 评测前必须先完成渲染（README Step 3）。
    """
    p = rendered_states_path(render_root, task)
    if not p.exists():
        raise FileNotFoundError(f"渲染产物缺失: {p}；先运行 render_edit_pairs.py")
    df = pd.read_parquet(p)
    bad = df[~df["validated"]]
    if len(bad):
        raise RuntimeError(f"{len(bad)} 个渲染状态未通过校验（先修复再评测）")
    return df


def semantic_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """语义编辑对（E4-B/C/D 的分析对象）：交换与关系反转。"""
    return pairs[pairs["edit_kind"].isin(["swap", "flip"])].copy()


def complete_semantic_pairs(manifest: dict, rendered: pd.DataFrame) -> pd.DataFrame:
    """只保留 before/after 状态都已渲染并通过校验的语义对（smoke 局部渲染用）。"""
    have = set(rendered["state_id"])
    sp = semantic_pairs(manifest["pairs"])
    ok = sp["before_state_id"].isin(have) & sp["after_state_id"].isin(have)
    return sp[ok].copy()


def pair_state_ids(pool_cfg: dict, manifest: dict, rendered: pd.DataFrame,
                   pair_row, task: str) -> tuple:
    """E4-D 候选池（第 6 节）：返回 (pool, missing)。

    pool: [(state_id, role, word_set_changed), ...]，按登记顺序去重：
    - before：原图；
    - after：正确编辑图；
    - wrong_edit：同基础场景的错误编辑（绑定=换色，词集变化单列；关系=共同
      平移，关系未反转）；
    - other_after：同转换、不同基础场景的 after（按 state_id 升序固定取前
      n 个，不随渲染进度漂移），检查是否保留原场景信息。
    missing: 规定候选缺失清单（角色:state_id / 数量缺口）。正式运行出现
    非空 missing 必须报错，不得静默缩小候选池后继续算正式结果（审查
    问题 4）；smoke 运行仅警告。
    """
    have = set(rendered["state_id"])
    pool, missing = [], []

    def _add(sid, role, ws=False):
        if sid in have:
            pool.append((sid, role, ws))
        else:
            missing.append(f"{role}:{sid}")

    _add(pair_row["before_state_id"], "before")
    _add(pair_row["after_state_id"], "after")
    states = manifest["states"]
    task_wrong = "recolor" if task == "binding" else "translate"
    if pool_cfg.get("include_wrong_edit", True):
        wrong_id = pair_row["base_scene_id"] + "::" + task_wrong
        if wrong_id != pair_row["after_state_id"]:
            ws = bool(states.loc[states["state_id"] == wrong_id, "state_tag"]
                      .eq("recolor").any())
            _add(wrong_id, "wrong_edit", ws)
    n_other = int(pool_cfg.get("n_other_scene_after", 0))
    if n_other > 0:
        fam = pair_row["family"]
        others = manifest["pairs"][
            (manifest["pairs"]["family"] == fam)
            & (manifest["pairs"]["edit_kind"] == ("swap" if task == "binding" else "flip"))
            & (manifest["pairs"]["base_scene_id"] != pair_row["base_scene_id"])]
        cands = sorted(set(others["after_state_id"])
                       - {pair_row["after_state_id"]})
        for sid in cands[:n_other]:
            _add(sid, "other_after")
        if len(cands) < n_other:
            missing.append(f"other_after:同家族候选不足 {len(cands)}/{n_other}")
    seen, out = set(), []
    for sid, role, ws in pool:
        if sid not in seen:
            seen.add(sid)
            out.append((sid, role, ws))
    return out, missing
