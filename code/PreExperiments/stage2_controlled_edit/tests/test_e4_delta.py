"""e4.delta 单元测试：对照 Δv 诊断接入、方向可靠性、随机方向全组合、
双任务方向合并（审查问题 5、10 与随机方向/合并两处修复）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from e4 import pairs as P
from e4.delta import (control_pairs, merge_dirs, run_e4b_controls,
                      run_e4b_reliability, run_e4b_stats)
from e4.stats import cosine_rows

SEED = 20261103
DIM = 8


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


CFG = {
    "near_zero": {"train_quantile": 0.05},
    "bootstrap": {"n_boot": 50, "ci": 0.95},
    "transfer": {"n_random_directions": 4, "random_seed": 3},
    "direction_reliability": {"n_repeats": 10, "train_fraction": 0.5,
                              "seed": 5, "n_boot": 50, "near_zero_norm": 1e-8},
}


def _binding_setup(n_swap=48):
    """8 转换 × 6 场景（4/1/1）；z 使每家族共享一个固定变化方向 + 小噪声。"""
    data = P.build_task("binding", SEED, n_swap=n_swap, n_recolor=8,
                        n_control=4)
    rng = np.random.default_rng(1)
    fam_dir = {}
    z = {}
    for s in data["states"]:
        sid = s["state_id"]
        fam = s["family"]
        if fam not in fam_dir:
            fam_dir[fam] = rng.standard_normal(DIM)
            fam_dir[fam] /= np.linalg.norm(fam_dir[fam])
        if s["state_tag"] == "before":
            base = rng.standard_normal(DIM)
            z[sid] = base
            # after 态 = 同场景 before + 家族方向 + 小噪声（保证 dv 主向共享）
            for tag in ("swap", "recolor"):
                z[f"{sid.rsplit('::', 1)[0]}::{tag}"] = \
                    base + 0.8 * fam_dir[fam] + 0.02 * rng.standard_normal(DIM)
    pairs = pd.DataFrame(data["pairs"])
    states = pd.DataFrame(data["states"])
    manifest = {"pairs": pairs, "states": states, "meta": data["meta"]}
    return manifest, pairs, z


def test_stats_return_dirs_and_heldout_references():
    """run_e4b_stats 返回 d_r/阈值供对照复用；留出行带随机与错误方向参照。"""
    manifest, pairs, z = _binding_setup()
    sp = pairs[pairs["edit_kind"].isin(["swap", "flip"])]
    raw = dict(z)
    res = run_e4b_stats(sp, z, raw, CFG, _Logger())
    dirs = res["dirs"]
    assert len(dirs["d_r"]) == 8 and len(dirs["thr"]) == 8
    held = [r for r in res["rows"]
            if r["split"] == "heldout"
            and r.get("cos_to_random_mean", float("nan")) == r.get("cos_to_random_mean")]
    assert held, "留出行应包含 cos_to_random_mean"
    # 共享方向构造：留出 cos_to_d_r 应明显高于随机与错误方向参照
    for r in held:
        assert r["cos_to_d_r"] > 0.9
        assert r["cos_to_d_r"] > r["cos_to_random_mean"] + 0.3
        assert r["cos_to_d_r"] > r["cos_to_wrong_conversion"] + 0.3


def test_reliability_reports_subset_stability():
    """对半划分子集方向高度一致；重采样波动字段齐全（审查问题 10）。"""
    manifest, pairs, z = _binding_setup()
    sp = pairs[pairs["edit_kind"].isin(["swap", "flip"])]
    rows = run_e4b_reliability(sp, z, CFG, _Logger())
    assert len(rows) == 8
    for r in rows:
        assert r["n_train"] == 4
        assert r["half_split_cos_mean"] > 0.9
        assert r["half_split_near_zero_frac"] == 0.0
        assert r["boot_cos_to_full_mean"] > 0.9
        assert r["boot_norm_std"] >= 0.0


def test_controls_enter_evaluation():
    """换色对照进入 Δv 诊断（同口径可比较），不被 swap/flip 过滤掉。"""
    manifest, pairs, z = _binding_setup()
    sp = pairs[pairs["edit_kind"].isin(["swap", "flip"])]
    st = run_e4b_stats(sp, z, dict(z), CFG, _Logger())
    ctrl = control_pairs(pairs)
    assert {"recolor", "jpeg", "crop"} <= set(ctrl["edit_kind"])
    rows = run_e4b_controls(ctrl, z, st["dirs"], CFG, _Logger())
    kinds = {r["edit_kind"] for r in rows}
    assert "recolor" in kinds
    for r in rows:
        assert r["n"] >= 1 and r["dv_norm_mean"] > 0
        # 换色对照与结构方向同族：字段口径一致
        assert "cos_to_structural_d_r" in r or r["family"] not in st["dirs"]["d_r"]


def _manual_pairs(n_train, n_val, n_held, dim=8, seed=0):
    """单家族手工对：dv = 固定方向 + 小噪声，留出数可任意指定。"""
    rows, z = [], {}
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(dim)
    for split, n in (("train", n_train), ("val", n_val), ("heldout", n_held)):
        for i in range(n):
            scene = f"{split}_{i}"
            b, a = f"s_{scene}_b", f"s_{scene}_a"
            base = rng.standard_normal(dim)
            z[b], z[a] = base, base + d + 0.01 * rng.standard_normal(dim)
            rows.append({"task": "binding", "family": "fam0", "edit_kind": "swap",
                         "base_scene_id": scene, "split": split,
                         "before_state_id": b, "after_state_id": a})
    return pd.DataFrame(rows), z, d


def test_random_direction_reference_all_pairs():
    """留出数 ≠ 1 且 ≠ 随机方向数时不崩溃，值与逐场景全组合手工计算一致。

    修复前 cosine_rows 逐行配对要求行数可广播：留出 ≥2 即 ValueError，
    默认规模（绑定 20 / 关系 10 个留出 vs 32 个随机方向）必崩。
    """
    n_rand = 5
    cfg = {**CFG, "transfer": {"n_random_directions": n_rand, "random_seed": 7}}
    pairs, z, _ = _manual_pairs(n_train=8, n_val=2, n_held=3)
    res = run_e4b_stats(pairs, z, dict(z), cfg, _Logger())
    held = [r for r in res["rows"] if r["split"] == "heldout"][0]
    # 手工重放随机方向（单家族：heldout 块只抽一次），独立计算期望值
    rng = np.random.default_rng(7)
    rand_mat = rng.standard_normal((n_rand, 8))
    dv = np.stack([z[f"s_heldout_{i}_a"] - z[f"s_heldout_{i}_b"]
                   for i in range(3)])
    An = dv / np.linalg.norm(dv, axis=1, keepdims=True)
    Bn = rand_mat / np.linalg.norm(rand_mat, axis=1, keepdims=True)
    expected = float((An @ Bn.T).mean(axis=1).mean())
    assert abs(held["cos_to_random_mean"] - expected) < 1e-12
    # 每场景与自身固定方向高度同向，应明显优于随机方向参照
    assert held["cos_to_d_r"] > held["cos_to_random_mean"] + 0.3


def test_random_direction_one_scene_still_valid():
    """留出恰为 1 个场景（旧行为的隐含前提）仍给出同样的全组合口径。"""
    cfg = {**CFG, "transfer": {"n_random_directions": 4, "random_seed": 3}}
    pairs, z, _ = _manual_pairs(n_train=6, n_val=2, n_held=1)
    res = run_e4b_stats(pairs, z, dict(z), cfg, _Logger())
    held = [r for r in res["rows"] if r["split"] == "heldout"][0]
    rng = np.random.default_rng(3)
    rand_mat = rng.standard_normal((4, 8))
    dv = (z["s_heldout_0_a"] - z["s_heldout_0_b"]).reshape(1, -1)
    expected = float(cosine_rows(
        np.tile(dv / np.linalg.norm(dv), (4, 1)),
        rand_mat / np.linalg.norm(rand_mat, axis=1, keepdims=True)).mean())
    assert abs(held["cos_to_random_mean"] - expected) < 1e-12


def test_merge_dirs_keeps_both_tasks():
    """逐内层合并：两任务的 d_r/thr 都保留（顶层 update 会整体覆盖）。"""
    d1 = {"d_r": {"binding|f": "b"}, "thr": {"binding|f": 1.0}}
    d2 = {"d_r": {"relation|g": "r"}, "thr": {"relation|g": 2.0}}
    merged = merge_dirs([d1, d2])
    assert set(merged["d_r"]) == {"binding|f", "relation|g"}
    assert set(merged["thr"]) == {"binding|f", "relation|g"}


def test_controls_keep_direction_after_two_task_merge():
    """按 run.py 的实际编排（双任务 stats → merge_dirs → 对照诊断）：
    绑定对照仍能拿到自己任务的结构方向，不被关系任务覆盖。"""
    dirs_list, ctrl_ctx = [], {}
    for task, kw in (("binding", dict(n_swap=48, n_recolor=8, n_control=4)),
                     ("relation", dict(n_flip=40, n_control=4))):
        data = P.build_task(task, SEED, **kw)
        pairs = pd.DataFrame(data["pairs"])
        z = _two_task_features(data["states"])
        sp = pairs[pairs["edit_kind"].isin(["swap", "flip"])]
        st = run_e4b_stats(sp, z, dict(z),
                           {**CFG, "transfer": {"n_random_directions": 1,
                                                "random_seed": 3}}, _Logger())
        dirs_list.append(st["dirs"])
        ctrl = control_pairs(pairs)
        have = set(z)
        ctrl_ctx[task] = (ctrl[ctrl["before_state_id"].isin(have)
                               & ctrl["after_state_id"].isin(have)], z)
    dirs_all = merge_dirs(dirs_list)
    prefixes = {k.split("|")[0] for k in dirs_all["d_r"]}
    assert prefixes == {"binding", "relation"}
    for task, (ctrl, z) in ctrl_ctx.items():
        rows = run_e4b_controls(ctrl, z, dirs_all, CFG, _Logger())
        assert rows, f"{task} 应有对照行"
        for r in rows:
            assert "cos_to_structural_d_r" in r and "near_zero_frac" in r


def _two_task_features(states):
    """两任务通用特征：before 向量 + 家族方向的 after（含 flip/translate）。"""
    rng = np.random.default_rng(1)
    fam_dir, base_of, z = {}, {}, {}
    for s in states:
        if s["state_tag"] == "before":
            base_of[s["base_scene_id"]] = rng.standard_normal(DIM)
    for s in states:
        fam = s["family"]
        if fam not in fam_dir:
            fam_dir[fam] = rng.standard_normal(DIM)
        if s["state_tag"] == "before":
            z[s["state_id"]] = base_of[s["base_scene_id"]]
        else:
            z[s["state_id"]] = (base_of[s["base_scene_id"]] + 0.8 * fam_dir[fam]
                                + 0.02 * rng.standard_normal(DIM))
    return z
