"""e4.delta 单元测试：对照 Δv 诊断接入、方向可靠性（审查问题 5、10）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from e4 import pairs as P
from e4.delta import (control_pairs, run_e4b_controls, run_e4b_reliability,
                      run_e4b_stats)

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
