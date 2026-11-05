"""e4.transfer 单元测试：打乱配对自检的样本集合、候选池完整性（审查问题 4、9）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from e4 import pairs as P
from e4.transfer import run_e4d

SEED = 20261103


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


CFG = {
    "transfer": {"alpha_grid": [0.0, 0.5], "n_random_directions": 2,
                 "random_seed": 7},
    "candidates": {"include_wrong_edit": True, "n_other_scene_after": 1},
}


def _relation_setup():
    """10 个类型对 × 6 场景（4/1/1 划分），全部状态用确定随机向量充当 z。"""
    data = P.build_task("relation", SEED, n_flip=60)
    manifest = {"pairs": pd.DataFrame(data["pairs"]),
                "states": pd.DataFrame(data["states"]),
                "meta": data["meta"]}
    rng = np.random.default_rng(0)
    z = {s["state_id"]: rng.standard_normal(8)
         for s in data["states"]}
    rendered = pd.DataFrame({"state_id": [s["state_id"] for s in data["states"]]})
    return manifest, rendered, z


def test_shuffle_self_check_uses_same_train_set():
    """自检必须与 d_r 同批训练场景：差值应为浮点舍入（审查问题 9）。"""
    manifest, rendered, z = _relation_setup()
    res = run_e4d(manifest, rendered, z, {}, CFG, _Logger(), smoke=True)
    assert res["self_check"]["shuffled_mean_max_diff"] < 1e-10
    assert res["rows"]


def test_formal_run_raises_on_missing_candidates():
    """正式运行缺规定候选必须报错；smoke 只警告并继续（审查问题 4）。"""
    manifest, rendered, z = _relation_setup()
    # 去掉一个 heldout 场景的平移态（同场景错误编辑候选）
    held_flips = [p for p in manifest["pairs"].to_dict("records")
                  if p["edit_kind"] == "flip" and p["split"] == "heldout"]
    dropped = held_flips[0]["base_scene_id"] + "::translate"
    rendered2 = pd.DataFrame({"state_id": [s for s in rendered["state_id"]
                                           if s != dropped]})
    with pytest.raises(RuntimeError, match="候选池不完整"):
        run_e4d(manifest, rendered2, z, {}, CFG, _Logger(), smoke=False)
    # smoke：不抛，正常返回
    res = run_e4d(manifest, rendered2, z, {}, CFG, _Logger(), smoke=True)
    assert res["rows"]
    # 候选清单随行落盘（pool 列固定每个查询的实际候选）
    assert all("pool" in r and r["pool"] for r in res["rows"])
