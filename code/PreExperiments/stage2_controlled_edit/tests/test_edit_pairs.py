"""e4.pairs 单元测试：确定性、划分完整性、序列化后的编辑校验、账本去重。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from e4 import pairs as P
from generation.edits import EditValidationError, validate_edit
from generation.scene_spec import BaseSceneSpec

SEED = 20261103
_VALIDATE_KIND = {"swap": "binding_swap", "recolor": "recolor",
                  "flip": "relation_flip", "translate": "translate"}


def _build_binding():
    return P.build_task("binding", SEED, n_swap=24, n_recolor=24, n_control=8)


def _build_relation():
    return P.build_task("relation", SEED, n_flip=30)


def test_binding_determinism():
    a, b = _build_binding(), _build_binding()
    assert json.dumps(a["pairs"], sort_keys=True) == json.dumps(b["pairs"], sort_keys=True)
    assert json.dumps(a["states"], sort_keys=True) == json.dumps(b["states"], sort_keys=True)


def test_binding_all_pairs_validate_from_serialized():
    data = _build_binding()
    for p in data["pairs"]:
        kind = _VALIDATE_KIND.get(p["edit_kind"])
        before = BaseSceneSpec.from_dict(json.loads(p["before_spec"]))
        after = BaseSceneSpec.from_dict(json.loads(p["after_spec"]))
        if kind:
            validate_edit(before, after, kind)  # 不抛即通过
        else:
            assert p["before_spec"] == p["after_spec"]


def test_split_integrity():
    for data in (_build_binding(), _build_relation()):
        by_scene = {}
        cov = {}
        for p in data["pairs"]:
            by_scene.setdefault(p["base_scene_id"], set()).add(p["split"])
            cov.setdefault(p["family"], set()).add(p["split"])
        assert all(len(s) == 1 for s in by_scene.values()), "同场景必须同组"
        for fam, splits in cov.items():
            assert splits == {"train", "val", "heldout"}, f"{fam} 缺 split: {splits}"


def test_states_unique_and_shared_before():
    data = _build_binding()
    ids = [s["state_id"] for s in data["states"]]
    assert len(ids) == len(set(ids))
    swap_before = {p["base_scene_id"]: p["before_state_id"]
                   for p in data["pairs"] if p["edit_kind"] == "swap"}
    recolor_before = {p["base_scene_id"]: p["before_state_id"]
                      for p in data["pairs"] if p["edit_kind"] == "recolor"}
    shared = set(swap_before) & set(recolor_before)
    for sid in shared:
        assert swap_before[sid] == recolor_before[sid], "换色对应复用同一 before 状态"
    for s in data["states"]:
        if s["state_tag"] in ("jpeg", "crop"):
            assert not s["needs_render"] and s["source_state_id"]


def test_relation_flip_and_translate():
    data = _build_relation()
    kinds = {p["edit_kind"] for p in data["pairs"]}
    assert {"flip", "translate"} <= kinds
    for p in data["pairs"]:
        if p["edit_kind"] == "translate":
            before = BaseSceneSpec.from_dict(json.loads(p["before_spec"]))
            after = BaseSceneSpec.from_dict(json.loads(p["after_spec"]))
            a, b = after.obj("a"), after.obj("b")
            # 平移对照不得改变左右关系
            from generation.scene_spec import is_left_of
            assert is_left_of(a["pos"], b["pos"])


def test_manifest_roundtrip_and_tamper(tmp_path):
    data = _build_binding()
    P.write_manifest(tmp_path, "binding", data)
    loaded = P.load_manifest(tmp_path, "binding")
    assert len(loaded["pairs"]) == len(data["pairs"])
    pp = tmp_path / "binding_pairs.parquet"
    df = pd.read_parquet(pp)
    df.loc[0, "split"] = "heldout"
    df.to_parquet(pp, index=False)
    with pytest.raises(RuntimeError):
        P.load_manifest(tmp_path, "binding")


def test_wrong_edit_states_cover_eval_scenes():
    """审查问题 4：val/heldout 的语义编辑场景必须拥有同场景错误编辑状态。"""
    for data, wrong in ((_build_binding(), "recolor"), (_build_relation(), "translate")):
        states = {s["state_id"] for s in data["states"]}
        semantic = {p["base_scene_id"]: p["split"] for p in data["pairs"]
                    if p["edit_kind"] in ("swap", "flip")}
        for sid, split in semantic.items():
            if split == "train":
                continue
            assert f"{sid}::{wrong}" in states, \
                f"{split} 场景 {sid} 缺同场景错误编辑状态"


def test_pair_state_ids_reports_missing_candidates():
    """候选缺失必须显式报告（返回 missing），不得静默缩小候选池。"""
    from e4.data import pair_state_ids

    data = _build_binding()
    manifest = {"pairs": pd.DataFrame(data["pairs"]),
                "states": pd.DataFrame(data["states"])}
    swap = next(p for p in data["pairs"] if p["edit_kind"] == "swap")
    all_ids = [s["state_id"] for s in data["states"]]
    full_rendered = pd.DataFrame({"state_id": all_ids})
    pcfg = {"include_wrong_edit": True, "n_other_scene_after": 2}
    pool, missing = pair_state_ids(pcfg, manifest, full_rendered, swap, "binding")
    roles = {r for _, r, _ in pool}
    assert {"before", "after", "wrong_edit", "other_after"} <= roles
    assert missing == []
    # 换色态未渲染 → missing 必须列出 wrong_edit，而不是少放一张图
    partial = pd.DataFrame({"state_id":
                            [s for s in all_ids if not s.endswith("::recolor")]})
    pool2, missing2 = pair_state_ids(pcfg, manifest, partial, swap, "binding")
    assert any(m.startswith("wrong_edit:") for m in missing2)
    assert all(not sid.endswith("::recolor") for sid, _, _ in pool2)
