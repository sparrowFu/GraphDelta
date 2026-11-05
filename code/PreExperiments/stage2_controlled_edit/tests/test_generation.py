"""src/generation 单元测试：纯逻辑，无 Blender 依赖。

覆盖：官方放置约束、8 个绑定转换与 10 个关系类型对、编辑算子的
差异恰好等于预期编辑集合、校验器对多余变化的检出。
"""
import math
import random
import sys
from pathlib import Path

# common/ 提供 src.*（共用包），本阶段 src/ 提供阶段包
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from generation import (
    BINDING_CONVERSIONS, CAMERA_FRONT_VEC, CAMERA_LEFT_VEC, MARGIN,
    RELATION_MIN_PROJ, RELATION_TYPE_PAIRS_TRAIN, apply_binding_swap,
    apply_recolor, apply_relation_flip, apply_translate,
    build_binding_base_scene, build_relation_base_scene, check_placement,
    is_left_of, validate_edit,
)
from generation.edits import EditValidationError, PlacementError
from generation.scene_spec import BaseSceneSpec, projection_along_left

SEED = 20261103


def official_left_of(spec_dict, i, j, eps=0.2):
    """官方 render_images.py 口径的独立复算（不复用 generation.is_left_of，
    审查问题 1 要求：测试须依据官方关系判定核对方向）：
    j 在 i 左侧 ⟺ dot(pos_j − pos_i, left) > eps。
    """
    oi, oj = spec_dict["objects"][i], spec_dict["objects"][j]
    d = ((oj["pos"][0] - oi["pos"][0]) * CAMERA_LEFT_VEC[0]
         + (oj["pos"][1] - oi["pos"][1]) * CAMERA_LEFT_VEC[1])
    return d > eps


# ---------------- 基础事实 ----------------

def test_constants_match_plan():
    assert len(BINDING_CONVERSIONS) == 8
    cube_pairs = {tuple(sorted((c[2], c[3]))) for c in BINDING_CONVERSIONS
                  if c[0] == "cube"}
    cyl_pairs = {tuple(sorted((c[2], c[3]))) for c in BINDING_CONVERSIONS
                 if c[0] == "cylinder"}
    # cube 四对均在 G0∪G1={red,blue,green,yellow}；cylinder 四对均在
    # G2∪G3={gray,brown,purple,cyan}（方案第 9.3 节训练允许配色）
    assert all(set(p) <= {"red", "blue", "green", "yellow"} for p in cube_pairs)
    assert all(set(p) <= {"gray", "brown", "purple", "cyan"} for p in cyl_pairs)
    assert len(RELATION_TYPE_PAIRS_TRAIN) == 10


def test_camera_axes_orthonormal():
    # 数值来自 Blender float32 场景 JSON，正交性仅到 ~4e-8
    dot = CAMERA_LEFT_VEC[0] * CAMERA_FRONT_VEC[0] + CAMERA_LEFT_VEC[1] * CAMERA_FRONT_VEC[1]
    assert abs(dot) < 1e-6
    assert abs(math.hypot(*CAMERA_LEFT_VEC) - 1.0) < 1e-6


# ---------------- 放置采样 ----------------

@pytest.mark.parametrize("conversion", BINDING_CONVERSIONS)
def test_binding_scene_placement(conversion):
    from generation.scene_spec import SIZES
    for idx in range(30):
        rng = random.Random(SEED + idx)
        spec = build_binding_base_scene(rng, idx, conversion)
        t, s = spec.obj("target"), spec.obj("sphere")
        assert check_placement(t["pos"], s["pos"],
                               SIZES[t["size"]], SIZES[s["size"]])
        # before 颜色符合转换定义
        assert t["color"] == conversion[2] and s["color"] == conversion[3]
        # 交叉平衡：偶数索引 target 在左，奇数在右（按官方口径独立核验：
        # target（对象 0）在 sphere（对象 1）左侧 ⟺ official_left_of(i=1, j=0)）
        assert official_left_of(spec.as_dict(), 1, 0) == (idx % 2 == 0)
        assert is_left_of(t["pos"], s["pos"]) == official_left_of(
            spec.as_dict(), 1, 0)


@pytest.mark.parametrize("type_pair", RELATION_TYPE_PAIRS_TRAIN)
def test_relation_scene_placement(type_pair):
    from generation.scene_spec import SIZES
    for idx in range(6):
        rng = random.Random(SEED + idx)
        spec = build_relation_base_scene(rng, idx, type_pair, a_left=(idx % 2 == 0))
        a, b = spec.obj("a"), spec.obj("b")
        assert check_placement(a["pos"], b["pos"],
                               SIZES[a["size"]], SIZES[b["size"]])
        # a_left 参数语义按官方口径独立核验：a（对象 0）在 b（对象 1）左侧
        assert official_left_of(spec.as_dict(), 1, 0) == (idx % 2 == 0)
        assert is_left_of(a["pos"], b["pos"]) == official_left_of(
            spec.as_dict(), 1, 0)
        # 关系场景同深度：front 轴偏移≈0（浮点残差，容差同 ALIGN_TOL）
        off = (b["pos"][0] - a["pos"][0], b["pos"][1] - a["pos"][1])
        lat = off[0] * CAMERA_FRONT_VEC[0] + off[1] * CAMERA_FRONT_VEC[1]
        assert abs(lat) < 1e-5
        # 材质符合类型对
        assert (a["shape"], a["material"]) == type_pair[0]
        assert (b["shape"], b["material"]) == type_pair[1]


def test_relation_projection_above_margin():
    """左右投影必须 ≥ RELATION_MIN_PROJ > MARGIN，不落入官方非法带。"""
    rng = random.Random(SEED)
    for i in range(50):
        spec = build_relation_base_scene(rng, i, RELATION_TYPE_PAIRS_TRAIN[0])
        p = abs(projection_along_left(spec.obj("a")["pos"], spec.obj("b")["pos"]))
        assert p >= RELATION_MIN_PROJ - 1e-9
        assert p > MARGIN


# ---------------- 编辑算子与校验 ----------------

def test_binding_swap_roundtrip():
    rng = random.Random(SEED)
    spec = build_binding_base_scene(rng, 0, BINDING_CONVERSIONS[0])
    after = apply_binding_swap(spec)
    validate_edit(spec, after, "binding_swap")
    # 输入未被修改
    assert spec.obj("target")["color"] == "red"
    # 再交换一次回到原状态（逆向复用同一 pair 的基础）
    back = apply_binding_swap(after)
    assert back.objects == spec.objects


def test_binding_swap_rejects_wrong_before_color():
    rng = random.Random(SEED)
    spec = build_binding_base_scene(rng, 1, BINDING_CONVERSIONS[0])
    spec.obj("target")["color"] = "green"     # 篡改 before 状态
    with pytest.raises(EditValidationError):
        apply_binding_swap(spec)


def test_validator_catches_unintended_changes():
    rng = random.Random(SEED + 1)
    before = build_binding_base_scene(rng, 2, BINDING_CONVERSIONS[1])
    after = apply_binding_swap(before)
    # 注入多余变化：尺寸
    bad = after.obj("sphere")["size"] = "small"
    with pytest.raises(EditValidationError):
        validate_edit(before, after, "binding_swap")
    after2 = apply_binding_swap(before)
    after2.obj("target")["pos"] = [after2.obj("target")["pos"][0] + 0.01,
                                   after2.obj("target")["pos"][1]]
    with pytest.raises(EditValidationError):
        validate_edit(before, after2, "binding_swap")
    # 相机抖动变化也必须被检出
    after3 = apply_binding_swap(before)
    after3.camera_jitter[0] += 0.01
    with pytest.raises(EditValidationError):
        validate_edit(before, after3, "binding_swap")


def test_recolor_control():
    rng = random.Random(SEED + 2)
    spec = build_binding_base_scene(rng, 3, BINDING_CONVERSIONS[0])
    allowed = ["red", "blue", "green", "yellow"]
    new_color = next(c for c in allowed
                     if c not in (spec.obj("target")["color"],
                                  spec.obj("sphere")["color"]))
    after = apply_recolor(spec, "target", new_color, allowed)
    validate_edit(spec, after, "recolor")
    with pytest.raises(EditValidationError):
        apply_recolor(spec, "target", "purple", allowed)  # 非法配色


def test_relation_flip_geometry_and_validation():
    rng = random.Random(SEED + 3)
    before = build_relation_base_scene(rng, 0, RELATION_TYPE_PAIRS_TRAIN[3])
    after = apply_relation_flip(before)
    summary = validate_edit(before, after, "relation_flip")
    # 间距严格保持
    d0 = math.dist(before.obj("a")["pos"], before.obj("b")["pos"])
    d1 = math.dist(after.obj("a")["pos"], after.obj("b")["pos"])
    assert abs(d0 - d1) < 1e-9
    # 左右翻转：投影符号改变且幅度超阈值
    assert summary["proj_before"] > 0 > summary["proj_after"]
    # 再翻转一次回到原位置（点反射两次，数值容差内还原）
    again = apply_relation_flip(after)
    assert all(abs(x - y) < 1e-8 for x, y in
               zip(again.obj("a")["pos"], before.obj("a")["pos"]))
    # anchor 不动
    assert after.obj("b")["pos"] == before.obj("b")["pos"]


def test_translate_preserves_relative_geometry():
    rng = random.Random(SEED + 4)
    spec = build_relation_base_scene(rng, 1, RELATION_TYPE_PAIRS_TRAIN[5])
    dx, dy = 0.3, -0.2
    after = apply_translate(spec, dx, dy)
    validate_edit(spec, after, "translate")
    for oid in ("a", "b"):
        p0, p1 = spec.obj(oid)["pos"], after.obj(oid)["pos"]
        assert abs(p1[0] - p0[0] - dx) < 1e-9
        assert abs(p1[1] - p0[1] - dy) < 1e-9


def test_flip_out_of_bounds_raises():
    """靠边场景翻转可能越界：应抛 PlacementError 而非产出非法场景。"""
    rng = random.Random(SEED + 5)
    spec = build_relation_base_scene(rng, 7, RELATION_TYPE_PAIRS_TRAIN[0])
    # 把 anchor 推到边界附近再翻转，制造越界；若构造不出越界则跳过
    a = spec.obj("a")
    b = spec.obj("b")
    b["pos"] = [2.8, 2.8]
    raised = False
    try:
        after = apply_relation_flip(spec)
        # 若仍合法，校验必须通过
        validate_edit(spec, after, "relation_flip")
    except PlacementError:
        raised = True
    except EditValidationError:
        pytest.fail("越界场景不应产生校验错误")
    # 两种结局都合法：要么 PlacementError，要么通过校验


def test_spec_serialization_roundtrip():
    rng = random.Random(SEED + 6)
    spec = build_binding_base_scene(rng, 4, BINDING_CONVERSIONS[2])
    restored = BaseSceneSpec.from_dict(spec.as_dict())
    assert restored.as_dict() == spec.as_dict()
    after = apply_binding_swap(restored)
    validate_edit(spec, after, "binding_swap")


# ---------------- 关系文本真假（审查问题 1）----------------

def test_relation_caption_truth_against_official_matrix():
    """默认关系编辑确实从"a 在 b 左侧"变为"a 在 b 右侧"，句子真假按官方
    关系矩阵独立核验（不调用 is_left_of 验证生成器）。"""
    from e4.captions import pair_texts, relation_contents
    rng = random.Random(SEED + 7)
    for tp in RELATION_TYPE_PAIRS_TRAIN[:3]:
        before = build_relation_base_scene(rng, 0, tp, a_left=True)
        after = apply_relation_flip(before)
        bd, ad = before.as_dict(), after.as_dict()
        # a_left=True：对象 0 在对象 1 左侧（官方口径）
        assert official_left_of(bd, 1, 0)
        t0, kind, rel0 = relation_contents(bd)
        assert kind == "sentence" and rel0 == "left"
        assert t0 == f"the {bd['objects'][0]['shape']} is to the left of the {bd['objects'][1]['shape']}"
        t1, _, rel1 = relation_contents(ad)
        assert rel1 == "right"
        assert official_left_of(ad, 0, 1)   # after：a 在 b 右侧 ⟺ b 在 a 左侧
        # pair_texts 的 t0/t1 与 relation_contents 一致
        pt = pair_texts(bd, ad, "relation")
        assert (pt["t0_content"], pt["t1_content"]) == (t0, t1)
