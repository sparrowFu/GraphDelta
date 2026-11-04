"""编辑算子与场景图差异校验（方案第 3 节）。

每个算子输入一个 BaseSceneSpec（before 状态），返回编辑后的新 spec
（after 状态），不修改输入。validate_edit 校验 before/after 的场景图
差异恰好等于预期编辑集合：多余的差异（位置漂移、尺寸、材质、朝向、
相机、灯光变化）一律报错，不机械要求“只变一条边”（绑定交换同时改
两个对象的颜色，关系反转同步改变逆关系）。
"""
import copy
import math

from .scene_spec import (
    COLORS, REL_EPS, RELATION_MIN_PROJ, SIZES, BaseSceneSpec, check_placement,
    is_left_of, projection_along_left,
)


class EditValidationError(Exception):
    pass


class PlacementError(Exception):
    """编辑后位置违反放置约束（调用方应重采样基础场景）。"""
    pass


def _clone(spec: BaseSceneSpec) -> BaseSceneSpec:
    return copy.deepcopy(spec)


def apply_binding_swap(spec: BaseSceneSpec) -> BaseSceneSpec:
    """颜色归属交换（主任务）：target 与 sphere 的颜色互换，其余不变。"""
    if spec.task != "binding" or spec.conversion is None:
        raise EditValidationError("apply_binding_swap 需要绑定任务场景")
    out = _clone(spec)
    t, s = out.obj("target"), out.obj("sphere")
    # before 状态颜色必须等于转换定义的某一方向：target=c1、sphere=c2，
    # 或逆向 target=c2、sphere=c1（逆向复用同一 pair，方案第 3 节）
    _, _, c1, c2 = spec.conversion
    if (t["color"], s["color"]) not in ((c1, c2), (c2, c1)):
        raise EditValidationError(
            f"before 颜色与转换不符: target={t['color']}, sphere={s['color']} "
            f"(期望 ({c1},{c2}) 或其逆向)")
    t["color"], s["color"] = s["color"], t["color"]
    return out


def apply_recolor(spec: BaseSceneSpec, obj_id: str, new_color: str,
                  allowed_colors=None) -> BaseSceneSpec:
    """目标对象换色（正控制）：只改一个对象的颜色。

    allowed_colors 给定时校验新颜色在允许集合内（正式数据须遵守
    第 9.3 节训练允许配色）。
    """
    out = _clone(spec)
    o = out.obj(obj_id)
    if new_color == o["color"]:
        raise EditValidationError("换色正控制的新颜色与原颜色相同")
    if new_color not in COLORS:
        raise EditValidationError(f"非法颜色: {new_color}")
    if allowed_colors is not None and new_color not in allowed_colors:
        raise EditValidationError(f"新颜色 {new_color} 不在允许集合内")
    o["color"] = new_color
    return out


def apply_relation_flip(spec: BaseSceneSpec, mover_id="a",
                        anchor_id="b") -> BaseSceneSpec:
    """左右关系反转：mover 移动到 anchor 另一侧的镜像位置。

    点反射 p_mover' = 2·p_anchor − p_mover：间距不变、深度不变
    （same-depth 场景的 front 轴偏移保持 0），仅左右符号翻转。
    位置超出放置约束时抛 PlacementError，由调用方重采样基础场景。
    """
    if spec.task != "relation":
        raise EditValidationError("apply_relation_flip 需要关系任务场景")
    m, a = spec.obj(mover_id), spec.obj(anchor_id)
    if not is_left_of(m["pos"], a["pos"]) and not is_left_of(a["pos"], m["pos"]):
        raise EditValidationError("初始左右关系不明确（投影 ≤ eps）")
    out = _clone(spec)
    om = out.obj(mover_id)
    new_pos = [round(2 * a["pos"][0] - m["pos"][0], 9),
               round(2 * a["pos"][1] - m["pos"][1], 9)]
    other = out.obj(anchor_id)
    if not check_placement(new_pos, other["pos"],
                           SIZES[om["size"]], SIZES[other["size"]]):
        raise PlacementError(f"翻转后位置越界: {new_pos}")
    om["pos"] = new_pos
    return out


def apply_translate(spec: BaseSceneSpec, dx: float, dy: float) -> BaseSceneSpec:
    """共同平移（关系任务对照）：两对象同向位移，相对几何不变。"""
    out = _clone(spec)
    new_pos = {}
    for o in spec.objects:
        p = [o["pos"][0] + dx, o["pos"][1] + dy]
        new_pos[o["id"]] = p
    first, second = out.objects[0], out.objects[1]
    if not check_placement(new_pos[first["id"]], new_pos[second["id"]],
                           SIZES[first["size"]], SIZES[second["size"]]):
        raise PlacementError(f"平移后越界: dx={dx}, dy={dy}")
    for o in out.objects:
        o["pos"] = list(new_pos[o["id"]])
    return out


# ---------------- 校验 ----------------

def _ctx_equal(b: BaseSceneSpec, a: BaseSceneSpec, where: str):
    """相机、灯光、场景标识必须逐位一致。"""
    if (b.base_scene_id, b.family_id, b.task) != (
            a.base_scene_id, a.family_id, a.task):
        raise EditValidationError(f"{where}: 场景标识不一致")
    if b.camera_jitter != a.camera_jitter:
        raise EditValidationError(f"{where}: 相机抖动不一致")
    if b.lamp_jitter != a.lamp_jitter:
        raise EditValidationError(f"{where}: 灯光抖动不一致")


def _objects_equal_except(b_objs, a_objs, ignore_fields, where):
    """逐对象比较；ignore_fields 之外的全部字段必须一致。"""
    if [o["id"] for o in b_objs] != [o["id"] for o in a_objs]:
        raise EditValidationError(f"{where}: 对象 id/顺序不一致")
    for ob, oa in zip(b_objs, a_objs):
        for k in ob:
            if k in ignore_fields:
                continue
            if ob[k] != oa.get(k):
                raise EditValidationError(
                    f"{where}: 对象 {ob['id']} 字段 {k} 不一致 "
                    f"({ob[k]} -> {oa.get(k)})")


def validate_edit(before: BaseSceneSpec, after: BaseSceneSpec, kind: str,
                  tol: float = 1e-6) -> dict:
    """校验 before→after 的差异恰好等于 kind 的预期编辑集合。

    返回差异摘要；违反预期抛 EditValidationError。
    """
    _ctx_equal(before, after, kind)
    b_objs, a_objs = before.objects, after.objects

    if kind == "binding_swap":
        _objects_equal_except(b_objs, a_objs, ignore_fields={"color"},
                              where=kind)
        bt, bs = before.obj("target"), before.obj("sphere")
        at, as_ = after.obj("target"), after.obj("sphere")
        if at["color"] != bs["color"] or as_["color"] != bt["color"]:
            raise EditValidationError("binding_swap: 颜色未正确互换")
        if at["color"] == bt["color"]:
            raise EditValidationError("binding_swap: target 颜色未变化")
        return {"kind": kind, "changed": {"target.color": at["color"],
                                          "sphere.color": as_["color"]}}

    if kind == "recolor":
        changed = [ob["id"] for ob, oa in zip(b_objs, a_objs)
                   if ob["color"] != oa["color"]]
        _objects_equal_except(b_objs, a_objs, ignore_fields={"color"},
                              where=kind)
        if len(changed) != 1:
            raise EditValidationError(f"recolor: 应恰好改一个对象, 改了 {changed}")
        return {"kind": kind, "changed": {f"{changed[0]}.color":
                after.obj(changed[0])["color"]}}

    if kind == "relation_flip":
        # 仅允许 mover 的 pos 变化；其余对象（含 anchor）逐字段一致
        _objects_equal_except(b_objs, a_objs, ignore_fields={"pos"},
                              where=kind)
        bm, am = before.obj("a"), after.obj("a")
        ba = before.obj("b")
        if bm["pos"] == am["pos"]:
            raise EditValidationError("relation_flip: mover 位置未变化")
        if ba["pos"] != after.obj("b")["pos"]:
            raise EditValidationError("relation_flip: anchor 位置不应变化")
        d_b = math.dist(bm["pos"], ba["pos"])
        d_a = math.dist(am["pos"], after.obj("b")["pos"])
        if abs(d_b - d_a) > tol:
            raise EditValidationError(
                f"relation_flip: 间距变化 {d_b:.6f} -> {d_a:.6f}")
        p_before = projection_along_left(bm["pos"], ba["pos"])
        p_after = projection_along_left(am["pos"], after.obj("b")["pos"])
        for p, tag in ((p_before, "before"), (p_after, "after")):
            if abs(p) <= RELATION_MIN_PROJ - 1e-9:
                raise EditValidationError(
                    f"relation_flip: {tag} 投影 |{p:.4f}| 低于采样下限")
        if (p_before > 0) == (p_after > 0):
            raise EditValidationError("relation_flip: 左右关系未翻转")
        return {"kind": kind,
                "changed": {"a.pos": am["pos"]},
                "proj_before": p_before, "proj_after": p_after}

    if kind == "translate":
        deltas = set()
        for ob, oa in zip(b_objs, a_objs):
            deltas.add((round(oa["pos"][0] - ob["pos"][0], 9),
                        round(oa["pos"][1] - ob["pos"][1], 9)))
        if len(deltas) != 1:
            raise EditValidationError(f"translate: 平移量不一致 {deltas}")
        _objects_equal_except(b_objs, a_objs, ignore_fields={"pos"},
                              where=kind)
        dx, dy = deltas.pop()
        rel_b = (round(before.obj("a")["pos"][0] - before.obj("b")["pos"][0], 9),
                 round(before.obj("a")["pos"][1] - before.obj("b")["pos"][1], 9))
        rel_a = (round(after.obj("a")["pos"][0] - after.obj("b")["pos"][0], 9),
                 round(after.obj("a")["pos"][1] - after.obj("b")["pos"][1], 9))
        if rel_b != rel_a:
            raise EditValidationError("translate: 相对几何改变")
        return {"kind": kind, "changed": {"all.pos": (dx, dy)}}

    raise EditValidationError(f"未知编辑类型: {kind}")
