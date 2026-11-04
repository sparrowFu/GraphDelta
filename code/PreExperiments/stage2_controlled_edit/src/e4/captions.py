"""E4 文本内容构造（工作方案第 3、5 节）。

内容（content）不套模板存清单，评测时按 kind 套 configs/templates.yaml 的
外层模板（与阶段一 common/src/data/captions.py 同一约定）：
- 绑定（kind=phrase）："a red cube and a blue sphere"，对象列举顺序固定
  target 在前（配置 captions.object_order，全方法一致）；
- 关系（kind=sentence）："the cube is to the left of the sphere"，before 真、
  after 假（编辑后变为 to the right of）。两对象形状不同（类型对取不同形状），
  用形状指代即可无歧义。

E4-C 第一版只用颜色和形状的绑定短描述（第 5 节）；材质 / 位置上下文文本
若以后加入，须单列任务与核验规则，不与本套协议合并。
"""
from generation.scene_spec import is_left_of

REL_PHRASE = {"left": "to the left of", "right": "to the right of"}

# 官方 properties.json 的颜色 RGB（0–255），颜色直方图参照用
CLEVR_COLOR_RGB = {
    "gray": (87, 87, 87), "red": (173, 35, 35), "blue": (42, 75, 215),
    "green": (29, 105, 20), "brown": (129, 74, 25), "purple": (129, 38, 192),
    "cyan": (41, 208, 208), "yellow": (255, 238, 51),
}


def binding_contents(spec_dict: dict):
    """绑定状态的描述内容：返回 (content, kind)。target 在前（配置固定）。"""
    objs = spec_dict["objects"]
    content = " and ".join(f"a {o['color']} {o['shape']}" for o in objs)
    return content, "phrase"


def relation_contents(spec_dict: dict):
    """关系状态的描述内容：返回 (content, kind, rel)。

    rel ∈ {"left", "right"}：a 相对 b 的方位（相机坐标，与 generation 的
    is_left_of 同口径）。构建器保证 |投影| > eps，无歧义。
    """
    a, b = spec_dict["objects"][0], spec_dict["objects"][1]
    rel = "left" if is_left_of(a["pos"], b["pos"]) else "right"
    content = f"the {a['shape']} is {REL_PHRASE[rel]} the {b['shape']}"
    return content, "sentence", rel


def pair_texts(before_spec: dict, after_spec: dict, task: str) -> dict:
    """一个编辑对的文本对：(T0=before 真句, T1=after 真句) 的 content 与 kind。

    E4-A 用同一对文本分别评两张图；E4-C 的 Δt = emb(T1) − emb(T0)。
    """
    if task == "binding":
        t0, kind = binding_contents(before_spec)
        t1, _ = binding_contents(after_spec)
        return {"t0_content": t0, "t1_content": t1, "kind": kind}
    if task == "relation":
        t0, kind, rel0 = relation_contents(before_spec)
        t1, _, rel1 = relation_contents(after_spec)
        if rel0 == rel1:
            raise ValueError(f"关系文本前后相同（编辑未反转关系）: {rel0}")
        return {"t0_content": t0, "t1_content": t1, "kind": kind}
    raise ValueError(f"未知任务: {task}")
