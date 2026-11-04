"""两对象受控场景规格：基础场景采样与关系计算（纯逻辑，无 Blender 依赖）。

规格结构（BaseSceneSpec.as_dict）：
    base_scene_id / family_id / conversion（绑定任务）或 type_pair（关系任务）
    camera_jitter: [dx, dy, dz]     相机位置抖动（渲染侧应用）
    lamp_jitter: {key/fill/back: [dx, dy, dz]}
    objects: [                      恰好两个对象，id 为 target/sphere 或 a/b
        id, shape, color, material, size, pos:[x, y], rotation
    ]
before/after 两个状态共享同一基础场景的全部上下文（相机、灯光、材质、
尺寸、位置、朝向），仅被编辑算子（edits.py）按规则改变。

放置约束与官方 render_images.py 一致：
- 地板 x,y ∈ [-3+r, 3-r]；
- 表面间距 dist − r_a − r_b ≥ min_dist=0.25；
- 基数方向 margin：对象间偏移在每个轴上的投影为 0 或 |proj| ≥ 0.4
  （官方判据 0 < margin < 0.4 为非法，恰好 0 合法，即同轴对齐）。
放置检查统一用保守半径 r=0.7（large），小对象自动满足。
"""
import math
import random
from dataclasses import dataclass, field

# ---- 官方 properties.json 的取值（来源：clevr-dataset-gen f0ce2c8）----
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["rubber", "metal"]
SIZES = {"large": 0.7, "small": 0.35}
SHAPES = ["cube", "sphere", "cylinder"]

# ---- 官方放置与关系约束 ----
FLOOR_BOUND = 3.0        # 地板坐标范围 [-3, 3]
MIN_DIST = 0.25          # 对象表面最小间距（官方 min_dist）
MARGIN = 0.4             # 基数方向间隔（官方 margin）
REL_EPS = 0.2            # 官方关系判定阈值：dot(Δ, dir) > eps
# 轴对齐容差：同轴对齐（front 分量应为 0）受浮点误差影响为 ±1e-17，
# 不加容差会被 margin 检查误判为非法
ALIGN_TOL = 1e-6
# 关系任务沿 left 轴的投影采样下限：必须 ≥ MARGIN（否则违反官方放置
# 约束），并明显高于 REL_EPS，避免边界不清样本
RELATION_MIN_PROJ = 0.45

# 相机 left / front 轴（地板平面分量）。来源：base_scene.blend 相机朝向
# 恒定（官方脚本只抖动相机位置、不改朝向）；数值取自 2026-11-03 试渲染
# trial_new_000000.json 的 directions，z 分量为 0。
CAMERA_LEFT_VEC = (-0.6563112735748291, -0.7544902563095093)
CAMERA_FRONT_VEC = (0.754490315914154, -0.6563112735748291)

# ---- 方案第 3 节：8 个默认绑定转换（从第 9.3 节合法训练转换中预选）----
# 每项 (目标形状, 次对象形状, 首色 c1, 次色 c2)：
# 编辑前 target=c1、sphere=c2；编辑后交换。逆向复用同一 pair。
BINDING_CONVERSIONS = [
    ("cube", "sphere", "red", "blue"),
    ("cube", "sphere", "red", "green"),
    ("cube", "sphere", "blue", "yellow"),
    ("cube", "sphere", "green", "yellow"),
    ("cylinder", "sphere", "gray", "brown"),
    ("cylinder", "sphere", "gray", "purple"),
    ("cylinder", "sphere", "brown", "cyan"),
    ("cylinder", "sphere", "purple", "cyan"),
]

# ---- 方案第 9.3 节：关系面板类型对（(shape, material) 六类的异形无序对）----
_OBJECT_TYPES = [(s, m) for s in ("cube", "cylinder", "sphere")
                 for m in ("rubber", "metal")]
_ALL_TYPE_PAIRS = sorted(
    {tuple(sorted((a, b))) for i, a in enumerate(_OBJECT_TYPES)
     for b in _OBJECT_TYPES[i + 1:] if a[0] != b[0]})
RELATION_OOD_VAL = tuple(sorted((("cube", "metal"), ("cylinder", "metal"))))
RELATION_OOD_TEST = tuple(sorted((("cube", "rubber"), ("cylinder", "metal"))))
RELATION_TYPE_PAIRS_TRAIN = [p for p in _ALL_TYPE_PAIRS
                             if p not in (RELATION_OOD_VAL, RELATION_OOD_TEST)]
assert len(_ALL_TYPE_PAIRS) == 12 and len(RELATION_TYPE_PAIRS_TRAIN) == 10


def projection_along_left(pos_a, pos_b):
    """b 相对 a 在 left 轴上的投影：> eps 表示 a 在 b 左侧。"""
    return ((pos_b[0] - pos_a[0]) * CAMERA_LEFT_VEC[0]
            + (pos_b[1] - pos_a[1]) * CAMERA_LEFT_VEC[1])


def is_left_of(pos_a, pos_b, eps=REL_EPS):
    """a 是否在 b 的左侧（相机坐标）。同官方：dot(b−a, left) > eps。"""
    return projection_along_left(pos_a, pos_b) > eps


def check_placement(pos_a, pos_b, r_a=0.7, r_b=0.7):
    """官方约束：地板范围、表面间距、基数方向 margin（允许恰为 0）。"""
    for (x, y), r in ((pos_a, r_a), (pos_b, r_b)):
        if abs(x) > FLOOR_BOUND - r or abs(y) > FLOOR_BOUND - r:
            return False
    dist = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
    if dist - r_a - r_b < MIN_DIST:
        return False
    off = (pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])
    for axis in (CAMERA_LEFT_VEC, CAMERA_FRONT_VEC):
        d = off[0] * axis[0] + off[1] * axis[1]
        if ALIGN_TOL < abs(d) < MARGIN:
            return False
    return True


@dataclass
class BaseSceneSpec:
    base_scene_id: str
    family_id: str
    objects: list           # [{id, shape, color, material, size, pos, rotation}]
    camera_jitter: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    lamp_jitter: dict = field(default_factory=lambda: {
        "key": [0.0, 0.0, 0.0], "fill": [0.0, 0.0, 0.0],
        "back": [0.0, 0.0, 0.0]})
    conversion: tuple = None        # 绑定任务：(shape, shape, c1, c2)
    type_pair: tuple = None         # 关系任务：((shape, mat), (shape, mat))
    task: str = ""

    def as_dict(self):
        return {
            "base_scene_id": self.base_scene_id,
            "family_id": self.family_id,
            "task": self.task,
            "conversion": list(self.conversion) if self.conversion else None,
            "type_pair": [list(t) for t in self.type_pair] if self.type_pair else None,
            "camera_jitter": list(self.camera_jitter),
            "lamp_jitter": {k: list(v) for k, v in self.lamp_jitter.items()},
            "objects": [dict(o) for o in self.objects],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            base_scene_id=d["base_scene_id"], family_id=d["family_id"],
            objects=[dict(o) for o in d["objects"]],
            camera_jitter=list(d["camera_jitter"]),
            lamp_jitter={k: list(v) for k, v in d["lamp_jitter"].items()},
            conversion=tuple(d["conversion"]) if d.get("conversion") else None,
            type_pair=tuple(tuple(t) for t in d["type_pair"]) if d.get("type_pair") else None,
            task=d.get("task", ""),
        )

    def obj(self, obj_id):
        return next(o for o in self.objects if o["id"] == obj_id)


def _sample_jitter(rng, amp):
    return [round(2.0 * amp * (rng.random() - 0.5), 6) for _ in range(3)]


def _sample_context(rng, material=None):
    """采样不被任务定义固定的上下文：材质（可指定）、尺寸、朝向。"""
    return {
        "material": material if material else rng.choice(MATERIALS),
        "size": rng.choice(list(SIZES)),
        "rotation": round(360.0 * rng.random(), 4),
    }


def _sample_pair_positions(rng, first_left, same_depth, r_a=0.7, r_b=0.7):
    """采样一对位置，返回 (pos_first, pos_second)；失败返回 (None, None)。

    first 在左（first_left=True）或在右。偏移沿 left 轴分解：
    proj 下限取 max(RELATION_MIN_PROJ, 由表面间距要求推出的下限)；
    横向分量（front 轴）在 same_depth=True 时恒为 0（关系任务同深度），
    否则从 {0} ∪ [±0.45, ±0.9] 采样（binding 任务布局多样性）。
    """
    lx, ly = CAMERA_LEFT_VEC
    px, py = -ly, lx              # front 轴单位向量
    sign = 1.0 if first_left else -1.0
    need_dist = r_a + r_b + MIN_DIST + 0.05
    for _ in range(500):
        ax, ay = rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8)
        if same_depth:
            lat = 0.0
        elif rng.random() < 0.5:
            lat = 0.0
        else:
            lat = rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 0.9)
        # 间距约束换算到 proj 下限：hypot(proj, lat) ≥ need_dist
        proj_lo = 0.0 if abs(lat) >= need_dist else math.sqrt(
            need_dist ** 2 - lat ** 2)
        proj_lo = max(RELATION_MIN_PROJ, proj_lo)
        proj = rng.uniform(proj_lo, proj_lo + 0.9)
        off = (sign * proj * lx + lat * px, sign * proj * ly + lat * py)
        pos_a, pos_b = (ax, ay), (ax + off[0], ay + off[1])
        if not check_placement(pos_a, pos_b, r_a, r_b):
            continue
        if same_depth:
            # 关系任务：翻转后的镜像位置也必须在界内（apply_relation_flip
            # 做点反射 2·pos_b − pos_a），否则该场景无法生成 after 状态
            mirror = (2 * pos_b[0] - pos_a[0], 2 * pos_b[1] - pos_a[1])
            if not check_placement(mirror, pos_b, r_a, r_b):
                continue
        return pos_a, pos_b
    return None, None


def build_binding_base_scene(rng: random.Random, scene_idx: int,
                             conversion: tuple, jitter_amp=0.5):
    """绑定任务基础场景：target(cube/cylinder) + sphere，两色来自转换。

    交叉平衡 target 的左右位置（方案第 3 节）：偶数索引 target 在左、
    奇数在右。材质、尺寸、朝向、布局、相机与灯光抖动跨场景变化。
    """
    shape_t, shape_s, c1, c2 = conversion
    target_left = (scene_idx % 2 == 0)
    ctx_t, ctx_s = _sample_context(rng), _sample_context(rng)
    pos_t, pos_s = _sample_pair_positions(
        rng, target_left, same_depth=False,
        r_a=SIZES[ctx_t["size"]], r_b=SIZES[ctx_s["size"]])
    if pos_t is None:
        raise RuntimeError(f"绑定场景放置失败: {conversion} idx={scene_idx}")
    cid = f"{shape_t[:3]}{shape_s[:3]}_{c1[:2]}{c2[:2]}_{scene_idx:05d}"
    return BaseSceneSpec(
        base_scene_id=cid, family_id=f"{shape_t}-{shape_s}|{c1}-{c2}",
        task="binding", conversion=conversion,
        camera_jitter=_sample_jitter(rng, jitter_amp),
        lamp_jitter={k: _sample_jitter(rng, jitter_amp)
                     for k in ("key", "fill", "back")},
        objects=[
            {"id": "target", "shape": shape_t, "color": c1, **ctx_t,
             "pos": list(pos_t)},
            {"id": "sphere", "shape": shape_s, "color": c2, **ctx_s,
             "pos": list(pos_s)},
        ],
    )


# 关系任务各形状的训练允许配色（方案第 9.3 节：关系面板全部使用训练允许配色）
RELATION_ALLOWED_COLORS = {
    "cube": ["red", "blue", "green", "yellow"],
    "cylinder": ["gray", "brown", "purple", "cyan"],
    "sphere": COLORS,
}


def build_relation_base_scene(rng: random.Random, scene_idx: int,
                              type_pair: tuple, a_left=True, jitter_amp=0.5):
    """关系任务基础场景：类型对 (shape,material) 固定两对象的形状与材质。

    same_depth=True：两对象沿 left 轴排列、front 轴偏移为 0（方案第 3 节：
    A、B 间距与相对深度前后保持不变）。a_left 控制初始方向；配额生成
    时左右两侧等量，编辑后翻转。
    """
    (shape_a, mat_a), (shape_b, mat_b) = type_pair
    ctx_a = _sample_context(rng, material=mat_a)
    ctx_b = _sample_context(rng, material=mat_b)
    pos_a, pos_b = _sample_pair_positions(
        rng, a_left, same_depth=True,
        r_a=SIZES[ctx_a["size"]], r_b=SIZES[ctx_b["size"]])
    if pos_a is None:
        raise RuntimeError(f"关系场景放置失败: {type_pair} idx={scene_idx}")
    cid = f"rel_{shape_a[:2]}{mat_a[:2]}-{shape_b[:2]}{mat_b[:2]}_{scene_idx:05d}"
    return BaseSceneSpec(
        base_scene_id=cid,
        family_id=f"{type_pair[0]}|{type_pair[1]}",
        task="relation", type_pair=type_pair,
        camera_jitter=_sample_jitter(rng, jitter_amp),
        lamp_jitter={k: _sample_jitter(rng, jitter_amp)
                     for k in ("key", "fill", "back")},
        objects=[
            {"id": "a", "shape": shape_a, "color":
             rng.choice(RELATION_ALLOWED_COLORS[shape_a]),
             **ctx_a, "pos": list(pos_a)},
            {"id": "b", "shape": shape_b, "color":
             rng.choice(RELATION_ALLOWED_COLORS[shape_b]),
             **ctx_b, "pos": list(pos_b)},
        ],
    )
