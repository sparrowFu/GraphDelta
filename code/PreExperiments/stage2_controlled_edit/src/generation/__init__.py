"""受控场景规格生成与编辑算子（方案第 3 节模块）。

与官方 clevr-dataset-gen（commit f0ce2c8）的兼容约定：
- 对象属性取值与 data/properties.json 一致（8 色 / 2 材质 / 2 尺寸 / 3 形状）；
- 地板约束同官方：x,y ∈ [-3,3]、表面间距 ≥ min_dist=0.25、
  基数方向 margin ≥ 0.4；
- 关系判定同官方 compute_all_relationships：dot(Δ, dir) > eps=0.2；
- 方向轴由 base_scene.blend 相机朝向决定，官方脚本只抖动相机位置、
  不改朝向，因此 left 等轴是常量（本模块取实测值，见 LEFT_VEC 出处），
  关系可在渲染前用纯 Python 精确计算。

规格（spec）只含语义与布置参数；高度、网格与渲染由官方 utils.add_object
在渲染驱动中完成，3d_coords 由渲染侧回填。
"""
from .scene_spec import (
    BINDING_CONVERSIONS,
    CAMERA_FRONT_VEC,
    CAMERA_LEFT_VEC,
    COLORS,
    FLOOR_BOUND,
    MARGIN,
    MIN_DIST,
    RELATION_MIN_PROJ,
    REL_EPS,
    RELATION_TYPE_PAIRS_TRAIN,
    SIZES,
    BaseSceneSpec,
    build_binding_base_scene,
    build_relation_base_scene,
    check_placement,
    is_left_of,
)
from .edits import (
    apply_binding_swap,
    apply_recolor,
    apply_relation_flip,
    apply_translate,
    validate_edit,
)

__all__ = [
    "BINDING_CONVERSIONS", "CAMERA_FRONT_VEC", "CAMERA_LEFT_VEC", "COLORS", "FLOOR_BOUND",
    "MARGIN", "MIN_DIST", "RELATION_MIN_PROJ", "REL_EPS",
    "RELATION_TYPE_PAIRS_TRAIN", "SIZES", "BaseSceneSpec",
    "build_binding_base_scene", "build_relation_base_scene", "check_placement", "is_left_of",
    "apply_binding_swap", "apply_recolor", "apply_relation_flip",
    "apply_translate", "validate_edit",
]
