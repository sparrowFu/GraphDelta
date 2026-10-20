"""CLEVR v1.0 场景标注加载与工具。

场景 JSON 为单个文件 {"scenes": [...]}，每个 scene 含：
  image_filename / objects / relationships / image_index
关系负例直接使用官方 relationships 字段（不自行从坐标推导），键为
left / right / front / behind，值为 [i, j]（第 i 个对象在第 j 个对象的该方位）。
"""
import json
from collections import Counter
from pathlib import Path

COLORS = ["gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan"]
SHAPES = ["cube", "sphere", "cylinder"]
SIZES = ["small", "large"]
MATERIALS = ["rubber", "metal"]

REL_PHRASE = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
}
REL_ANTONYM = {"left": "right", "right": "left", "front": "behind", "behind": "front"}

VALUE_SETS = {"color": COLORS, "shape": SHAPES, "size": SIZES, "material": MATERIALS}


def load_scenes(scene_file):
    scene_file = Path(scene_file)
    with open(scene_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scenes"]


def combo(obj):
    """对象完整属性组合键 (size, color, material, shape)。"""
    return (obj["size"], obj["color"], obj["material"], obj["shape"])


def combo_with(obj, key, value):
    o = dict(obj)
    o[key] = value
    return combo(o)


def scene_combo_counts(scene):
    return Counter(combo(o) for o in scene["objects"])


def color_shape_pairs(scene):
    """图中出现的 (color, shape) 集合。"""
    return {(o["color"], o["shape"]) for o in scene["objects"]}


def obj_full_desc(obj):
    """'a small red metal cube'"""
    return f"a {obj['size']} {obj['color']} {obj['material']} {obj['shape']}"


def obj_full_desc_with(obj, key, value):
    o = dict(obj)
    o[key] = value
    return obj_full_desc(o)


def image_path(scene, image_dir):
    return Path(image_dir) / scene["image_filename"]
