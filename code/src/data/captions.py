"""确定性 caption 构造（内容模板）与外层 prompt 包装。

内容 caption 不带句号；外层模板自带句号。
- phrase 类：对象 / 属性 / 场景 / binding / color-shape
- sentence 类：relation / position
"""
from .clevr import REL_ANTONYM, REL_PHRASE, obj_full_desc, obj_full_desc_with
from .pug_spar import POS_ANTONYM
from ..utils.io import load_config

_TEMPLATES = None


def templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = load_config("templates")
    return _TEMPLATES


def wrap(content: str, template: str) -> str:
    return template.format(content)


# ---------------- CLEVR / CoGenT ----------------

def clevr_attribute_contents(obj, key, neg_value):
    """单属性负例：object / color / size / material 共用。"""
    pos = obj_full_desc(obj)
    neg = obj_full_desc_with(obj, key, neg_value)
    return pos, neg


def clevr_relation_contents(obj_a, obj_b, rel):
    pos = f"{obj_full_desc(obj_a)} is {REL_PHRASE[rel]} {obj_full_desc(obj_b)}"
    anti = REL_ANTONYM[rel]
    neg = f"{obj_full_desc(obj_a)} is {REL_PHRASE[anti]} {obj_full_desc(obj_b)}"
    return pos, neg


def clevr_binding_contents(obj_a, obj_b):
    pos = f"a {obj_a['color']} {obj_a['shape']} and a {obj_b['color']} {obj_b['shape']}"
    neg = f"a {obj_b['color']} {obj_a['shape']} and a {obj_a['color']} {obj_b['shape']}"
    return pos, neg


def cogent_color_shape_contents(obj, neg_color=None, neg_shape=None):
    """E3：'a red cube'，负例换色或换形。"""
    pos = f"a {obj['color']} {obj['shape']}"
    neg_color = neg_color or obj["color"]
    neg_shape = neg_shape or obj["shape"]
    neg = f"a {neg_color} {neg_shape}"
    return pos, neg


# ---------------- PUG: SPAR ----------------

def _lower(tex):
    return str(tex).lower()


def spar_scene_contents(row, other_animal):
    c1 = row["character_name"]
    world = row["world_name"]
    if row["character2_name"] == "blank":
        pos = f"a {c1} in the {world}"
        neg = f"a {other_animal} in the {world}"
    else:
        c2 = row["character2_name"]
        pos = f"a {c1} and a {c2} in the {world}"
        neg = f"a {c1} and a {other_animal} in the {world}"
    return pos, neg


def spar_position_contents(row):
    c1, c2 = row["character_name"], row["character2_name"]
    rel = row["character1_pos"]              # left / below
    anti = POS_ANTONYM[rel]                  # right / above
    pos = f"the {c1} is to the {rel} of the {c2}"
    neg = f"the {c1} is to the {anti} of the {c2}"
    return pos, neg


def spar_attribute_contents(row):
    c1, c2 = row["character_name"], row["character2_name"]
    t1, t2 = _lower(row["character_texture"]), _lower(row["character2_texture"])
    pos = f"a {t1} {c1} and a {t2} {c2}"
    neg = f"a {t2} {c1} and a {t1} {c2}"
    return pos, neg


def spar_relation_contents(row):
    """role swap：主客体互换，方位词不变。"""
    c1, c2 = row["character_name"], row["character2_name"]
    rel = row["character1_pos"]
    pos = f"the {c1} is to the {rel} of the {c2}"
    neg = f"the {c2} is to the {rel} of the {c1}"
    return pos, neg


# ---------------- PUG: Animals（文本 nuisance 用） ----------------

def pug_animals_caption(row):
    return f"a {row['animal']} in the {row['environment']}"
