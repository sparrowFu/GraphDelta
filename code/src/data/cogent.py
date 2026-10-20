"""CLEVR-CoGenT：按 condition 加载 valA / valB 场景。格式与 CLEVR 相同。"""
from .clevr import load_scenes


def load_condition_scenes(cogent_root, condition):
    assert condition in ("A", "B"), condition
    scene_file = f"scenes/CLEVR_val{condition}_scenes.json"
    return load_scenes(f"{cogent_root}/{scene_file}")


def condition_image_dir(cogent_root, condition):
    return f"{cogent_root}/images/val{condition}"
