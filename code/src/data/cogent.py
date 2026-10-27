"""CLEVR-CoGenT：按 condition 加载 valA / valB 场景。格式与 CLEVR 相同。

路径可由 configs/e3_cogent.yaml 的 conditions.<cond>.scene_file / image_dir
覆盖（四轮补充 A：此前配置字段从未生效，实验入口与独立抽取脚本都固定读
scenes/CLEVR_val{A,B}_scenes.json 与 images/val{A,B}）；非默认路径时缓存
key 由 condition_cache_key 附加路径指纹——更换数据源后不能复用默认路径的
同名旧特征。
"""
from ..utils.io import md5
from .clevr import load_scenes


def load_condition_scenes(cogent_root, condition, scene_file=None):
    assert condition in ("A", "B"), condition
    scene_file = scene_file or f"scenes/CLEVR_val{condition}_scenes.json"
    return load_scenes(f"{cogent_root}/{scene_file}")


def condition_image_dir(cogent_root, condition, image_dir=None):
    image_dir = image_dir or f"images/val{condition}"
    return f"{cogent_root}/{image_dir}"


def condition_cache_key(base_key, scene_file=None, image_dir=None):
    """非默认数据路径时给缓存 key 附加路径指纹（四轮补充 A 的缓存身份核对）。

    默认路径（两者均为 None/空）保持原 key，历史缓存继续可用；任一路径
    自定义后 key 变化，旧特征不会被静默命中。
    """
    if not scene_file and not image_dir:
        return base_key
    return f"{base_key}@{md5(f'{scene_file}|{image_dir}')[:12]}"
