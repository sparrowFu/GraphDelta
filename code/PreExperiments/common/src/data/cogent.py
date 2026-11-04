"""CLEVR-CoGenT：按 condition 加载 valA / valB 场景。格式与 CLEVR 相同。

路径可由 configs/e3_cogent.yaml 的 conditions.<cond>.scene_file / image_dir
覆盖（四轮补充 A：此前配置字段从未生效，实验入口与独立抽取脚本都固定读
scenes/CLEVR_val{A,B}_scenes.json 与 images/val{A,B}）。
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


def condition_cache_key(base_key, cogent_root, condition,
                        scene_file=None, image_dir=None):
    """cogent 图像缓存的完整身份 key（四轮补充 A + 复核缺口 4/5）。

    key = base + (数据根目录, scene_file, image_dir) 的指纹：

    - 指纹包含**解析后的数据根目录**——只换根目录、保留相对配置与同名
      image id 也会得到不同 key，不再复用原数据的特征（复核缺口 5）；
    - 默认路径同样带指纹（配置里 scene_file/image_dir 恒有值；复核说明 1：
      不存在"默认保持裸 key"的分支）。与不带后缀的历史 key 不通用，
      首次使用需重抽一次；
    - 抽取（extract_embeddings.py）与 --force 删除共用本函数，force 不会
      删错目标（复核缺口 4）。
    """
    assert condition in ("A", "B"), condition
    scene_file = scene_file or f"scenes/CLEVR_val{condition}_scenes.json"
    image_dir = image_dir or f"images/val{condition}"
    return f"{base_key}@{md5(f'{cogent_root}|{scene_file}|{image_dir}')[:12]}"
