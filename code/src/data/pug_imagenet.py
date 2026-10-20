"""PUG: ImageNet（辅助数据集，第一阶段默认不运行）。

字段：world_name / character_name(hash) / character_label / rotation(yaw/roll/pitch) /
character_scale / camera_roll / camera_pitch / camera_yaw / character_texture / scene_light
"""
import pandas as pd

from .pug_common import read_metadata

META_COLUMNS = [
    "world_name", "character_name", "character_label", "character_scale",
    "camera_yaw", "camera_pitch", "camera_roll", "character_texture", "scene_light",
]


def build_metadata(parquet_dir) -> pd.DataFrame:
    rows = read_metadata(parquet_dir, META_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "row_id", range(len(df)))
    df["sample_id"] = df["row_id"].map(lambda i: f"pug_imagenet_{i:06d}")
    return df
