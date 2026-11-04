"""PUG: ImageNet（辅助数据集，方案 §14：验证 Animals 的单因素规律是否只在动物场景成立）。

字段（官方 schema）：world_name / character_name(hash) / character_label /
character_rotation_yaw/roll/pitch / character_scale / camera_roll / camera_pitch /
camera_yaw / character_texture / scene_light

E2 因素映射（与 Animals 统一命名，审查 A1）：
- environment ← world_name
- texture     ← character_texture
- size        ← character_scale（保留原始数值，不做 small/medium/large 映射）
- camera      ← camera_yaw

配对的固定 context 除上述语义列外，还包含 character_name（实例身份，复核 B1：
官方 724 个 assets 分属 151 个类，character_label 只是类别——只固定类别时
同类多实例会被 by_val 折叠、实例覆盖被压缩）、camera_pitch、camera_roll、
scene_light 与 character_rotation_*——实例、其他相机角度、灯光和物体姿态
都必须固定，保证每对只有目标因素变化。
"""
import pandas as pd

from .pug_common import read_metadata

META_COLUMNS = [
    "world_name", "character_name", "character_label", "character_scale",
    "camera_yaw", "camera_pitch", "camera_roll", "character_texture", "scene_light",
    "character_rotation_yaw", "character_rotation_roll", "character_rotation_pitch",
]

# E2 配对使用的语义列：前两列用于 smoke 子集截取，全部列参与 context 固定。
# character_name 在 character_label 之前——实例先于类别固定（复核 B1）。
META_COLS = [
    "character_name", "character_label", "environment", "texture", "size", "camera",
    "camera_pitch", "camera_roll", "scene_light",
    "character_rotation_yaw", "character_rotation_roll", "character_rotation_pitch",
]


def build_metadata(parquet_dir) -> pd.DataFrame:
    rows = read_metadata(parquet_dir, META_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "row_id", range(len(df)))
    df["sample_id"] = df["row_id"].map(lambda i: f"pug_imagenet_{i:06d}")
    df = df.rename(columns={"world_name": "environment"})
    df["texture"] = df["character_texture"]
    df["size"] = df["character_scale"]
    df["camera"] = df["camera_yaw"]
    cols = (["row_id", "sample_id", "character_scale", "camera_yaw",
             "character_texture"] + META_COLS)
    return df[cols]
