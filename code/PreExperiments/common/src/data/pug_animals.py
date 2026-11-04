"""PUG: Animals（215,040 行 = 70 动物 × 64 环境 × 3 尺度 × 4 纹理 × 4 相机角，组合唯一）。

字段：world_name / character_name / character_scale / camera_yaw / character_texture

已核实的取值：
- character_scale ∈ {0.7, 1.0, 1.3} → size ∈ {small, medium, large}
- camera_yaw ∈ {0, 45, 225, 315}
- character_texture ∈ {Asphalt, Default, Grass, Sky}
"""
import pandas as pd

from .pug_common import read_metadata

META_COLUMNS = [
    "world_name",
    "character_name",
    "character_scale",
    "camera_yaw",
    "character_texture",
]

SCALE_TO_SIZE = {0.7: "small", 1.0: "medium", 1.3: "large"}


def build_metadata(parquet_dir) -> pd.DataFrame:
    rows = read_metadata(parquet_dir, META_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "row_id", range(len(df)))
    df["sample_id"] = df["row_id"].map(lambda i: f"pug_animals_{i:06d}")
    df = df.rename(columns={"world_name": "environment", "character_name": "animal"})
    df["size"] = df["character_scale"].map(SCALE_TO_SIZE)
    df["camera"] = df["camera_yaw"].astype(int)
    df["texture"] = df["character_texture"]
    return df[
        ["row_id", "sample_id", "animal", "environment", "texture", "size", "camera",
         "character_scale", "camera_yaw", "character_texture"]
    ]
