"""PUG: SPAR（43,560 行，纯评测集）。

字段：world_name / character_name / character2_name / character1_pos /
character2_pos / character_texture / character2_texture

已核实的取值：
- character2_name 含 "blank"（单动物场景，共 1,320 行）
- character1_pos ∈ {left, below}，character2_pos ∈ {above, right}，且 pos1 != pos2 恒成立
- character_texture（主体）∈ {Blue, Default, Grass}；character2_texture ∈ {Default, Red, Stone}
"""
import pandas as pd

from .pug_common import read_metadata

META_COLUMNS = [
    "world_name",
    "character_name",
    "character2_name",
    "character1_pos",
    "character2_pos",
    "character_texture",
    "character2_texture",
]

POS_ANTONYM = {"left": "right", "below": "above"}


def build_metadata(parquet_dir) -> pd.DataFrame:
    rows = read_metadata(parquet_dir, META_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "row_id", range(len(df)))
    df["sample_id"] = df["row_id"].map(lambda i: f"pug_spar_{i:06d}")
    df["two_object"] = df["character2_name"] != "blank"
    return df
