"""PUG: SPAR（43,560 行，纯评测集）。

字段：world_name / character_name / character2_name / character1_pos /
character2_pos / character_texture / character2_texture

已核实的取值：
- character_name 与 character2_name 均含 "blank"（各 1,320 行）：blank 可在
  任意一侧；两侧均非 blank 的双动物场景 40,960 行，单动物场景 2,560 行，
  双侧均 blank 的纯背景场景 40 行（审查报告 R3）
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
# 水平与垂直关系使用不同短语形式，避免 "to the below of" 这类错误
POS_PHRASE = {
    "left": "to the left of",
    "right": "to the right of",
    "below": "below",
    "above": "above",
}


def build_metadata(parquet_dir) -> pd.DataFrame:
    rows = read_metadata(parquet_dir, META_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "row_id", range(len(df)))
    df["sample_id"] = df["row_id"].map(lambda i: f"pug_spar_{i:06d}")
    # 双动物 = 两侧均非 blank（R3：blank 可能在任意一侧，不能只看 character2）
    df["two_object"] = (df["character_name"] != "blank") & (df["character2_name"] != "blank")
    return df
