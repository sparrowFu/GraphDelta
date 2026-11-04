"""PUG parquet 公共工具（不依赖 `datasets` 库，直接用 pyarrow）。

约定：
- 行 ID（row_id）= 按分片文件名字典序、片内行序累计的全局行号。
- image 列为 struct{bytes, path}，用 PIL 解码；only_row_ids 之外的行不解码。
"""
import glob
import io

import pyarrow.parquet as pq


def shard_files(parquet_dir):
    files = sorted(glob.glob(f"{parquet_dir}/data/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {parquet_dir}/data/")
    return files


def read_metadata(parquet_dir, columns):
    """读取全部元数据列（不含 image），返回 list[dict]，顺序即全局行序。"""
    rows = []
    for f in shard_files(parquet_dir):
        t = pq.read_table(f, columns=columns)
        rows.extend(t.to_pylist())
    return rows


def count_rows(parquet_dir) -> int:
    """统计全部行数（只读 parquet footer，不解码图像），用于抽取进度分母。"""
    return sum(pq.ParquetFile(f).metadata.num_rows for f in shard_files(parquet_dir))


def decode_image(cell):
    from PIL import Image

    buf = io.BytesIO(cell["bytes"])
    return Image.open(buf).convert("RGB")


def iter_image_batches(parquet_dir, columns, batch_size=48, only_row_ids=None):
    """按全局 row_id 顺序产出 (row_ids, images, meta_rows)。"""
    only = set(only_row_ids) if only_row_ids is not None else None
    row_id = 0
    for f in shard_files(parquet_dir):
        pf = pq.ParquetFile(f)
        cols = ["image"] + list(columns)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            n = batch.num_rows
            d = batch.to_pylist()
            ids = list(range(row_id, row_id + n))
            row_id += n
            if only is not None:
                picked = [(rid, r) for rid, r in zip(ids, d) if rid in only]
                if not picked:
                    continue
                ids = [p[0] for p in picked]
                images = [decode_image(p[1]["image"]) for p in picked]
                meta = [{k: p[1][k] for k in columns} for p in picked]
            else:
                images = [decode_image(r["image"]) for r in d]
                meta = [{k: r[k] for k in columns} for r in d]
            yield ids, images, meta
