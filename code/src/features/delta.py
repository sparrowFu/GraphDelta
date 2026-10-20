"""Delta 计算与 evaluation-only nuisance 扰动（方案 §38 / §39a）。

nuisance 扰动只在内存中产生两个增强视图用于编码，不落盘、不修改公开数据。
"""
import io

import numpy as np
from PIL import Image


def augment_view(img: Image.Image, seed: int, scale_min: float, scale_max: float,
                 quality: int) -> Image.Image:
    """确定性轻量增强：随机裁剪（固定面积比例区间）+ JPEG 压缩。"""
    rng = np.random.default_rng(seed)
    w, h = img.size
    s = float(rng.uniform(scale_min, scale_max))
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    x0 = int(rng.integers(0, w - nw + 1))
    y0 = int(rng.integers(0, h - nh + 1))
    img2 = img.crop((x0, y0, x0 + nw, y0 + nh))
    buf = io.BytesIO()
    img2.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compute_deltas(pairs, row_index, raw, norm):
    """pairs: DataFrame（含 row_id_a / row_id_b）；返回 Δ_raw、Δ_norm (N, D) float32。"""
    ra = np.array([row_index[int(r)] for r in pairs["row_id_a"]], dtype=np.int64)
    rb = np.array([row_index[int(r)] for r in pairs["row_id_b"]], dtype=np.int64)
    delta_raw = raw[rb].astype(np.float32) - raw[ra].astype(np.float32)
    delta_norm = norm[rb].astype(np.float32) - norm[ra].astype(np.float32)
    return delta_raw, delta_norm
