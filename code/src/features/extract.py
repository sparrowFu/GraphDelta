"""图像特征抽取：CLEVR/CoGenT（文件图）与 PUG（parquet 内嵌图）。只抽取，不修改数据。"""
import numpy as np
from PIL import Image

from ..data.clevr import image_path
from ..data.pug_common import iter_image_batches


def _pack_save(cache, key, ids, raw_list, norm_list):
    raw = np.concatenate(raw_list, axis=0) if raw_list else np.zeros((0, 1), np.float32)
    norm = np.concatenate(norm_list, axis=0) if norm_list else np.zeros((0, 1), np.float32)
    cache.save(key, ids, raw, norm)


def extract_clevr_images(model, cache, dataset_key, scenes, image_dir, logger,
                         batch_size=None, only_ids=None):
    """only_ids: 只抽取指定 image_filename（smoke 模式用）。"""
    if cache.exists(dataset_key):
        logger.info(f"[extract] cache exists, skip: {dataset_key}")
        return
    only = set(only_ids) if only_ids is not None else None
    todo = scenes if only is None else [s for s in scenes if s["image_filename"] in only]
    ids, raws, norms = [], [], []
    bs = batch_size or model.bs_img
    buf_imgs, buf_ids = [], []

    def flush():
        if not buf_imgs:
            return
        out = model.encode_images(buf_imgs)
        raws.append(out["raw"])
        norms.append(out["normalized"])
        ids.extend(buf_ids)
        buf_imgs.clear()
        buf_ids.clear()

    for i, scene in enumerate(todo):
        img = Image.open(image_path(scene, image_dir)).convert("RGB")
        buf_imgs.append(img)
        buf_ids.append(scene["image_filename"])
        if len(buf_imgs) >= bs:
            flush()
        if (i + 1) % 2000 == 0:
            logger.info(f"[extract:{dataset_key}] {i + 1}/{len(todo)}")
    flush()
    _pack_save(cache, dataset_key, ids, raws, norms)
    logger.info(f"[extract] saved {len(ids)} images -> {cache.path(dataset_key)}")


def extract_pug_images(model, cache, dataset_key, parquet_dir, meta_columns, logger,
                       only_row_ids=None, row_batch=48, id_prefix=None):
    """按全局 row_id 顺序抽取 PUG 图像特征。

    id_prefix: 生成 sample_id（如 'pug_spar' -> 'pug_spar_000123'）。
    only_row_ids: set[int]，只抽取选中的行（smoke / nuisance 用，不解码其余图像）。
    """
    if cache.exists(dataset_key):
        logger.info(f"[extract] cache exists, skip: {dataset_key}")
        return
    fmt = (lambda rid: f"{id_prefix}_{rid:06d}") if id_prefix else (lambda rid: str(rid))
    ids, raws, norms = [], [], []
    total = 0
    for row_ids, images, _meta in iter_image_batches(parquet_dir, meta_columns,
                                                     batch_size=row_batch,
                                                     only_row_ids=only_row_ids):
        out = model.encode_images(images)
        raws.append(out["raw"])
        norms.append(out["normalized"])
        ids.extend(fmt(rid) for rid in row_ids)
        total += len(images)
        if total % 9600 < row_batch:
            logger.info(f"[extract:{dataset_key}] encoded ~{total} images")
    _pack_save(cache, dataset_key, ids, raws, norms)
    logger.info(f"[extract] saved {len(ids)} images -> {cache.path(dataset_key)}")
