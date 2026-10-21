"""图像特征抽取：CLEVR/CoGenT（文件图）与 PUG（parquet 内嵌图）。只抽取，不修改数据。

缓存按 id 集合做增量校验：已存在的样本跳过，缺失样本补抽后与旧缓存
合并写回（审查报告 P2-12）。
"""
import numpy as np
from PIL import Image

from ..data.clevr import image_path
from ..data.pug_common import iter_image_batches


def _pack_save(cache, key, ids, raw_list, norm_list):
    raw = np.concatenate(raw_list, axis=0) if raw_list else np.zeros((0, 1), np.float32)
    norm = np.concatenate(norm_list, axis=0) if norm_list else np.zeros((0, 1), np.float32)
    cache.save(key, ids, raw, norm)


def _existing(cache, dataset_key, logger):
    """返回缓存已有的 id 列表（无缓存返回空）。"""
    if not cache.exists(dataset_key):
        return []
    eids, _index, _raw, _norm = cache.arrays(dataset_key)
    logger.info(f"[extract] cache {dataset_key}: {len(eids)} ids present")
    return eids.tolist()


def _merge_and_save(cache, dataset_key, existing_ids, existing_arrays, new_ids, raws, norms):
    if existing_ids and new_ids:
        eraw, enorm = existing_arrays
        raws = [eraw.astype(np.float32)] + raws
        norms = [enorm.astype(np.float32)] + norms
        new_ids = existing_ids + new_ids
    _pack_save(cache, dataset_key, new_ids, raws, norms)
    return len(new_ids)


def extract_clevr_images(model, cache, dataset_key, scenes, image_dir, logger,
                         batch_size=None, only_ids=None):
    """only_ids: 只抽取指定 image_filename（smoke 模式用）；已缓存的样本跳过。"""
    existing_ids = _existing(cache, dataset_key, logger)
    have = set(existing_ids)
    only = set(only_ids) if only_ids is not None else None
    todo = [s for s in scenes
            if s["image_filename"] not in have
            and (only is None or s["image_filename"] in only)]
    if not todo:
        logger.info(f"[extract] cache complete, skip: {dataset_key}")
        return
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
    old = cache.arrays(dataset_key)[2:] if existing_ids else None
    n = _merge_and_save(cache, dataset_key, existing_ids, old, ids, raws, norms)
    logger.info(f"[extract] saved {len(ids)} new / {n} total images -> {cache.path(dataset_key)}")


def extract_pug_images(model, cache, dataset_key, parquet_dir, meta_columns, logger,
                       only_row_ids=None, row_batch=48, id_prefix=None):
    """按全局 row_id 顺序抽取 PUG 图像特征（增量：已缓存的行跳过）。

    id_prefix: 生成 sample_id（如 'pug_spar' -> 'pug_spar_000123'）。
    only_row_ids: set[int]，只抽取选中的行（smoke / nuisance 用，不解码其余图像）。
    """
    fmt = (lambda rid: f"{id_prefix}_{rid:06d}") if id_prefix else (lambda rid: str(rid))
    existing_ids = _existing(cache, dataset_key, logger)
    have = set(existing_ids)
    ids, raws, norms = [], [], []
    total = 0
    for row_ids, images, _meta in iter_image_batches(parquet_dir, meta_columns,
                                                     batch_size=row_batch,
                                                     only_row_ids=only_row_ids):
        todo = [(r, im) for r, im in zip(row_ids, images) if fmt(r) not in have]
        if not todo:
            continue
        out = model.encode_images([im for _r, im in todo])
        raws.append(out["raw"])
        norms.append(out["normalized"])
        ids.extend(fmt(r) for r, _im in todo)
        total += len(todo)
        if total % 9600 < row_batch:
            logger.info(f"[extract:{dataset_key}] encoded ~{total} images")
    if not ids:
        logger.info(f"[extract] cache complete, skip: {dataset_key}")
        return
    old = cache.arrays(dataset_key)[2:] if existing_ids else None
    n = _merge_and_save(cache, dataset_key, existing_ids, old, ids, raws, norms)
    logger.info(f"[extract] saved {len(ids)} new / {n} total images -> {cache.path(dataset_key)}")
