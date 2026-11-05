"""E4 特征接入：float32 命名空间缓存 + 状态图片抽取（工作方案第 8 节）。

- 缓存目录 data/cache/f32/<model>/，与阶段一 float16 缓存（data/cache/<model>/）
  完全隔离；protocol.json 指纹机制照常生效；
- dataset_key 带清单指纹与图片内容指纹（范式同 CoGenT 的
  condition_cache_key）：清单重建（md5 变化）或任一图片重渲染 / 被替换
  （字节变化）都会开新键，不会串用旧特征（审查问题 6）。
"""
import numpy as np

from src.features.cache import ImageFeatureCache, TextCache


def state_dataset_key(task: str, manifest_meta: dict, rendered=None) -> str:
    """缓存键 = 任务 + 清单指纹（+ 可选的图片内容指纹）。

    rendered 传 rendered_states 表时逐图累加字节摘要：同一清单下重渲染或
    替换图片（清单 md5 不变）也会失效旧缓存。
    """
    import hashlib
    fp = f"{task}|{manifest_meta['pairs_md5']}|{manifest_meta['states_md5']}"
    if rendered is not None:
        h = hashlib.md5()
        for sid, path in zip(rendered["state_id"], rendered["image_path"]):
            h.update(str(sid).encode("utf-8"))
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        fp += "|" + h.hexdigest()
    return f"{task}_states@{hashlib.md5(fp.encode()).hexdigest()[:12]}"


def e4_caches(paths: dict, model) -> tuple:
    """返回 (ImageFeatureCache, TextCache)，均 float32 独立命名空间。"""
    f32_dir = paths["dirs"]["cache"] / "f32"
    icache = ImageFeatureCache(f32_dir, model.name, protocol=model.cache_protocol,
                               dtype=np.float32)
    tcache = TextCache(f32_dir, model.name, protocol=model.cache_protocol,
                       dtype=np.float32)
    return icache, tcache


def extract_state_images(model, cache: ImageFeatureCache, dataset_key: str,
                         state_ids: list, image_dir, logger, batch_size=None):
    """把渲染状态图编码进缓存；按 id 增量补齐（机制同 extract_clevr_images）。

    state_ids 中已在缓存里的跳过；新图按 batch 编码后与旧缓存合并写回。
    """
    from pathlib import Path
    from PIL import Image

    image_dir = Path(image_dir)
    cached_ids = []
    if cache.exists(dataset_key):
        cached_ids = list(cache.arrays(dataset_key)[0])
    todo = [s for s in state_ids if s not in cached_ids]
    if not todo:
        logger.info(f"[e4-extract] {dataset_key}: 全部 {len(state_ids)} 态已缓存")
        return
    bs = batch_size or model.bs_img
    new_ids, new_raw, new_norm = [], [], []
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        images = [Image.open(image_dir / f"{s}.png").convert("RGB") for s in chunk]
        out = model.encode_images(images)
        new_ids.extend(chunk)
        new_raw.append(out["raw"])
        new_norm.append(out["normalized"])
        if (i // bs) % 20 == 0:
            logger.info(f"[e4-extract] {dataset_key}: {i + len(chunk)}/{len(todo)}")
    ids = cached_ids + new_ids
    raw = (np.concatenate([cache.arrays(dataset_key)[2]] + new_raw)
           if cached_ids else np.concatenate(new_raw))
    norm = (np.concatenate([cache.arrays(dataset_key)[3]] + new_norm)
            if cached_ids else np.concatenate(new_norm))
    cache.save(dataset_key, ids, raw, norm)
    logger.info(f"[e4-extract] {dataset_key}: 新抽 {len(todo)} 态，共 {len(ids)}")
