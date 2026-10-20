"""特征缓存（方案 §62）：图像特征按 dataset 存一份；文本特征按内容哈希去重缓存。"""
from pathlib import Path

import numpy as np

from ..utils.io import md5


class ImageFeatureCache:
    def __init__(self, cache_dir, model_name: str):
        self.dir = Path(cache_dir) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._loaded = {}

    def path(self, dataset_key: str) -> Path:
        return self.dir / f"{dataset_key}_images.npz"

    def exists(self, dataset_key: str) -> bool:
        return self.path(dataset_key).exists()

    def save(self, dataset_key: str, ids, raw: np.ndarray, normalized: np.ndarray):
        np.savez(self.path(dataset_key),
                 ids=np.array(ids, dtype=np.str_),
                 raw=raw.astype(np.float16),
                 normalized=normalized.astype(np.float16))
        self._loaded.pop(dataset_key, None)

    def arrays(self, dataset_key: str):
        """返回 (ids, index, raw, normalized)；raw/norm 为 fp16 视图。"""
        if dataset_key not in self._loaded:
            z = np.load(self.path(dataset_key), allow_pickle=False)
            ids = z["ids"].astype(str)
            index = {sid: i for i, sid in enumerate(ids)}
            self._loaded[dataset_key] = (ids, index, z["raw"], z["normalized"])
        return self._loaded[dataset_key]

    def get(self, dataset_key: str, wanted_ids):
        """返回 (raw, normalized)，float32，顺序与 wanted_ids 一致。"""
        _, index, raw, norm = self.arrays(dataset_key)
        rows = [index[w] for w in wanted_ids]
        return raw[rows].astype(np.float32), norm[rows].astype(np.float32)


class TextCache:
    """文本特征按 md5(text) 去重，跨实验复用。"""

    def __init__(self, cache_dir, model_name: str):
        self.dir = Path(cache_dir) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "text_cache.npz"
        self.keys = []
        self.index = {}
        self.raw = np.zeros((0, 0), dtype=np.float16)
        self.norm = np.zeros((0, 0), dtype=np.float16)
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        z = np.load(self.path, allow_pickle=False)
        self.keys = list(z["keys"].astype(str))
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.raw, self.norm = z["raw"], z["normalized"]

    def encode(self, model, texts):
        """返回 (raw, normalized) float32，顺序与 texts 一致；缺失部分批量编码后写回。"""
        keys = [md5(t) for t in texts]
        missing_keys, first_text = [], {}
        for t, k in zip(texts, keys):
            if k not in self.index and k not in first_text:
                first_text[k] = t
                missing_keys.append(k)
        if missing_keys:
            out = model.encode_texts([first_text[k] for k in missing_keys])
            new_raw = out["raw"].astype(np.float16)
            new_norm = out["normalized"].astype(np.float16)
            if self.keys:
                self.raw = np.concatenate([self.raw, new_raw], axis=0)
                self.norm = np.concatenate([self.norm, new_norm], axis=0)
            else:
                self.raw, self.norm = new_raw, new_norm
            offset = len(self.keys)
            for i, k in enumerate(missing_keys):
                self.index[k] = offset + i
            self.keys.extend(missing_keys)
            np.savez(self.path, keys=np.array(self.keys, dtype=np.str_),
                     raw=self.raw, normalized=self.norm)
        rows = [self.index[k] for k in keys]
        return self.raw[rows].astype(np.float32), self.norm[rows].astype(np.float32)
