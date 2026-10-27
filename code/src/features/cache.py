"""特征缓存（方案 §62）：图像特征按 dataset 存一份；文本特征按内容哈希去重缓存。

缓存按 ID / 文本键命中，不含任何预处理语义信息：encoder 的图像预处理或文本
预处理变更后（如 R1 的 FixRes 恢复、R4 的小写化），旧缓存会被静默命中、绕过
新 encoder（审查报告 §3.8）。因此每个 encoder 声明 cache_protocol 指纹，
缓存目录内用 protocol.json 记录；指纹不一致时直接报错，要求删除后重抽。
"""
import json
from pathlib import Path

import numpy as np

from ..utils.io import md5


def _check_protocol(model_dir: Path, protocol: str):
    """校验/登记缓存目录的 encoder 协议指纹（审查报告 §3.8 / 审查 A2）。

    - 已有 protocol.json：指纹不一致即报错，要求整套重建；
    - 无 protocol.json 但目录里已有任何 NPZ（协议化之前的历史缓存）：
      拒绝登记——新写入的协议不能证明旧特征按当前语义抽取；
    - 无 protocol.json 且无任何 NPZ（空目录）：首次登记。
    """
    pf = model_dir / "protocol.json"
    if pf.exists():
        try:
            obj = json.loads(pf.read_text())
            if not isinstance(obj, dict):
                raise ValueError(f"不是 JSON 对象：{type(obj).__name__}")
            stored = obj.get("protocol")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            # 复核 B7 / 三轮审查问题 5：损坏、不可读或结构不对（[]/null/字符串
            # 都是合法 JSON 但不是协议对象）→ 来源未知，不能按任何协议复用
            raise RuntimeError(
                f"protocol.json 无法解析或不是对象（{e}）：{pf}。缓存来源未知，"
                f"不能复用——请用 extract_embeddings.py --force 整套重建。")
        if stored != protocol:
            raise RuntimeError(
                f"缓存协议不匹配：{model_dir} 记录 protocol={stored!r}，"
                f"当前 encoder 要求 {protocol!r}。encoder 语义已变化，旧特征不能"
                f"复用——请删除整个目录后重抽：rm -rf {model_dir}")
        return
    if any(model_dir.glob("*.npz")):
        raise RuntimeError(
            f"缓存目录无 protocol.json 但已有特征文件：{model_dir}。这些特征"
            f"无法证明按当前协议 {protocol!r} 抽取，不能直接登记复用——"
            f"请用 extract_embeddings.py --force 整套重建，或把旧目录移入"
            f"历史命名空间后从空目录重抽。")
    pf.write_text(json.dumps({"protocol": protocol}))


def _check_features(arr, name: str, ctx: str) -> None:
    """特征块有效性校验（四轮补充 C）：二维矩阵 + 有限值。

    NaN/Inf 相似度的 s_pos > s_neg 恒为 False，会被计成确定的答错；
    形状异常（如一维）则可能在下游被广播悄悄扩展。两者都在缓存边界
    明确停止，并给出位置（ctx）便于定位 encoder/缓存/样本。
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise RuntimeError(f"{ctx}: {name} 不是二维特征矩阵（shape={a.shape}）")
    if not np.isfinite(a).all():
        bad = np.where((~np.isfinite(a)).any(axis=1))[0]
        raise RuntimeError(
            f"{ctx}: {name} 含 NaN/Inf（{len(bad)} 行，首行 #{int(bad[0])}）——"
            f"无效特征不得进入评分，请检查 encoder 输出或删除该缓存重抽")


def _check_ids_rows(ids, raw: np.ndarray, norm: np.ndarray, ctx: str) -> None:
    if raw.shape != norm.shape or len(ids) != raw.shape[0]:
        raise RuntimeError(
            f"{ctx}: ids/raw/normalized 行数或维度不一致（ids={len(ids)}, "
            f"raw={raw.shape}, normalized={norm.shape}）——按 id 取行会错位")


class ImageFeatureCache:
    def __init__(self, cache_dir, model_name: str, protocol: str = None):
        self.dir = Path(cache_dir) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        if protocol is not None:
            _check_protocol(self.dir, protocol)
        self._loaded = {}

    def path(self, dataset_key: str) -> Path:
        return self.dir / f"{dataset_key}_images.npz"

    def exists(self, dataset_key: str) -> bool:
        return self.path(dataset_key).exists()

    def save(self, dataset_key: str, ids, raw: np.ndarray, normalized: np.ndarray):
        raw = np.asarray(raw)
        normalized = np.asarray(normalized)
        ctx = f"cache save {self.path(dataset_key)}"
        _check_features(raw, "raw", ctx)
        _check_features(normalized, "normalized", ctx)
        _check_ids_rows(ids, raw, normalized, ctx)
        np.savez(self.path(dataset_key),
                 ids=np.array(ids, dtype=np.str_),
                 raw=raw.astype(np.float16),
                 normalized=normalized.astype(np.float16))
        self._loaded.pop(dataset_key, None)

    def arrays(self, dataset_key: str):
        """返回 (ids, index, raw, normalized)；raw/norm 为 fp16 视图。"""
        if dataset_key not in self._loaded:
            p = self.path(dataset_key)
            z = np.load(p, allow_pickle=False)
            ids = z["ids"].astype(str)
            raw, norm = z["raw"], z["normalized"]
            ctx = f"cache load {p}"
            _check_features(raw, "raw", ctx)
            _check_features(norm, "normalized", ctx)
            _check_ids_rows(ids, raw, norm, ctx)
            index = {sid: i for i, sid in enumerate(ids)}
            self._loaded[dataset_key] = (ids, index, raw, norm)
        return self._loaded[dataset_key]

    def get(self, dataset_key: str, wanted_ids):
        """返回 (raw, normalized)，float32，顺序与 wanted_ids 一致。"""
        _, index, raw, norm = self.arrays(dataset_key)
        rows = [index[w] for w in wanted_ids]
        return raw[rows].astype(np.float32), norm[rows].astype(np.float32)


class TextCache:
    """文本特征按 md5(text) 去重，跨实验复用。

    注意缓存键是**传入时**的文本（含大小写）；encoder 内部的小写化等预处理
    变化不会改变键（R4 教训），因此同样依赖 protocol 指纹失效旧缓存。
    """

    def __init__(self, cache_dir, model_name: str, protocol: str = None):
        self.dir = Path(cache_dir) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        if protocol is not None:
            _check_protocol(self.dir, protocol)
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
        self.raw, self.norm = z["raw"], z["normalized"]
        ctx = f"cache load {self.path}"
        _check_features(self.raw, "raw", ctx)
        _check_features(self.norm, "normalized", ctx)
        _check_ids_rows(self.keys, self.raw, self.norm, ctx)
        self.index = {k: i for i, k in enumerate(self.keys)}

    def encode(self, model, texts, ensure_period=True):
        """返回 (raw, normalized) float32，顺序与 texts 一致；缺失部分批量编码后写回。

        缓存键 = **实际编码文本**的 md5：ensure_period=True 时先补句号再计算，
        同一原文在两种模式下不会互相污染缓存（二轮审查问题 8）。
        """
        eff = [t if (not ensure_period or t.endswith(".")) else t + "." for t in texts]
        keys = [md5(t) for t in eff]
        missing_keys, first_text = [], {}
        for t, k in zip(eff, keys):
            if k not in self.index and k not in first_text:
                first_text[k] = t
                missing_keys.append(k)
        if missing_keys:
            out = model.encode_texts([first_text[k] for k in missing_keys],
                                     ensure_period=False)
            new_raw = out["raw"].astype(np.float16)
            new_norm = out["normalized"].astype(np.float16)
            # 四轮补充 C：encoder 新产出的块同样校验（二维 + 有限值 + 行数一致）
            ctx = f"text cache encode {self.path}"
            _check_features(new_raw, "raw", ctx)
            _check_features(new_norm, "normalized", ctx)
            _check_ids_rows(missing_keys, new_raw, new_norm, ctx)
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
