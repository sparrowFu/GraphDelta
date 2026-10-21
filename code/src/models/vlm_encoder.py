"""冻结 VLM baseline 统一接口（方案 §61）。

返回 dict：
  raw        —— normalization 前的 projected feature（模型支持时）
  normalized —— L2 归一化后的 embedding

所有 baseline 必须 eval + frozen；不同模型不跨模型计算 cosine。
"""
from abc import ABC, abstractmethod


class BaseVLMEncoder(ABC):
    name: str = "base"

    @abstractmethod
    def encode_images(self, images):
        """list[PIL.Image] -> {"raw": np(N,D) float32, "normalized": np(N,D)}"""

    @abstractmethod
    def encode_texts(self, texts, ensure_period=True):
        """list[str] -> {"raw": np(N,D) float32, "normalized": np(N,D)}

        ensure_period=False 时不补句号（官方协议 caption 用）。
        """

    def encode_image_tensors(self, tensors):
        """已预处理 tensor (N,3,H,W) -> 特征 dict；仅官方协议评测需要。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 tensor 输入")

    def requires_no_grad(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
