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
    def encode_texts(self, texts):
        """list[str] -> {"raw": np(N,D) float32, "normalized": np(N,D)}"""

    def requires_no_grad(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
