"""冻结 VLM baseline 统一接口（方案 §61、§84-22）。

返回 dict：
  raw        —— normalization 前的 projected feature（模型支持时）
  normalized —— L2 归一化后的 embedding

所有 baseline 必须 eval + frozen；不同模型不跨模型计算 cosine。

official_preprocess：SPAR 官方协议（PUG 官方脚本）的图像预处理入口。
基类默认 = 官方脚本对所有模型统一的 Resize(224,224) + ImageNet mean/std
（CLIP 路径口径）；新 baseline 按方案 §84-22 覆写为各自官方预处理。
"""
from abc import ABC, abstractmethod

import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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

    def official_preprocess(self, images):
        """list[PIL.Image] -> tensor(N,3,224,224)。

        默认 = PUG 官方脚本对全部模型统一的 Resize(224,224) + ImageNet
        mean/std（复现其 CLIP 路径，P2-11）；新 baseline 按 §84-22 覆写
        为各自官方 image processor。
        """
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return torch.stack([tf(im) for im in images])

    def requires_no_grad(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
