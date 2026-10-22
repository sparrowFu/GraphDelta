"""SigLIP2 so400m-patch14-384（HuggingFace Siglip2Model，本地权重，冻结）。

第一阶段第二轮对照 baseline（方案 §84-2 / §84-22）。加载
scripts/convert_siglip2_checkpoint.py 生成的 -v5 目录（transformers 5.13 重构了
Siglip2：Linear patch 嵌入 + processor 侧 patch 化；旧权重需转换）。

- 图像 / 文本一律走 SigLIP2 官方预处理（384 resize、mean/std=0.5、patch 14 →
  729 patches；so400m 文本上限 64——tokenizer_config 的 model_max_length 是
  哨兵值 1e30，不能读，显式用常量 TEXT_MAX_LENGTH=64）；
- get_image_features / get_text_features 返回 projection 后、归一化前的特征，
  与 CLIPEncoder 同口径（方案 §37 的 h；缓存 raw + normalized，§84-9）；
- transformers 5.13 怪癖：Siglip2Processor.__call__ 会用默认 patch 16/256 覆盖
  实例属性，因此图像走 image_processor、文本走 tokenizer，不用包装层 __call__；
- official_preprocess 覆写为自身 processor（§84-22）。SPAR 官方协议对 SigLIP2
  因此是"官方任务协议 + SigLIP2 官方预处理"，数字不复现任何官方发布值，
  只作 backbone 对照；
- fp16 推理；不训练任何参数；不与 CLIP 跨模型计算 cosine。
"""
import math

import numpy as np
import torch
from transformers import Siglip2Model, Siglip2ImageProcessor, AutoTokenizer

from .vlm_encoder import BaseVLMEncoder

TEXT_MAX_LENGTH = 64  # siglip2-so400m 官方文本上限


def _unwrap_feat(f):
    """v5 的 get_*_features 可能返回 tensor / tuple / ModelOutput，统一取 tensor。"""
    if torch.is_tensor(f):
        return f
    if isinstance(f, tuple):
        f = f[0]
    if torch.is_tensor(f):
        return f
    return f.pooler_output


class SigLIP2Encoder(BaseVLMEncoder):

    def __init__(self, model_path: str, device: str = "auto", fp16: bool = True,
                 batch_size_image: int = 256, batch_size_text: int = 512, logger=None,
                 name: str = "siglip2_so400m"):
        self.name = name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")

        self.model = Siglip2Model.from_pretrained(model_path)
        self.model.eval()
        if self.fp16:
            self.model.half()
        self.model.to(device)
        self.requires_no_grad()

        self.image_processor = Siglip2ImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.grid = int(round(math.sqrt(self.image_processor.max_num_patches)))
        self.bs_img = batch_size_image
        self.bs_txt = batch_size_text

    def _forward_images(self, enc):
        """enc: image_processor 输出（patchified pixel_values + spatial_shapes 等）。"""
        kw = {k: v.to(self.device) for k, v in enc.items()
              if k in ("pixel_values", "spatial_shapes", "pixel_attention_mask")}
        if self.fp16:
            kw["pixel_values"] = kw["pixel_values"].half()
        with torch.no_grad():
            f = _unwrap_feat(self.model.get_image_features(**kw))
        return f.float().cpu().numpy()

    # ---------- image ----------
    def encode_images(self, images):
        feats = []
        n = len(images)
        for i in range(0, n, self.bs_img):
            batch = images[i:i + self.bs_img]
            enc = self.image_processor(images=list(batch), return_tensors="pt")
            feats.append(self._forward_images(enc))
        return self._pack(np.concatenate(feats, axis=0))

    def encode_image_tensors(self, tensors):
        """输入为本类 official_preprocess 的产物：patchified (N, 729, 588)。"""
        feats = []
        pv = tensors
        for i in range(0, len(pv), self.bs_img):
            chunk = pv[i:i + self.bs_img]
            ss = torch.full((len(chunk), 2), self.grid, dtype=torch.long)
            feats.append(self._forward_images(
                {"pixel_values": chunk, "spatial_shapes": ss}))
        return self._pack(np.concatenate(feats, axis=0))

    def official_preprocess(self, images):
        """§84-22：官方协议实验也用 SigLIP2 自身 processor（patchified 输出）。"""
        return self.image_processor(images=list(images), return_tensors="pt")["pixel_values"]

    # ---------- text ----------
    def encode_texts(self, texts, ensure_period=True):
        feats = []
        n = len(texts)
        for i in range(0, n, self.bs_txt):
            if ensure_period:
                batch = [t if t.endswith(".") else t + "." for t in texts[i:i + self.bs_txt]]
            else:
                batch = texts[i:i + self.bs_txt]
            # SigLIP2 池化取序列最后一个位置（含 padding），必须 pad 到模型
            # 训练时的固定长度 64（官方 docstring：padding="max_length"）
            enc = self.tokenizer(
                batch, padding="max_length", truncation=True,
                return_tensors="pt", max_length=TEXT_MAX_LENGTH)
            enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                f = _unwrap_feat(self.model.get_text_features(**enc))
            feats.append(f.float().cpu().numpy())
        return self._pack(np.concatenate(feats, axis=0))

    @staticmethod
    def _pack(raw: np.ndarray):
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        return {"raw": raw, "normalized": raw / norm}
