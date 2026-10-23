"""SigLIP2 so400m-patch14-384（FixRes checkpoint，HuggingFace SiglipModel，本地权重，冻结）。

第一阶段第二轮对照 baseline（方案 §84-2 / §84-22）。

FixRes 与 NaFlex 的区分（审查报告 R1，已实测验证）：
- 本 checkpoint（google/siglip2-so400m-patch14-384，config model_type=siglip）是
  FixRes 版本：Conv2d patch 嵌入 + SiglipImageProcessor 固定 384×384 resize。
  transformers 5.13 的 SiglipModel / SiglipImageProcessor 仍完整保留该路径，
  原始权重可直接加载，无需任何转换。
- Siglip2Model / Siglip2ImageProcessor 是 NaFlex 路径：Linear patch 嵌入 +
  动态分辨率（保持纵横比、按 max_num_patches 缩放）。曾用
  scripts/convert_siglip2_checkpoint.py 把权重转成 -v5 目录按 NaFlex 加载，
  该做法会改变预处理语义（512×512 输入被按 378×378 处理、非方形图纵横比
  被改变；同图两路特征实测 cosine 0.994，非等价），已废弃。转换脚本仅留档。

- 文本一律小写后编码（审查报告 R4，实测验证）：SigLIP2 官方要求与训练一致的
  小写预处理；本 checkpoint 的 tokenizer 是 GemmaTokenizer，
  tokenizer_config 的 do_lower_case=true 字段对其不生效（"This" 与 "this"、
  "Elephant" 与 "elephant" 的 token id 不同），必须显式 lower()。
- 文本 padding="max_length" 到 64（so400m 官方上限；tokenizer_config 的
  model_max_length 是哨兵值 1e30，不能读，显式用常量 TEXT_MAX_LENGTH=64）。
- get_image_features / get_text_features 返回 projection 后、归一化前的特征，
  与 CLIPEncoder 同口径（方案 §37 的 h；缓存 raw + normalized，§84-9）。
- official_preprocess 覆写为自身 processor（§84-22）：官方任务协议 +
  SigLIP2 官方预处理（固定 384）。SPAR 官方协议数字不复现任何官方发布值，
  只作 backbone 对照。
- fp16 推理；不训练任何参数；不与 CLIP 跨模型计算 cosine。
"""
import numpy as np
import torch
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipModel

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

        # FixRes 路径：SiglipModel（Conv2d）+ SiglipImageProcessor（固定 384×384）
        self.model = SiglipModel.from_pretrained(model_path)
        self.model.eval()
        if self.fp16:
            self.model.half()
        self.model.to(device)
        self.requires_no_grad()

        self.image_processor = SiglipImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bs_img = batch_size_image
        self.bs_txt = batch_size_text

    def _forward_images(self, pixel_values):
        if self.fp16:
            pixel_values = pixel_values.half()
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            f = _unwrap_feat(self.model.get_image_features(pixel_values=pixel_values))
        return f.float().cpu().numpy()

    # ---------- image ----------
    def encode_images(self, images):
        feats = []
        n = len(images)
        for i in range(0, n, self.bs_img):
            batch = images[i:i + self.bs_img]
            pixel = self.image_processor(images=list(batch), return_tensors="pt")["pixel_values"]
            feats.append(self._forward_images(pixel))
        return self._pack(np.concatenate(feats, axis=0))

    def encode_image_tensors(self, tensors):
        """输入为本类 official_preprocess 的产物：(N, 3, 384, 384) 张量。"""
        feats = []
        for i in range(0, len(tensors), self.bs_img):
            feats.append(self._forward_images(tensors[i:i + self.bs_img]))
        return self._pack(np.concatenate(feats, axis=0))

    def official_preprocess(self, images):
        """§84-22：官方协议实验也用 SigLIP2 自身 processor（固定 384×384）。"""
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
            # R4：GemmaTokenizer 不执行 do_lower_case，SigLIP2 官方要求小写
            batch = [t.lower() for t in batch]
            # SigLIP 池化取序列最后一个位置（含 padding），必须 pad 到模型
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
