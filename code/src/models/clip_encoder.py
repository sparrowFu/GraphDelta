"""CLIP ViT-L/14（HuggingFace CLIPModel，本地权重，冻结）。

- 官方 preprocess（224，短边 resize + center crop）与 tokenizer；
- get_image_features / get_text_features 返回 projection 后、归一化前的特征，即方案 §37 的 h；
- fp16 推理；不训练任何参数。
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from .vlm_encoder import BaseVLMEncoder


class CLIPEncoder(BaseVLMEncoder):

    def __init__(self, model_path: str, device: str = "auto", fp16: bool = True,
                 batch_size_image: int = 512, batch_size_text: int = 512, logger=None):
        self.name = "clip_vitl14"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")

        self.model = CLIPModel.from_pretrained(model_path)
        self.model.eval()
        if self.fp16:
            self.model.half()
        self.model.to(device)
        self.requires_no_grad()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.bs_img = batch_size_image
        self.bs_txt = batch_size_text
        # logger 参数保留给后续 baseline（如 SigLIP2）；本类自身不输出日志，
        # 抽取/实验进度统一由 features/extract.py 与各实验模块报告

    # ---------- image ----------
    def encode_images(self, images):
        feats = []
        n = len(images)
        for i in range(0, n, self.bs_img):
            batch = images[i:i + self.bs_img]
            pixel = self.processor(images=batch, return_tensors="pt")["pixel_values"]
            pixel = pixel.to(self.device)
            if self.fp16:
                pixel = pixel.half()
            with torch.no_grad():
                f = self.model.get_image_features(pixel_values=pixel)
            if not torch.is_tensor(f):          # transformers v5 返回 ModelOutput
                f = f.pooler_output
            feats.append(f.float().cpu().numpy())
        return self._pack(np.concatenate(feats, axis=0))

    def encode_image_tensors(self, tensors):
        """输入为已预处理的 float tensor (N,3,224,224)，跳过 CLIPProcessor。

        PUG:SPAR 官方协议（run_eval_vlms_on_spar.py）对全部模型统一使用
        Resize(224,224) + ImageNet mean/std 预处理，与 CLIPProcessor 不同，
        因此官方评测单独走此入口（审查报告 P2-11）。
        """
        feats = []
        for i in range(0, len(tensors), self.bs_img):
            pixel = tensors[i:i + self.bs_img].to(self.device)
            if self.fp16:
                pixel = pixel.half()
            with torch.no_grad():
                f = self.model.get_image_features(pixel_values=pixel)
            if not torch.is_tensor(f):
                f = f.pooler_output
            feats.append(f.float().cpu().numpy())
        return self._pack(np.concatenate(feats, axis=0))

    # ---------- text ----------
    def encode_texts(self, texts, ensure_period=True):
        feats = []
        n = len(texts)
        for i in range(0, n, self.bs_txt):
            if ensure_period:
                batch = [t if t.endswith(".") else t + "." for t in texts[i:i + self.bs_txt]]
            else:
                batch = texts[i:i + self.bs_txt]
            enc = self.processor(text=batch, padding=True, truncation=True,
                                 return_tensors="pt", max_length=77)
            enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                f = self.model.get_text_features(**enc)
            if not torch.is_tensor(f):          # transformers v5 返回 ModelOutput
                f = f.pooler_output
            feats.append(f.float().cpu().numpy())
        return self._pack(np.concatenate(feats, axis=0))

    @staticmethod
    def _pack(raw: np.ndarray):
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        return {"raw": raw, "normalized": raw / norm}
