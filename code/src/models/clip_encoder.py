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
        self._n_img_calls = 0
        self.log = (lambda msg: logger.info(msg)) if logger else (lambda msg: print(msg))

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
        self._n_img_calls += 1
        if self._n_img_calls % 100 == 0 or len(feats) == 0:
            self.log(f"[{self.name}] encode_images: {self._n_img_calls} calls (~"
                     f"{self._n_img_calls * self.bs_img} images)")
        return self._pack(np.concatenate(feats, axis=0))

    # ---------- text ----------
    def encode_texts(self, texts):
        feats = []
        n = len(texts)
        for i in range(0, n, self.bs_txt):
            batch = [t if t.endswith(".") else t + "." for t in texts[i:i + self.bs_txt]]
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
