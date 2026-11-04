#!/usr/bin/env python
"""【已废弃，仅留档】一次性：把旧格式 SigLIP2 checkpoint 转成 transformers v5 结构。

废弃原因（审查报告 R1，2026-10-23 实测验证）：本脚本把 FixRes checkpoint
（SiglipModel + SiglipImageProcessor 固定 384×384）转成 NaFlex 布局
（Siglip2Model + Siglip2ImageProcessor 动态分辨率、保持纵横比），改变的是
预处理语义而非等价升级：512×512 输入被按 378×378 处理、320×480 被按
308×462 处理（patch 网格 22×33），不再是对指定 checkpoint 的官方预处理。
transformers 5.13 的 SiglipModel / SiglipImageProcessor 仍完整支持原始权重
直接 from_pretrained（Conv2d patch 嵌入），无需本转换。siglip_encoder.py
已改回原始目录 + SiglipModel 路径，paths.yaml 同步指向原始目录。
（-v5 产物目录保留在原处，不再被任何代码引用。）

历史背景（转换当时的技术判断，其中"v5 无自动转换映射、直接加载会 size
mismatch"仅对 Siglip2Model 成立，对 SiglipModel 不成立——此为当时的误判）：
transformers 5.13 重构了 Siglip2——
- patch 嵌入从 Conv2d(C→H, k=P) 改为 Linear(C*P*P→H)（patch 化移到 processor）；
- 配置从 image_size 改为 num_patches（位置嵌入数）；
- image processor 改为 Siglip2ImageProcessor（输出 patchified pixel_values
  (N, 729, 588) + spatial_shapes + pixel_attention_mask）。
本脚本做过四件事并在随机输入上数值验证等价：
1. vision patch_embedding.weight: (H, C, P, P) → permute(0,2,3,1).reshape(H, P*P*C)。
   Siglip2ImageProcessor 的 patch 向量按 (P,P,C) 顺序展平（已实测：与 (C,P,P)
   不符、与 (P,P,C) 差 3e-8），权重必须同序；
   等价性断言：conv2d(x, W) == Linear(W_flat)(每个 patch 按 (P,P,C) 展平)。
2. config 增加 num_patches = (image_size // patch_size)²；text_config 的
   bos/eos_token_id（上游误带 CLIP 词表 id 49406/49407，越界）置 null——
   只影响生成，特征提取不用。
3. 用 Siglip2ImageProcessor.save_pretrained 重写 preprocessor_config
   （沿用原 mean/std=0.5、384 resize、resample=2；patch 14、max_num_patches 729）。
4. 其余 tokenizer / 配置文件原样复制。
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Siglip2ImageProcessor

SRC = Path("/mnt/disk1/xpfu/PreTrainedModels/siglip2-so400m-patch14-384")
DST = Path("/mnt/disk1/xpfu/PreTrainedModels/siglip2-so400m-patch14-384-v5")


def main():
    cfg = json.loads((SRC / "config.json").read_text())
    vis = cfg["vision_config"]
    image_size, patch = vis["image_size"], vis["patch_size"]
    num_patches = (image_size // patch) ** 2
    print(f"image_size={image_size} patch={patch} → num_patches={num_patches}")

    sd = load_file(str(SRC / "model.safetensors"))
    key = "vision_model.embeddings.patch_embedding.weight"
    w = sd[key]                       # (H, C, P, P) conv
    H, C, P, _ = w.shape
    pos_key = "vision_model.embeddings.position_embedding.weight"
    assert sd[pos_key].shape[0] == num_patches, \
        f"position embedding {sd[pos_key].shape} 与 num_patches={num_patches} 不符"

    # 数值等价：conv(x) == Linear(w_flat)(每个 patch 按 (P,P,C) 展平)
    # 注意 384 = 14×27+6，卷积输出 27×27（floor），patch 用同口径 unfold 提取
    g = image_size // patch
    assert g * g == num_patches, (g, num_patches)
    x = torch.randn(2, C, image_size, image_size, dtype=torch.float32)
    w32 = w.to(torch.float32)
    ref = torch.nn.functional.conv2d(x, w32, stride=P)          # (2, H, 27, 27)
    patches = x.unfold(2, P, P).unfold(3, P, P)                 # (2, C, 27, 27, P, P)
    patches = patches.permute(0, 2, 3, 4, 5, 1).reshape(2, g * g, P * P * C)
    w_flat = w32.permute(0, 2, 3, 1).reshape(H, P * P * C)
    lin = torch.nn.functional.linear(patches, w_flat)           # (2, 729, H)
    lin = lin.reshape(2, g, g, H).permute(0, 3, 1, 2)           # (2, H, 27, 27)
    err = (ref - lin).abs().max().item()
    print(f"conv vs linear max abs err = {err:.2e}")
    assert err < 1e-3, "reshape 等价性验证失败"

    sd[key] = w.permute(0, 2, 3, 1).reshape(H, P * P * C).contiguous()
    cfg["vision_config"]["num_patches"] = num_patches
    # 兼容字段：v5 忽略 image_size，但保留无害
    cfg["vision_config"]["image_size"] = image_size
    # 上游误带 CLIP 词表 id，v5 加载时警告越界；只影响生成，置 null
    if cfg.get("text_config", {}).get("bos_token_id", 0) not in (None, 0, 1, 2):
        cfg["text_config"]["bos_token_id"] = None
        cfg["text_config"]["eos_token_id"] = None

    DST.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(DST / "model.safetensors"))
    (DST / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    # 重写 v5 口径的 image processor 配置（参数沿用原 preprocessor_config）
    old_pc = json.loads((SRC / "preprocessor_config.json").read_text())
    ip = Siglip2ImageProcessor(
        size=old_pc["size"], patch_size=patch, resample=old_pc["resample"],
        image_mean=old_pc["image_mean"], image_std=old_pc["image_std"],
        max_num_patches=num_patches)
    ip.save_pretrained(DST)
    for f in ["tokenizer.json", "tokenizer.model",
              "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(SRC / f, DST / f)
    print(f"written {DST}")


if __name__ == "__main__":
    sys.exit(main())
