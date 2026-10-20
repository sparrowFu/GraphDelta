"""SigLIP2 冻结对照（第二阶段复现用占位）。

本地 /mnt/disk1/xpfu/PreTrainedModels/ 中暂无 SigLIP2 权重。
需要时下载（例如 google/siglip2-so400m-patch14-384）并把路径写入 paths.yaml，
本类按各自官方 preprocess / tokenizer 实现（方案 §84 第 22 条）。
"""


class SigLIP2Encoder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SigLIP2 baseline 未启用：本地无权重。"
            "第一阶段第一轮只运行 CLIP ViT-L/14；现象明确后再下载 SigLIP2 权重启用本类。"
        )
