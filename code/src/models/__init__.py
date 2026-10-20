"""冻结 baseline 工厂。新增模型（如 SigLIP2）在此注册，仍只做评估。"""
from ..utils.io import load_config
from .clip_encoder import CLIPEncoder


def build_encoder(model_name: str, paths: dict, logger=None):
    if model_name != "clip_vitl14":
        raise ValueError(f"unknown baseline: {model_name}（SigLIP2 为第二轮对照，本地暂无权重）")
    cfg = load_config("clip_vitl14")
    return CLIPEncoder(
        model_path=paths["models"]["clip_vitl14"],
        device=cfg.get("device", "auto"),
        fp16=cfg.get("fp16", True),
        batch_size_image=cfg.get("batch_size_image", 512),
        batch_size_text=cfg.get("batch_size_text", 512),
        logger=logger,
    )
