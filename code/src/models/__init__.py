"""冻结 baseline 工厂（方案 §84-2）。按 configs/<model>.yaml 的 model_type 分发，
新增模型在此注册，仍只做评估、完全冻结。"""
from ..utils.io import load_config
from .clip_encoder import CLIPEncoder
from .siglip_encoder import SigLIP2Encoder

_ENCODERS = {"clip": CLIPEncoder, "siglip2": SigLIP2Encoder}


def build_encoder(model_name: str, paths: dict, logger=None):
    cfg = load_config(model_name)
    model_type = cfg.get("model_type")
    if model_type not in _ENCODERS:
        raise ValueError(
            f"unknown baseline {model_name}: model_type={model_type!r}，"
            f"已注册: {sorted(_ENCODERS)}")
    if model_name not in paths["models"]:
        raise ValueError(f"paths.yaml models 缺少 {model_name} 的本地权重路径")
    cls = _ENCODERS[model_type]
    return cls(
        model_path=paths["models"][model_name],
        device=cfg.get("device", "auto"),
        fp16=cfg.get("fp16", True),
        batch_size_image=cfg.get("batch_size_image", 512),
        batch_size_text=cfg.get("batch_size_text", 512),
        logger=logger,
        name=model_name,
    )
