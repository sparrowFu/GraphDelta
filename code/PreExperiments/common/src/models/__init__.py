"""冻结 baseline 工厂（方案 §84-2）。按 configs/<model>.yaml 的 model_type 分发，
新增模型在此注册，仍只做评估、完全冻结。"""
import gc

from ..utils.io import CONFIG_DIR, load_config
from .clip_encoder import CLIPEncoder
from .siglip_encoder import SigLIP2Encoder

_ENCODERS = {"clip": CLIPEncoder, "siglip2": SigLIP2Encoder}
ALL = "all"


def available_models(paths):
    """当前可用的冻结 baseline：paths.yaml models 段与 configs/<name>.yaml
    同时存在的名字，顺序与 paths.yaml 一致（首选 baseline 在前）。"""
    return [m for m in paths.get("models", {})
            if (CONFIG_DIR / f"{m}.yaml").exists()]


def resolve_models(model_name, paths):
    """--model 取值解析（串行多模型入口）：

    'all' → 全部已配置 baseline 的列表，调用方依次运行、一次只加载一个模型；
    其他名字 → 单元素列表；未知名直接报错并列出可用值（不等到加载才失败）。
    """
    models = available_models(paths)
    if model_name == ALL:
        if not models:
            raise ValueError("paths.yaml models 段没有可用的 baseline")
        return models
    if model_name not in models:
        raise ValueError(
            f"unknown baseline {model_name!r}，可用: {['all'] + models}")
    return [model_name]


def free_encoder_memory():
    """串行切换模型前释放上一个模型的显存：gc 后清空 CUDA 缓存。
    调用方需先 del 掉旧的 encoder 引用。"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
