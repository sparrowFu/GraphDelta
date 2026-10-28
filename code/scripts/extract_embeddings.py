#!/usr/bin/env python
"""抽取图像特征到缓存（方案 §62）。所有特征只抽一次，后续实验读缓存。

进度日志统一为 "[extract:key] N/M"（每 2000 张）；缓存按 id 增量补齐，
--force 删除后重抽。

--dataset all：全部数据集串行依次抽取（辅助集 pug_imagenet 排最后）；
--model all：全部已配置模型串行运行，一次只加载一个模型（模型在外层循环，
数据集在内层，同一模型连抽全部数据集后再换下一个模型）。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clevr
from src.data.cogent import (condition_cache_key, condition_image_dir,
                             load_condition_scenes)
from src.data.pug_animals import META_COLUMNS as ANIMALS_META
from src.data.pug_imagenet import META_COLUMNS as IMAGENET_META
from src.data.pug_spar import META_COLUMNS as SPAR_META
from src.experiments.e1_spar_official import extract_official_features
from src.features.cache import ImageFeatureCache
from src.features.extract import extract_clevr_images, extract_pug_images
from src.models import build_encoder, free_encoder_memory, resolve_models
from src.utils.io import load_config, load_paths
from src.utils.logging import get_logger

# --dataset 的全部取值；all 的串行顺序与此列表一致（辅助集最后）
EXTRACT_DATASETS = ["clevr_val", "pug_spar", "pug_spar_official", "pug_animals",
                    "cogent_valA", "cogent_valB", "pug_imagenet"]


def resolve_datasets(dataset):
    """--dataset 取值解析：'all' → 全部数据集（串行依次抽取）。"""
    return list(EXTRACT_DATASETS) if dataset == "all" else [dataset]


def _force_reset(cache_root, model, dataset, logger):
    """--force 的整套重建入口（审查 A2）：必须在协议校验**之前**执行，
    否则协议不匹配的历史目录会先抛异常，删除分支永远不可达。

    - 协议匹配：只删本数据集图像缓存 + 文本缓存（文本与图像共享 encoder
      语义，重建不能只覆盖图像而保留旧文本特征）；
    - 协议不匹配、无 protocol.json 或协议文件损坏（复核 B7：损坏 JSON 此前
      使 force 抛 JSONDecodeError、旧 NPZ 删不掉）：目录内新旧特征无法区分，
      整目录删除。
    删除前校验目标路径确实位于 cache_root 下一级（复核 B7 的路径安全建议）。
    """
    root = Path(cache_root).resolve()
    model_dir = root / model.name
    if model_dir.resolve().parent != root:
        raise RuntimeError(f"异常缓存路径 {model_dir}（不在 {root} 下一级），拒绝删除")
    stored = None
    pf = model_dir / "protocol.json"
    if pf.exists():
        try:
            obj = json.loads(pf.read_text())
            if not isinstance(obj, dict):
                raise ValueError(f"不是 JSON 对象：{type(obj).__name__}")
            stored = obj.get("protocol")
        except (json.JSONDecodeError, OSError, ValueError):
            # 损坏/不可读/结构不对（[]/null/字符串）→ 来源未知，整套重建
            # （复核 B7 / 三轮审查问题 5）
            stored = None
            logger.info("[extract] force: protocol.json 损坏、不可读或不是对象，"
                        "按来源未知处理")
    if stored == model.cache_protocol:
        for p in (model_dir / f"{dataset}_images.npz", model_dir / "text_cache.npz"):
            if p.exists():
                p.unlink()
                logger.info(f"[extract] force: removed {p}")
    else:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        logger.info(f"[extract] force: removed entire cache dir {model_dir} "
                    f"(protocol mismatch/missing/corrupt; mixed-version features)")


def _cogent_key_cfg(paths, dataset):
    """cogent 数据集的生效配置与完整缓存 key（复核缺口 4：--force 删除与
    实际抽取必须使用同一个 key；key 指纹含数据根目录与生效路径）。"""
    cond = dataset[-1]
    ccfg = load_config("e3_cogent")["conditions"][cond]
    key = condition_cache_key(dataset, paths["datasets"]["cogent"], cond,
                              ccfg.get("scene_file"), ccfg.get("image_dir"))
    return ccfg, key


def extract_dataset(model, paths, logger, dataset, force=False):
    """单个 (model, dataset) 的特征抽取入口（含 --force 的缓存重建）。"""
    # cogent 的生效缓存 key 在 force 之前解析（复核缺口 4：此前 force 删裸
    # dataset 名的 NPZ，实际抽取用带指纹 key，同协议 force 删错目标、不重抽）
    cogent_ccfg = cogent_key = None
    if dataset in ("cogent_valA", "cogent_valB"):
        cogent_ccfg, cogent_key = _cogent_key_cfg(paths, dataset)
    if force:
        _force_reset(paths["dirs"]["cache"], model, cogent_key or dataset,
                     logger)
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name,
                              protocol=model.cache_protocol)

    if dataset == "clevr_val":
        cfg = load_config("e1_state")["clevr"]
        scenes = clevr.load_scenes(f"{paths['datasets']['clevr']}/{cfg['scene_file']}")
        extract_clevr_images(model, cache, "clevr_val", scenes,
                             f"{paths['datasets']['clevr']}/{cfg['image_dir']}", logger)
    elif dataset in ("cogent_valA", "cogent_valB"):
        cond = dataset[-1]
        # 四轮补充 A：与 run_e3 统一读取 configs/e3_cogent.yaml 的
        # conditions.<cond>.scene_file / image_dir；缓存 key 含数据根目录与
        # 生效路径的指纹，与实验入口、--force 三处共用同一 key
        scenes = load_condition_scenes(paths["datasets"]["cogent"], cond,
                                       cogent_ccfg.get("scene_file"))
        extract_clevr_images(model, cache, cogent_key, scenes,
                             condition_image_dir(paths["datasets"]["cogent"], cond,
                                                 cogent_ccfg.get("image_dir")), logger)
    elif dataset == "pug_spar":
        extract_pug_images(model, cache, "pug_spar", paths["datasets"]["pug_spar"],
                           SPAR_META, logger, id_prefix="pug_spar")
    elif dataset == "pug_animals":
        extract_pug_images(model, cache, "pug_animals", paths["datasets"]["pug_animals"],
                           ANIMALS_META, logger, id_prefix="pug_animals")
    elif dataset == "pug_imagenet":
        extract_pug_images(model, cache, "pug_imagenet",
                           paths["datasets"]["pug_imagenet"],
                           IMAGENET_META, logger, id_prefix="pug_imagenet")
    elif dataset == "pug_spar_official":
        # 官方协议预处理（Resize224 + ImageNet mean/std），缓存 key 同名
        extract_official_features(model, cache, "pug_spar_official",
                                  paths["datasets"]["pug_spar"], logger)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["all"] + EXTRACT_DATASETS,
                    help="要抽取特征的数据集（all = 全部数据集串行依次抽取）")
    ap.add_argument("--model", default="clip_vitl14",
                    help="冻结 baseline（all = 依次运行全部已配置模型，"
                         "串行执行、一次只加载一个）")
    ap.add_argument("--force", action="store_true", help="删除已有缓存重新抽取")
    args = ap.parse_args()

    paths = load_paths()
    datasets = resolve_datasets(args.dataset)
    logger = get_logger(f"extract_{args.dataset}", paths["dirs"]["outputs"] / "logs")
    # 嵌套顺序：模型在外层（一次只加载一个），数据集在内层；
    # 某个 (model, dataset) 中途失败即停止，后面的不再运行
    for name in resolve_models(args.model, paths):
        logger.info(f"[extract] model {name}：{len(datasets)} 个数据集串行")
        model = build_encoder(name, paths, logger)
        for ds in datasets:
            logger.info(f"[extract:{ds}] model {name}")
            extract_dataset(model, paths, logger, ds, force=args.force)
        del model
        free_encoder_memory()


if __name__ == "__main__":
    main()
