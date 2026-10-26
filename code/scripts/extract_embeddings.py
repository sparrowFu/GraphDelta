#!/usr/bin/env python
"""抽取图像特征到缓存（方案 §62）。所有特征只抽一次，后续实验读缓存。

进度日志统一为 "[extract:key] N/M"（每 2000 张）；缓存按 id 增量补齐，
--force 删除后重抽。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clevr
from src.data.cogent import condition_image_dir, load_condition_scenes
from src.data.pug_animals import META_COLUMNS as ANIMALS_META
from src.data.pug_imagenet import META_COLUMNS as IMAGENET_META
from src.data.pug_spar import META_COLUMNS as SPAR_META
from src.experiments.e1_spar_official import extract_official_features
from src.features.cache import ImageFeatureCache
from src.features.extract import extract_clevr_images, extract_pug_images
from src.models import build_encoder
from src.utils.io import load_config, load_paths
from src.utils.logging import get_logger


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["clevr_val", "cogent_valA", "cogent_valB",
                             "pug_spar", "pug_animals", "pug_imagenet",
                             "pug_spar_official"])
    ap.add_argument("--model", default="clip_vitl14")
    ap.add_argument("--force", action="store_true", help="删除已有缓存重新抽取")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger(f"extract_{args.dataset}", paths["dirs"]["outputs"] / "logs")
    model = build_encoder(args.model, paths, logger)
    if args.force:
        _force_reset(paths["dirs"]["cache"], model, args.dataset, logger)
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name, protocol=model.cache_protocol)

    if args.dataset == "clevr_val":
        cfg = load_config("e1_state")["clevr"]
        scenes = clevr.load_scenes(f"{paths['datasets']['clevr']}/{cfg['scene_file']}")
        extract_clevr_images(model, cache, "clevr_val", scenes,
                             f"{paths['datasets']['clevr']}/{cfg['image_dir']}", logger)
    elif args.dataset in ("cogent_valA", "cogent_valB"):
        cond = args.dataset[-1]
        scenes = load_condition_scenes(paths["datasets"]["cogent"], cond)
        extract_clevr_images(model, cache, args.dataset, scenes,
                             condition_image_dir(paths["datasets"]["cogent"], cond), logger)
    elif args.dataset == "pug_spar":
        extract_pug_images(model, cache, "pug_spar", paths["datasets"]["pug_spar"],
                           SPAR_META, logger, id_prefix="pug_spar")
    elif args.dataset == "pug_animals":
        extract_pug_images(model, cache, "pug_animals", paths["datasets"]["pug_animals"],
                           ANIMALS_META, logger, id_prefix="pug_animals")
    elif args.dataset == "pug_imagenet":
        extract_pug_images(model, cache, "pug_imagenet",
                           paths["datasets"]["pug_imagenet"],
                           IMAGENET_META, logger, id_prefix="pug_imagenet")
    elif args.dataset == "pug_spar_official":
        # 官方协议预处理（Resize224 + ImageNet mean/std），缓存 key 同名
        extract_official_features(model, cache, "pug_spar_official",
                                  paths["datasets"]["pug_spar"], logger)


if __name__ == "__main__":
    main()
