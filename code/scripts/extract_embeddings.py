#!/usr/bin/env python
"""抽取图像特征到缓存（方案 §62）。所有特征只抽一次，后续实验读缓存。

进度日志统一为 "[extract:key] N/M"（每 2000 张）；缓存按 id 增量补齐，
--force 删除后重抽。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clevr
from src.data.cogent import condition_image_dir, load_condition_scenes
from src.data.pug_animals import META_COLUMNS as ANIMALS_META
from src.data.pug_spar import META_COLUMNS as SPAR_META
from src.experiments.e1_spar_official import extract_official_features
from src.features.cache import ImageFeatureCache
from src.features.extract import extract_clevr_images, extract_pug_images
from src.models import build_encoder
from src.utils.io import load_config, load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["clevr_val", "cogent_valA", "cogent_valB",
                             "pug_spar", "pug_animals", "pug_spar_official"])
    ap.add_argument("--model", default="clip_vitl14")
    ap.add_argument("--force", action="store_true", help="删除已有缓存重新抽取")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger(f"extract_{args.dataset}", paths["dirs"]["outputs"] / "logs")
    model = build_encoder(args.model, paths, logger)
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name, protocol=model.cache_protocol)

    if args.force and cache.exists(args.dataset):
        cache.path(args.dataset).unlink()

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
    elif args.dataset == "pug_spar_official":
        # 官方协议预处理（Resize224 + ImageNet mean/std），缓存 key 同名
        extract_official_features(model, cache, "pug_spar_official",
                                  paths["datasets"]["pug_spar"], logger)


if __name__ == "__main__":
    main()
