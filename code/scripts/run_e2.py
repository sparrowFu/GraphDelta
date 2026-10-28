#!/usr/bin/env python
"""运行 E2: Single-Factor Representation Response Diagnosis（PUG: Animals / ImageNet）。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import e2_change_diagnosis
from src.models import build_encoder, free_encoder_memory, resolve_models
from src.utils.io import load_paths
from src.utils.logging import get_logger

# --dataset 的全部取值；all 的串行顺序与此列表一致（核心数据集在前）
E2_DATASETS = ["pug_animals", "pug_imagenet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14",
                    help="冻结 baseline（all = 依次运行全部已配置模型，"
                         "串行执行、一次只加载一个）")
    ap.add_argument("--dataset", default="pug_animals",
                    choices=["all"] + E2_DATASETS,
                    help="E2 数据集（all = 两个数据集串行；pug_imagenet 为"
                         "方案 §14 辅助验证，审查 A1）")
    ap.add_argument("--smoke", action="store_true", help="小样本快速自检")
    ap.add_argument("--seed", type=int, default=None,
                    help="配对与 nuisance 采样种子（默认取配置 seeds[0]）")
    ap.add_argument("--allow-empty-factors", action="store_true",
                    help="某因素 0 对配对时跳过而非停止（复核 B2；结果不构成完整因素集）")
    args = ap.parse_args()

    paths = load_paths()
    datasets = list(E2_DATASETS) if args.dataset == "all" else [args.dataset]
    logger = get_logger(f"run_e2_{args.dataset}", paths["dirs"]["outputs"] / "logs")
    logger.info(f"[run_e2] datasets={datasets}")
    # 嵌套顺序：模型在外层（一次只加载一个），数据集在内层；
    # 某个 (model, dataset) 中途失败即停止，后面的不再运行
    for name in resolve_models(args.model, paths):
        logger.info(f"[run_e2] model {name}：{len(datasets)} 个数据集串行")
        model = build_encoder(name, paths, logger)
        for ds in datasets:
            ds_logger = get_logger(f"run_e2_{ds}", paths["dirs"]["outputs"] / "logs")
            ds_logger.info(f"[run_e2_{ds}] model {name}")
            e2_change_diagnosis.run(model, paths, ds_logger, smoke=args.smoke,
                                    seed=args.seed, dataset=ds,
                                    allow_empty_factors=args.allow_empty_factors)
        del model
        free_encoder_memory()


if __name__ == "__main__":
    main()
