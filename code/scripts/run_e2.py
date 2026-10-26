#!/usr/bin/env python
"""运行 E2: Single-Factor Representation Response Diagnosis（PUG: Animals / ImageNet）。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import e2_change_diagnosis
from src.models import build_encoder
from src.utils.io import load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14")
    ap.add_argument("--dataset", default="pug_animals",
                    choices=["pug_animals", "pug_imagenet"],
                    help="E2 数据集（pug_imagenet 为方案 §14 辅助验证，审查 A1）")
    ap.add_argument("--smoke", action="store_true", help="小样本快速自检")
    ap.add_argument("--seed", type=int, default=None,
                    help="配对与 nuisance 采样种子（默认取配置 seeds[0]）")
    ap.add_argument("--allow-empty-factors", action="store_true",
                    help="某因素 0 对配对时跳过而非停止（复核 B2；结果不构成完整因素集）")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger(f"run_e2_{args.dataset}", paths["dirs"]["outputs"] / "logs")
    model = build_encoder(args.model, paths, logger)
    e2_change_diagnosis.run(model, paths, logger, smoke=args.smoke, seed=args.seed,
                            dataset=args.dataset,
                            allow_empty_factors=args.allow_empty_factors)


if __name__ == "__main__":
    main()
