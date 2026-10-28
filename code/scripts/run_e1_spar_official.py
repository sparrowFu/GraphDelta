#!/usr/bin/env python
"""运行 PUG:SPAR 官方协议评测（检索 top1/top5 + hard-negative，P2-11）。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import e1_spar_official
from src.models import build_encoder, free_encoder_memory, resolve_models
from src.utils.io import load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14",
                    help="冻结 baseline（all = 依次运行全部已配置模型，"
                         "串行执行、一次只加载一个）")
    ap.add_argument("--smoke", action="store_true", help="小样本快速自检")
    ap.add_argument("--seed", type=int, default=None,
                    help="子集抽样种子（默认取配置 seeds[0]；检验多次采样稳定性用）")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger("run_e1_spar_official", paths["dirs"]["outputs"] / "logs")
    # --model all：串行依次评估各模型，一次只加载一个模型；
    # 某个模型中途失败即停止，后面的模型不再运行
    for name in resolve_models(args.model, paths):
        logger.info(f"[run_e1_spar_official] model {name}")
        model = build_encoder(name, paths, logger)
        e1_spar_official.run(model, paths, logger, smoke=args.smoke, seed=args.seed)
        del model
        free_encoder_memory()


if __name__ == "__main__":
    main()
