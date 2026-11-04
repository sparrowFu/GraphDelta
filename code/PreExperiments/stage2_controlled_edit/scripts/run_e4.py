#!/usr/bin/env python
"""运行 E4: Controlled-Edit Diagnosis（阶段二 Step 4，工作方案第 5–7 节）。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e4 import run as e4_run
from src.models import build_encoder, free_encoder_memory, resolve_models
from src.utils.io import load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14",
                    help="冻结 baseline（all = 依次运行全部已配置模型，串行执行）")
    ap.add_argument("--stage", default="all", choices=["all", "a", "b", "c", "d"],
                    help="运行哪个子实验；all = 依次运行 E4-A→D")
    ap.add_argument("--smoke", action="store_true",
                    help="小样本快速自检（配合小批清单 / --limit 渲染），输出带 _smoke 后缀")
    ap.add_argument("--seed", type=int, default=None,
                    help="置换 / bootstrap / 探针种子（默认取配置 seeds[0]）")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger("run_e4", paths["dirs"]["outputs"] / "logs")
    for name in resolve_models(args.model, paths):
        logger.info(f"[run_e4] model {name}")
        model = build_encoder(name, paths, logger)
        e4_run.run(model, paths, logger, stage=args.stage,
                   smoke=args.smoke, seed=args.seed)
        del model
        free_encoder_memory()


if __name__ == "__main__":
    main()
