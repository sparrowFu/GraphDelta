#!/usr/bin/env python
"""运行 E3: Composition Difficulty Diagnosis（CoGenT valA/valB + sphere 对照）。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import e3_cogent_diagnosis
from src.models import build_encoder
from src.utils.io import load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitl14")
    ap.add_argument("--smoke", action="store_true", help="小样本快速自检")
    ap.add_argument("--seed", type=int, default=None,
                    help="候选采样种子（默认取配置 seeds[0]；检验多次采样稳定性用）")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger("run_e3", paths["dirs"]["outputs"] / "logs")
    model = build_encoder(args.model, paths, logger)
    e3_cogent_diagnosis.run(model, paths, logger, smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()
