#!/usr/bin/env python
"""运行 E2: Single-Factor Representation Response Diagnosis（PUG: Animals）。"""
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
    ap.add_argument("--smoke", action="store_true", help="小样本快速自检")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger("run_e2", paths["dirs"]["outputs"] / "logs")
    model = build_encoder(args.model, paths, logger)
    e2_change_diagnosis.run(model, paths, logger, smoke=args.smoke)


if __name__ == "__main__":
    main()
