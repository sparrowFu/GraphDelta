#!/usr/bin/env python
"""汇总 E4 结果并给出阶段判断（阶段二 Step 5，工作方案第 7 节）。

读取 outputs/<model>/seed_<seed>/tables/ 的 e4*.csv，写 e4_summary.csv 与
e4_stage_decision.json（结构可辨识 / 跨模态对应 / 方向迁移三级标注）。
不加载模型。正式汇总只消费已成功提交的正式产物（run_meta 核验），不回退
smoke 文件；--smoke 只读 *_smoke.csv，写独立文件名，仅记录流程是否跑通。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e4.summarize import summarize, summarize_smoke
from src.models import resolve_models
from src.utils.io import load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    help="汇总的模型；all = 依次汇总全部已配置模型（串行，不加载模型）")
    ap.add_argument("--seed", type=int, default=0,
                    help="结果所属种子（定位 tables 目录）")
    ap.add_argument("--smoke", action="store_true",
                    help="只汇总 *_smoke 产物（独立文件名，仅记录流程是否跑通，"
                         "不给出正式机制支持标签）")
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger("summarize_e4", paths["dirs"]["outputs"] / "logs")
    for name in resolve_models(args.model, paths):
        if args.smoke:
            summarize_smoke(name, args.seed, paths, logger)
        else:
            summarize(name, args.seed, paths, logger)


if __name__ == "__main__":
    main()
