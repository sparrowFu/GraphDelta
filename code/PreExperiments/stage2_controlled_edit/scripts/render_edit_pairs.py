#!/usr/bin/env python
"""渲染编辑对（阶段二 Step 3，工作方案第 3、8 节）。

读清单 → 唯一状态 → Blender（规格驱动）渲染 → 场景图复核 → rendered_states
回填 → 账本登记 → jpeg/crop 后处理对照。吞吐含校验与重试；账本已完成的
状态默认跳过。渲染环境路径来自 configs/paths.yaml 的 render 段。

用法（任意目录）：
  python code/PreExperiments/stage2_controlled_edit/scripts/render_edit_pairs.py \
      [--task all] [--workers 1] [--limit N] [--force]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e4.pairs import load_manifest
from e4.render import render_manifest
from src.utils.io import load_config, load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "binding", "relation"],
                    help="渲染哪个任务的清单")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 Blender 进程数；加速比须实测写入账本，不按进程数线性外推")
    ap.add_argument("--limit", type=int, default=None,
                    help="只渲染前 N 个唯一状态（小批试运行 / 冒烟）")
    ap.add_argument("--force", action="store_true",
                    help="重渲染已完成状态；默认跳过账本中已登记的状态")
    args = ap.parse_args()

    paths = load_paths()
    cfg = load_config("e4_protocol")
    logger = get_logger("render_edit_pairs", paths["dirs"]["outputs"] / "logs")
    tasks = ["binding", "relation"] if args.task == "all" else [args.task]
    for task in tasks:
        m = load_manifest(paths["dirs"]["edit_pairs"], task)
        render_manifest(task, m, paths, cfg, logger,
                        workers=args.workers, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
