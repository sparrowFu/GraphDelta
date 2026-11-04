#!/usr/bin/env python
"""生成编辑对清单（阶段二 Step 2，工作方案第 3、4 节）。

采样基础场景 → 应用编辑算子 → 逐对 validate_edit → 家族内 60/20/20 划分 →
写 <task>_pairs.parquet / <task>_states.parquet / <task>_splits.json 到
data/edit_pairs/。全部由 --seed 决定，可复现；逆向复用同一 pair_id。

用法（任意目录）：
  python code/PreExperiments/stage2_controlled_edit/scripts/build_edit_pairs.py \
      [--task all] [--binding-n 800] [--relation-n 500]
      [--recolor-n 200] [--control-n 200] [--seed 20261103]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from e4.pairs import build_task, write_manifest
from src.utils.io import load_config, load_paths
from src.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "binding", "relation"],
                    help="生成哪个任务的编辑对；all = 两个任务依次生成")
    ap.add_argument("--binding-n", type=int, default=800,
                    help="颜色归属交换对数（8 个转换均分，每转换 N/8 个基础场景）")
    ap.add_argument("--relation-n", type=int, default=500,
                    help="左右关系反转对数（上限 1,000；缩范围须在正式批量生成时决定并记录）")
    ap.add_argument("--recolor-n", type=int, default=200,
                    help="目标对象换色正控制对数（复用绑定场景；新颜色取其余 6 色）")
    ap.add_argument("--control-n", type=int, default=200,
                    help="无语义变化扰动对照对数（jpeg/crop 各半，仅需后处理不占渲染账本）")
    ap.add_argument("--seed", type=int, default=20261103,
                    help="场景采样与划分种子；换种子须分别报告，不合并样本")
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["dirs"]["edit_pairs"]
    cfg = load_config("e4_protocol")
    logger = get_logger("build_edit_pairs", paths["dirs"]["outputs"] / "logs")
    tasks = ["binding", "relation"] if args.task == "all" else [args.task]
    for task in tasks:
        data = build_task(task, args.seed, n_swap=args.binding_n,
                          n_recolor=args.recolor_n, n_flip=args.relation_n,
                          n_control=args.control_n,
                          ratios=dict(cfg["split"]),
                          postprocess=dict(cfg["postprocess"]))
        meta = write_manifest(out_dir, task, data)
        logger.info(
            f"[build] {task}: {meta['counts']['pairs']} 对 / "
            f"{meta['counts']['render_states']} 渲染态 / "
            f"重采样 {meta['counts']['resamples']} -> {out_dir}")
        cov = {}
        for p in data["pairs"]:
            cov.setdefault(p["family"], set()).add(p["split"])
        incomplete = sorted(f for f, s in cov.items()
                            if s != {"train", "val", "heldout"})
        if incomplete:
            logger.warning(f"[build] {task} 以下家族 split 不全（场景数过少）: "
                           f"{incomplete}")


if __name__ == "__main__":
    main()
