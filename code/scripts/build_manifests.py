#!/usr/bin/env python
"""生成统一 manifest（方案 §57–§58）：只整理已有 metadata，不生成新图像。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pug_animals import build_metadata as build_animals
from src.data.pug_imagenet import build_metadata as build_imagenet
from src.data.pug_spar import build_metadata as build_spar
from src.utils.io import load_paths
from src.utils.logging import get_logger

BUILDERS = {
    "pug_spar": build_spar,
    "pug_animals": build_animals,
    "pug_imagenet": build_imagenet,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all"] + list(BUILDERS))
    args = ap.parse_args()

    paths = load_paths()
    logger = get_logger("build_manifests", paths["dirs"]["outputs"] / "logs")
    man_dir = paths["dirs"]["manifests"]
    names = list(BUILDERS) if args.dataset == "all" else [args.dataset]

    inventory = {}
    for name in names:
        df = BUILDERS[name](paths["datasets"][name])
        out = man_dir / f"{name}_meta.parquet"
        df.to_parquet(out, index=False)
        inv = {c: int(df[c].nunique()) for c in df.columns if c not in ("row_id", "sample_id")}
        inventory[name] = {"rows": len(df), "uniques": inv}
        logger.info(f"[manifest] {name}: {len(df)} rows -> {out}")
        logger.info(json.dumps(inventory[name]["uniques"], ensure_ascii=False))

    with open(man_dir / "inventory.json", "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
