"""路径与配置加载。所有相对路径以 code/ 为根解析。"""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name):
    """加载 configs/<name>.yaml"""
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def load_paths():
    """加载 paths.yaml 并把相对目录解析为绝对路径。"""
    cfg = load_config("paths")
    dirs = {}
    for key, val in cfg["dirs"].items():
        p = Path(val)
        dirs[key] = p if p.is_absolute() else (ROOT / p)
        dirs[key].mkdir(parents=True, exist_ok=True)
    cfg["dirs"] = dirs
    return cfg


def resolve(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def output_tables_dir(paths, model_name: str, seed: int) -> Path:
    """结果目录按模型与种子隔离（审查报告 R2）：outputs/<model>/seed_<seed>/tables/。

    切换 --model / --seed 不再互相覆盖；旧布局 outputs/tables、outputs/tables_seed{k}
    为历史结果，只读保留（summarize_seed_stability / make_figures 有回退逻辑）。

    目录内写 run_meta.json（model/seed/created，R2 读取端身份校验用）；
    后处理入口据此核对数据身份，避免把历史 CLIP-L/14 结果当成其他模型。
    """
    d = paths["dirs"]["outputs"] / model_name / f"seed_{seed}" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "run_meta.json"
    if not meta.exists():
        from datetime import datetime
        meta.write_text(json.dumps(
            {"model": model_name, "seed": seed,
             "created": datetime.now().isoformat(timespec="seconds")},
            ensure_ascii=False, indent=2))
    return d


def verify_run_meta(tables_dir, model_name: str, seed: int) -> None:
    """核对结果目录的 run_meta.json 身份（审查报告 R2/R7 读取端）。

    目录无 run_meta.json（旧布局或显式 --tables-dir 指向的历史目录）时不校验；
    有则 model/seed 必须与请求一致，否则报错。
    """
    meta = Path(tables_dir) / "run_meta.json"
    if not meta.exists():
        return
    m = json.loads(meta.read_text())
    if m.get("model") != model_name or m.get("seed") != seed:
        raise SystemExit(
            f"结果目录身份不符：{tables_dir} 的 run_meta.json 记录 "
            f"model={m.get('model')!r}, seed={m.get('seed')!r}，"
            f"与请求的 model={model_name!r}, seed={seed!r} 不一致。"
            f"请核对 --model/--seed 或 --tables-dir。")


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
