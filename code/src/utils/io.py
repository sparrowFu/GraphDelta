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


def _code_identity():
    """粗粒度代码身份（复核 R1）：当前 commit 与是否有未提交改动。
    仓库缺 git 或命令失败时返回空 dict，不阻塞实验运行。"""
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True,
                                timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True,
                                    timeout=10).stdout.strip())
        if commit:
            return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        pass
    return {}


def output_tables_dir(paths, model_name: str, seed: int, extra: dict = None) -> Path:
    """结果目录按模型与种子隔离（审查报告 R2）：outputs/<model>/seed_<seed>/tables/。

    切换 --model / --seed 不再互相覆盖；旧布局 outputs/tables、outputs/tables_seed{k}
    为历史结果，只读保留（summarize_seed_stability / make_figures 有回退逻辑）。

    目录内写 run_meta.json（model/seed/created + git 身份；复核 R1 的低成本
    来源记录：调用方经 extra 传入 cache_protocol 等预处理身份）；后处理入口
    据此核对数据身份，避免把历史 CLIP-L/14 结果当成其他模型。
    """
    d = paths["dirs"]["outputs"] / model_name / f"seed_{seed}" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "run_meta.json"
    if not meta.exists():
        from datetime import datetime
        info = {"model": model_name, "seed": seed,
                "created": datetime.now().isoformat(timespec="seconds")}
        info.update(_code_identity())
        if extra:
            info.update(extra)
        meta.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return d


def verify_run_meta(tables_dir, model_name: str, seed: int) -> None:
    """核对结果目录的 run_meta.json 身份（审查报告 R2/R7 读取端）。

    目录无 run_meta.json（旧布局或显式 --tables-dir 指向的历史目录）时不校验；
    有则 model/seed 必须与请求一致，否则报错。显式目录的完整校验用
    verify_tables_identity（审查 A5）。
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


# 旧布局目录的固定属主（R2 隔离前的历史结果，无 run_meta.json）
LEGACY_TABLES_IDENTITY = {
    "tables": ("clip_vitl14", 0),
    "tables_seed1": ("clip_vitl14", 1),
    "tables_seed2": ("clip_vitl14", 2),
}


def verify_tables_identity(tables_dir, out_dir, model_name: str, seed: int) -> None:
    """显式指定的 tables 目录也必须有可验证身份（审查 A5）。

    - 有 run_meta.json → 核对 model/seed；
    - 无 run_meta.json → 只接受已知的旧布局目录（outputs/tables、
      outputs/tables_seed1/2，属主固定为 clip_vitl14 对应种子），且请求的
      model/seed 必须与属主一致；
    - 其余无元数据目录一律拒绝：不能被标成任何模型的正式结果。
    """
    d = Path(tables_dir)
    if (d / "run_meta.json").exists():
        verify_run_meta(d, model_name, seed)
        return
    try:
        rel = str(d.resolve().relative_to(Path(out_dir).resolve()))
    except ValueError:
        rel = None
    ident = LEGACY_TABLES_IDENTITY.get(rel) if rel else None
    if ident is None:
        raise SystemExit(
            f"目录 {d} 没有 run_meta.json，也不属于已知的旧布局目录，无法验证"
            f"其模型/种子身份——不能作为 {model_name} seed {seed} 的正式结果"
            f"来源（审查 A5）。")
    if ident != (model_name, seed):
        raise SystemExit(
            f"旧布局目录 {d} 属于 {ident[0]} seed {ident[1]}，与请求的 "
            f"{model_name} seed {seed} 不一致。")


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
