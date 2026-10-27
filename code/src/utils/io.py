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


def _experiment_ident(rec):
    """实验记录里的身份字段（问题 1 / 补充 D 的比较基准）。"""
    return {k: rec.get(k) for k in ("git_commit", "git_dirty", "cache_protocol")}


def output_tables_dir(paths, model_name: str, seed: int, extra: dict = None,
                      experiment: str = None) -> Path:
    """结果目录按模型与种子隔离（审查报告 R2）：outputs/<model>/seed_<seed>/tables/。

    切换 --model / --seed 不再互相覆盖；旧布局 outputs/tables、outputs/tables_seed{k}
    为历史结果，只读保留（summarize_seed_stability / make_figures 有回退逻辑）。

    run_meta.json 的写入规则（四轮审查问题 1：失败运行不能改写成功来源）：
    - 首次创建：写 model/seed/created + git 身份 + extra（cache_protocol 等）；
    - 复用目录：顶层字段保留首次记录；experiment 参数只登记**本次尝试**
      （status=running + 身份 + started），不触碰 last_success——运行在数据
      读取、配对验证或写盘中途失败时，last_success 仍指向旧的成功产出，
      失败的尝试不能冒充当前来源；
    - 输出全部成功写出后由 commit_experiment 提交 last_success（含产物清单）
      并把 status 置 success；
    - replaces（四轮补充 D）：身份变化的历史以**列表**追加，同版本重复重跑
      不再使留痕消失；旧格式（单条 dict）读取时自动转为列表。
    """
    from datetime import datetime
    d = paths["dirs"]["outputs"] / model_name / f"seed_{seed}" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "run_meta.json"
    now = datetime.now().isoformat(timespec="seconds")
    ident = _code_identity()
    if extra:
        ident.update(extra)
    if meta.exists():
        info = json.loads(meta.read_text())
    else:
        info = {"model": model_name, "seed": seed, "created": now}
        info.update(ident)
    if experiment:
        exps = info.setdefault("experiments", {})
        rec = {**ident, "status": "running", "started": now}
        prev = exps.get(experiment)
        if prev is not None:
            # 旧 schema 迁移（复核缺口 1）：只匹配明确的三轮旧格式——身份在
            # 记录顶层，且没有 status / last_success 字段。新格式的 running
            # 记录同样没有 last_success，但有 status 字段；从未成功过的运行
            # 失败后留下的正是这种记录，不能推定成历史成功来源——成功来源
            # 只能由 commit_experiment 提交产生。
            if ("last_success" not in prev and "status" not in prev
                    and any(k in prev for k in ident)):
                prev = {**prev, "last_success": {**_experiment_ident(prev),
                                                 "updated": prev.get("updated")}}
            if prev.get("last_success") is not None:
                rec["last_success"] = prev["last_success"]
            if prev.get("replaces"):
                rec["replaces"] = prev["replaces"]
        exps[experiment] = rec
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return d


def commit_experiment(tables_dir, experiment: str, outputs) -> None:
    """实验输出成功写出后提交来源（四轮审查问题 1）。

    - 用本次尝试的身份更新 experiments/<name>.last_success（含产物清单
      outputs），status 置 success；失败/中断的运行不会走到这里，
      last_success 保持旧值；
    - 新 last_success 与上一次身份不同时，把旧来源**追加**进 replaces 列表
      （补充 D：同版本重跑不追加也不丢失，变更留痕一直保留）；
    - 未先经 output_tables_dir 登记尝试时按错误处理，防止凭空提交。
    """
    from datetime import datetime
    meta = Path(tables_dir) / "run_meta.json"
    if not meta.exists():
        raise RuntimeError(f"{meta} 不存在：commit_experiment 必须在 "
                           f"output_tables_dir 之后调用")
    info = json.loads(meta.read_text())
    rec = info.setdefault("experiments", {}).get(experiment)
    if rec is None:
        raise RuntimeError(f"实验 {experiment!r} 未登记尝试："
                           f"commit_experiment 必须在 output_tables_dir 之后调用")
    now = datetime.now().isoformat(timespec="seconds")
    prev_success = rec.get("last_success")
    new_success = {**_experiment_ident(rec), "updated": now,
                   "outputs": sorted(str(o) for o in outputs)}
    replaces = rec.get("replaces") or []
    if isinstance(replaces, dict):     # 旧格式（单条 dict）→ 列表
        replaces = [replaces]
    if prev_success is not None and \
            _experiment_ident(prev_success) != _experiment_ident(new_success):
        replaces = replaces + [{**_experiment_ident(prev_success),
                                "updated": prev_success.get("updated")}]
    rec["replaces"] = replaces
    rec["last_success"] = new_success
    rec["status"] = "success"
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=2))


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


def e3_source_fingerprint(tables_dir):
    """E3 派生单元表的源输入指纹（四轮审查问题 3）。

    analyze_e3_units 消费 e3_items_cogent.jsonl 与 e3_difficulty_gap.csv，
    指纹 = 两个文件内容哈希的组合；任一文件缺失返回 None（无法核对的
    旧目录不比对，有指纹的目录必须匹配）。
    """
    d = Path(tables_dir)
    parts = []
    for name in ("e3_items_cogent.jsonl", "e3_difficulty_gap.csv"):
        p = d / name
        if not p.exists():
            return None
        parts.append(f"{name}={md5_file(p)}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def verify_experiment_committed(tables_dir, experiment: str,
                                required_files=()) -> None:
    """正式消费前核对实验来源已完成（复核问题 1 缺口 2：部分写出的结果
    不得进入正式汇总/绘图）。

    运行中途失败时 status 停在 running、last_success 指向旧产物，但磁盘上
    的文件可能已被本次运行部分改写（例如 E2 写出第一张表后中断）。只校验
    model/seed 不够，正式消费者必须拒绝未完成、产物身份无法对应的实验：

    - 无 run_meta.json（旧布局/显式历史目录）：不校验（与 verify_run_meta
      的兼容口径一致）；
    - 实验记录 status=running：该实验有未完成的运行，直接停止；
    - last_success 带产物清单（outputs）时：required_files 必须都在清单内
      ——清单外的文件不属于任何已提交的成功运行；
    - 旧 schema 记录（无 status / 无 outputs 清单）：无证据表明未完成，
      放行。
    """
    meta = Path(tables_dir) / "run_meta.json"
    if not meta.exists():
        return
    info = json.loads(meta.read_text())
    rec = (info.get("experiments") or {}).get(experiment)
    if rec is None:
        return
    if rec.get("status") == "running":
        raise SystemExit(
            f"{meta} 中实验 {experiment!r} 的 status=running——该实验存在未"
            f"完成的运行（启动后未成功提交产物），目录里的文件可能部分来自"
            f"本次中断的运行，不能作为正式结果消费。请重跑该实验至完成。")
    outputs = (rec.get("last_success") or {}).get("outputs")
    if outputs is not None:
        missing = [str(f) for f in required_files if str(f) not in outputs]
        if missing:
            raise SystemExit(
                f"{meta} 中实验 {experiment!r} 的成功产物清单不含 {missing}"
                f"——这些文件不属于任何已提交的成功运行，不能作为正式结果"
                f"消费。")


def write_text_atomic(path, text) -> None:
    """同目录临时文件 + rename（复核缺口 3）：中断不会留下半写的目标文件。"""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(p)


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_e3_consistency(tables_dir) -> None:
    """消费 e3_unit_gaps.csv 前核对一致性 sidecar（三轮问题 2 + 四轮问题 3
    + 复核缺口 3）。

    - status=failed：items 与主表冲突，旧单元表已删除/失效，不得消费；
    - status=stale：E3 源结果已重写（run_e3 全量成功后主动失效），待重算；
    - status=pending：分析进行中或上次分析中断（analyze 先发 pending、
      单元表安全写出后才提交最终状态），当前单元表不能消费；
    - source_fingerprint：核对当前源文件指纹——E3 源结果更新后残留的旧
      单元表即使 sidecar 仍是 passed 也会因指纹不匹配被拦截；
    - unit_table_md5：核对当前 e3_unit_gaps.csv 内容哈希——sidecar 的
      passed 只对"由该次分析写出的那份单元表"有效，旧表顶替新表（哈希
      不一致）同样拦截；
    - not_checked / passed（无指纹的旧 sidecar 兼容）不拦截。
    """
    d = Path(tables_dir)
    p = d / "e3_unit_consistency.json"
    if not p.exists():
        return
    c = json.loads(p.read_text())
    status = c.get("status")
    if status is None:
        status = "passed" if c.get("all_ok") else "failed"
    if status == "failed":
        raise SystemExit(
            f"{p} 记录一致性检查失败（items 与主表冲突）——该目录的"
            f"e3_unit_gaps.csv 已失效，请先重跑 analyze_e3_units 解决冲突再消费。")
    if status == "stale":
        raise SystemExit(
            f"{p} 记录 status=stale——E3 源结果已重写，e3_unit_gaps.csv 待重跑 "
            f"analyze_e3_units 后才能消费。")
    if status == "pending":
        raise SystemExit(
            f"{p} 记录 status=pending——单元表分析未完成或上次中断，"
            f"e3_unit_gaps.csv 不能消费，请重跑 analyze_e3_units。")
    fp = c.get("source_fingerprint")
    if fp is not None:
        cur = e3_source_fingerprint(d)
        if cur != fp:
            raise SystemExit(
                f"{p} 记录的源输入指纹与当前 e3_items_cogent.jsonl / "
                f"e3_difficulty_gap.csv 不匹配——E3 源结果已更新，"
                f"e3_unit_gaps.csv 是过期派生表，请重跑 analyze_e3_units 再消费。")
    ut = c.get("unit_table_md5")
    if ut is not None:
        units = d / "e3_unit_gaps.csv"
        if not units.exists() or md5_file(units) != ut:
            raise SystemExit(
                f"{p} 记录的 e3_unit_gaps.csv 内容哈希与当前文件不匹配——"
                f"当前单元表不是该次分析写出的版本（旧表残留或被改写），"
                f"请重跑 analyze_e3_units 再消费。")


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
