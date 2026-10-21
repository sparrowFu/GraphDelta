"""Verify the fix for the P2 result-directory regression found by the style review.

Style review (8ebdd3f): a fresh standalone run of E2 / E3 / official SPAR fails at
to_csv because outputs/tables is never created. The fix adds
out_tables.mkdir(parents=True, exist_ok=True) to every experiment run().

This script re-runs those scenarios against the working tree without
pre-creating any output directory. Historical review results are untouched;
results go to fix_outputs_tables_results.json.

Run: python -B code/review/fix_outputs_tables_checks.py
Exit 1 indicates the regression is still reproducible, not a script crash.
"""
import ast
import importlib
import json
import logging
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

FIXED_FILES = [
    "code/src/experiments/e1_state_diagnosis.py",
    "code/src/experiments/e1_spar_official.py",
    "code/src/experiments/e2_change_diagnosis.py",
    "code/src/experiments/e3_cogent_diagnosis.py",
]
MKDIR_LINE = "    out_tables.mkdir(parents=True, exist_ok=True)"


class Encoder:
    name = "fix_tables"

    def encode_images(self, images):
        raw = np.array([[float(np.asarray(im).mean()), 1.] for im in images], dtype=np.float32)
        return dict(raw=raw, normalized=raw / np.linalg.norm(raw, axis=1, keepdims=True))

    def encode_texts(self, texts, ensure_period=True):
        eff = [t if not ensure_period or t.endswith(".") else t + "." for t in texts]
        raw = np.array([[float(len(t)), 1.] for t in eff], dtype=np.float32)
        return dict(raw=raw, normalized=raw / np.linalg.norm(raw, axis=1, keepdims=True))


def mkdir_follows_out_tables_init(source):
    """run() 内每处 out_tables 赋值后紧跟 mkdir(parents=True, exist_ok=True)。"""
    tree = ast.parse(source)
    seen = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "run"):
            continue
        for i, stmt in enumerate(node.body):
            if not (isinstance(stmt, ast.Assign)
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "out_tables"):
                continue
            seen += 1
            nxt = node.body[i + 1] if i + 1 < len(node.body) else None
            ok = (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Call)
                  and isinstance(nxt.value.func, ast.Attribute)
                  and nxt.value.func.attr == "mkdir"
                  and isinstance(nxt.value.func.value, ast.Name)
                  and nxt.value.func.value.id == "out_tables"
                  and [k.arg for k in nxt.value.keywords] == ["parents", "exist_ok"]
                  and all(getattr(k.value, "value", None) is True for k in nxt.value.keywords))
            if not ok:
                return False, seen
    return seen > 0, seen


def fresh_paths(tmp, scenario, dataset_key, dataset_value):
    """每个场景独立的全新 outputs/cache 根：tables / logs 均不存在。"""
    root = tmp / scenario
    outputs = root / "outputs"
    assert not (outputs / "tables").exists()
    return dict(dirs=dict(cache=root / "cache", outputs=outputs),
                datasets={dataset_key: dataset_value})


def main():
    results = []

    def check(name, passed, evidence):
        row = dict(check=name, passed=bool(passed), evidence=evidence)
        results.append(row)
        print(json.dumps(row, ensure_ascii=True))

    for path in FIXED_FILES:
        ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    init_ok = {p: mkdir_follows_out_tables_init((ROOT / p).read_text(encoding="utf-8"))
               for p in FIXED_FILES}
    check("every_run_creates_out_tables", all(ok and seen == 1 for ok, seen in init_ok.values()),
          {p: dict(followed=ok, assignments=seen) for p, (ok, seen) in init_ok.items()})

    diff = subprocess.run(["git", "diff", "-U0", "--", *FIXED_FILES],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout
    added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    check("diff_only_adds_mkdir_lines", not removed and added and set(added) == {MKDIR_LINE},
          dict(added=added, removed=len(removed)))

    base = ROOT.resolve()
    with tempfile.TemporaryDirectory(prefix=".fix_tables_", dir=base) as tmpstr:
        tmp = Path(tmpstr).resolve()
        assert tmp.is_relative_to(base)

        # --- E3：风格审查复现的原场景，独立首跑 ---
        from src.experiments import e3_cogent_diagnosis as e3
        from src.utils import io as project_io
        from src.utils.logging import get_logger
        cfg = project_io.load_config("e3_cogent")
        cfg["bootstrap"]["n_boot"] = 10
        cfg["sampling"]["per_scene_cap"] = 3
        cfg["sampling"]["target_per_condition_shape"] = 20
        paths = fresh_paths(tmp, "e3", "cogent", "fixture")
        logger = get_logger("fix_tables_e3", paths["dirs"]["outputs"] / "logs")
        logger.setLevel(logging.CRITICAL)

        def scenes(_root, condition):
            colors = ["blue", "red", "green"] if condition == "A" else ["red", "blue", "green"]
            return [dict(image_filename=f"{condition}.png", objects=[dict(color=c, shape=s)
                         for c, s in zip(colors, ["cube", "cylinder", "sphere"])])]

        def extract(model, cache, key, selected, *args, **kwargs):
            raw = np.ones((len(selected), 2), dtype=np.float32)
            cache.save(key, [s["image_filename"] for s in selected], raw, raw / np.sqrt(2))

        error = None
        try:
            with patch.object(e3, "load_config", return_value=cfg), \
                    patch.object(e3, "load_condition_scenes", side_effect=scenes), \
                    patch.object(e3, "extract_clevr_images", side_effect=extract):
                e3.run(Encoder(), paths, logger, smoke=True)
        except OSError as exc:
            error = str(exc)
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
        expected = ["e3_cogent_smoke.csv", "e3_difficulty_gap_smoke.csv",
                    "e3_cogent_by_combo_smoke.csv", "e3_cogent_by_color_smoke.csv"]
        tables = paths["dirs"]["outputs"] / "tables"
        written = sorted(p.name for p in tables.glob("*.csv")) if tables.exists() else []
        check("fresh_standalone_e3_writes_all_tables",
              error is None and written == sorted(expected),
              dict(error=error, tables_created=tables.exists(), files=written))

        # --- 官方 SPAR：第三轮检查曾预建 tables，本次不预建 ---
        fake_torch, fake_tv = types.ModuleType("torch"), types.ModuleType("torchvision")
        fake_tv.transforms = types.SimpleNamespace()
        with patch.dict(sys.modules, {"torch": fake_torch, "torchvision": fake_tv}):
            official = importlib.import_module("src.experiments.e1_spar_official")
        row = dict(sample_id="fixture", character_name="cat", character2_name="dog", world_name="desert",
                   character1_pos="left", character2_pos="right", character_texture="Default",
                   character2_texture="Red")
        meta = pd.DataFrame([{**row, "row_id": i, "sample_id": f"s{i}", "character2_name": "dog",
                              "character_texture": tex, "world_name": world}
                             for i, (tex, world) in enumerate([("Default", "desert"), ("Blue", "arena")])])
        paths = fresh_paths(tmp, "spar", "pug_spar", "fixture")

        def extract_official(model, cache, key, *args, **kwargs):
            raw = np.ones((len(meta), 2), dtype=np.float32)
            cache.save(key, [f"pug_spar_official_{i:06d}" for i in range(len(meta))], raw, raw / np.sqrt(2))

        error = None
        try:
            with patch.object(official, "build_metadata", return_value=meta), \
                    patch.object(official, "extract_official_features", side_effect=extract_official), \
                    patch.object(official, "load_config", return_value={}):
                official.run(Encoder(), paths, logging.getLogger("fix_tables_spar"))
        except OSError as exc:
            error = str(exc)
        tables = paths["dirs"]["outputs"] / "tables"
        written = sorted(p.name for p in tables.glob("*.csv")) if tables.exists() else []
        check("fresh_standalone_spar_official_writes_tables",
              error is None and written == ["e1_spar_official.csv"],
              dict(error=error, tables_created=tables.exists(), files=written))

        # --- E2：同样不预建 tables 的独立首跑 ---
        from src.experiments import e2_change_diagnosis as e2
        cfg = project_io.load_config("e2_change")
        cfg["bootstrap"]["n_boot"] = 20
        rows = []
        rid = 0
        for animal in ["cat", "dog"]:
            for env in ["arena", "desert"]:
                for tex in ["Default", "Grass"]:
                    for size in ["small", "large"]:
                        for cam in [0, 45]:
                            rows.append(dict(row_id=rid, sample_id=f"pug_animals_{rid:06d}",
                                             animal=animal, environment=env, texture=tex,
                                             size=size, camera=cam))
                            rid += 1
        meta = pd.DataFrame(rows)
        paths = fresh_paths(tmp, "e2", "pug_animals", "fixture")

        def extract_pug(model, cache, key, parquet_dir, meta_columns, logger_,
                        only_row_ids=None, id_prefix="pug_animals"):
            ids = [f"pug_animals_{int(r):06d}" for r in sorted(only_row_ids if only_row_ids is not None
                                                               else meta["row_id"])]
            raw = np.stack([np.full(2, float(int(i.split('_')[-1]) % 3) + 1., dtype=np.float32) for i in ids])
            cache.save(key, ids, raw, raw / np.sqrt(2))

        def nuisance(kind, n=4):
            raw = np.tile(np.array([[1.5, 1.]], dtype=np.float32), (n, 1))
            out = dict(raw=raw, z=raw / np.sqrt(2))
            if kind == "visual":
                out["ids"] = [f"pug_animals_{i:06d}" for i in range(n)]
            return out

        error = None
        try:
            with patch.object(e2, "load_config", return_value=cfg), \
                    patch.object(e2, "build_animals_metadata", return_value=meta), \
                    patch.object(e2, "extract_pug_images", side_effect=extract_pug), \
                    patch.object(e2, "_nuisance_visual", side_effect=lambda *a, **k: nuisance("visual")), \
                    patch.object(e2, "_nuisance_text", side_effect=lambda *a, **k: nuisance("text")):
                e2.run(Encoder(), paths, logging.getLogger("fix_tables_e2"), smoke=True)
        except OSError as exc:
            error = str(exc)
        tables = paths["dirs"]["outputs"] / "tables"
        written = sorted(p.name for p in tables.glob("*.csv")) if tables.exists() else []
        check("fresh_standalone_e2_writes_tables",
              error is None and written == ["e2_delta_stats_smoke.csv", "e2_transition_stats_smoke.csv"],
              dict(error=error, tables_created=tables.exists(), files=written))

    revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    summary = dict(checks=len(results), passed=sum(r["passed"] for r in results),
                   failed=sum(not r["passed"] for r in results))
    (Path(__file__).parent / "fix_outputs_tables_results.json").write_text(
        json.dumps(dict(date="2026-10-21", revision=revision, summary=summary, results=results),
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(summary=summary)))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
