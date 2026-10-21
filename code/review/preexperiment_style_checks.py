"""Check behavioral compatibility of style commit 8ebdd3f against 5642dfb.

Only review artifacts are written; deleted experiment outputs stay deleted.
"""
import ast
import io
import json
import logging
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent


def main():
    source = Path(__file__).with_name("preexperiment_audit_round3_checks.py").read_text(encoding="utf-8")
    replacements = {
        "def capture(boots, point, ci):": "def capture(boots, point, ci, target=None):",
        "return old_ci(boots, point, ci)": "return old_ci(boots, point, ci, target=target)",
        "preexperiment_audit_round3_results.json": "preexperiment_style_results.json",
        'revision="4081130"': 'revision="8ebdd3f"',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(f"Historical regression interface changed: {before}")
        source = source.replace(before, after)
    namespace = {"__name__": "style_regressions", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, "style_regressions", "exec"), namespace)
    namespace["main"]()
    result_path = Path(__file__).with_name("preexperiment_style_results.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))

    def check(name, passed, evidence):
        row = dict(check=name, passed=bool(passed), evidence=evidence)
        result["results"].append(row)
        print(json.dumps(row, ensure_ascii=True))

    def previous(path):
        return subprocess.check_output(["git", "show", f"5642dfb:{path}"],
                                       cwd=ROOT, text=True, encoding="utf-8")

    core_files = ["code/src/experiments/e2_change_diagnosis.py", "code/src/metrics/bootstrap.py",
                  "code/src/metrics/delta_stats.py", "code/src/data/pair_matcher.py",
                  "code/src/experiments/e3_cogent_diagnosis.py", "code/src/experiments/common.py"]
    equality = {p: ast.dump(ast.parse(previous(p))) == ast.dump(ast.parse((ROOT / p).read_text(encoding="utf-8")))
                for p in core_files}
    check("core_scoring_and_statistics_ast_unchanged", all(equality.values()), equality)

    from src.features.cache import ImageFeatureCache
    import src.features.extract as current
    import src.data.pug_common as common
    legacy = types.ModuleType("src.features.style_previous")
    legacy.__package__ = "src.features"
    exec(compile(previous("code/src/features/extract.py"), "previous_extract", "exec"), legacy.__dict__)

    class Encoder:
        name = "style_fixture"
        bs_img = 2

        def encode_images(self, images):
            raw = np.array([[float(np.asarray(im).mean()), 1.] for im in images], dtype=np.float32)
            return dict(raw=raw, normalized=raw / np.linalg.norm(raw, axis=1, keepdims=True))

        def encode_texts(self, texts, ensure_period=True):
            eff = [t if not ensure_period or t.endswith(".") else t + "." for t in texts]
            raw = np.array([[float(len(t)), 1.] for t in eff], dtype=np.float32)
            return dict(raw=raw, normalized=raw / np.linalg.norm(raw, axis=1, keepdims=True))

    with tempfile.TemporaryDirectory(prefix=".style_review_", dir=ROOT) as tmpstr:
        tmp = Path(tmpstr).resolve()
        assert tmp.is_relative_to(ROOT.resolve())
        parquet_root = tmp / "parquet"
        (parquet_root / "data").mkdir(parents=True)
        rows = []
        for i in range(5):
            buf = io.BytesIO()
            Image.new("RGB", (4, 4), (i * 20, 30, 40)).save(buf, format="PNG")
            rows.append(dict(image=dict(bytes=buf.getvalue(), path=f"{i}.png"), label=str(i)))
        pq.write_table(pa.Table.from_pylist(rows), parquet_root / "data/shard.parquet")
        outputs, decode_counts = {}, {}
        for name, module in [("before", legacy), ("after", current)]:
            cache = ImageFeatureCache(tmp / name, "fixture")
            original_decode, count = common.decode_image, [0]

            def decode(cell):
                count[0] += 1
                return original_decode(cell)

            with patch.object(common, "decode_image", side_effect=decode):
                module.extract_pug_images(Encoder(), cache, "set", parquet_root, ["label"],
                                          logging.getLogger("style"), only_row_ids={1, 3}, id_prefix="pug")
                n1 = count[0]
                module.extract_pug_images(Encoder(), cache, "set", parquet_root, ["label"],
                                          logging.getLogger("style"), id_prefix="pug")
                n2 = count[0]
                module.extract_pug_images(Encoder(), cache, "set", parquet_root, ["label"],
                                          logging.getLogger("style"), id_prefix="pug")
                decode_counts[name] = dict(subset=n1, completion=n2 - n1, complete_cache=count[0] - n2)
            ids, _, raw, norm = cache.arrays("set")
            outputs[name] = (ids, raw, norm)
        same = all(np.array_equal(a, b) for a, b in zip(outputs["before"], outputs["after"]))
        check("pug_cache_output_and_order_unchanged", same and common.count_rows(parquet_root) == 5,
              dict(decode_counts=decode_counts, ids=outputs["after"][0].tolist()))

        from src.experiments import e3_cogent_diagnosis as e3
        from src.utils import io as project_io
        from src.utils.logging import get_logger
        cfg = project_io.load_config("e3_cogent")
        cfg["bootstrap"]["n_boot"] = 10
        cfg["sampling"]["per_scene_cap"] = 3
        cfg["sampling"]["target_per_condition_shape"] = 20
        paths_config = dict(dirs=dict(cache="cache", manifests="manifests", outputs="outputs"),
                            datasets=dict(cogent="fixture"))
        fresh = tmp / "fresh"
        with patch.object(project_io, "ROOT", fresh), patch.object(project_io, "load_config", return_value=paths_config):
            paths = project_io.load_paths()
        logger = get_logger("style_review_fresh_run", paths["dirs"]["outputs"] / "logs")
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
        check("fresh_standalone_e3_creates_its_result_directory", error is None,
              dict(error=error, outputs_root_created=paths["dirs"]["outputs"].exists(),
                   tables_created=(paths["dirs"]["outputs"] / "tables").exists()))

    summary = dict(checks=len(result["results"]), passed=sum(r["passed"] for r in result["results"]),
                   failed=sum(not r["passed"] for r in result["results"]))
    result["summary"] = summary
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(summary=summary)))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
