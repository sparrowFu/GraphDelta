"""Behavioral checks for configuration fidelity; no production files are changed.

Run with D:\\conda\\python.exe from any current directory. Tests patch model/cache
boundaries because this audit host has no torch or installed model checkpoints.
All experiment scratch outputs live in a temporary workspace and are removed.
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from src.experiments import e1_state_diagnosis as e1
from src.experiments import e3_cogent_diagnosis as e3
from src.utils.io import load_config


class Log:
    def info(self, *args):
        pass

    def warning(self, *args):
        pass


MODEL = SimpleNamespace(name="audit_stub", cache_protocol="audit_stub")


def check_cogent_paths(root):
    """Call actual E3.run and actual loader with existing custom scene files."""
    cfg = copy.deepcopy(load_config("e3_cogent"))
    default_conditions = copy.deepcopy(cfg["conditions"])
    for cond in ("A", "B"):
        cfg["conditions"][cond] = {
            "scene_file": f"custom/{cond}.json",
            "image_dir": f"custom_images/{cond}",
        }
    data = root / "cogent"
    (data / "custom").mkdir(parents=True)
    for cond in ("A", "B"):
        (data / "custom" / f"{cond}.json").write_text(
            json.dumps({"scenes": []}), encoding="utf-8")
        (data / "custom_images" / cond).mkdir(parents=True)
    paths = {"datasets": {"cogent": str(data)},
             "dirs": {"cache": root / "cache", "outputs": root / "outputs"}}
    result = {"configured_conditions": cfg["conditions"],
              "default_conditions": default_conditions,
              "custom_scene_exists": (data / "custom/A.json").exists()}
    with patch.object(e3, "load_config", return_value=cfg), \
         patch.object(e3, "ImageFeatureCache", return_value=object()), \
         patch.object(e3, "TextCache", return_value=object()), \
         patch.object(e3, "output_tables_dir", return_value=root):
        try:
            e3.run(MODEL, paths, Log())
        except FileNotFoundError as err:
            result["custom_scene_exception"] = type(err).__name__
            result["attempted_scene_relative"] = Path(err.filename).relative_to(data).as_posix()
        else:
            raise AssertionError("Custom scene path unexpectedly honored")

        # Give the hardcoded scene loader its expected file, then observe which
        # image directory the actual experiment passes into extraction.
        (data / "scenes").mkdir()
        (data / "scenes/CLEVR_valA_scenes.json").write_text(
            json.dumps({"scenes": []}), encoding="utf-8")

        class StopAtExtract(Exception):
            pass

        def capture(model, cache, key, scenes, image_dir, logger, **kwargs):
            result["actual_image_dir_relative"] = Path(image_dir).relative_to(data).as_posix()
            raise StopAtExtract()

        with patch.object(e3, "extract_clevr_images", side_effect=capture):
            try:
                e3.run(MODEL, paths, Log())
            except StopAtExtract:
                pass
    assert result["attempted_scene_relative"] == "scenes/CLEVR_valA_scenes.json"
    assert result["actual_image_dir_relative"] == "images/valA"
    result["default_matches_hardcoded_layout"] = (
        default_conditions["A"]["scene_file"] == result["attempted_scene_relative"]
        and default_conditions["A"]["image_dir"] == result["actual_image_dir_relative"])
    result["impact"] = "Custom conditions scene/image paths are ignored; official default layout is unaffected."
    return result


def spar_fixture():
    base = {"world_name": "w", "character_name": "cat", "character2_name": "dog",
            "character1_pos": "left", "character2_pos": "right",
            "character_texture": "Blue", "character2_texture": "Red"}
    return pd.DataFrame([
        {**base, "sample_id": "valid_two"},
        {**base, "character2_name": "blank", "sample_id": "single"},
        {**base, "character_name": "horse", "character2_name": "horse", "sample_id": "same_animal"},
    ])


def check_spar_switches(root):
    """Call actual E1.run twice and compare its real generated item/stat files.

    CLEVR loading, encoding/evaluation and metadata I/O are isolated; SPAR
    item generation and config propagation execute production code unchanged.
    """
    cfg = copy.deepcopy(load_config("e1_state"))
    cfg["clevr"]["factors"] = []  # isolate the SPAR path
    meta = spar_fixture()
    paths = {"datasets": {"clevr": "unused", "pug_spar": "unused"},
             "dirs": {"cache": root / "cache", "outputs": root / "outputs"}}

    def fake_eval(model, tcache, cache, key, items, **kwargs):
        out = {field: 0.0 for field in e1.EXPORT_FIELDS}
        out.update(n_samples=len(items), item_acc=np.ones(len(items)),
                   item_margin=np.ones(len(items)))
        return out

    runs = []
    for label, same_filter, two_only in [
        ("defaults", True, ["position", "attribute", "relation"]),
        ("changed", False, []),
    ]:
        variant = copy.deepcopy(cfg)
        variant["spar"]["same_animal_filter"] = same_filter
        variant["spar"]["two_object_only"] = two_only
        out = root / label
        out.mkdir()
        with patch.object(e1, "load_config", return_value=variant), \
             patch.object(e1, "ImageFeatureCache", return_value=object()), \
             patch.object(e1, "TextCache", return_value=object()), \
             patch.object(e1, "output_tables_dir", return_value=out), \
             patch.object(e1.clevr, "load_scenes", return_value=[]), \
             patch.object(e1, "extract_clevr_images"), \
             patch.object(e1, "extract_pug_images"), \
             patch.object(e1, "build_spar_metadata", return_value=meta), \
             patch.object(e1, "eval_2afc_items", side_effect=fake_eval), \
             patch.object(e1, "cluster_gap_ci", return_value=(0.0, 0.0, 0.0)):
            e1.run(MODEL, paths, Log())
        runs.append({
            "config": {"same_animal_filter": same_filter, "two_object_only": two_only},
            "items": [json.loads(line) for line in (out / "e1_items_spar.jsonl").read_text(encoding="utf-8").splitlines()],
            "filter_stats": json.loads((out / "e1_spar_filter_stats.json").read_text(encoding="utf-8")),
        })
    same_items = runs[0]["items"] == runs[1]["items"]
    same_stats = runs[0]["filter_stats"] == runs[1]["filter_stats"]
    assert same_items and same_stats
    assert runs[1]["filter_stats"]["same_animal_filtered"] == 1
    assert all(it["image_id"] != "single" or it["factor"] == "scene"
               for it in runs[1]["items"])
    return {"runs": runs, "same_items": same_items, "same_filter_stats": same_stats,
            "default_matches_hardcoded_filters": True,
            "impact": "Two exposed filter settings are inert; default protocol remains in effect."
                      " Do not simply disable the validity filters: remove/fix obsolete settings or reject unsupported alternatives."}


def main():
    with tempfile.TemporaryDirectory(prefix="config_audit_", dir=CODE.parents[0]) as scratch:
        root = Path(scratch)
        results = {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE,
                                         capture_output=True, text=True, check=True).stdout.strip(),
            "scope": "Configuration flow only; no actual pretrained-model forward pass.",
            "e3_custom_paths_ignored": check_cogent_paths(root),
            "e1_filter_switches_inert": check_spar_switches(root),
        }
    target = Path(__file__).with_name("config_supplement_results.json")
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(target), "checks": 2,
                      "e3_default_layout_unaffected": results["e3_custom_paths_ignored"]["default_matches_hardcoded_layout"],
                      "e1_default_filtering_unaffected": results["e1_filter_switches_inert"]["default_matches_hardcoded_filters"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
