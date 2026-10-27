"""Independent configuration/provenance recheck; production sources stay untouched.

Run: D:\\conda\\python.exe -B <this-file>
Uses deterministic encoders, tiny PNG/scene fixtures and temporary cache/output
directories.  extract_embeddings main/_force_reset are executed unchanged from
their AST function nodes, injecting only model construction/config/log boundaries
to avoid importing unavailable torch/transformers.
"""
import ast
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True
from src.data.cogent import condition_cache_key
from src.experiments import e1_spar_official as official
from src.experiments import e1_state_diagnosis as e1
from src.experiments import e3_cogent_diagnosis as e3
from src.features.cache import ImageFeatureCache
from src.features.extract import extract_clevr_images, extract_pug_images
from src.utils.io import load_config


class Log:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass


class Encoder:
    name = "audit_config_encoder"
    cache_protocol = "audit-config-v1"
    bs_img = 8
    def __init__(self): self.image_calls = 0
    def pack(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        return {"raw": raw, "normalized": raw / np.linalg.norm(raw, axis=1, keepdims=True)}
    def encode_images(self, images):
        self.image_calls += 1
        return self.pack([[1., *list(np.asarray(im).mean(axis=(0, 1)) / 255)] for im in images])
    def encode_texts(self, texts, ensure_period=True):
        return self.pack([[1., len(t) / 100, sum(t.encode()) % 100 / 100, .5] for t in texts])


def fixtures(root, cfg):
    for cond in ("A", "B"):
        ccfg = cfg["conditions"][cond]
        scene_path = root / ccfg["scene_file"]
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        image_dir = root / ccfg["image_dir"]
        image_dir.mkdir(parents=True, exist_ok=True)
        scenes = [{"image_filename": f"{cond}.png", "objects": [
            {"color": "gray" if cond == "A" else "red", "shape": "cube"},
            {"color": "blue", "shape": "sphere"}]}]
        scene_path.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
        Image.new("RGB", (8, 8), (90, 20, 30)).save(image_dir / f"{cond}.png")


def make_extract_functions():
    import argparse
    import shutil
    tree = ast.parse((CODE / "scripts/extract_embeddings.py").read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("_force_reset", "main")]
    ns = dict(Path=Path, argparse=argparse, json=json, shutil=shutil,
              ImageFeatureCache=ImageFeatureCache, condition_cache_key=condition_cache_key,
              condition_image_dir=e3.condition_image_dir,
              load_condition_scenes=e3.load_condition_scenes,
              extract_clevr_images=extract_clevr_images, extract_pug_images=extract_pug_images)
    exec(compile(ast.Module(body=body, type_ignores=[]), "production_extract_functions", "exec"), ns)
    return ns


def main():
    evidence = {"scope": "Synthetic PNG/scene data; deterministic encoder; no model checkpoints.", "checks": {}}
    checks = evidence["checks"]
    cfg = copy.deepcopy(load_config("e3_cogent"))
    cfg["bootstrap"]["n_boot"] = 6
    cfg["sampling"]["target_per_condition_shape"] = 12
    for cond in ("A", "B"):
        cfg["conditions"][cond] = {"scene_file": f"custom_scenes/{cond}.json",
                                     "image_dir": f"custom_images/{cond}"}
    with tempfile.TemporaryDirectory(prefix="fixtures_", dir=Path(__file__).parent) as td:
        root = Path(td)
        fixtures(root / "data", cfg)
        model = Encoder()
        paths = {"datasets": {"cogent": str(root / "data")},
                 "dirs": {"cache": root / "cache", "outputs": root / "outputs"}}
        with patch.object(e3, "load_config", return_value=cfg):
            e3.run(model, paths, Log(), smoke=False)
            e3.run(model, paths, Log(), smoke=True)
        table_dir = paths["dirs"]["outputs"] / model.name / "seed_0/tables"
        meta = json.loads((table_dir / "run_meta.json").read_text())
        keys = sorted(meta["experiments"])
        assert keys == ["e3_cogent", "e3_cogent_smoke"]
        assert all(v["status"] == "success" for v in meta["experiments"].values())
        assert not (root / "data/scenes/CLEVR_valA_scenes.json").exists()
        checks["A_custom_paths_and_e3_smoke_keys"] = {
            "completed_full_and_smoke": True, "experiments": keys,
            "default_scene_absent": True, "image_encoder_calls": model.image_calls,
            "full_outputs": meta["experiments"]["e3_cogent"]["last_success"]["outputs"],
            "smoke_outputs": meta["experiments"]["e3_cogent_smoke"]["last_success"]["outputs"]}

        # Capture actual E1/SPAR entry calls before independent data loading.
        class StopAtOutput(Exception): pass
        entry_keys = {}
        for name, module in [("e1_state", e1), ("e1_spar_official", official)]:
            captured = []
            def capture(*args, **kwargs):
                captured.append(kwargs["experiment"])
                raise StopAtOutput()
            for smoke in (False, True):
                with patch.object(module, "ImageFeatureCache", return_value=object()), \
                     patch.object(module, "TextCache", return_value=object()), \
                     patch.object(module, "output_tables_dir", side_effect=capture):
                    try: module.run(model, paths, Log(), smoke=smoke)
                    except StopAtOutput: pass
            assert captured == [name, name + "_smoke"]
            entry_keys[name] = captured
        checks["problem2_E1_entry_keys"] = entry_keys

        default_sp = load_config("e1_state")["spar"]
        e1._validate_spar_protocol(default_sp)
        rejected = {}
        for field, value in [("same_animal_filter", False), ("two_object_only", [])]:
            variant = {**default_sp, field: value}
            try: e1._validate_spar_protocol(variant)
            except SystemExit as err: rejected[field] = str(err)
            else: raise AssertionError(f"unsupported {field} accepted")
        checks["B_fixed_protocol_rejection"] = {"default_accepted": True, "rejected": rejected}

        default = load_config("e3_cogent")["conditions"]["A"]
        default_key = condition_cache_key("cogent_valA", default["scene_file"], default["image_dir"])
        assert default_key != "cogent_valA"
        checks["A_default_cache_compatibility_regression"] = {
            "default_yaml": default, "old_key": "cogent_valA", "current_key": default_key,
            "report_claim_default_key_unchanged": False}

        # Execute real force+extract branch with a matching protocol. A fresh
        # in-memory cache ensures the stale hit is from the on-disk NPZ.
        functions = make_extract_functions()
        force_paths = {"datasets": {"cogent": str(root / "force_data")},
                       "dirs": {"cache": root / "force_cache", "outputs": root / "force_out"}}
        fixtures(root / "force_data", cfg)
        force_model = Encoder()
        functions.update(load_paths=lambda: force_paths,
                         get_logger=lambda *args: Log(),
                         build_encoder=lambda *args: force_model,
                         load_config=lambda name: cfg)
        def invoke(force=False):
            argv = ["extract_embeddings.py", "--dataset", "cogent_valA", "--model", force_model.name]
            if force: argv.append("--force")
            with patch.object(sys, "argv", argv): functions["main"]()
        invoke()
        custom = cfg["conditions"]["A"]
        key = condition_cache_key("cogent_valA", custom["scene_file"], custom["image_dir"])
        cache_path = root / "force_cache" / force_model.name / (key + "_images.npz")
        before = cache_path.read_bytes()
        calls_before = force_model.image_calls
        Image.new("RGB", (8, 8), (5, 250, 5)).save(root / "force_data" / custom["image_dir"] / "A.png")
        invoke(force=True)
        assert force_model.image_calls == calls_before
        assert cache_path.read_bytes() == before
        checks["A_force_fails_to_reextract"] = {
            "actual_cached_key": key, "force_deletes_key": "cogent_valA",
            "image_changed_on_disk": True,
            "extra_image_encoder_calls_after_force": force_model.image_calls - calls_before,
            "cached_npz_bytes_unchanged": cache_path.read_bytes() == before}

        # Root directory is omitted from the new fingerprint. Repointing only
        # paths.datasets.cogent to a second root with equal IDs reuses old data.
        fixtures(root / "second_data", cfg)
        Image.new("RGB", (8, 8), (0, 0, 255)).save(root / "second_data" / custom["image_dir"] / "A.png")
        force_paths["datasets"]["cogent"] = str(root / "second_data")
        invoke()
        assert force_model.image_calls == calls_before
        checks["A_dataset_root_not_in_cache_identity"] = {
            "changed_dataset_root_only": True, "same_relative_path_and_image_id": True,
            "extra_image_encoder_calls": force_model.image_calls - calls_before,
            "cached_npz_bytes_unchanged": cache_path.read_bytes() == before}
    evidence["source_sha256"] = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [CODE / "src/data/cogent.py", CODE / "scripts/extract_embeddings.py",
                                            CODE / "src/experiments/e1_state_diagnosis.py",
                                            CODE / "src/experiments/e1_spar_official.py",
                                            CODE / "src/experiments/e3_cogent_diagnosis.py"]}
    target = Path(__file__).with_name("config_results.json")
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(target), "checks": len(checks), "remaining_defects_reproduced": 3}))


if __name__ == "__main__": main()
