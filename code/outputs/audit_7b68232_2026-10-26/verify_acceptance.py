"""Offline repair verification. Reproduced defects are recorded, not called passes.

Uses only temporary fixtures and the previously saved public metadata sample.
Does not load weights, download images, modify experiment code, or clear real caches.
"""
import ast
import copy
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from src.data.pug_imagenet import META_COLUMNS, META_COLS, build_metadata
from src.data.pair_matcher import build_factor_pairs
from src.experiments import e2_change_diagnosis as e2
from src.features import extract as extraction
from src.features.cache import ImageFeatureCache
from src.utils import io as utils_io


class Log:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass


class FakeEncoder:
    name = "audit_encoder"
    cache_protocol = "audit-v1"
    bs_img = 5
    def __init__(self): self.image_batches = []
    def pack(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        return {"raw": raw, "normalized": raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)}
    def encode_images(self, images):
        self.image_batches.append(len(images))
        return self.pack([[1., np.asarray(im).mean() / 255, np.asarray(im).std() / 255, im.width / 32] for im in images])
    def encode_texts(self, texts, ensure_period=True):
        return self.pack([[1., len(t) / 100, sum(t.encode()) % 256 / 256, .5] for t in texts])


def script(name):
    spec = importlib.util.spec_from_file_location("accept_" + name, CODE / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(asset="asset_A", label="class_A", env="forest", tex="Default", size=1., cam=0):
    return dict(character_name=asset, character_label=label, world_name=env,
                character_texture=tex, character_scale=size, camera_yaw=cam,
                camera_pitch=0, camera_roll=0, scene_light="Default",
                character_rotation_yaw=0, character_rotation_roll=0, character_rotation_pitch=0)


def parquet(root, rows, images=False):
    records = [{k: r[k] for k in META_COLUMNS} for r in rows]
    if images:
        for i, record in enumerate(records):
            buffer = io.BytesIO()
            Image.new("RGB", (24, 24), (17 * i % 256, 43 * i % 256, 71 * i % 256)).save(buffer, format="PNG")
            record["image"] = {"bytes": buffer.getvalue(), "path": None}
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records), root / "data" / "sample.parquet")
    return build_metadata(root)


def invoke(module, argv):
    previous = sys.argv
    sys.argv = [module.__name__] + argv
    try: return module.main()
    finally: sys.argv = previous


def summary_tables(root):
    for seed in range(3):
        t = root / "clip_vitl14" / f"seed_{seed}" / "tables"
        t.mkdir(parents=True)
        (t / "run_meta.json").write_text(json.dumps({"model": "clip_vitl14", "seed": seed}))
        pd.DataFrame([dict(dataset="clevr", factor="color", accuracy=.7, acc_ci_low=.6, acc_ci_high=.8)]).to_csv(t / "e1_results.csv", index=False)
        pd.DataFrame([dict(dataset="clevr", factor_a="color", factor_b="object", gap=.2, gap_ci_low=.1, gap_ci_high=.3)]).to_csv(t / "e1_factor_gaps.csv", index=False)
        official = [dict(task="two_objects_relation", world=w, add_background=False, texture="Default", pos_pair="left/right", hardneg=False, n_images=20, top1=value, top5=.8) for w, value in [("all", .5), ("desert", [.1, .9, .1][seed])]]
        official.append(dict(task="one_object_position", world="all", add_background=False, texture="Default", pos_pair="left/right", hardneg=True, n_images=20, top1=.5, top5=np.nan))
        pd.DataFrame(official).to_csv(t / "e1_spar_official.csv", index=False)
        pd.DataFrame([dict(factor="size", norm_gap_visual_raw=.2, norm_gap_visual_raw_ci_low=.1, norm_gap_visual_raw_ci_high=.3, transition_similarity_gap=.3, gap_ci_low=.2, gap_ci_high=.4)]).to_csv(t / "e2_delta_stats.csv", index=False)
        pd.DataFrame([dict(shape_group=g, difficulty_gap=.1, gap_ci_low=.05, gap_ci_high=.15) for g in ["cube_cylinder", "sphere"]]).to_csv(t / "e3_difficulty_gap.csv", index=False)
        pd.DataFrame([dict(negative_type="pooled_equal_type", shape_group="sphere", color="red", gap=[1e-5, -1e-5, 1e-5][seed])]).to_csv(t / "e3_unit_gaps.csv", index=False)


def main():
    evidence = {"revision": "7b68232", "checks": {}}
    checks = evidence["checks"]
    files = list((CODE / "src").rglob("*.py")) + list((CODE / "scripts").rglob("*.py"))
    for path in files: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    evidence["source_sha256"] = {p.relative_to(CODE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checks["syntax"] = {"production_files": len(files)}
    sample_path = CODE / "outputs" / "audit_2026-10-26" / "imagenet_metadata_sample.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="audit_accept_", dir=CODE.parent) as temp:
        root = Path(temp)
        meta = parquet(root / "real_sample", sample["rows"])
        pairs = build_factor_pairs(meta, "camera", None, 4, 30000, 0, meta_cols=META_COLS)
        assets = dict(zip(meta.row_id, meta.character_name))
        endpoints = set(pairs.row_id_a) | set(pairs.row_id_b)
        checks["B1_real_sample"] = {"pairs": len(pairs), "contexts": int(pairs.context_key.nunique()), "assets": len({assets[i] for i in endpoints}), "cross_asset": int(sum(assets[r.row_id_a] != assets[r.row_id_b] for r in pairs.itertuples())), "sample_file_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest()}
        assert checks["B1_real_sample"]["assets"] == 18
        try: build_factor_pairs(pd.concat([meta, meta.iloc[[0]]], ignore_index=True), "camera", None, 4, 30000, 0, meta_cols=META_COLS)
        except ValueError: checks["B1_duplicate_guard"] = "rejected"
        else: raise AssertionError("duplicate metadata accepted")

        reversed_rows = [row(f"hash{i:02d}", f"class{11-i:02d}", env, tex, size, cam) for i in range(12) for env, tex, size, cam in itertools.product(["e0", "e1"], ["t0", "t1"], [1., 2.], [0, 90])]
        meta = parquet(root / "reverse", reversed_rows)
        full_counts = {factor: len(build_factor_pairs(meta, factor, None, 4, 30000, 0, meta_cols=META_COLS)) for factor in ["environment", "texture", "size", "camera"]}
        assert all(full_counts.values())
        m0, m1 = META_COLS[:2]
        selected = meta[meta[m0].astype(str).isin(sorted(meta[m0].astype(str).unique())[:6]) & meta[m1].astype(str).isin(sorted(meta[m1].astype(str).unique())[:6])]
        paths = {"datasets": {"pug_imagenet": root / "reverse"}, "dirs": {"cache": root / "reverse_cache", "outputs": root / "reverse_outputs"}}
        try: e2.run(FakeEncoder(), paths, Log(), smoke=True, dataset="pug_imagenet")
        except SystemExit: error = "SystemExit: no pairs after smoke selection"
        else: error = "unexpected success"
        checks["new_smoke_selection_regression"] = {"input_rows": len(meta), "full_factor_pairs": full_counts, "smoke_rows": len(selected), "actual_run": error}
        try: e2.run(FakeEncoder(), paths, Log(), smoke=True, dataset="pug_imagenet", allow_empty_factors=True)
        except ValueError as error: checks["all_factors_skipped"] = str(error)

        partial_rows = [row(env=env, tex=tex, cam=cam) for env, tex, cam in itertools.product(["e0", "e1"], ["t0", "t1"], [0, 90])]
        parquet(root / "partial", partial_rows, images=True)
        paths = {"datasets": {"pug_imagenet": root / "partial"}, "dirs": {"cache": root / "partial_cache", "outputs": root / "partial_outputs"}}
        original_config = e2.load_config
        config = copy.deepcopy(original_config("e2_change"))
        config["bootstrap"]["n_boot"] = 8
        config["similarity"]["max_comparisons_per_group"] = 20
        e2.load_config = lambda name: config
        try:
            try: e2.run(FakeEncoder(), paths, Log(), dataset="pug_imagenet")
            except SystemExit: checks["B2_strict_empty_factor"] = "explicit stop"
            encoder = FakeEncoder()
            result = e2.run(encoder, paths, Log(), dataset="pug_imagenet", allow_empty_factors=True)
            checks["B2_allow_partial"] = {"returned_factors": result.factor.tolist(), "image_encoder_batches": encoder.image_batches, "synthetic_encoder_only": True}
        finally: e2.load_config = original_config

        nuisance_cfg = copy.deepcopy(original_config("e2_change"))
        nuisance_cfg["nuisance"]["n_images"] = 200
        nuisance_meta = pd.DataFrame({"row_id": range(200), "sample_id": [f"nuisance_{i}" for i in range(200)]})
        original_iterator = e2.iter_image_batches
        images = [Image.new("RGB", (24, 24), (i % 256, 3 * i % 256, 7 * i % 256)) for i in range(200)]
        def nuisance_iterator(*args, **kwargs):
            for begin in range(0, 200, 48):
                end = min(begin + 48, 200)
                yield list(range(begin, end)), images[begin:end], [{}] * (end - begin)
        e2.iter_image_batches = nuisance_iterator
        encoder = FakeEncoder()
        encoder.bs_img = 37
        try:
            batched = e2._nuisance_visual(encoder, {"datasets": {"pug_imagenet": "unused"}}, nuisance_meta, nuisance_cfg, Log(), False, 0, e2.DATASETS["pug_imagenet"])
        finally: e2.iter_image_batches = original_iterator
        from src.features.delta import augment_view
        scales = nuisance_cfg["nuisance"]["crop_scale"]
        qualities = nuisance_cfg["nuisance"]["jpeg_quality"]
        reference_model = FakeEncoder()
        a = reference_model.encode_images([augment_view(im, 100000 + i, *scales, qualities[0]) for i, im in enumerate(images)])
        b = reference_model.encode_images([augment_view(im, 200000 + i, *scales, qualities[1]) for i, im in enumerate(images)])
        checks["R2_nuisance_batching"] = {"max_batch": max(encoder.image_batches), "images": len(batched["ids"]), "ids_in_row_order": batched["ids"] == nuisance_meta.sample_id.tolist(), "raw_matches_fake_reference": bool(np.array_equal(batched["raw"], b["raw"] - a["raw"])), "normalized_matches_fake_reference": bool(np.array_equal(batched["z"], b["normalized"] - a["normalized"])), "real_weights_not_tested": True}

        summary = script("summarize_seed_stability")
        summary_root = root / "summary"
        summary_tables(summary_root)
        summary.load_paths = lambda: {"dirs": {"outputs": summary_root}}
        summary.get_logger = lambda *args, **kwargs: Log()
        result = invoke(summary, [])
        spar = result[result.section == "spar_official"]
        desert = spar[(spar.metric == "top1") & spar.key.str.contains("w=desert")].iloc[0]
        checks["B3_B4_B8_summary"] = {"SPAR_all_stable_NA": bool(spar.stable.isna().all()), "undefined_hardneg_top5_rows": int(((spar.metric == "top5") & spar.key.str.startswith("one_object_position")).sum()), "tiny_E3_gap_stable": bool(result[result.section == "e3_unit"].iloc[0].stable), "desert_seeds": [desert.seed0, desert.seed1, desert.seed2], "desert_note": desert.note}

        analyze = script("analyze_e3_units")
        tables = root / "analysis"
        tables.mkdir()
        items = [dict(image_id=f"{condition}_{group}", condition=condition, shape_group=group, negative_type=kind, color="red", shape="sphere" if group == "sphere" else "cube", item_acc=float(condition == "A"), item_margin=0.) for condition in ["A", "B"] for group in ["cube_cylinder", "sphere"] for kind in ["color", "shape"]]
        (tables / "e3_items_cogent.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        def reference(gap): pd.DataFrame([dict(shape_group=g, difficulty_gap=gap) for g in ["cube_cylinder", "sphere"]]).to_csv(tables / "e3_difficulty_gap.csv", index=False)
        analyze.load_paths = lambda: {"dirs": {"outputs": root / "analysis_out"}}
        analyze.load_config = lambda name: {"bootstrap": {"n_boot": 8, "ci": .95}, "sampling": {"negative_types": ["color", "shape"]}, "colors": ["red"]}
        analyze.get_logger = lambda *args, **kwargs: Log()
        reference(1.)
        invoke(analyze, ["--tables-dir", str(tables)])
        old_unit = (tables / "e3_unit_gaps.csv").read_bytes()
        reference(0.)
        try: invoke(analyze, ["--tables-dir", str(tables)])
        except SystemExit: stopped = True
        else: stopped = False
        sidecar = json.loads((tables / "e3_unit_consistency.json").read_text())
        checks["B6_strict_rerun"] = {"stopped": stopped, "all_ok": sidecar["all_ok"], "old_unit_unchanged": (tables / "e3_unit_gaps.csv").read_bytes() == old_unit}
        (tables / "e3_difficulty_gap.csv").unlink()
        invoke(analyze, ["--tables-dir", str(tables)])
        sidecar = json.loads((tables / "e3_unit_consistency.json").read_text())
        checks["B6_missing_reference"] = {"all_ok": sidecar["all_ok"], "checks": sidecar["checks"]}

        function = next(n for n in ast.parse((CODE / "scripts/extract_embeddings.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "_force_reset")
        namespace = {"Path": Path, "json": json, "shutil": shutil}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "current_force_reset", "exec"), namespace)
        protocols = {}
        for protocol in ['{"protocol":', '[]', 'null']:
            cache = root / "force" / FakeEncoder.name
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "protocol.json").write_text(protocol)
            (cache / "old_images.npz").write_bytes(b"fixture")
            assert cache.resolve().is_relative_to(root.resolve())
            try: namespace["_force_reset"](cache.parent, FakeEncoder(), "clevr_val", Log())
            except Exception as error: protocols[protocol] = type(error).__name__
            else: protocols[protocol] = "rebuilt"
        checks["B7_force_JSON_boundaries"] = protocols

        identity = utils_io._code_identity
        try:
            utils_io._code_identity = lambda: {"git_commit": "A", "git_dirty": False}
            paths = {"dirs": {"outputs": root / "identity"}}
            table = utils_io.output_tables_dir(paths, "clip_vitl14", 0, extra={"cache_protocol": "v1"})
            utils_io._code_identity = lambda: {"git_commit": "B", "git_dirty": False}
            utils_io.output_tables_dir(paths, "clip_vitl14", 0, extra={"cache_protocol": "v2"})
            meta = json.loads((table / "run_meta.json").read_text())
            checks["R1_rerun_metadata"] = {"requested_second": {"git_commit": "B", "cache_protocol": "v2"}, "stored_second": meta}
        finally: utils_io._code_identity = identity

        cache = ImageFeatureCache(root / "extract_cache", FakeEncoder.name, protocol=FakeEncoder.cache_protocol)
        cache.save("fixture", ["d_000000"], np.ones((1, 4)), np.ones((1, 4)))
        iterator = extraction.iter_image_batches
        calls = []
        def fail_iterator(*args, **kwargs):
            calls.append(kwargs["only_row_ids"])
            raise AssertionError("cached image should not be iterated")
        extraction.iter_image_batches = fail_iterator
        try: extraction.extract_pug_images(FakeEncoder(), cache, "fixture", "unused", [], Log(), only_row_ids={0}, id_prefix="d")
        finally: extraction.iter_image_batches = iterator
        checks["B9_full_cache_hit"] = {"iterator_calls": len(calls)}

    evidence["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(__file__).with_name("acceptance_checks.json")
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"result": str(output), "evidence_categories": len(checks), "defect_reproductions_are_not_fix_passes": True}))


if __name__ == "__main__": main()
