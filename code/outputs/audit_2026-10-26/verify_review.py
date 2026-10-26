"""Offline audit reproductions; exit 0 means checks ran, not that defects are fixed.

Run with a Python environment containing numpy/pandas/pyarrow/Pillow/PyYAML.
Only temporary fixtures and audit_checks.json are written. No model or image download.
"""
import ast
import hashlib
import importlib.util
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

from src.data.pair_matcher import build_factor_pairs
from src.data.pug_imagenet import META_COLS, META_COLUMNS, build_metadata
from src.experiments import e2_change_diagnosis as e2
from src.features.cache import ImageFeatureCache
from src.features.delta import augment_view
from src.metrics.delta_stats import build_delta_frame, group_similarities
from src.utils.io import load_config, output_tables_dir, verify_tables_identity


class Log:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class Encoder:
    name = "audit_encoder"
    cache_protocol = "audit-v1"


def script(name):
    spec = importlib.util.spec_from_file_location("audit_" + name, CODE / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_row(asset, camera):
    return dict(world_name="SaltFlats", character_name=asset,
                character_label="African_crocodile", character_scale=1.,
                camera_yaw=camera, camera_pitch=240, camera_roll=0,
                character_texture="Default", scene_light="Default",
                character_rotation_yaw=0, character_rotation_roll=0,
                character_rotation_pitch=0)


def metadata_fixture(root, rows):
    data = root / "data"
    data.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in META_COLUMNS} for r in rows]), data / "train.parquet")
    return build_metadata(root)


def main():
    evidence = {"head_reviewed": "3cc6dae", "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__}, "checks": {}}
    checks = evidence["checks"]
    files = list((CODE / "src").rglob("*.py")) + list((CODE / "scripts").rglob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    evidence["production_source_sha256"] = {path.relative_to(CODE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    checks["syntax"] = {"production_python_files": len(files), "passed": True}
    summary = script("summarize_seed_stability")
    checks["nan_ci_repair"] = {
        "classification": summary.ci_sign(np.nan, np.nan),
        "all_missing_stable": summary.stable_classes(["invalid"] * 3)}
    assert checks["nan_ci_repair"] == {"classification": "invalid", "all_missing_stable": False}
    image = augment_view(Image.new("RGB", (1000, 1000)), 0, .85, .850000001, 90)
    areas = []
    for seed in range(200):
        crop = augment_view(Image.new("RGB", (317, 199)), seed, .85, 1., 90)
        areas.append(crop.width * crop.height / (317 * 199))
    assert min(areas) >= .85 and max(areas) <= 1.
    checks["crop_repair"] = {"fixed_size": list(image.size), "random_cases": len(areas), "minimum_area": min(areas)}

    with tempfile.TemporaryDirectory(prefix="audit_review_", dir=CODE.parent) as temporary:
        root = Path(temporary)
        cache_dir = root / "cache" / Encoder.name
        cache_dir.mkdir(parents=True)
        np.savez(cache_dir / "old_images.npz", raw=np.ones((1, 2)))
        try:
            ImageFeatureCache(root / "cache", Encoder.name, protocol=Encoder.cache_protocol)
        except RuntimeError:
            checks["unversioned_cache_repair"] = {"rejected": True, "marker_created": (cache_dir / "protocol.json").exists()}
        else:
            raise AssertionError("unversioned cache was accepted")

        force_node = next(n for n in ast.parse((CODE / "scripts" / "extract_embeddings.py").read_text(encoding="utf-8")).body
                          if isinstance(n, ast.FunctionDef) and n.name == "_force_reset")
        namespace = {"Path": Path, "json": json, "shutil": shutil}
        exec(compile(ast.Module(body=[force_node], type_ignores=[]), "actual_force_reset", "exec"), namespace)
        assert cache_dir.resolve().is_relative_to(root.resolve())
        (cache_dir / "protocol.json").write_text('{"protocol":', encoding="utf-8")
        try:
            namespace["_force_reset"](root / "cache", Encoder(), "clevr_val", Log())
        except json.JSONDecodeError:
            checks["corrupt_protocol_force"] = {"error": "JSONDecodeError", "old_file_remains": (cache_dir / "old_images.npz").exists()}
        else:
            raise AssertionError("expected the documented corrupt-protocol boundary")

        known = root / "identity" / "tables"
        known.mkdir(parents=True)
        try:
            verify_tables_identity(known, known.parent, "siglip2_so400m", 1)
        except SystemExit:
            checks["identity_repair"] = {"wrong_legacy_owner_rejected": True}
        else:
            raise AssertionError("wrong legacy owner was accepted")
        verify_tables_identity(known, known.parent, "clip_vitl14", 0)

        rows = [metadata_row(a, c) for a in ("asset_A", "asset_B") for c in (0, 90)]
        meta = metadata_fixture(root / "mesh_grid", rows)
        pairs = build_factor_pairs(meta, "camera", None, 4, 30000, 0, meta_cols=META_COLS)
        checks["mesh_grid_collapse"] = {"input_assets": 2, "expected_instance_pairs": 2,
                                        "actual_pairs": len(pairs), "character_name_retained": "character_name" in meta.columns}
        sparse_rows = [metadata_row("asset_A", 0), metadata_row("asset_B", 90)]
        sparse = root / "sparse"
        meta = metadata_fixture(sparse, sparse_rows)
        pairs = build_factor_pairs(meta, "camera", None, 4, 30000, 0, meta_cols=META_COLS)
        checks["synthetic_cross_asset"] = {"pairs": len(pairs), "assets": [sparse_rows[int(pairs.iloc[0][k])]["character_name"] for k in ("row_id_a", "row_id_b")]}
        try:
            e2.run(Encoder(), {"datasets": {"pug_imagenet": sparse}, "dirs": {"cache": root / "empty_cache", "outputs": root / "empty_outputs"}}, Log(), smoke=True, dataset="pug_imagenet")
        except KeyError as error:
            checks["sparse_input"] = {"error": repr(error), "synthetic_subset_only": True}
        else:
            raise AssertionError("expected the documented empty-factor failure")

        outroot = root / "summary"
        for seed in (0, 1, 2):
            table = output_tables_dir({"dirs": {"outputs": outroot}}, "clip_vitl14", seed)
            checks["run_meta_fields"] = sorted(json.loads((table / "run_meta.json").read_text()))
            pd.DataFrame([dict(dataset="clevr", factor="color", accuracy=.7, acc_ci_low=.6, acc_ci_high=.8)]).to_csv(table / "e1_results.csv", index=False)
            pd.DataFrame([dict(dataset="clevr", factor_a="color", factor_b="object", gap=.2, gap_ci_low=.1, gap_ci_high=.3)]).to_csv(table / "e1_factor_gaps.csv", index=False)
            official = [dict(task="two_objects_relation", world=w, add_background=False, texture="Default", pos_pair="left/right", hardneg=False, n_images=20, top1=v, top5=.8)
                        for w, v in [("all", .5), ("desert", [.1, .9, .1][seed])]]
            official.append(dict(task="one_object_position", world="all", add_background=False, texture="Default", pos_pair="left/right", hardneg=True, n_images=20, top1=.5, top5=np.nan))
            pd.DataFrame(official).to_csv(table / "e1_spar_official.csv", index=False)
            pd.DataFrame([dict(factor="size", norm_gap_visual_raw=.2, norm_gap_visual_raw_ci_low=.1, norm_gap_visual_raw_ci_high=.3, transition_similarity_gap=.3, gap_ci_low=.2, gap_ci_high=.4)]).to_csv(table / "e2_delta_stats.csv", index=False)
            pd.DataFrame([dict(shape_group=g, difficulty_gap=.1, gap_ci_low=.05, gap_ci_high=.15) for g in ("cube_cylinder", "sphere")]).to_csv(table / "e3_difficulty_gap.csv", index=False)
            pd.DataFrame([dict(negative_type="pooled_equal_type", shape_group="sphere", color="red", gap=[1e-5, -1e-5, 1e-5][seed])]).to_csv(table / "e3_unit_gaps.csv", index=False)
        summary.load_paths = lambda: {"dirs": {"outputs": outroot}}
        summary.get_logger = lambda *args, **kwargs: Log()
        previous_argv = sys.argv
        sys.argv = ["summarize_seed_stability.py"]
        try:
            result = summary.main()
            per_world = result[(result.section == "spar_official") & result.key.str.startswith("two_objects_relation") & (result.metric == "top1")]
            checks["world_dimension_omitted"] = {"input_desert_seeds": [.1, .9, .1], "output_rows": per_world[["key", "seed0", "seed1", "seed2", "note"]].to_dict("records")}
            missing = result[(result.section == "spar_official") & result.key.str.startswith("one_object_position") & (result.metric == "top5")].iloc[0]
            checks["undefined_top5_stable"] = bool(missing.stable)
            unit = result[result.section == "e3_unit"].iloc[0]
            checks["round_before_sign"] = {"input_gaps": [1e-5, -1e-5, 1e-5], "reported_gaps": [unit.seed0, unit.seed1, unit.seed2], "stable": bool(unit.stable)}
            source = outroot / "clip_vitl14" / "seed_1" / "tables"
            target = outroot / "tables_seed1"
            assert source.resolve().is_relative_to(root.resolve()) and target.resolve().is_relative_to(root.resolve())
            source.rename(target)
            try:
                summary.main()
            except SystemExit:
                checks["mixed_layout_repair"] = {"default_rejected": True}
            else:
                raise AssertionError("mixed layout accepted")
        finally:
            sys.argv = previous_argv

        sample_path = Path(__file__).with_name("imagenet_metadata_sample.json")
        if sample_path.exists():
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            sample_rows = sample["rows"]
            meta = metadata_fixture(root / "official_sample", sample_rows)
            duplicate_rows = int(meta.duplicated(subset=META_COLS, keep=False).sum())
            current = build_factor_pairs(meta, "camera", None, 4, 30000, 0, meta_cols=META_COLS)
            # In-memory reference experiment; the repository implementation is not changed.
            meta["character_name"] = [r["character_name"] for r in sample_rows]
            reference = build_factor_pairs(meta, "camera", None, 4, 30000, 0, meta_cols=META_COLS + ["character_name"])
            def describe(frame):
                endpoints = set(frame.row_id_a) | set(frame.row_id_b)
                cross = sum(sample_rows[int(r.row_id_a)]["character_name"] != sample_rows[int(r.row_id_b)]["character_name"] for r in frame.itertuples())
                return {"pairs": len(frame), "contexts": int(frame.context_key.nunique()), "endpoint_assets": len({sample_rows[int(i)]["character_name"] for i in endpoints}), "cross_asset_pairs": cross}
            checks["official_metadata_sample"] = {"source": sample["source"], "metadata_file_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(), "canonical_rows_sha256": hashlib.sha256(json.dumps(sample_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(), "rows": len(sample_rows), "rows_in_duplicate_context_factor_cells": duplicate_rows, "current": describe(current), "instance_fixed_reference": describe(reference), "not_a_random_or_full_dataset_sample": True}

        analyze = script("analyze_e3_units")
        analyze_dir = root / "analyze"
        analyze_dir.mkdir()
        items = [dict(image_id=f"{condition}_{group}", condition=condition, shape_group=group,
                      negative_type=kind, color="red", shape="sphere" if group == "sphere" else "cube",
                      item_acc=float(condition == "A"), item_margin=0.)
                 for condition in ("A", "B") for group in ("cube_cylinder", "sphere")
                 for kind in ("color", "shape")]
        (analyze_dir / "e3_items_cogent.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items), encoding="utf-8")
        pd.DataFrame([dict(shape_group=group, difficulty_gap=0.) for group in ("cube_cylinder", "sphere")]).to_csv(analyze_dir / "e3_difficulty_gap.csv", index=False)
        analyze.load_paths = lambda: {"dirs": {"outputs": root / "analyze_outputs"}}
        analyze.load_config = lambda name: {"bootstrap": {"n_boot": 8, "ci": .95}, "sampling": {"negative_types": ["color", "shape"]}, "colors": ["red"]}
        analyze.get_logger = lambda *args, **kwargs: Log()
        previous_argv = sys.argv
        sys.argv = ["analyze_e3_units.py", "--tables-dir", str(analyze_dir)]
        try:
            analyze.main()
        finally:
            sys.argv = previous_argv
        checks["inconsistent_e3_inputs"] = {"reference_total_gap": 0., "item_total_gap": 1., "unit_file_written": (analyze_dir / "e3_unit_gaps.csv").exists(), "input_conflict_was_deliberately_constructed": True}

    values = itertools.product(["a", "b"], ["w0", "w1", "w2", "w3"], ["Asphalt", "Default", "Grass", "Sky"], ["small", "medium", "large"], [0, 45, 225, 315])
    meta = pd.DataFrame(values, columns=["animal", "environment", "texture", "size", "camera"])
    meta["row_id"] = range(len(meta))
    meta["sample_id"] = meta.row_id.map(lambda n: f"audit_{n}")
    orders = load_config("e2_change")["datasets"]["pug_animals"]["canonical_orders"]
    factors = ["environment", "texture", "size", "camera"]
    frame = build_delta_frame({f: build_factor_pairs(meta, f, orders, 4, 30000, 0) for f in factors})
    deltas = np.random.default_rng(0).normal(size=(len(frame), 8)).astype(np.float32)
    counts = {f: {g: len(v["values"]) for g, v in group_similarities(frame, f, deltas, 100, np.random.default_rng(1000)).items()} for f in factors}
    assert all(n <= 100 for group in counts.values() for n in group.values())
    checks["comparison_cap_repair"] = counts
    palettes = load_config("e3_cogent")["palettes"]
    shape_accuracy = {"cube": .9, "cylinder": .3}
    gap = {}
    for color in load_config("e3_cogent")["colors"]:
        shape_a = next(s for s, colors in palettes["A"].items() if color in colors)
        shape_b = next(s for s, colors in palettes["B"].items() if color in colors)
        gap[color] = shape_accuracy[shape_a] - shape_accuracy[shape_b]
    checks["shape_only_counterexample"] = {"assumed_shape_accuracy": shape_accuracy, "color_unit_gaps": gap, "balanced_color_average": float(np.mean(list(gap.values()))), "counterexample_not_measured_model_performance": True}
    nuisance_count = load_config("e2_change")["nuisance"]["n_images"]
    logical_rgb_bytes = 2 * nuisance_count * 512 * 512 * 3
    checks["nuisance_memory_estimate"] = {"assumed_original_size": [512, 512], "configured_images": nuisance_count,
                                           "logical_rgb_GiB_approx": [.85 * logical_rgb_bytes / 2**30, logical_rgb_bytes / 2**30],
                                           "formula_estimate_not_measured_RAM_or_OOM": True}
    output = Path(__file__).with_name("audit_checks.json")
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "checks_executed": len(checks), "known_defects_are_reproductions_not_passes": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
