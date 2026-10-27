"""Offline verification for ef329ad. Records fixes and lifecycle counterexamples.

Uses saved fixture helpers without executing the historical audit main().
No model/image downloads, production edits, or operations on real caches.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
import pandas as pd
import numpy as np
from src.experiments import e2_change_diagnosis as e2
from src.features.cache import ImageFeatureCache
from src.utils import io as utils_io


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    helper_path = CODE / "outputs/audit_7b68232_2026-10-26/verify_acceptance.py"
    helper = load_module("saved_fixture_helpers", helper_path)
    result = {"revision": "ef329ad", "fixture_helpers_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(), "checks": {}}
    checks = result["checks"]
    source_files = list((CODE / "src").rglob("*.py")) + list((CODE / "scripts").rglob("*.py"))
    for p in source_files: ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    result["source_sha256"] = {p.relative_to(CODE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    checks["syntax"] = {"production_files": len(source_files)}
    with tempfile.TemporaryDirectory(prefix="audit_current_", dir=CODE.parent) as temp:
        root = Path(temp)
        rows = [helper.row(f"hash{i:02d}", f"class{11-i:02d}", env, tex, size, cam)
                for i in range(12) for env in ["e0", "e1"] for tex in ["t0", "t1"]
                for size in [1., 2.] for cam in [0, 90]]
        helper.parquet(root / "reverse", rows, images=True)
        paths = {"datasets": {"pug_imagenet": root / "reverse"}, "dirs": {"cache": root / "reverse_cache", "outputs": root / "reverse_out"}}
        config_original = e2.load_config
        config = copy.deepcopy(config_original("e2_change"))
        config["bootstrap"]["n_boot"] = 8
        config["similarity"]["max_comparisons_per_group"] = 20
        e2.load_config = lambda name: config
        try:
            class RecordingLog(helper.Log):
                def __init__(self): self.messages = []
                def info(self, *args, **kwargs): self.messages.append(str(args[0]))
            smoke_log = RecordingLog()
            out = e2.run(helper.FakeEncoder(), paths, smoke_log, smoke=True, dataset="pug_imagenet")
            selected_rows = next(int(re.search(r"smoke subset: (\d+)", line).group(1)) for line in smoke_log.messages if "smoke subset:" in line)
            checks["issue1_smoke_fixed"] = {"input_rows": len(rows), "selected_rows_by_current_rule": selected_rows,
                                             "factors": out.factor.tolist(), "pairs": out.n_pairs.astype(int).tolist(),
                                             "synthetic_encoder": True}
            helper.parquet(root / "empty", [helper.row()], images=False)
            paths = {"datasets": {"pug_imagenet": root / "empty"}, "dirs": {"cache": root / "empty_cache", "outputs": root / "empty_out"}}
            try: e2.run(helper.FakeEncoder(), paths, helper.Log(), dataset="pug_imagenet", allow_empty_factors=True)
            except SystemExit as error: checks["issue4_all_empty_fixed"] = {"error_type": "SystemExit", "message": str(error)}
            else: raise AssertionError("all-empty run should stop explicitly")
        finally: e2.load_config = config_original

        force_source = CODE / "scripts/extract_embeddings.py"
        node = next(n for n in ast.parse(force_source.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "_force_reset")
        namespace = {"Path": Path, "json": json, "shutil": shutil}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_force_reset", "exec"), namespace)
        protocol_checks = {}
        for protocol in ["[]", "null", '"text"', '{"protocol":']:
            cache_root = root / "protocols"
            model_dir = cache_root / helper.FakeEncoder.name
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "protocol.json").write_text(protocol)
            (model_dir / "old_images.npz").write_bytes(b"temporary fixture")
            try: ImageFeatureCache(cache_root, helper.FakeEncoder.name, protocol=helper.FakeEncoder.cache_protocol)
            except RuntimeError: rejected = True
            else: rejected = False
            assert model_dir.resolve().is_relative_to(root.resolve())
            namespace["_force_reset"](cache_root, helper.FakeEncoder(), "clevr_val", helper.Log())
            protocol_checks[protocol] = {"reader_rejected": rejected, "force_removed_directory": not model_dir.exists()}
        checks["issue5_protocol_types_fixed"] = protocol_checks

        analyze = load_module("actual_analyze", CODE / "scripts/analyze_e3_units.py")
        tables = root / "analysis"
        tables.mkdir()
        items = [dict(image_id=f"{condition}_{group}", condition=condition, shape_group=group,
                      negative_type=kind, color="red", shape="sphere" if group == "sphere" else "cube",
                      item_acc=float(condition == "A"), item_margin=0.)
                 for condition in ["A", "B"] for group in ["cube_cylinder", "sphere"] for kind in ["color", "shape"]]
        (tables / "e3_items_cogent.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        def reference(gap): pd.DataFrame([dict(shape_group=g, difficulty_gap=gap) for g in ["cube_cylinder", "sphere"]]).to_csv(tables / "e3_difficulty_gap.csv", index=False)
        analyze.load_paths = lambda: {"dirs": {"outputs": root / "analysis_out"}}
        analyze.load_config = lambda name: {"bootstrap": {"n_boot": 8, "ci": .95}, "sampling": {"negative_types": ["color", "shape"]}, "colors": ["red"]}
        analyze.get_logger = lambda *args, **kwargs: helper.Log()
        reference(1.)
        helper.invoke(analyze, ["--tables-dir", str(tables)])
        reference(0.)
        try: helper.invoke(analyze, ["--tables-dir", str(tables)])
        except SystemExit: stopped = True
        else: stopped = False
        sidecar = json.loads((tables / "e3_unit_consistency.json").read_text())
        try: utils_io.verify_e3_consistency(tables)
        except SystemExit: downstream_blocked = True
        else: downstream_blocked = False
        checks["issue2_failed_units_fixed"] = {"strict_stopped": stopped, "old_unit_removed": not (tables / "e3_unit_gaps.csv").exists(), "status": sidecar["status"], "consumer_guard_blocked": downstream_blocked}
        (tables / "e3_difficulty_gap.csv").unlink()
        try: helper.invoke(analyze, ["--tables-dir", str(tables)])
        except SystemExit: missing_strict = True
        else: missing_strict = False
        helper.invoke(analyze, ["--tables-dir", str(tables), "--no-strict"])
        sidecar = json.loads((tables / "e3_unit_consistency.json").read_text())
        checks["issue6_missing_reference_fixed"] = {"strict_stopped": missing_strict, "exploration_status": sidecar["status"], "exploration_all_ok": sidecar["all_ok"]}

        summary = load_module("actual_summary", CODE / "scripts/summarize_seed_stability.py")
        summary_root = root / "summary"
        helper.summary_tables(summary_root)
        for seed in range(3):
            t = summary_root / "clip_vitl14" / f"seed_{seed}" / "tables"
            (t / "e3_items_cogent.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
            # Actual exploration mode generates failed sidecars and new unit tables.
            helper.invoke(analyze, ["--tables-dir", str(t), "--no-strict"])
        summary.load_paths = lambda: {"dirs": {"outputs": summary_root}}
        summary.get_logger = lambda *args, **kwargs: helper.Log()
        summary_out = helper.invoke(summary, [])
        checks["failed_exploration_units_consumed_by_summary"] = {
            "sidecar_statuses": [json.loads((summary_root / "clip_vitl14" / f"seed_{seed}" / "tables/e3_unit_consistency.json").read_text())["status"] for seed in range(3)],
            "formal_summary_written": (summary_root / "clip_vitl14/seed_0/tables/seed_stability.csv").exists(),
            "unit_rows_stable": summary_out[summary_out.section == "e3_unit"].stable.tolist()}

        from src.experiments import e3_cogent_diagnosis as e3
        e3_originals = {name: getattr(e3, name) for name in ["load_config", "load_condition_scenes", "extract_clevr_images", "eval_2afc_items"]}
        identity_before_e3 = utils_io._code_identity
        e3_config = copy.deepcopy(e3.load_config("e3_cogent"))
        e3_config["bootstrap"]["n_boot"] = 8
        e3_config["sampling"]["per_scene_cap"] = 3
        score_state = {"A": 1., "B": 0.}
        def e3_scenes(unused, condition):
            shapes = ["cube", "cylinder"] if condition == "A" else ["cylinder", "cube"]
            return [{"image_filename": f"{condition}.png", "objects": [{"color": "gray", "shape": shapes[0]}, {"color": "red", "shape": shapes[1]}, {"color": "blue", "shape": "sphere"}]}]
        def e3_scores(model, text_cache, image_cache, key, items, **kwargs):
            score = score_state[items[0]["condition"]]
            values = {field: 0. for field in e3.EXPORT_FIELDS}
            values.update(n_samples=len(items), accuracy=score, acc_ci_low=score, acc_ci_high=score,
                          item_acc=np.full(len(items), score), item_margin=np.zeros(len(items)))
            return values
        e3.load_config = lambda name: e3_config
        e3.load_condition_scenes = e3_scenes
        e3.extract_clevr_images = lambda *args, **kwargs: None
        e3.eval_2afc_items = e3_scores
        e3_paths = {"datasets": {"cogent": "unused"}, "dirs": {"cache": root / "e3_cache", "outputs": root / "e3_out"}}
        try:
            utils_io._code_identity = lambda: {"git_commit": "full_A", "git_dirty": False}
            e3.run(helper.FakeEncoder(), e3_paths, helper.Log(), smoke=False)
            t = root / "e3_out" / helper.FakeEncoder.name / "seed_0/tables"
            analyze.load_config = lambda name: e3_config
            helper.invoke(analyze, ["--tables-dir", str(t)])
            old_unit = (t / "e3_unit_gaps.csv").read_bytes()
            score_state["A"] = 0.
            utils_io._code_identity = lambda: {"git_commit": "full_B", "git_dirty": False}
            e3.run(helper.FakeEncoder(), e3_paths, helper.Log(), smoke=False)
            sidecar = json.loads((t / "e3_unit_consistency.json").read_text())
            try: utils_io.verify_e3_consistency(t)
            except SystemExit: blocked = True
            else: blocked = False
            checks["successful_e3_rerun_keeps_stale_passed_units"] = {
                "new_main_gaps": pd.read_csv(t / "e3_difficulty_gap.csv").difficulty_gap.tolist(),
                "old_nonmissing_unit_gaps": pd.read_csv(t / "e3_unit_gaps.csv").gap.dropna().unique().tolist(),
                "unit_bytes_unchanged": (t / "e3_unit_gaps.csv").read_bytes() == old_unit,
                "sidecar_status": sidecar["status"], "guard_blocked": blocked,
                "loading_extraction_scoring_replaced": True}
            full_bytes = (t / "e3_cogent.csv").read_bytes()
            utils_io._code_identity = lambda: {"git_commit": "smoke_C", "git_dirty": False}
            e3.run(helper.FakeEncoder(), e3_paths, helper.Log(), smoke=True)
            metadata = json.loads((t / "run_meta.json").read_text())
            checks["smoke_overwrites_full_experiment_provenance"] = {
                "full_CSV_unchanged": (t / "e3_cogent.csv").read_bytes() == full_bytes,
                "full_CSV_source_before_smoke": "full_B",
                "recorded_full_experiment_source": metadata["experiments"]["e3_cogent"]["git_commit"],
                "experiment_keys": sorted(metadata["experiments"]),
                "smoke_CSV_written": (t / "e3_cogent_smoke.csv").exists()}
        finally:
            for name, value in e3_originals.items(): setattr(e3, name, value)
            utils_io._code_identity = identity_before_e3

        identity_original = utils_io._code_identity
        try:
            paths = {"dirs": {"outputs": root / "identity"}}
            utils_io._code_identity = lambda: {"git_commit": "A", "git_dirty": False}
            t = utils_io.output_tables_dir(paths, "clip_vitl14", 0, extra={"cache_protocol": "v1"}, experiment="e1_state")
            utils_io._code_identity = lambda: {"git_commit": "B", "git_dirty": False}
            utils_io.output_tables_dir(paths, "clip_vitl14", 0, extra={"cache_protocol": "v2"}, experiment="e1_state")
            utils_io.output_tables_dir(paths, "clip_vitl14", 0, extra={"cache_protocol": "v2"}, experiment="e2_pug_imagenet")
            metadata = json.loads((t / "run_meta.json").read_text())
            checks["issue3_normal_update_fixed"] = {"first_directory_commit": metadata["git_commit"],
                                                    "e1_current_commit": metadata["experiments"]["e1_state"]["git_commit"],
                                                    "e1_current_protocol": metadata["experiments"]["e1_state"]["cache_protocol"],
                                                    "e1_replaces_commit": metadata["experiments"]["e1_state"]["replaces"]["git_commit"],
                                                    "separate_experiments": sorted(metadata["experiments"])}
            utils_io.output_tables_dir(paths, "clip_vitl14", 0, extra={"cache_protocol": "v2"}, experiment="e1_state")
            metadata = json.loads((t / "run_meta.json").read_text())
            checks["same_identity_repeat_history"] = {"replaces_still_present": "replaces" in metadata["experiments"]["e1_state"]}

            failing_paths = {"datasets": {"pug_imagenet": root / "empty"}, "dirs": {"cache": root / "failing_cache", "outputs": root / "failing_out"}}
            utils_io._code_identity = lambda: {"git_commit": "old_success", "git_dirty": False}
            t = utils_io.output_tables_dir(failing_paths, helper.FakeEncoder.name, 0, extra={"cache_protocol": helper.FakeEncoder.cache_protocol}, experiment="e2_pug_imagenet")
            old_csv = t / "e2_delta_stats_pug_imagenet.csv"
            old_csv.write_text("dataset,factor,source\npug_imagenet,camera,old_success\n")
            old_bytes = old_csv.read_bytes()
            utils_io._code_identity = lambda: {"git_commit": "failed_attempt", "git_dirty": False}
            try: e2.run(helper.FakeEncoder(), failing_paths, helper.Log(), dataset="pug_imagenet")
            except SystemExit: failed = True
            else: failed = False
            metadata = json.loads((t / "run_meta.json").read_text())
            checks["failed_run_provenance_mismatch"] = {"actual_run_stopped": failed,
                                                       "old_csv_unchanged": old_csv.read_bytes() == old_bytes,
                                                       "recorded_experiment_commit": metadata["experiments"]["e2_pug_imagenet"]["git_commit"],
                                                       "actual_csv_source": "old_success"}
        finally: utils_io._code_identity = identity_original

    result["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(__file__).with_name("verification_results.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "evidence_categories": len(checks), "records_include_both_repairs_and_remaining_defects": True}))


if __name__ == "__main__": main()
