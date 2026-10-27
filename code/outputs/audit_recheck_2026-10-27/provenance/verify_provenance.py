"""Independent offline recheck; no real datasets, weights, or caches are touched."""
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
sys.path.insert(0, str(CODE))
import pandas as pd
from src.experiments import e2_change_diagnosis as e2
from src.utils import io


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helper = module("saved_audit_helpers", CODE / "outputs/audit_7b68232_2026-10-26/verify_acceptance.py")


def ident(name):
    return {"git_commit": name, "git_dirty": False}


def record(tables, experiment):
    return json.loads((tables / "run_meta.json").read_text())["experiments"][experiment]


def main():
    results = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip(),
               "checks": {}, "controlled_encoder": True}
    checks = results["checks"]
    files = sorted(list((CODE / "src").rglob("*.py")) + list((CODE / "scripts").rglob("*.py")))
    for file in files:
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    results["source_sha256"] = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checks["syntax"] = {"files": len(files), "valid": True}
    with tempfile.TemporaryDirectory(prefix="fixtures_", dir=HERE) as temporary:
        root = Path(temporary)
        paths = {"dirs": {"outputs": root / "history"}}
        exp = "e1_state"
        for revision in ["AAA", "BBB", "BBB", "CCC"]:
            with patch.object(io, "_code_identity", lambda r=revision: ident(r)):
                tables = io.output_tables_dir(paths, "audit_encoder", 0, {"cache_protocol": "v1"}, exp)
                (tables / "result.csv").write_text(revision)
                io.commit_experiment(tables, exp, ["result.csv"])
        rec = record(tables, exp)
        assert [v["git_commit"] for v in rec["replaces"]] == ["AAA", "BBB"]
        checks["D_history_preserved"] = rec

        legacy = {"model": "audit_encoder", "seed": 0, "experiments": {exp: {
            **ident("legacy_B"), "cache_protocol": "v1", "updated": "legacy_time",
            "replaces": {**ident("legacy_A"), "cache_protocol": "v0"}}}}
        (tables / "run_meta.json").write_text(json.dumps(legacy))
        with patch.object(io, "_code_identity", lambda: ident("legacy_B")):
            io.output_tables_dir(paths, "audit_encoder", 0, {"cache_protocol": "v1"}, exp)
            io.commit_experiment(tables, exp, ["result.csv"])
        rec = record(tables, exp)
        assert rec["last_success"]["git_commit"] == "legacy_B"
        assert isinstance(rec["replaces"], list) and rec["replaces"][0]["git_commit"] == "legacy_A"
        checks["D_legacy_migration"] = rec

        helper.parquet(root / "empty", [helper.row()], images=False)
        failure_paths = {"datasets": {"pug_imagenet": root / "empty"},
                         "dirs": {"cache": root / "empty_cache", "outputs": root / "empty_outputs"}}
        exp = "e2_pug_imagenet"
        tables = failure_paths["dirs"]["outputs"] / helper.FakeEncoder.name / "seed_0/tables"
        attempts = []
        for revision in ["failed_A", "failed_B"]:
            with patch.object(io, "_code_identity", lambda r=revision: ident(r)):
                try:
                    e2.run(helper.FakeEncoder(), failure_paths, helper.Log(), dataset="pug_imagenet")
                except SystemExit as error:
                    attempts.append({"error": str(error), "metadata": record(tables, exp)})
                else:
                    raise AssertionError("Expected pair validation to stop")
        assert "last_success" not in attempts[0]["metadata"]
        assert attempts[1]["metadata"]["last_success"]["git_commit"] == "failed_A"
        assert not list(tables.glob("*.csv"))
        checks["P1_repeated_failed_run_fabricates_success"] = {
            "reproduced": True, "actual_e2_entrypoint": True, "attempts": attempts,
            "csv_files": [p.name for p in tables.glob("*.csv")]}

        success_paths = {"datasets": {"pug_imagenet": root / "empty"},
                         "dirs": {"cache": root / "known_cache", "outputs": root / "known_outputs"}}
        with patch.object(io, "_code_identity", lambda: ident("old_success")):
            tables = io.output_tables_dir(success_paths, helper.FakeEncoder.name, 0,
                                         {"cache_protocol": helper.FakeEncoder.cache_protocol}, exp)
            csv = tables / "e2_delta_stats_pug_imagenet.csv"
            csv.write_text("source\nold_success\n")
            io.commit_experiment(tables, exp, [csv.name])
        old = csv.read_bytes()
        with patch.object(io, "_code_identity", lambda: ident("new_failure")):
            try:
                e2.run(helper.FakeEncoder(), success_paths, helper.Log(), dataset="pug_imagenet")
            except SystemExit:
                pass
            else:
                raise AssertionError("Expected failure")
        rec = record(tables, exp)
        assert rec["last_success"]["git_commit"] == "old_success" and csv.read_bytes() == old
        checks["P1_original_single_failure_fixed"] = {"csv_unchanged": True, "metadata": rec}

        summary = module("actual_summary", CODE / "scripts/summarize_seed_stability.py")
        outputs = root / "mixed_outputs"
        helper.summary_tables(outputs)
        rows = [helper.row(animal, animal, env, texture, scale, camera)
                for animal, env, texture, scale, camera in itertools.product(
                    ["cat", "dog"], ["forest", "desert"], ["Asphalt", "Default"], [.7, 1.], [0, 45])]
        helper.parquet(root / "animals", rows, images=True)
        run_paths = {"datasets": {"pug_animals": root / "animals"},
                     "dirs": {"cache": root / "mixed_cache", "outputs": outputs}}
        model = helper.FakeEncoder()
        model.name = "clip_vitl14"
        cfg = copy.deepcopy(e2.load_config("e2_change"))
        cfg["bootstrap"]["n_boot"] = 8
        cfg["similarity"]["max_comparisons_per_group"] = 20
        cfg["nuisance"]["n_images"] = 12
        cfg["nuisance"]["n_captions"] = 12
        with patch.object(e2, "load_config", lambda name: cfg), patch.object(io, "_code_identity", lambda: ident("actual_success")):
            e2.run(model, run_paths, helper.Log(), seed=0)
        tables = outputs / model.name / "seed_0/tables"
        stat = tables / "e2_delta_stats.csv"
        transition = tables / "e2_transition_stats.csv"
        before_stat, before_transition = stat.read_bytes(), transition.read_bytes()
        old_values = pd.read_csv(stat).norm_gap_visual_raw.tolist()
        for seed in (1, 2):
            (outputs / model.name / f"seed_{seed}/tables/e2_delta_stats.csv").write_bytes(before_stat)
        original_compute = e2.compute_deltas

        def scaled(*args, **kwargs):
            raw, norm = original_compute(*args, **kwargs)
            return raw * 3, norm * 3

        with patch.object(e2, "load_config", lambda name: cfg), \
                patch.object(io, "_code_identity", lambda: ident("partial_failure")), \
                patch.object(e2, "compute_deltas", scaled), \
                patch.object(e2, "transition_table", side_effect=RuntimeError("controlled interruption after first CSV")):
            try:
                e2.run(model, run_paths, helper.Log(), seed=0)
            except RuntimeError as error:
                interruption = str(error)
            else:
                raise AssertionError("Expected interruption")
        rec = record(tables, "e2_pug_animals")
        assert rec["status"] == "running" and rec["last_success"]["git_commit"] == "actual_success"
        assert stat.read_bytes() != before_stat and transition.read_bytes() == before_transition
        new_values = pd.read_csv(stat).norm_gap_visual_raw.tolist()
        with patch.object(summary, "load_paths", lambda: {"dirs": {"outputs": outputs}}), \
                patch.object(summary, "get_logger", lambda *a, **kw: helper.Log()):
            helper.invoke(summary, ["--model", model.name])
        official = tables / "seed_stability.csv"
        assert official.exists()
        generated = pd.read_csv(official)
        consumed = generated[(generated.section == "e2") & (generated.metric == "norm_gap_visual_raw")]
        assert consumed.seed0.tolist() == [round(v, 4) for v in new_values]
        checks["P1_partial_output_consumed_by_official_summary"] = {
            "reproduced": True, "actual_e2_and_summary_entrypoints": True,
            "controlled_change": "factor deltas scaled by 3; interruption before transition table calculation",
            "interruption": interruption, "first_csv_changed": True, "second_csv_unchanged": True,
            "metadata": rec, "old_values": old_values, "new_values": new_values,
            "official_summary_generated": True, "consumed_rows": consumed.to_dict("records")}
    results["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = HERE / "results.json"
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"results": str(target), "check_groups": len(checks),
                      "P1_repeated_failure_reproduced": True, "P1_partial_consumption_reproduced": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
