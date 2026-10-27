"""Independent E3 lifecycle checks. Uses actual entrypoints with small synthetic scores.

Does not load VLM weights or modify production sources/caches.
"""
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
CODE = BASE.parents[2]
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
from src.experiments import e3_cogent_diagnosis as e3
from src.utils import io


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


helper = module("fixture_helpers_e3", CODE / "outputs/audit_7b68232_2026-10-26/verify_acceptance.py")
analyze = module("actual_analyze_e3", CODE / "scripts/analyze_e3_units.py")
summary = module("actual_summary_e3", CODE / "scripts/summarize_seed_stability.py")


def blocked(tables):
    try:
        io.verify_e3_consistency(tables)
    except SystemExit as error:
        return str(error)
    return None


def record(tables):
    return json.loads((tables / "e3_unit_consistency.json").read_text())


def main():
    result = {"checks": {}, "synthetic_encoder": True}
    checks = result["checks"]
    config = copy.deepcopy(e3.load_config("e3_cogent"))
    config["bootstrap"]["n_boot"] = 8
    config["colors"] = ["red", "blue", "gray"]
    config["sampling"]["per_scene_cap"] = 3
    state = {"A": 1., "B": 0.}

    def scenes(root, cond, configured=None):
        return [{"image_filename": f"{cond}.png", "objects": [
            {"color": "gray", "shape": "cube"},
            {"color": "red", "shape": "cylinder"},
            {"color": "blue", "shape": "sphere"}]}]

    def scores(model, tcache, cache, key, items, **kwargs):
        value = state[items[0]["condition"]]
        out = {field: 0. for field in e3.EXPORT_FIELDS}
        out.update(n_samples=len(items), accuracy=value, acc_ci_low=value, acc_ci_high=value,
                   item_acc=np.full(len(items), value), item_margin=np.zeros(len(items)))
        return out

    with tempfile.TemporaryDirectory(prefix="e3_check_", dir=BASE) as temporary:
        root = Path(temporary)
        paths = {"datasets": {"cogent": root / "data"},
                 "dirs": {"cache": root / "cache", "outputs": root / "out"}}
        with patch.object(e3, "load_config", return_value=config), \
             patch.object(e3, "load_condition_scenes", side_effect=scenes), \
             patch.object(e3, "extract_clevr_images", return_value=None), \
             patch.object(e3, "eval_2afc_items", side_effect=scores), \
             patch.object(io, "_code_identity", return_value={"git_commit": "controlled", "git_dirty": False}), \
             patch.object(analyze, "load_paths", return_value=paths), \
             patch.object(analyze, "load_config", return_value=config), \
             patch.object(analyze, "get_logger", return_value=helper.Log()):
            e3.run(helper.FakeEncoder(), paths, helper.Log())
            tables = root / "out/audit_encoder/seed_0/tables"
            helper.invoke(analyze, ["--tables-dir", str(tables)])
            checks["first_analysis"] = {"status": record(tables)["status"],
                                        "fingerprint_present": bool(record(tables)["source_fingerprint"]),
                                        "guard_error": blocked(tables)}
            state["A"] = 0.
            e3.run(helper.FakeEncoder(), paths, helper.Log())
            checks["successful_rerun_invalidates"] = {
                "units_removed": not (tables / "e3_unit_gaps.csv").exists(),
                "status": record(tables)["status"], "guard_error": blocked(tables)}
            helper.invoke(analyze, ["--tables-dir", str(tables)])
            checks["reanalysis_recovers"] = {"status": record(tables)["status"], "guard_error": blocked(tables)}

            # Simulate an interrupted E3 run after all sources were written but before
            # deletion of the previous derived table. In real runs unlink can fail
            # (e.g. a file opened/locked by another process on Windows).
            old_units = (tables / "e3_unit_gaps.csv").read_bytes()
            original_unlink = Path.unlink
            def fail_unit_unlink(self, *args, **kwargs):
                if self.name == "e3_unit_gaps.csv":
                    raise PermissionError("controlled failure while invalidating unit table")
                return original_unlink(self, *args, **kwargs)
            state["A"] = 1.
            with patch.object(Path, "unlink", fail_unit_unlink):
                try:
                    e3.run(helper.FakeEncoder(), paths, helper.Log())
                except PermissionError as error:
                    stopped = str(error)
                else:
                    raise AssertionError("expected controlled failure")
            checks["interrupted_e3_new_sidecar"] = {
                "error": stopped, "status": record(tables)["status"],
                "units_unchanged": (tables / "e3_unit_gaps.csv").read_bytes() == old_units,
                "experiment_status": json.loads((tables / "run_meta.json").read_text())["experiments"]["e3_cogent"]["status"],
                "guard_error": blocked(tables)}

            # The reanalysis publishes passed + current fingerprint before writing
            # new units. A write failure then releases the old unit table.
            original_csv = pd.DataFrame.to_csv
            def fail_units_csv(self, path_or_buf=None, *args, **kwargs):
                if Path(path_or_buf).name == "e3_unit_gaps.csv":
                    raise OSError("controlled unit CSV write failure")
                return original_csv(self, path_or_buf, *args, **kwargs)
            with patch.object(pd.DataFrame, "to_csv", fail_units_csv):
                try:
                    helper.invoke(analyze, ["--tables-dir", str(tables)])
                except OSError as error:
                    stopped = str(error)
                else:
                    raise AssertionError("expected controlled failure")
            checks["analysis_failure_publishes_passed_for_old_units"] = {
                "error": stopped, "status": record(tables)["status"],
                "fingerprint_matches_current_sources": record(tables)["source_fingerprint"] == io.e3_source_fingerprint(tables),
                "units_unchanged": (tables / "e3_unit_gaps.csv").read_bytes() == old_units,
                "old_nonmissing_unit_gaps": pd.read_csv(tables / "e3_unit_gaps.csv").gap.dropna().unique().tolist(),
                "current_source_gaps": pd.read_csv(tables / "e3_difficulty_gap.csv").difficulty_gap.tolist(),
                "guard_error": blocked(tables)}
            failed_analysis_snapshot = {
                name: (tables / name).read_bytes() for name in (
                    "e3_difficulty_gap.csv", "e3_items_cogent.jsonl",
                    "e3_unit_gaps.csv", "e3_unit_consistency.json")}
            helper.invoke(analyze, ["--tables-dir", str(tables)])
            with (tables / "e3_items_cogent.jsonl").open("a") as output:
                output.write("\n")
            checks["fingerprint_tamper_blocked"] = {"guard_error": blocked(tables)}

            # Historical sidecar compatibility leaves interrupted E3 writes open.
            helper.invoke(analyze, ["--tables-dir", str(tables)])
            rec = record(tables)
            rec.pop("source_fingerprint", None)
            (tables / "e3_unit_consistency.json").write_text(json.dumps(rec))
            state["A"] = 0.
            with patch.object(Path, "unlink", fail_unit_unlink):
                try:
                    e3.run(helper.FakeEncoder(), paths, helper.Log())
                except PermissionError:
                    pass
            checks["interrupted_e3_old_sidecar"] = {
                "status": record(tables)["status"],
                "guard_error": blocked(tables),
                "old_nonmissing_unit_gaps": pd.read_csv(tables / "e3_unit_gaps.csv").gap.dropna().unique().tolist(),
                "current_source_gaps": pd.read_csv(tables / "e3_difficulty_gap.csv").difficulty_gap.tolist()}

            # Regression of the prior six-fix acceptance cases: conflicting items
            # must invalidate existing units; missing reference has distinct status.
            reference = pd.read_csv(tables / "e3_difficulty_gap.csv")
            reference["difficulty_gap"] = 1.
            reference.to_csv(tables / "e3_difficulty_gap.csv", index=False)
            try:
                helper.invoke(analyze, ["--tables-dir", str(tables)])
            except SystemExit as error:
                stopped = str(error)
            else:
                raise AssertionError("strict conflicting analysis should stop")
            checks["regression_strict_conflict"] = {
                "strict_error": stopped,
                "units_removed": not (tables / "e3_unit_gaps.csv").exists(),
                "status": record(tables)["status"],
                "guard_error": blocked(tables)}
            (tables / "e3_difficulty_gap.csv").unlink()
            try:
                helper.invoke(analyze, ["--tables-dir", str(tables)])
            except SystemExit as error:
                stopped = str(error)
            else:
                raise AssertionError("strict missing-reference analysis should stop")
            helper.invoke(analyze, ["--tables-dir", str(tables), "--no-strict"])
            checks["regression_missing_reference"] = {
                "strict_error": stopped, "exploration_status": record(tables)["status"],
                "all_ok": record(tables)["all_ok"],
                "fingerprint": record(tables)["source_fingerprint"],
                "units_written": (tables / "e3_unit_gaps.csv").exists()}

        summary_root = root / "summary"
        helper.summary_tables(summary_root)
        seed_tables = [summary_root / "clip_vitl14" / f"seed_{s}/tables" for s in range(3)]
        with patch.object(summary, "load_paths", return_value={"dirs": {"outputs": summary_root}}), \
             patch.object(summary, "get_logger", return_value=helper.Log()):
            for status in ("failed", "stale"):
                (seed_tables[1] / "e3_unit_consistency.json").write_text(json.dumps({"status": status}))
                try:
                    helper.invoke(summary, [])
                except SystemExit as error:
                    message = str(error)
                else:
                    raise AssertionError("formal summary should reject invalid state")
                checks[f"summary_{status}_blocked"] = {
                    "error": message, "formal_file_exists": (seed_tables[0] / "seed_stability.csv").exists()}
            out = helper.invoke(summary, ["--allow-exploratory-e3"])
            checks["summary_exploratory_separate"] = {
                "exploratory_exists": (seed_tables[0] / "seed_stability_exploratory.csv").exists(),
                "formal_file_exists": (seed_tables[0] / "seed_stability.csv").exists(), "rows": len(out)}
            (seed_tables[1] / "e3_unit_consistency.json").write_text(json.dumps({"status": "not_checked", "source_fingerprint": None, "all_ok": False}))
            out = helper.invoke(summary, [])
            checks["summary_not_checked_compatibility"] = {
                "formal_file_exists": (seed_tables[0] / "seed_stability.csv").exists(), "rows": len(out)}
            for t in seed_tables:
                for name, data in failed_analysis_snapshot.items():
                    (t / name).write_bytes(data)
            out = helper.invoke(summary, [])
            cells = out[(out.section == "e3_unit") & (out.stable == True)]
            checks["summary_consumes_stale_units_after_analysis_failure"] = {
                "formal_file_exists": (seed_tables[0] / "seed_stability.csv").exists(),
                "main_gap_seed0": pd.read_csv(seed_tables[0] / "e3_difficulty_gap.csv").difficulty_gap.tolist(),
                "stable_units": cells[["key", "seed0", "seed1", "seed2"]].to_dict(orient="records")}
    sources = [CODE / p for p in ("src/utils/io.py", "src/experiments/e3_cogent_diagnosis.py", "scripts/analyze_e3_units.py", "scripts/summarize_seed_stability.py", "scripts/make_figures.py")]
    result["source_sha256"] = {p.relative_to(CODE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output = BASE / "results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "checks": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
