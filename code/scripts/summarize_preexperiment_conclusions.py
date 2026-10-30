"""Summarize saved stage-one results; no model/data access or new inference.

Run from any directory: python code/scripts/summarize_preexperiment_conclusions.py
Seed means/ranges are descriptive, not confidence intervals. Existing CIs are
counted separately; overlapping seed samples are never pooled as independent.
"""
import csv
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis"
MODELS = ("clip_vitb32", "clip_vitl14", "siglip2_so400m")
DATE = "2026-10-28"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}
    sources = {}
    checks = []

    def read(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def add(section, model, seed, key, metric, row, lo=None, hi=None, ref=0):
        value = float(row[metric])
        signature = (section, model, key, metric)
        sign = "unavailable"
        if lo is not None:
            lower, upper = float(row[lo]), float(row[hi])
            assert lower <= upper
            sign = "above" if lower > ref else "below" if upper < ref else "contains"
        groups.setdefault(signature, []).append((seed, value, sign))

    for model in MODELS:
        for seed in range(3):
            tables = ROOT / "outputs" / model / f"seed_{seed}" / "tables"
            meta_path = tables / "run_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sources[str(meta_path.relative_to(ROOT))] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
            expected = ("e1_state", "e1_spar_official", "e2_pug_animals", "e2_pug_imagenet", "e3_cogent")
            assert all(meta["experiments"][key]["status"] == "success" for key in expected)
            checks.append({"model": model, "seed": seed, "five_experiments_success": True,
                           "git_dirty": meta["git_dirty"]})
            consistency_path = tables / "e3_unit_consistency.json"
            consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
            sources[str(consistency_path.relative_to(ROOT))] = hashlib.sha256(consistency_path.read_bytes()).hexdigest()
            assert consistency["status"] == "passed" and consistency["all_ok"]
            source_payloads = {}
            for filename in ("e3_items_cogent.jsonl", "e3_difficulty_gap.csv"):
                source_path = tables / filename
                source_bytes = source_path.read_bytes()
                sources[str(source_path.relative_to(ROOT))] = hashlib.sha256(source_bytes).hexdigest()
                source_payloads[filename] = source_bytes
            units_bytes = (tables / "e3_unit_gaps.csv").read_bytes()
            for mode in ("exact_bytes", "lf_normalized"):
                normalize = (lambda b: b) if mode == "exact_bytes" else (lambda b: b.replace(b"\r\n", b"\n"))
                parts = [f"{name}={hashlib.md5(normalize(payload)).hexdigest()}"
                         for name, payload in source_payloads.items()]
                fp = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
                if fp == consistency["source_fingerprint"] and hashlib.md5(normalize(units_bytes)).hexdigest() == consistency["unit_table_md5"]:
                    checks[-1]["e3_fingerprints_verified"] = mode
                    break
            else:
                raise ValueError(f"E3 source/unit fingerprint mismatch: {tables}")
            e1_rows = read(tables / "e1_results.csv")
            for row in e1_rows:
                key = f'{row["dataset"]}/{row["factor"]}'
                add("e1", model, seed, key, "accuracy", row, "acc_ci_low", "acc_ci_high", .5)
                add("e1", model, seed, key, "mean_margin", row, "margin_ci_low", "margin_ci_high")
            items_path = tables / "e1_items_clevr.jsonl"
            sources[str(items_path.relative_to(ROOT))] = hashlib.sha256(items_path.read_bytes()).hexdigest()
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line]
            size_items = [item for item in items if item["factor"] == "size"]
            assert len(size_items) == 2000
            assert {item["content_pos"].split()[1] for item in size_items} == {"small", "large"}
            small_prior = sum(item["content_pos"].split()[1] == "small" for item in size_items)/len(size_items)
            size_accuracy = float(next(row for row in e1_rows if row["dataset"] == "clevr" and row["factor"] == "size")["accuracy"])
            add("e1_size_baseline", model, seed, "clevr/size", "always_small_accuracy", {"always_small_accuracy": small_prior})
            add("e1_size_baseline", model, seed, "clevr/size", "accuracy_minus_small_prior", {"accuracy_minus_small_prior": size_accuracy-small_prior})
            for row in read(tables / "e1_factor_gaps.csv"):
                key = f'{row["dataset"]}/{row["factor_a"]}-{row["factor_b"]}'
                add("e1_gap", model, seed, key, "gap", row, "gap_ci_low", "gap_ci_high")
            for filename in ("e2_delta_stats.csv", "e2_delta_stats_pug_imagenet.csv"):
                for row in read(tables / filename):
                    key = f'{row["dataset"]}/{row["factor"]}'
                    for metric, lo, hi in (
                        ("same_transition_sim_diff_context", "same_transition_ci_low", "same_transition_ci_high"),
                        ("transition_similarity_gap", "gap_ci_low", "gap_ci_high"),
                        ("norm_gap_visual_raw", "norm_gap_visual_raw_ci_low", "norm_gap_visual_raw_ci_high"),
                        ("norm_gap_visual", "norm_gap_visual_ci_low", "norm_gap_visual_ci_high"),
                    ):
                        add("e2", model, seed, key, metric, row, lo, hi)
                    for metric in ("same_factor_dt_sc_sim", "same_factor_dt_dc_sim", "diff_factor_dc_sim",
                                   "nuisance_sim_visual_z", "nuisance_norm_visual_z", "mean_delta_norm_z",
                                   "factor_over_nuisance_ratio"):
                        add("e2", model, seed, key, metric, row)
                    specific = float(row["same_transition_sim_diff_context"])-float(row["same_factor_dt_dc_sim"])
                    add("e2", model, seed, key, "same_vs_other_transition_excess", {"same_vs_other_transition_excess": specific})
            for row in read(tables / "e3_difficulty_gap.csv"):
                add("e3", model, seed, row["shape_group"], "difficulty_gap", row, "gap_ci_low", "gap_ci_high")
            for row in read(tables / "e3_unit_gaps.csv"):
                key = f'{row["shape_group"]}/{row["color"]}/{row["negative_type"]}'
                add("e3_unit", model, seed, key, "gap", row, "ci_low", "ci_high")
            for row in read(tables / "e3_cogent.csv"):
                key = f'{row["condition"]}/{row["shape_group"]}/{row["negative_type"]}'
                add("e3_acc", model, seed, key, "accuracy", row, "acc_ci_low", "acc_ci_high", .5)
            shape_sums = {}
            for row in read(tables / "e3_cogent_by_combo.csv"):
                key = "/".join(row[k] for k in ("condition", "color", "shape", "negative_type"))
                add("e3_combo", model, seed, key, "accuracy", row)
                shape_key = "/".join(row[k] for k in ("condition", "shape", "negative_type"))
                total, count = shape_sums.get(shape_key, (0., 0))
                n = int(row["n"])
                shape_sums[shape_key] = (total + float(row["accuracy"])*n, count+n)
            for key, (total, count) in shape_sums.items():
                add("e3_shape", model, seed, key, "accuracy", {"accuracy": total/count})

    records = []
    for (section, model, key, metric), entries in sorted(groups.items()):
        entries.sort()
        assert [entry[0] for entry in entries] == [0, 1, 2], (section, model, key, metric)
        values = [entry[1] for entry in entries]
        records.append(dict(section=section, model=model, key=key, metric=metric,
                            seed0=values[0], seed1=values[1], seed2=values[2],
                            mean=statistics.mean(values), minimum=min(values), maximum=max(values),
                            seed_range=max(values)-min(values),
                            ci_above=sum(e[2] == "above" for e in entries),
                            ci_contains=sum(e[2] == "contains" for e in entries),
                            ci_below=sum(e[2] == "below" for e in entries)))
    csv_path = OUT / f"preexperiment_deep_metrics_{DATE}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    audit_path = OUT / f"preexperiment_source_audit_{DATE}.json"
    audit_path.write_text(json.dumps({"runs": checks, "source_sha256": sources,
                                     "note": "Seed ranges are descriptive; source CIs are unadjusted pointwise intervals."},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(checks)} runs, {len(sources)} source files, {len(records)} summaries")
    print(csv_path)
    with tempfile.TemporaryDirectory(prefix=".mplcache-", dir=OUT) as cache:
        os.environ["MPLCONFIGDIR"] = cache
        make_figure(records)


def make_figure(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(r["section"], r["model"], r["key"], r["metric"]): r for r in records}
    colors = ("#3475a7", "#dd8542", "#499b78")
    labels = ("CLIP B/32", "CLIP L/14", "SigLIP2")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.45, 1, 1]})
    panels = [("e1", "clevr", "accuracy", ["object", "color", "size", "material", "relation", "binding"],
               "E1: CLEVR matching", "Accuracy", (.35, .8)),
              ("e2", "pug_animals", "same_transition_sim_diff_context", ["environment", "texture", "size", "camera"],
               "E2: PUG Animals", "Same-transition cosine", (-.03, .85)),
              ("e2", "pug_imagenet", "same_transition_sim_diff_context", ["environment", "texture", "size", "camera"],
               "E2: PUG ImageNet", "Same-transition cosine", (-.03, .85))]
    for ax, (section, dataset, metric, factors, title, ylabel, ylim) in zip(axes, panels):
        x = np.arange(len(factors))
        for i, (model, color, label) in enumerate(zip(MODELS, colors, labels)):
            rows = [lookup[(section, model, f"{dataset}/{factor}", metric)] for factor in factors]
            means = np.array([row["mean"] for row in rows])
            errors = np.array([[row["mean"]-row["minimum"] for row in rows],
                               [row["maximum"]-row["mean"] for row in rows]])
            ax.bar(x+(i-1)*.24, means, .23, color=color, label=label, zorder=3)
            ax.errorbar(x+(i-1)*.24, means, yerr=errors, fmt="none", ecolor="#263344", capsize=2, lw=1, zorder=4)
        ax.axhline(.5 if section == "e1" else 0, color="#697586", ls="--", lw=1)
        ax.set(title=title, ylabel=ylabel, ylim=ylim)
        ax.set_xticks(x, factors, rotation=28, ha="right")
        ax.grid(axis="y", color="#e1e6ec", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
    fig.suptitle("Frozen model diagnostics: 3-seed means and min-max ranges", fontsize=14, y=.98)
    fig.text(.02, .02, "Whiskers show seed ranges, not confidence intervals. E1 and E2 measure different properties.", fontsize=10, color="#475467")
    fig.tight_layout(rect=(0, .06, 1, .94))
    target = OUT / f"preexperiment_evidence_{DATE}.png"
    fig.savefig(target, dpi=170, facecolor="white")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
