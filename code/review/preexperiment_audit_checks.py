"""Reproduce review findings without model weights or public image datasets.

Run from any directory: python -B <absolute path to this script>
PASS/FAIL refers to the intended scientific or engineering invariant.
This audit deliberately leaves the experiment implementation unchanged.
"""
import ast
import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.data.pair_matcher import build_factor_pairs
from src.experiments.e1_state_diagnosis import build_clevr_items, build_spar_items
from src.features.cache import ImageFeatureCache
from src.features.extract import extract_clevr_images
from src.metrics.bootstrap import cluster_gap_ci
from src.metrics.delta_stats import group_similarities, transition_table
from src.metrics.matching import per_template_scores
from src.utils.io import load_config


def main():
    results = []

    def record(name, passed, evidence):
        row = {"check": name, "passed": bool(passed), "evidence": evidence}
        results.append(row)
        print(json.dumps(row, ensure_ascii=True))

    pyfiles = sorted((CODE / "src").rglob("*.py")) + sorted((CODE / "scripts").glob("*.py"))
    for path in pyfiles:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    record("python_syntax", True, {"files": len(pyfiles)})

    one = np.array([[1.0, 0.0]])
    correct, margin = per_template_scores(one, one, one)
    record("2afc_ties_are_errors", not correct.any() and not margin.any(), {})

    objects = [
        dict(size="small", color="red", material="metal", shape="cube"),
        dict(size="large", color="blue", material="rubber", shape="sphere"),
    ]
    # Official CLEVR semantics: j in left[i] means object j is left of object i.
    scene = {"image_filename": "fixture.png", "objects": objects,
             "relationships": {"left": [[], [0]], "right": [[1], []],
                               "front": [[], []], "behind": [[], []]}}
    generated = build_clevr_items([scene], ["relation"], np.random.default_rng(0))["relation"]
    true_texts = {
        "a small red metal cube is to the left of a large blue rubber sphere",
        "a large blue rubber sphere is to the right of a small red metal cube",
    }
    record("clevr_relation_positive_is_true",
           all(item["content_pos"] in true_texts for item in generated), generated)

    row = dict(sample_id="same_species", character_name="cat", character2_name="cat",
               world_name="park", character1_pos="left", character2_pos="right",
               character_texture="Blue", character2_texture="Red")
    generated = build_spar_items(pd.DataFrame([row]), ["attribute", "relation"],
                                 np.random.default_rng(0), {"attribute", "relation"}, True)
    invalid = generated["attribute"] + generated["relation"]
    record("spar_ambiguous_same_species_pairs_are_rejected", len(invalid) == 0, invalid)

    from src.data.captions import spar_position_contents
    vertical = spar_position_contents({**row, "character2_name": "dog", "character1_pos": "below"})
    record("spar_vertical_relation_is_grammatical",
           not any("to the below of" in t or "to the above of" in t for t in vertical), vertical)

    items_path = CODE / "outputs/tables/e1_items_clevr_smoke.jsonl"
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line]
    repeated = {}
    for factor in sorted({it["factor"] for it in items}):
        counts = Counter(it["image_id"] for it in items if it["factor"] == factor)
        repeated[factor] = dict(items=sum(counts.values()), images=len(counts), maximum=max(counts.values()))
    record("e1_configured_one_item_per_scene_is_respected",
           all(c["maximum"] <= 1 for c in repeated.values()), repeated)

    metadata = pd.DataFrame([
        dict(row_id=i, sample_id=f"s{i}", animal="cat", environment="park",
             texture=texture, size=size, camera=0)
        for i, (size, texture) in enumerate((s, t) for s in ["small", "large"] for t in ["A", "B"])
    ])
    pairs = build_factor_pairs(metadata, "texture", {"texture": ["A", "B"]}, 4, 100, 0)
    lookup = metadata.set_index("row_id")
    fixed = ["animal", "environment", "size", "camera"]
    valid = all(lookup.loc[p.row_id_a, fixed].equals(lookup.loc[p.row_id_b, fixed])
                and lookup.loc[p.row_id_a, "texture"] != lookup.loc[p.row_id_b, "texture"]
                for p in pairs.itertuples())
    record("e2_pairs_change_one_factor", valid, {"pairs": len(pairs)})

    frame = pd.DataFrame({
        "factor": ["texture"] * 4, "source_value": ["A"] * 4, "target_value": ["B"] * 4,
        "transition_key": ["A->B"] * 4, "delta_index": np.arange(4),
        "context_key": [f"context{i}" for i in range(4)],
        "ae_key": [f"animal{i}|env" for i in range(4)],
    })
    deltas = np.ones((4, 2), dtype=np.float32)
    groups = group_similarities(frame, "texture", deltas, 100, np.random.default_rng(0))
    count = len(groups["same_factor_dt_dc"]["values"])
    record("e2_different_transition_excludes_identical_transition", count == 0,
           {"number_of_transitions": 1, "reported_different_transition_comparisons": count})

    values = [0, 0, 1, 1]
    clusters = [f"c{i}" for i in range(4)]
    gap = cluster_gap_ci(values, clusters, values, clusters, n_boot=1000, seed=0)
    record("shared_cluster_identical_groups_have_zero_gap_in_every_resample",
           np.allclose(gap, [0, 0, 0]), {"gap_and_ci": gap})

    sparse = transition_table(frame.iloc[:2], deltas[:2], 2000, np.random.default_rng(0))
    observed = float(sparse.iloc[0]["mean_sim"])
    record("two_context_transition_keeps_its_unique_comparison", np.isclose(observed, 1.0),
           {"expected_similarity": 1.0, "observed_similarity": observed if np.isfinite(observed) else None})

    rng = np.random.default_rng(0)
    counts = np.zeros(2016, dtype=int)
    # Exact random choices made for 3360 full-grid environment contexts, 4 pairs each.
    for _ in range(70 * 4 * 3 * 4):
        counts[rng.choice(2016, size=4, replace=False)] += 1
    record("environment_sampling_covers_all_2016_transitions_with_two_contexts",
           (counts >= 2).all(), {"covered": int((counts > 0).sum()), "required": 2016,
                                  "single_context": int((counts == 1).sum())})

    table = pd.read_csv(CODE / "outputs/tables/e3_cogent_smoke.csv")
    main_rows = table[table.shape_group == "cube_cylinder"]
    pooled = main_rows[main_rows.negative_type == "pooled"].set_index("condition").accuracy
    typed = main_rows[main_rows.negative_type != "pooled"]
    balanced = typed.groupby("condition").accuracy.mean()
    raw_gap, balanced_gap = float(pooled["A"] - pooled["B"]), float(balanced["A"] - balanced["B"])
    weights = typed.pivot(index="condition", columns="negative_type", values="n_samples")
    weights = weights.div(weights.sum(axis=1), axis=0)
    record("e3_conditions_use_same_negative_type_weights", np.allclose(weights.loc["A"], weights.loc["B"]),
           {"pooled_gap": raw_gap, "equal_type_weight_gap": balanced_gap,
            "type_counts": typed[["condition", "negative_type", "n_samples"]].to_dict("records")})

    combo = pd.read_csv(CODE / "outputs/tables/e3_cogent_by_combo_smoke.csv")
    record("e3_per_combo_retains_condition_and_margin",
           {"condition", "mean_margin"}.issubset(combo.columns), {"columns": combo.columns.tolist()})
    record("relation_prompt_count_is_between_five_and_seven",
           5 <= len(load_config("templates")["sentence"]) <= 7,
           {"sentence_prompt_count": len(load_config("templates")["sentence"])})

    class FakeEncoder:
        bs_img = 2
        calls = 0

        def encode_images(self, images):
            self.calls += 1
            raw = np.ones((len(images), 3), dtype=np.float32)
            return {"raw": raw, "normalized": raw / np.sqrt(3)}

    # Temporary test fixtures stay inside the workspace and are removed on exit.
    base = CODE.parent.resolve()
    with tempfile.TemporaryDirectory(prefix=".graphdelta_audit_", dir=base) as tmpstr:
        tmp = Path(tmpstr).resolve()
        assert tmp.is_relative_to(base)
        for name in ["a.png", "b.png"]:
            Image.new("RGB", (4, 4)).save(tmp / name)
        scenes = [{"image_filename": name} for name in ["a.png", "b.png"]]
        cache = ImageFeatureCache(tmp / "cache", "fixture")
        model = FakeEncoder()
        logger = logging.getLogger("audit")
        extract_clevr_images(model, cache, "fixture", scenes, tmp, logger, only_ids=["a.png"])
        extract_clevr_images(model, cache, "fixture", scenes, tmp, logger)
        ids = cache.arrays("fixture")[0].tolist()
        record("image_extraction_completes_an_existing_partial_cache", set(ids) == {"a.png", "b.png"},
               {"cached_ids": ids, "encode_calls": model.calls})

    summary = {"checks": len(results), "passed": sum(r["passed"] for r in results),
               "failed": sum(not r["passed"] for r in results)}
    print(json.dumps({"summary": summary}, ensure_ascii=True))
    (Path(__file__).parent / "preexperiment_audit_results.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
