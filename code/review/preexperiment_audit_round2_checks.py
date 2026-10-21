"""Second review checks for revision 224f37b; no model weights are required.

Run: python -B code/review/preexperiment_audit_round2_checks.py
Failed checks reproduce defects or protocol differences; source files are unchanged.
"""
import ast
import importlib
import json
import logging
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.data.captions import spar_position_contents, templates
from src.experiments.e1_state_diagnosis import build_clevr_items, build_spar_items
from src.features.cache import ImageFeatureCache, TextCache
from src.features.extract import extract_clevr_images
from src.metrics.bootstrap import (cluster_gap_ci, cluster_gap_ci_equal_types,
                                   cluster_mean_ci, diff_proportions_ci, mean_diff_ci)
from src.metrics.delta_stats import group_similarities, transition_table


def main():
    results = []

    def check(name, passed, evidence):
        row = dict(check=name, passed=bool(passed), evidence=evidence)
        results.append(row)
        print(json.dumps(row, ensure_ascii=True))

    files = sorted((CODE / "src").rglob("*.py")) + sorted((CODE / "scripts").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    check("source_syntax", True, dict(files=len(files)))

    objects = [dict(size="small", color="red", material="metal", shape="cube"),
               dict(size="large", color="blue", material="rubber", shape="sphere")]
    scene = dict(image_filename="fixture.png", objects=objects,
                 relationships=dict(left=[[], [0]], right=[[1], []],
                                    front=[[], []], behind=[[], []]))
    items, stats = build_clevr_items([scene], ["relation"], np.random.default_rng(0))
    true_texts = {"a small red metal cube is to the left of a large blue rubber sphere",
                  "a large blue rubber sphere is to the right of a small red metal cube"}
    check("fixed_clevr_relation_labels", all(i["content_pos"] in true_texts for i in items["relation"]),
          dict(items=len(items["relation"]), filter_stats_present=bool(stats)))

    row = dict(sample_id="fixture", character_name="cat", character2_name="cat",
               world_name="park", character1_pos="left", character2_pos="right",
               character_texture="Blue", character2_texture="Red")
    factors = ["position", "attribute", "relation"]
    items, stats = build_spar_items(pd.DataFrame([row]), factors, np.random.default_rng(0), set(factors), True)
    check("fixed_spar_same_species_filter", not any(items.values()), stats)
    vertical = spar_position_contents({**row, "character2_name": "dog", "character1_pos": "below"})
    check("fixed_vertical_phrases", "to the below of" not in vertical[0], vertical)
    check("fixed_prompt_counts", all(5 <= len(v) <= 7 for v in templates().values()),
          {k: len(v) for k, v in templates().items()})

    cells = {}
    for ds in ["clevr", "spar"]:
        path = CODE / f"outputs/tables/e1_items_{ds}_smoke.jsonl"
        data = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]
        for f in sorted({i["factor"] for i in data}):
            selected = [i for i in data if i["factor"] == f]
            cells[f"{ds}/{f}"] = [len(selected), len({i["image_id"] for i in selected})]
    check("fixed_e1_one_item_per_image", all(n == m for n, m in cells.values()), cells)

    frame_rows, D = [], []
    for factor in ["texture", "camera", "size", "environment"]:
        for j in range(2):
            frame_rows.append(dict(factor=factor, source_value="A", target_value="B",
                                   transition_key="A->B", delta_index=len(frame_rows),
                                   context_key=f"{factor}|c{j}", cluster_key=f"{factor}|c{j}",
                                   animal=f"{factor}_animal{j}", environment=f"{factor}_env{j}",
                                   texture="A->B" if factor == "texture" else "T",
                                   size="A->B" if factor == "size" else "small",
                                   camera="A->B" if factor == "camera" else "0"))
            D.append([1., 0.] if factor in ["texture", "camera"] else [-1., 0.])
    frame, D = pd.DataFrame(frame_rows), np.asarray(D)
    g = group_similarities(frame, "texture", D, 100, np.random.default_rng(0))
    check("fixed_same_factor_different_transition_filter", len(g["same_factor_dt_dc"]["values"]) == 0,
          dict(comparisons=len(g["same_factor_dt_dc"]["values"])))
    sparse = transition_table(frame.iloc[:2], D, 2000, np.random.default_rng(0))
    check("fixed_two_context_comparison", np.isclose(sparse.iloc[0].mean_sim, 1.),
          dict(mean_sim=float(sparse.iloc[0].mean_sim)))
    same = cluster_gap_ci([0, 0, 1, 1], list("abcd"), [0, 0, 1, 1], list("abcd"))
    check("fixed_shared_cluster_gap", np.allclose(same, [0, 0, 0]), dict(gap_ci=same))

    observed = float(np.mean(g["diff_factor_dc"]["values"]))
    check("e2_reference_retains_all_other_factors", np.isclose(observed, -1/3, atol=.25),
          dict(observed_mean=observed, equal_size_reference_mean=-1/3,
               comparisons=len(g["diff_factor_dc"]["values"])))

    v = np.array([0.] * 100 + [1.] * 100)
    gap = diff_proportions_ci(v, v, n_boot=10000, seed=0)
    check("e1_shared_image_factor_gap_uses_shared_resampling", np.allclose(gap, [0, 0, 0]),
          dict(current_gap_ci=gap, expected=[0, 0, 0]))

    a = cluster_mean_ci([1, -1], ["center", "center"], n_boot=5000)
    b = cluster_mean_ci([1, -1], ["other1", "other2"], n_boot=5000)
    check("e2_similarity_ci_keeps_both_endpoints", np.allclose(a, b),
          dict(first_endpoint_center_ci=a, reversed_pair_endpoint_ci=b,
               note="Same unordered cosine pairs; only arbitrary endpoint order changed."))

    ng1 = mean_diff_ci([0, 1, 2, 3], np.ones(8), n_boot=5000, seed=0)
    ng2 = mean_diff_ci(np.repeat([0, 1, 2, 3], 4), np.ones(8), n_boot=5000, seed=0)
    check("e2_norm_gap_does_not_treat_context_duplicates_as_independent", np.allclose(ng1, ng2),
          dict(one_pair_per_context_ci=ng1, four_identical_pairs_per_context_ci=ng2))

    # Normal-size check: type imbalance must not alter the equal-type statistic.
    ac = [1.] * 80 + [0.] * 20
    bc = [1.] * 20 + [0.] * 80
    ta, tb = ["color"] * 80 + ["shape"] * 20, ["color"] * 20 + ["shape"] * 80
    eq = cluster_gap_ci_equal_types(ac, [f"a{i}" for i in range(100)], ta,
                                    bc, [f"b{i}" for i in range(100)], tb,
                                    ["color", "shape"], n_boot=1000)
    check("fixed_e3_equal_type_main_gap", np.allclose(eq, [0, 0, 0]), dict(gap_ci=eq))
    combo = pd.read_csv(CODE / "outputs/tables/e3_cogent_by_combo_smoke.csv")
    expected_cols = {"condition", "shape_group", "color", "shape", "negative_type", "mean_margin"}
    check("fixed_e3_combo_dimensions", expected_cols.issubset(combo.columns), dict(columns=combo.columns.tolist()))

    sparse_gap = cluster_gap_ci_equal_types([1, 1], ["a1", "a2"], ["color", "shape"],
                                           [1, 1], ["b1", "b2"], ["color", "shape"],
                                           ["color", "shape"], n_boot=10000)
    disjoint = cluster_gap_ci([1, 1], ["a0", "a1"], [1, 1], ["b0", "b1"], n_boot=10000)
    check("sparse_bootstrap_does_not_fabricate_zero_for_empty_group",
          np.allclose(sparse_gap, [0, 0, 0]) and np.allclose(disjoint, [0, 0, 0]),
          dict(equal_type_gap_ci=sparse_gap, disjoint_cluster_gap_ci=disjoint))

    repeated = np.repeat([0.] * 50 + [1.] * 50, 4)
    pooled = diff_proportions_ci(repeated, repeated, n_boot=5000)
    by_image = diff_proportions_ci(repeated[::4], repeated[::4], n_boot=5000)
    check("e3_pooled_sensitivity_ci_clusters_images", np.allclose(pooled, by_image),
          dict(current_item_ci=pooled, image_cluster_equivalent_ci=by_image))

    class Encoder:
        name = "audit2"
        calls = 0
        bs_img = 2

        def encode_images(self, images):
            raw = np.ones((len(images), 2), dtype=np.float32)
            return {"raw": raw, "normalized": raw / np.sqrt(2)}

        def encode_texts(self, texts, ensure_period=True):
            self.calls += 1
            raw = np.full((len(texts), 2), 1. if ensure_period else 2., dtype=np.float32)
            return {"raw": raw, "normalized": raw / np.linalg.norm(raw, axis=1, keepdims=True)}

    base = CODE.parent.resolve()
    with tempfile.TemporaryDirectory(prefix=".audit2_", dir=base) as tempstr:
        temp = Path(tempstr).resolve()
        assert temp.is_relative_to(base)
        c, m = TextCache(temp, "fixture"), Encoder()
        c.encode(m, ["same caption"], ensure_period=True)
        after = c.encode(m, ["same caption"], ensure_period=False)[0]
        check("text_cache_distinguishes_period_mode", np.allclose(after, 2.),
              dict(observed=after.tolist(), expected=2., encoder_calls=m.calls))

        for name in ["a.png", "b.png"]:
            Image.new("RGB", (4, 4)).save(temp / name)
        fixtures = [dict(image_filename=name) for name in ["a.png", "b.png"]]
        icache = ImageFeatureCache(temp / "incremental", "fixture")
        extract_clevr_images(Encoder(), icache, "fixture", fixtures, temp,
                             logging.getLogger("audit2"), only_ids=["a.png"])
        extract_clevr_images(Encoder(), icache, "fixture", fixtures, temp, logging.getLogger("audit2"))
        ids = icache.arrays("fixture")[0].tolist()
        check("fixed_incremental_image_cache", set(ids) == {"a.png", "b.png"}, dict(cached_ids=ids))

        # Stub only unavailable imports. Actual protocol task construction and metrics run below.
        fake_torch, fake_tv = types.ModuleType("torch"), types.ModuleType("torchvision")
        fake_tv.transforms = types.SimpleNamespace()
        with patch.dict(sys.modules, {"torch": fake_torch, "torchvision": fake_tv}):
            official = importlib.import_module("src.experiments.e1_spar_official")
        fixture = pd.DataFrame([{**row, "row_id": i, "sample_id": f"s{i}",
                                 "character2_name": "dog", "character_texture": tex}
                                for i, tex in enumerate(["Default", "Blue"])])
        paths = dict(dirs=dict(cache=temp / "cache", outputs=temp / "outputs"),
                     datasets=dict(pug_spar="fixture"))
        (temp / "outputs/tables").mkdir(parents=True)

        def extract(model, cache, key, *args, **kwargs):
            raw = np.ones((len(fixture), 2), dtype=np.float32)
            cache.save(key, [f"pug_spar_official_{i:06d}" for i in range(len(fixture))],
                       raw, raw / np.sqrt(2))

        with patch.object(official, "build_metadata", return_value=fixture), \
                patch.object(official, "extract_official_features", side_effect=extract), \
                patch.object(official, "load_config", return_value={}):
            table = official.run(Encoder(), paths, logging.getLogger("audit2"))
        select = table[(table.task == "two_objects_relation") & ~table.hardneg & ~table.add_background
                       & (table.pos_pair == "left/right")]
        count = int(select.iloc[0].n_captions)
        check("official_retrieval_candidates_only_use_metadata_positive_captions", count == 1,
              dict(actual_candidates=count, reference_positive_candidates=1))

    summary = dict(checks=len(results), passed=sum(r["passed"] for r in results),
                   failed=sum(not r["passed"] for r in results))
    output = dict(date="2026-10-21", revision="224f37b", summary=summary, results=results)
    (Path(__file__).parent / "preexperiment_audit_round2_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(summary=summary)))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
