"""Regression checks for review 3, revision 4081130; no model weights required.

Run: python -B code/review/preexperiment_audit_round3_checks.py
Exit 1 indicates a reproduced defect or protocol mismatch, not a script crash.
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

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from src.experiments.e1_state_diagnosis import build_clevr_items, build_spar_items
from src.features.cache import TextCache
from src.metrics import bootstrap as bs
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
                 relationships=dict(left=[[], [0]], right=[[1], []], front=[[], []], behind=[[], []]))
    items, _ = build_clevr_items([scene], ["relation"], np.random.default_rng(0))
    true_texts = {"a small red metal cube is to the left of a large blue rubber sphere",
                  "a large blue rubber sphere is to the right of a small red metal cube"}
    check("clevr_relation_labels", all(i["content_pos"] in true_texts for i in items["relation"]), {})
    duplicate = {**scene, "objects": objects + [{**objects[0], "size": "large", "material": "rubber"}]}
    items, stats = build_clevr_items([duplicate], ["binding"], np.random.default_rng(0))
    check("binding_caption_uniqueness", not items["binding"], stats["binding"])

    row = dict(sample_id="fixture", character_name="cat", character2_name="cat", world_name="desert",
               character1_pos="left", character2_pos="right", character_texture="Blue", character2_texture="Red")
    items, stats = build_spar_items(pd.DataFrame([row]), ["position", "attribute", "relation"],
                                    np.random.default_rng(0), {"position", "attribute", "relation"}, True)
    check("spar_same_species_filter", not any(items.values()), stats)

    records, D = [], []
    for f in ["texture", "camera", "size", "environment"]:
        for j in range(2):
            records.append(dict(factor=f, source_value="A", target_value="B", transition_key="A->B",
                                delta_index=len(records), context_key=f"{f}|c{j}",
                                animal=f"{f}_animal{j}", environment=f"{f}_env{j}", texture="T", size="small", camera="0"))
            D.append([1., 0.] if f in ["texture", "camera"] else [-1., 0.])
    frame, D = pd.DataFrame(records), np.asarray(D)
    groups = group_similarities(frame, "texture", D, 100, np.random.default_rng(0))
    source = groups["diff_factor_dc"]["source_counts"]
    check("different_factor_reference_uses_all_sources", set(source) == {"camera", "size", "environment"}, source)
    check("different_transition_excludes_identical_transition", not len(groups["same_factor_dt_dc"]["values"]), {})
    tt = transition_table(frame.iloc[:2], D, 2000, np.random.default_rng(0))
    check("two_context_transition_keeps_comparison", np.isclose(tt.iloc[0].mean_sim, 1.), {})

    shared = bs.cluster_gap_ci([0, 0, 1, 1], list("abcd"), [0, 0, 1, 1], list("abcd"))
    check("shared_images_are_resampled_together", np.allclose(shared, [0, 0, 0]), dict(gap_ci=shared))
    args = ([1, -1], ["center", "center"], ["other1", "other2"],
            [0, 0], ["center", "center"], ["other1", "other2"])
    forward = bs.pair_cluster_gap_ci(*args, n_boot=200)
    reverse = bs.pair_cluster_gap_ci(args[0], args[2], args[1], args[3], args[5], args[4], n_boot=200)
    check("pair_gap_endpoint_order_invariance", np.allclose(forward, reverse), dict(forward=forward, reverse=reverse))
    ng = bs.cluster_gap_ci([0, 1, 2, 3], list("abcd"), np.ones(8), [f"n{i}" for i in range(8)])
    ng_rep = bs.cluster_gap_ci(np.repeat([0, 1, 2, 3], 4), np.repeat(list("abcd"), 4),
                               np.ones(8), [f"n{i}" for i in range(8)])
    check("norm_gap_keeps_context_clustering", np.allclose(ng, ng_rep), dict(original=ng, duplicated=ng_rep))

    old_ci, captured = bs._ci, []

    def capture(boots, point, ci):
        captured.append(len(boots))
        return old_ci(boots, point, ci)

    with patch.object(bs, "_ci", side_effect=capture):
        mean_ci = bs.pair_cluster_mean_ci([.1, .5, .9], ["a", "a", "b"], ["b", "c", "c"], n_boot=1000)
    check("pair_mean_honors_configured_bootstrap_count", captured == [1000],
          dict(requested=1000, effective=captured[0], returned_ci=mean_ci))

    sparse = bs.cluster_gap_ci_equal_types([1, 1], ["a1", "a2"], ["color", "shape"],
                                          [1, 1], ["b1", "b2"], ["color", "shape"],
                                          ["color", "shape"], n_boot=1000)
    check("sparse_type_draws_are_not_zero_filled", np.allclose(sparse, [0, 0, 0]), dict(gap_ci=sparse))
    missing = bs.cluster_gap_ci_equal_types([1], ["a"], ["color"], [1], ["b"], ["color"], ["color", "shape"])
    check("missing_type_is_explicitly_nan", np.isnan(missing).all(), {})
    constant = bs.cluster_gap_ci([1, 1], ["a0", "a1"], [1, 1], ["b0", "b1"], n_boot=1000)
    check("empty_group_draws_are_rejected", np.allclose(constant, [0, 0, 0]), dict(gap_ci=constant))
    repeated = np.repeat([0.] * 50 + [1.] * 50, 4)
    pooled = bs.cluster_gap_ci(repeated, np.repeat([f"a{i}" for i in range(100)], 4),
                               repeated, np.repeat([f"b{i}" for i in range(100)], 4), n_boot=2000)
    check("pooled_gap_uses_picture_clusters", pooled[2] - pooled[1] > .2, dict(gap_ci=pooled))

    class Encoder:
        name = "audit3"
        calls = 0

        def encode_texts(self, texts, ensure_period=True):
            self.calls += 1
            eff = [t if not ensure_period or t.endswith(".") else t + "." for t in texts]
            raw = np.array([[float(len(t)), 1.] for t in eff], dtype=np.float32)
            return dict(raw=raw, normalized=raw / np.linalg.norm(raw, axis=1, keepdims=True))

    base = CODE.parent.resolve()
    with tempfile.TemporaryDirectory(prefix=".audit3_", dir=base) as tmpstr:
        tmp = Path(tmpstr).resolve()
        assert tmp.is_relative_to(base)
        cache, model = TextCache(tmp, "fixture"), Encoder()
        with_period = cache.encode(model, ["same"], ensure_period=True)[0]
        no_period = cache.encode(model, ["same"], ensure_period=False)[0]
        check("text_cache_keys_effective_text", with_period[0, 0] == 5 and no_period[0, 0] == 4,
              dict(with_period=with_period.tolist(), no_period=no_period.tolist(), calls=model.calls))

        # Only inference dependencies and input features are replaced; real task construction runs.
        fake_torch, fake_tv = types.ModuleType("torch"), types.ModuleType("torchvision")
        fake_tv.transforms = types.SimpleNamespace()
        with patch.dict(sys.modules, {"torch": fake_torch, "torchvision": fake_tv}):
            official = importlib.import_module("src.experiments.e1_spar_official")
        meta = pd.DataFrame([{**row, "row_id": i, "sample_id": f"s{i}", "character2_name": "dog",
                              "character_texture": tex, "world_name": world}
                             for i, (tex, world) in enumerate([("Default", "desert"), ("Blue", "arena")])])
        paths = dict(dirs=dict(cache=tmp / "cache", outputs=tmp / "outputs"), datasets=dict(pug_spar="fixture"))
        (tmp / "outputs/tables").mkdir(parents=True)

        def extract(model, cache, key, *args, **kwargs):
            raw = np.ones((len(meta), 2), dtype=np.float32)
            cache.save(key, [f"pug_spar_official_{i:06d}" for i in range(len(meta))], raw, raw / np.sqrt(2))

        with patch.object(official, "build_metadata", return_value=meta), \
                patch.object(official, "extract_official_features", side_effect=extract), \
                patch.object(official, "load_config", return_value={}):
            out = official.run(Encoder(), paths, logging.getLogger("audit3"))
        relation = out[(out.task == "two_objects_relation") & ~out.hardneg & (out.pos_pair == "left/right")]
        no_bg = int(relation[~relation.add_background].iloc[0].n_captions)
        with_bg = int(relation[relation.add_background].iloc[0].n_captions)
        check("official_retrieval_excludes_role_swap_distractors", no_bg == 1, dict(candidates=no_bg))
        check("official_retrieval_expands_every_background", with_bg == 2,
              dict(actual_candidates=with_bg, reference_candidates=2, worlds=sorted(meta.world_name.unique())))

    summary = dict(checks=len(results), passed=sum(r["passed"] for r in results), failed=sum(not r["passed"] for r in results))
    (Path(__file__).parent / "preexperiment_audit_round3_results.json").write_text(
        json.dumps(dict(date="2026-10-21", revision="4081130", summary=summary, results=results),
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(summary=summary)))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
