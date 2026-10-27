"""Reproduce conditional numeric-input handling gaps without model dependencies.

This probes malformed/nonfinite features, not real model output. It does not
establish that production model inference or existing feature caches are faulty.
All feature-cache writes stay inside a temporary directory that is removed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from src.experiments.common import eval_2afc_items
from src.features.cache import ImageFeatureCache, TextCache
from src.metrics.bootstrap import cluster_gap_ci
from src.metrics.delta_stats import _distinct_pairs


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else str(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


class ControlledModel:
    """Deterministic unit vectors: all red captions beat blue captions."""

    def encode_texts(self, texts, ensure_period=False):
        vectors = np.asarray(
            [[1.0, 0.0] if "red" in text else [-1.0, 0.0]
             for text in texts], dtype=np.float32)
        return {"raw": vectors, "normalized": vectors}


class ControlledTwoDimensionalText:
    def encode(self, model, texts):
        unit = np.asarray([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
        vectors = np.stack([unit if i % 2 == 0 else -unit
                            for i in range(len(texts))])
        return vectors, vectors


def main():
    items = [{"image_id": "im", "content_pos": "a red cube",
              "content_neg": "a blue cube", "kind": "phrase"}]
    results = {
        "scope": "Synthetic abnormal feature inputs; no real weights or dataset inference.",
        "classification": "Conditional numeric-input validation gap, not a demonstrated normal-input model failure.",
        "production_files_modified": False,
    }
    with tempfile.TemporaryDirectory(prefix="tmp_numeric_supplement_",
                                     dir=CODE_ROOT.parent) as temporary:
        image_cache = ImageFeatureCache(temporary, "controlled", protocol="test")
        text_cache = TextCache(temporary, "controlled", protocol="test")
        model = ControlledModel()

        finite_image = np.asarray([[1.0, 0.0]], dtype=np.float32)
        image_cache.save("finite", ["im"], finite_image, finite_image)
        baseline = eval_2afc_items(
            model, text_cache, image_cache, "finite", items,
            n_boot=20, clusters=["im"])
        assert baseline["accuracy"] == 1.0

        nan_image = np.asarray([[np.nan, 0.0]], dtype=np.float32)
        image_cache.save("nonfinite", ["im"], nan_image, nan_image)
        nan_result = eval_2afc_items(
            model, text_cache, image_cache, "nonfinite", items,
            n_boot=20, clusters=["im"])
        assert nan_result["accuracy"] == 0.0
        assert nan_result["acc_ci_low"] == nan_result["acc_ci_high"] == 0.0
        assert np.isnan(nan_result["mean_margin"])
        results["nonfinite_feature"] = {
            "input": "One image embedding [NaN, 0] with two-dimensional finite unit text embeddings.",
            "cache_save_and_reload_accepted": True,
            "finite_control_accuracy": baseline["accuracy"],
            "observed_metrics": nan_result,
            "finding": "NaN dot products become False comparisons and therefore finite accuracy 0 with CI [0, 0].",
            "source_locations": ["src/features/cache.py:66-79",
                                 "src/metrics/matching.py:13-16",
                                 "src/experiments/common.py:47-60"],
        }

        one_dimensional_image = np.asarray([[1.0]], dtype=np.float32)
        image_cache.save("dimension_mismatch", ["im"],
                         one_dimensional_image, one_dimensional_image)
        dimension_result = eval_2afc_items(
            None, ControlledTwoDimensionalText(), image_cache,
            "dimension_mismatch", items, n_boot=20, clusters=["im"])
        assert dimension_result["accuracy"] == 1.0
        assert dimension_result["mean_margin"] > 2.0
        results["feature_dimension_mismatch"] = {
            "image_shape": [1, 1], "text_shape_per_item": [2, 2],
            "both_original_inputs_unit_norm": True,
            "observed_metrics": dimension_result,
            "finding": "Image scalar dimension broadcasts to both text dimensions; the invalid comparison is accepted and margin exceeds 2.",
            "source_locations": ["src/features/cache.py:66-79",
                                 "src/experiments/common.py:41-47"],
        }

    values = np.asarray([0.0, 1.0])
    disjoint = cluster_gap_ci(values, ["factor|a", "factor|b"],
                             values, ["image_a", "image_b"],
                             n_boot=10000, seed=0)
    shared = cluster_gap_ci(values, ["factor|a", "factor|b"],
                           values, ["factor|a", "factor|b"],
                           n_boot=10000, seed=0)
    first, second = _distinct_pairs(np.random.default_rng(0), 3, 3)
    comparisons = list(zip(first.tolist(), second.tolist()))
    results["methodology_notes_not_counted_as_bugs"] = {
        "cluster_label_example": {
            "disjoint_labels": disjoint, "shared_labels": shared,
            "interpretation": "Label identity changes the CI. This example is wider for disjoint labels; it does not prove systematic undercoverage. The scheme does not define how nuisance image units should be embedded in factor-context resampling.",
            "source_locations": ["src/experiments/e2_change_diagnosis.py:318-325"],
        },
        "sampling_with_replacement": {
            "m": 3, "k": 3, "seed": 0, "comparisons": comparisons,
            "unique_unordered_comparisons": len({tuple(sorted(pair)) for pair in comparisons}),
            "interpretation": "Monte Carlo draws can repeat; k counts draws, not unique unordered comparisons. Uniform nonself draws remain unbiased for the complete pairwise mean, so this is not by itself a proven implementation bug.",
            "source_locations": ["src/metrics/delta_stats.py:50-54"],
        },
    }

    output = Path(__file__).with_name("numeric_supplement_results.json")
    output.write_text(json.dumps(jsonable(results), ensure_ascii=False,
                                 indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "nonfinite_accuracy": nan_result["accuracy"],
                      "nonfinite_acc_ci": [nan_result["acc_ci_low"], nan_result["acc_ci_high"]],
                      "mismatched_dimension_margin": dimension_result["mean_margin"],
                      "temporary_cache_removed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
