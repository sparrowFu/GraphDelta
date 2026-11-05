"""e4.summarize 单元测试：来源核验、方向检查、Holm 校正（审查问题 3、7、8）。

正式汇总只消费 run_meta 中成功提交的正式产物；smoke 文件不回退；E4-C
必须检查 C 方向；E4-D 用同场景配对差 + Holm，对照缺失记数据不足。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from e4.summarize import _holm, summarize, summarize_smoke


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _tables(tmp_path, files: dict, committed=None):
    """构造 outputs/<model>/seed_0/tables 与 run_meta；files: {名: DataFrame}。"""
    tables = tmp_path / "clip_vitl14" / "seed_0" / "tables"
    tables.mkdir(parents=True)
    for name, df in files.items():
        df.to_csv(tables / name, index=False)
    meta = {"model": "clip_vitl14", "seed": 0,
            "experiments": {} if committed is None else {
                "e4_all": {"status": "success", "started": "t",
                           "last_success": {"outputs": sorted(committed),
                                            "updated": "t"}}}}
    (tables / "run_meta.json").write_text(json.dumps(meta))
    return {"dirs": {"outputs": tmp_path}}, tables


def _e4a(p=0.001, acc=0.9, null=0.3):
    rows = []
    for task in ("binding", "relation"):
        rows.append({"task": task, "family": "__overall__", "split": "heldout",
                     "n_scenes": 8, "pair_acc": acc, "perm_p": p,
                     "null_mean": null})
    return pd.DataFrame(rows)


def _e4c(c_mean=0.5, p=0.002, n_fam={"binding": 8, "relation": 10}):
    rows = []
    for task, n in n_fam.items():
        for i in range(n):
            rows.append({"task": task, "family": f"f{i}",
                         "p_heldout": np.nan})
        rows.append({"task": task, "family": "__overall__",
                     "n_conditions": n, "n_heldout": 4,
                     "C_mean_heldout": c_mean, "C_ci_low_heldout": c_mean - 0.1,
                     "C_ci_high_heldout": c_mean + 0.1,
                     "p_heldout": p, "n_perm": 16, "exact": True,
                     "direction_positive": c_mean > 0})
    return pd.DataFrame(rows)


def _e4d(correct=1.0, ctrl=0.0, n_scenes=8, with_controls=True,
         tasks=("binding", "relation")):
    rows = []
    methods = (["alpha0", "correct", "wrong_conversion", "random", "reverse",
                "text_delta"] if with_controls else ["alpha0", "correct"])
    for task in tasks:
        for s in range(n_scenes):
            for k, m in enumerate(methods):
                for rep in ([0, 1] if m == "random" else [None]):
                    r = {"task": task, "family": "fam", "scene": f"s{s}",
                         "method": m,
                         "top1": correct if m == "correct" else ctrl,
                         "alpha": 0.1, "pool_size": 5, "pool": "a,b,c"}
                    if rep is not None:
                        r["random_k"] = rep
                    rows.append(r)
    return pd.DataFrame(rows)


def test_holm_stepdown():
    adj = _holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert abs(adj["a"]["p_holm"] - 0.03) < 1e-12
    assert abs(adj["c"]["p_holm"] - 0.06) < 1e-12
    assert abs(adj["b"]["p_holm"] - 0.06) < 1e-12
    # 校正后单调不减、不超过 1
    vals = [v["p_holm"] for v in adj.values()]
    assert all(0 <= v <= 1 for v in vals)


def test_summary_rejects_smoke_fallback(tmp_path):
    """只有 smoke 文件时正式汇总不得读取（审查问题 7）。"""
    tables = tmp_path / "clip_vitl14" / "seed_0" / "tables"
    tables.mkdir(parents=True)
    _e4a().to_csv(tables / "e4a_results_smoke.csv", index=False)
    (tables / "run_meta.json").write_text(json.dumps(
        {"experiments": {"e4_all_smoke": {"status": "success",
          "last_success": {"outputs": ["e4a_results_smoke.csv"]}}}}))
    decision = summarize("clip_vitl14", 0, {"dirs": {"outputs": tmp_path}},
                         _Logger())
    assert any("e4a_results.csv" in m for m in decision["missing"])
    assert decision["structure_identifiability"] == {}


def test_crossmodal_direction_guard(tmp_path):
    """C 显著为负（p<α）不能判有支持证据（审查问题 3）。"""
    paths, tables = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(),
         "e4c_conversion_stats.csv": _e4c(c_mean=-1.0, p=0.0078),
         "e4d_retrieval.csv": _e4d()},
        committed=["e4a_results.csv", "e4c_conversion_stats.csv",
                   "e4d_retrieval.csv"])
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        e = decision["crossmodal_alignment"][task]
        assert e["C_mean_heldout"] < 0 and e["p_heldout"] < 0.05
        assert e["label"] == "证据不足"
    # 正负分开写：A/D 正常数据不受 C 反向影响
    assert decision["structure_identifiability"]["binding"]["label"] == "有支持证据"


def test_transfer_empty_controls_is_insufficient(tmp_path):
    """对照缺失时 all(空) 不得判支持（审查问题 8）。"""
    paths, tables = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(),
         "e4c_conversion_stats.csv": _e4c(),
         "e4d_retrieval.csv": _e4d(with_controls=False)},
        committed=["e4a_results.csv", "e4c_conversion_stats.csv",
                   "e4d_retrieval.csv"])
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        e = decision["direction_transfer"][task]
        assert e["label"] == "数据不足"
        assert e["missing_controls"]
    assert any("对照未覆盖" in m for m in decision["missing"])


def test_transfer_paired_holm_supported(tmp_path):
    """正确方向显著优于全部规定对照 → 有支持证据（Holm 后）。"""
    paths, tables = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(),
         "e4c_conversion_stats.csv": _e4c(),
         "e4d_retrieval.csv": _e4d(n_scenes=9)},
        committed=["e4a_results.csv", "e4c_conversion_stats.csv",
                   "e4d_retrieval.csv"])
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        e = decision["direction_transfer"][task]
        # 9 场景全 1 vs 0：符号翻转精确 p = 2/512；Holm 族共 12 个检验
        #（A2 + C2 + D8），D 的校正 p = 8×2/512 < 0.05
        assert e["label"] == "有支持证据"
        v = e["controls"]["random"]
        assert abs(v["p_signflip"] - 2 / 512) < 1e-9
        assert v["p_holm"] < 0.05
        assert v["mean_diff"] == 1.0 and v["diff_ci_low"] == 1.0


def test_missing_relation_task_reported(tmp_path):
    """e4c 缺 relation 行：任务级缺失逐项列出，该轴记数据不足。"""
    paths, tables = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(),
         "e4c_conversion_stats.csv": _e4c(n_fam={"binding": 8}),
         "e4d_retrieval.csv": _e4d(tasks=("binding",))},
        committed=["e4a_results.csv", "e4c_conversion_stats.csv",
                   "e4d_retrieval.csv"])
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    assert any("e4c:relation" in m for m in decision["missing"])
    assert any("e4d:relation" in m for m in decision["missing"])
    assert decision["crossmodal_alignment"]["relation"]["label"] == "数据不足"
    assert decision["direction_transfer"]["relation"]["label"] == "数据不足"


def test_smoke_summary_separate_files_only(tmp_path):
    tables = tmp_path / "clip_vitl14" / "seed_0" / "tables"
    tables.mkdir(parents=True)
    _e4a().to_csv(tables / "e4a_results_smoke.csv", index=False)
    out = summarize_smoke("clip_vitl14", 0, {"dirs": {"outputs": tmp_path}},
                          _Logger())
    assert out["axes"]["a"]["ran"] and out["axes"]["a"]["n_rows"] == 2
    assert out["axes"]["c"]["ran"] is False
    assert (tables / "e4_summary_smoke.csv").exists()
    assert (tables / "e4_stage_decision_smoke.json").exists()
    assert not (tables / "e4_stage_decision.json").exists()
    # smoke 汇总不产出机制支持标签
    assert "label" not in json.dumps(out)
