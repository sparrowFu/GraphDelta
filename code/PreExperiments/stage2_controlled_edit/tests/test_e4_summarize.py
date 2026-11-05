"""e4.summarize 单元测试：来源核验、方向检查、Holm 校正、完整性前置
（审查问题 3、7、8 及完整性/摘要两处修复）。

正式汇总只消费 run_meta 中成功提交且内容摘要相符的正式产物；smoke 文件
不回退；E4-C 必须检查 C 方向；E4-D 用同场景配对差 + Holm，任一规定对照、
规定转换或清单规定留出场景缺失即记数据不足；Holm 族规模冻结。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

import e4.summarize as S
from e4.pairs import family_of_binding, family_of_relation
from e4.summarize import _holm, summarize, summarize_smoke
from generation.scene_spec import (BINDING_CONVERSIONS,
                                   RELATION_TYPE_PAIRS_TRAIN)
from src.utils.io import md5_file

FAM_NAMES = {"binding": [family_of_binding(c) for c in BINDING_CONVERSIONS],
             "relation": [family_of_relation(p)
                          for p in RELATION_TYPE_PAIRS_TRAIN]}


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _tables(tmp_path, files: dict, committed=None):
    """构造 outputs/<model>/seed_0/tables 与 run_meta；files: {名: DataFrame}。

    committed 不为 None 时按 e4_all 成功提交登记，含逐产物内容摘要
    （outputs_md5），与 commit_experiment 的写出格式一致。
    """
    tables = tmp_path / "clip_vitl14" / "seed_0" / "tables"
    tables.mkdir(parents=True)
    for name, df in files.items():
        df.to_csv(tables / name, index=False)
    meta = {"model": "clip_vitl14", "seed": 0,
            "experiments": {} if committed is None else {
                "e4_all": {"status": "success", "started": "t",
                           "last_success": {
                               "outputs": sorted(committed), "updated": "t",
                               "outputs_md5": {n: md5_file(tables / n)
                                               for n in committed}}}}}
    (tables / "run_meta.json").write_text(json.dumps(meta))
    return {"dirs": {"outputs": tmp_path,
                     "edit_pairs": tmp_path / "edit_pairs"}}, tables


def _e4a(p=0.001, acc=0.9, null=0.3, drop_families=()):
    """E4-A 夹具：每任务全部规定转换的 heldout 行 + 任务级 __overall__ 行。"""
    rows = []
    for task in ("binding", "relation"):
        for fam in FAM_NAMES[task]:
            if fam in drop_families and task == "binding":
                continue
            rows.append({"task": task, "family": fam, "split": "heldout",
                         "n_scenes": 1, "pair_acc": acc})
        rows.append({"task": task, "family": "__overall__", "split": "heldout",
                     "n_scenes": 8, "pair_acc": acc, "perm_p": p,
                     "null_mean": null})
    return pd.DataFrame(rows)


def _e4c(c_mean=0.5, p=0.002, n_fam=None):
    """E4-C 夹具：规定转换名的家族行（n_heldout 齐全）+ __overall__ 行。"""
    n_fam = n_fam or {"binding": 8, "relation": 10}
    rows = []
    for task, n in n_fam.items():
        for fam in FAM_NAMES[task][:n]:
            rows.append({"task": task, "family": fam, "p_heldout": np.nan,
                         "n_heldout": 2})
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
    # 文件不可消费时该轴显式记数据不足（不留空）
    assert all(e["label"] == "数据不足"
               for e in decision["structure_identifiability"].values())
    assert set(decision["structure_identifiability"]) == {"binding", "relation"}


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
    assert out["axes"]["a"]["ran"] and out["axes"]["a"]["n_rows"] == len(_e4a())
    assert out["axes"]["c"]["ran"] is False
    assert (tables / "e4_summary_smoke.csv").exists()
    assert (tables / "e4_stage_decision_smoke.json").exists()
    assert not (tables / "e4_stage_decision.json").exists()
    # smoke 汇总不产出机制支持标签
    assert "label" not in json.dumps(out)


_COMMITTED_ALL = ["e4a_results.csv", "e4c_conversion_stats.csv",
                  "e4d_retrieval.csv"]


def test_partial_controls_insufficient_and_frozen_holm_family(tmp_path):
    """缺任一规定对照（哪怕只剩一个对照可用）即数据不足；Holm 族保持
    冻结规模 12，缺失的检验按 p=1 计入（computed=False），不因缺数据
    缩小族、放宽其余检验的校正。"""
    d = _e4d(n_scenes=9)
    d = d[~d["method"].isin(["wrong_conversion", "random", "text_delta"])]
    paths, _ = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(), "e4c_conversion_stats.csv": _e4c(),
         "e4d_retrieval.csv": d},
        committed=_COMMITTED_ALL)
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        e = decision["direction_transfer"][task]
        assert e["label"] == "数据不足"
        assert set(e["missing_controls"]) == {"wrong_conversion", "random",
                                              "text_delta"}
        assert e["controls"] == {}
    assert len(decision["holm_family"]) == 12
    # 不完整任务的 D 检验全部不进入确认族（按 p=1 填充，computed=False）
    uncomputed = {h["key"] for h in decision["holm_family"]
                  if not h["computed"]}
    assert uncomputed == {f"D:{t}:{m}" for t in ("binding", "relation")
                          for m in ("wrong_conversion", "random",
                                    "reverse", "text_delta")}
    assert all(h["p_raw"] == 1.0 and h["p_holm"] == 1.0
               for h in decision["holm_family"] if not h["computed"])
    # 完整的轴不受影响
    assert decision["structure_identifiability"]["binding"]["label"] == "有支持证据"
    assert decision["crossmodal_alignment"]["binding"]["label"] == "有支持证据"


def test_e4c_incomplete_conversions_insufficient(tmp_path):
    """E4-C 转换不齐全（1/8）：只登记 missing 不够，必须判数据不足；
    转换齐全的任务正常判定。"""
    paths, _ = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(),
         "e4c_conversion_stats.csv": _e4c(n_fam={"binding": 1, "relation": 10}),
         "e4d_retrieval.csv": _e4d()},
        committed=_COMMITTED_ALL)
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    assert decision["crossmodal_alignment"]["binding"]["label"] == "数据不足"
    assert decision["crossmodal_alignment"]["relation"]["label"] == "有支持证据"
    assert any("e4c:binding:转换留出数据缺失" in m for m in decision["missing"])


def test_e4a_missing_family_insufficient(tmp_path):
    """E4-A 缺规定转换的 heldout 行：数据不足，不能凭 __overall__ 行判支持。"""
    paths, _ = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(drop_families=FAM_NAMES["binding"][:2]),
         "e4c_conversion_stats.csv": _e4c(),
         "e4d_retrieval.csv": _e4d()},
        committed=_COMMITTED_ALL)
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    assert decision["structure_identifiability"]["binding"]["label"] == "数据不足"
    assert decision["structure_identifiability"]["relation"]["label"] == "有支持证据"
    assert any("e4a:binding:转换 heldout 行缺失" in m for m in decision["missing"])


def test_shared_file_rejected_when_overwritten(tmp_path):
    """共享输出路径被中断的单阶段重跑覆盖：内容与 e4_all 的提交摘要不符，
    必须拒绝消费（即使 e4_all 仍是 success、e4_a 只是 running）。"""
    paths, tables = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(), "e4c_conversion_stats.csv": _e4c(),
         "e4d_retrieval.csv": _e4d(n_scenes=9)},
        committed=_COMMITTED_ALL)
    # 模拟：重跑 e4_a 写出了新的 e4a_results.csv 后中断（未提交）
    a = pd.read_csv(tables / "e4a_results.csv")
    a.loc[a["family"] == "__overall__", "pair_acc"] = 0.42
    a.to_csv(tables / "e4a_results.csv", index=False)
    meta = json.loads((tables / "run_meta.json").read_text())
    meta["experiments"]["e4_a"] = {"status": "running", "started": "t2"}
    (tables / "run_meta.json").write_text(json.dumps(meta))
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    assert any("摘要" in m and "e4a_results.csv" in m
               for m in decision["missing"])
    assert all(e["label"] == "数据不足"
               for e in decision["structure_identifiability"].values())
    assert "e4a_results.csv" not in decision["provenance"]
    # 其余产物未被覆盖，正常判定
    assert decision["crossmodal_alignment"]["binding"]["label"] == "有支持证据"


def test_scene_coverage_against_frozen_manifest(tmp_path, monkeypatch):
    """留出场景覆盖与冻结清单逐一核对：e4d 少清单规定场景、e4a 的
    n_scenes 少于清单规定数、e4c 各转换场景数不足，对应轴都记数据不足。"""
    def _fake_manifest(d, t):
        fams = FAM_NAMES[t]
        rows = [{"base_scene_id": f"t{i}", "family": fams[i % len(fams)],
                 "edit_kind": "swap", "split": "train"} for i in range(6)]
        for i in range(len(fams) + 1):     # 每家族 1 个留出 + 1 个多余
            rows.append({"base_scene_id": f"s{i}", "family": fams[i % len(fams)],
                         "edit_kind": "swap", "split": "heldout"})
        return {"pairs": pd.DataFrame(rows)}

    monkeypatch.setattr(S, "load_manifest", _fake_manifest)
    paths, _ = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(), "e4c_conversion_stats.csv": _e4c(),
         "e4d_retrieval.csv": _e4d(n_scenes=8)},   # 只覆盖 s0..s7
        committed=_COMMITTED_ALL)
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        assert decision["structure_identifiability"][task]["label"] == "数据不足"
        assert decision["direction_transfer"][task]["label"] == "数据不足"
    assert any("留出场景覆盖不足 8/9" in m for m in decision["missing"])
    assert any("清单规定留出场景缺失 1 个" in m for m in decision["missing"])


def test_e4c_partial_scenes_insufficient(tmp_path, monkeypatch):
    """转换全部出现、但每个转换只有部分留出场景（渲染不完整被静默剔除）：
    E4-C 记数据不足；场景数与清单一致时正常判定。"""
    def _manifest(n_per_fam):
        def _f(d, t):
            rows = []
            for i, fam in enumerate(FAM_NAMES[t]):
                for k in range(n_per_fam):
                    rows.append({"base_scene_id": f"s{i}_{k}", "family": fam,
                                 "edit_kind": "swap", "split": "heldout"})
            return {"pairs": pd.DataFrame(rows)}
        return _f

    files = {"e4a_results.csv": _e4a(),
             "e4c_conversion_stats.csv": _e4c(),   # 每转换 n_heldout=2
             "e4d_retrieval.csv": _e4d()}
    monkeypatch.setattr(S, "load_manifest", _manifest(3))
    paths, _ = _tables(tmp_path, files, committed=_COMMITTED_ALL)
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        e = decision["crossmodal_alignment"][task]
        assert e["label"] == "数据不足"
        assert "cond_raw" not in e          # 被拦截，不进入确认性判定
    assert any("e4c:binding:留出场景覆盖不足" in m for m in decision["missing"])
    assert any("e4c:relation:留出场景覆盖不足" in m for m in decision["missing"])
    # 场景数与清单一致（每转换 2 个）→ 正常判定
    monkeypatch.setattr(S, "load_manifest", _manifest(2))
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        e = decision["crossmodal_alignment"][task]
        assert e["label"] == "有支持证据" and "cond_raw" in e


def test_e4c_missing_heldout_column_no_crash(tmp_path, monkeypatch):
    """整份 e4c CSV 无 n_heldout 列（全部转换无留出数据）：不崩溃，
    两任务记数据不足，逐项列出转换缺失。"""
    def _fake_manifest(d, t):
        return {"pairs": pd.DataFrame(
            [{"base_scene_id": f"s{i}", "family": fam, "edit_kind": "swap",
              "split": "heldout"}
             for i, fam in enumerate(FAM_NAMES[t])])}

    monkeypatch.setattr(S, "load_manifest", _fake_manifest)
    no_held = pd.DataFrame([{"task": t, "family": fam}
                            for t in ("binding", "relation")
                            for fam in FAM_NAMES[t]])
    paths, _ = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(), "e4c_conversion_stats.csv": no_held,
         "e4d_retrieval.csv": _e4d()},
        committed=_COMMITTED_ALL)
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    for task in ("binding", "relation"):
        e = decision["crossmodal_alignment"][task]
        assert e["label"] == "数据不足" and "cond_raw" not in e
    assert any("e4c:binding:转换留出数据缺失" in m for m in decision["missing"])


def test_newer_committed_single_stage_accepted(tmp_path):
    """e4_all 成功 → e4_a 成功重跑且内容变化并提交新摘要 → 汇总接受
    e4_a 的新提交；旧 e4_all 摘要不匹配不得提前拒绝整个文件。"""
    paths, tables = _tables(
        tmp_path,
        {"e4a_results.csv": _e4a(), "e4c_conversion_stats.csv": _e4c(),
         "e4d_retrieval.csv": _e4d(n_scenes=9)},
        committed=_COMMITTED_ALL)
    a = pd.read_csv(tables / "e4a_results.csv")
    a.loc[a["family"] == "__overall__", "pair_acc"] = 0.55
    a.to_csv(tables / "e4a_results.csv", index=False)
    meta = json.loads((tables / "run_meta.json").read_text())
    meta["experiments"]["e4_a"] = {
        "status": "success", "started": "t2",
        "last_success": {"outputs": ["e4a_results.csv"], "updated": "t2",
                         "outputs_md5": {"e4a_results.csv":
                                         md5_file(tables / "e4a_results.csv")}}}
    (tables / "run_meta.json").write_text(json.dumps(meta))
    decision = summarize("clip_vitl14", 0, paths, _Logger())
    assert decision["provenance"]["e4a_results.csv"] == "e4_a"
    assert not any("e4a_results.csv" in m for m in decision["missing"])
    for task in ("binding", "relation"):
        assert decision["structure_identifiability"][task]["label"] == "有支持证据"
