"""渲染账本复用与特征缓存失效测试（审查问题 6）。

用假渲染器（monkeypatch _render_one）验证：账本复用必须同时匹配
spec_md5 与 render_fp；旧格式 / 换规格 / 换设置的记录作废重渲染；
rendered_states 只登记当前清单的有效产物；缓存键随图片字节变化。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from e4 import render as R
from e4.features import state_dataset_key


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


CFG = {"render": {"width": 480, "height": 320, "samples": 512,
                  "min_pixels_per_object": 200, "max_retries": 0},
       "postprocess": {"jpeg_quality": 75, "crop_scale": 0.9}}


def _fake_render_one(state, root, gp, paths, rcfg, logger,
                     spec_md5="", render_fp=""):
    sid = state["state_id"]
    img = Path(root) / "images" / f"{sid}.png"
    scn = Path(root) / "scenes" / f"{sid}.json"
    img.parent.mkdir(parents=True, exist_ok=True)
    scn.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"PNG" + sid.encode())
    scn.write_text(json.dumps({"objects": [], "directions": {},
                               "relationships": {}}))
    rec = {"state_id": sid, "seconds": 0.0, "exit_code": 0, "retries": 0,
           "validated": True, "problems": [], "spec_md5": spec_md5,
           "render_fp": render_fp, "image_bytes": img.stat().st_size}
    return rec, scn.read_text()


def _manifest(specs: dict):
    rows = [{"state_id": sid, "base_scene_id": sid.split("::")[0],
             "task": "binding", "family": "f", "state_tag": sid.split("::")[1],
             "spec": spec, "needs_render": True, "source_state_id": "",
             "postprocess": ""} for sid, spec in specs.items()]
    return {"states": pd.DataFrame(rows), "pairs": pd.DataFrame([]),
            "meta": {}}


def _paths(tmp_path, monkeypatch):
    gen = tmp_path / "gen" / "image_generation"
    for sub in ("data/base_scene.blend", "data/properties.json",
                "data/shapes", "data/materials"):
        p = gen / sub
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == "":
            p.mkdir(exist_ok=True)
        else:
            p.write_bytes(b"asset")
    paths = {"render": {"blender": str(tmp_path / "blender"),
                        "clevr_gen": str(tmp_path / "gen"),
                        "output_dir": str(tmp_path / "out")}}
    fake_driver = tmp_path / "driver.py"
    fake_driver.write_text("# driver")
    monkeypatch.setattr(R, "DRIVER", fake_driver)
    return paths


def test_ledger_reuse_requires_matching_fingerprints(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_render_one", _fake_render_one)
    paths = _paths(tmp_path, monkeypatch)
    spec = json.dumps({"objects": [{"id": "target", "pos": [0.0, 0.0]}]})
    m1 = _manifest({"s1::before": spec})
    R.render_manifest("binding", m1, paths, CFG, _Logger())
    # 同清单同设置：复用，不重渲染
    summary = R.render_manifest("binding", m1, paths, CFG, _Logger())
    assert summary["attempted"] == 0 and summary["validated_total"] == 1
    # 换规格（同 state_id）：指纹不符，必须重渲染
    m2 = _manifest({"s1::before": json.dumps({"objects": [{"id": "t", "pos": [1.0, 0.0]}]})})
    summary = R.render_manifest("binding", m2, paths, CFG, _Logger())
    assert summary["attempted"] == 1 and summary["stale_fingerprint"] >= 1
    # 换渲染设置：同样作废
    cfg2 = {"render": {**CFG["render"], "samples": 256},
            "postprocess": CFG["postprocess"]}
    summary = R.render_manifest("binding", m2, paths, cfg2, _Logger())
    assert summary["attempted"] == 1


def test_rendered_states_excludes_stale_and_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_render_one", _fake_render_one)
    paths = _paths(tmp_path, monkeypatch)
    spec_old = json.dumps({"v": 0})
    spec_new = json.dumps({"v": 1})
    # 旧清单渲染一次
    m_old = _manifest({"s1::before": spec_old, "s2::before": spec_old})
    R.render_manifest("binding", m_old, paths, CFG, _Logger())
    root = Path(paths["render"]["output_dir"]) / "binding"
    # 手工追加一条无指纹的旧格式账本行（升级前的遗留）
    with (root / "ledger.jsonl").open("a") as f:
        f.write(json.dumps({"state_id": "s3::before", "validated": True,
                            "seconds": 0.1}) + "\n")
    # 新清单（s1 改规格、删 s2、加 s3）：只有 s1/s3 属于当前清单
    m_new = _manifest({"s1::before": spec_new, "s3::before": spec_old})
    R.render_manifest("binding", m_new, paths, CFG, _Logger())
    df = pd.read_parquet(root / "rendered_states.parquet")
    assert set(df["state_id"]) == {"s1::before", "s3::before"}


def test_state_dataset_key_changes_with_image_bytes(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"old")
    meta = {"pairs_md5": "p", "states_md5": "s"}
    rendered = pd.DataFrame({"state_id": ["s1"], "image_path": [str(img)]})
    k1 = state_dataset_key("binding", meta, rendered)
    k_noimg = state_dataset_key("binding", meta)
    assert k1 != k_noimg
    img.write_bytes(b"new")          # 同清单下图片被替换
    k2 = state_dataset_key("binding", meta, rendered)
    assert k1 != k2
    meta2 = {"pairs_md5": "p2", "states_md5": "s"}
    assert k1 != state_dataset_key("binding", meta2, rendered)
