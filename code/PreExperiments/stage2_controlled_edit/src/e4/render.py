"""规格驱动渲染调度（阶段二 Step 3 的核心逻辑，脚本 render_edit_pairs.py 调用）。

流程：读清单 → 展开唯一渲染状态 → 查账本跳过已完成（须同时匹配场景
规格 spec_md5 与渲染设置 / 资产指纹 render_fp，审查问题 6）→ 子进程调
Blender（blend_render_driver.py）→ 用渲染回填的场景图复核规格 → 回填
rendered_states.parquet（只登记当前清单对应的有效产物）→ 账本登记
（状态数 / 秒 / 重试 / 字节数）→ 生成 jpeg / crop 后处理对照图。

账本口径（工作方案第 8 节）：唯一状态是渲染单位；吞吐含校验与重试的
重试不重复计费；--force 重渲染已完成状态；--limit 只渲前 N 态（小批试运行）。
"""
import io
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from generation.scene_spec import (
    CAMERA_FRONT_VEC, CAMERA_LEFT_VEC, REL_EPS, SIZES,
)

DRIVER = Path(__file__).resolve().parents[2] / "render" / "blend_render_driver.py"
BLENDER_TIMEOUT = 900  # 秒；单态标称约 39 秒，留足余量
EXIT_VISIBILITY = 3    # 驱动约定：可见性检查失败（确定性，不重试）


def _gen_paths(paths: dict) -> dict:
    gen = Path(paths["render"]["clevr_gen"]) / "image_generation"
    return {"base": gen / "data" / "base_scene.blend",
            "props": gen / "data" / "properties.json",
            "shapes": gen / "data" / "shapes",
            "mats": gen / "data" / "materials"}


def _spec_md5(spec: str) -> str:
    import hashlib
    return hashlib.md5(spec.encode("utf-8")).hexdigest()


def _asset_fingerprint(paths: dict, rcfg: dict) -> str:
    """渲染设置与资产指纹（审查问题 6）。

    覆盖：渲染参数、Blender 与生成器路径、本驱动脚本、base_scene.blend、
    properties.json 及 shapes/materials 全部资产文件。账本复用必须同时
    匹配场景规格（spec_md5）与本指纹；任一变化即视为不同产物，不复用
    旧图。state_id 不含 seed / 规格指纹，单凭它判断"已完成"会让换种子
    或改规格后的新清单直接跳过渲染并沿用旧场景。
    """
    import hashlib
    gp = _gen_paths(paths)
    h = hashlib.md5()

    def _add_file(p):
        p = Path(p)
        h.update(str(p).encode("utf-8"))
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)

    h.update(json.dumps({k: rcfg.get(k) for k in sorted(rcfg)},
                        sort_keys=True).encode("utf-8"))
    h.update(str(paths["render"]["blender"]).encode("utf-8"))
    h.update(str(paths["render"]["clevr_gen"]).encode("utf-8"))
    _add_file(DRIVER)
    for p in (gp["base"], gp["props"]):
        _add_file(p)
    for d in (gp["shapes"], gp["mats"]):
        for f in sorted(Path(d).iterdir()):
            if f.is_file():
                _add_file(f)
    return h.hexdigest()


def expected_relationships(spec: dict, eps: float = REL_EPS) -> dict:
    """按规格（2D 位置 + 常量方向轴）推关系矩阵，与官方 compute_all_relationships
    同口径：output[rel][i] 列出 j 满足 dot(pos_j − pos_i, dir_rel) > eps。"""
    objs = spec["objects"]
    pos = [(o["pos"][0], o["pos"][1], 0.0) for o in objs]
    dirs = {"left": CAMERA_LEFT_VEC, "right": (-CAMERA_LEFT_VEC[0], -CAMERA_LEFT_VEC[1]),
            "front": CAMERA_FRONT_VEC, "behind": (-CAMERA_FRONT_VEC[0], -CAMERA_FRONT_VEC[1])}
    out = {rel: [] for rel in dirs}
    for rel, d in dirs.items():
        for i in range(len(objs)):
            related = sorted(
                j for j in range(len(objs)) if j != i and
                (pos[j][0] - pos[i][0]) * d[0] + (pos[j][1] - pos[i][1]) * d[1] > eps)
            out[rel].append(related)
    return out


def validate_scene_json(scene: dict, spec: dict, tol: float = 1e-4) -> list:
    """渲染回填的场景图 vs 规格；返回问题列表（空 = 通过）。"""
    problems = []
    objs, sobjs = scene["objects"], spec["objects"]
    if len(objs) != len(sobjs):
        return [f"对象数不符: 渲染 {len(objs)} vs 规格 {len(sobjs)}"]
    for o, s in zip(objs, sobjs):
        for k in ("shape", "color", "material", "size"):
            if o[k] != s[k]:
                problems.append(f"对象 {s['id']} 的 {k}: 渲染 {o[k]} vs 规格 {s[k]}")
        cx, cy, cz = o["3d_coords"]
        if abs(cx - s["pos"][0]) > tol or abs(cy - s["pos"][1]) > tol:
            problems.append(f"对象 {s['id']} 位置漂移: ({cx:.5f},{cy:.5f}) vs {s['pos']}")
        if abs(cz - SIZES[s["size"]]) > 1e-3:
            problems.append(f"对象 {s['id']} 高度异常: {cz} vs {SIZES[s['size']]}")
        if abs(o["rotation"] - s["rotation"]) > 1e-3:
            problems.append(f"对象 {s['id']} 旋转漂移: {o['rotation']} vs {s['rotation']}")
    for name, vec in (("left", CAMERA_LEFT_VEC), ("front", CAMERA_FRONT_VEC)):
        d = scene["directions"][name]
        if abs(d[0] - vec[0]) > 1e-3 or abs(d[1] - vec[1]) > 1e-3 or abs(d[2]) > 1e-6:
            problems.append(f"方向轴 {name} 偏离常量: {d}")
    exp = expected_relationships(spec)
    for rel in ("left", "right", "front", "behind"):
        got = [list(x) for x in scene["relationships"][rel]]
        if got != exp[rel]:
            problems.append(f"关系 {rel}: 渲染 {got} vs 预期 {exp[rel]}")
    return problems


def _render_one(state, root: Path, gp: dict, paths: dict, rcfg: dict, logger,
                spec_md5: str = "", render_fp: str = ""):
    sid = state["state_id"]
    img, scn = root / "images" / f"{sid}.png", root / "scenes" / f"{sid}.json"
    spec_p = root / "specs" / f"{sid}.json"
    spec_p.write_text(state["spec"])
    cmd = [str(paths["render"]["blender"]), "--background", "-noaudio",
           "--python", str(DRIVER), "--",
           "--spec", str(spec_p), "--output_image", str(img),
           "--output_scene", str(scn),
           "--base_scene_blendfile", str(gp["base"]),
           "--properties_json", str(gp["props"]),
           "--shape_dir", str(gp["shapes"]), "--material_dir", str(gp["mats"]),
           "--width", str(rcfg["width"]), "--height", str(rcfg["height"]),
           "--render_num_samples", str(rcfg["samples"]),
           "--min_pixels_per_object", str(rcfg["min_pixels_per_object"])]
    start = time.time()
    retries, code = 0, None
    for attempt in range(rcfg.get("max_retries", 3) + 1):
        log_p = root / "logs" / f"{sid}_try{attempt}.txt"
        log_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=BLENDER_TIMEOUT)
            code = r.returncode
            log_p.write_text(r.stdout[-20000:] + "\n==STDERR==\n" + r.stderr[-5000:])
        except subprocess.TimeoutExpired:
            code = -9
            log_p.write_text("TIMEOUT")
        if code == EXIT_VISIBILITY:
            break  # 确定性失败：规格固定，重试结果相同
        if code == 0 and img.exists() and scn.exists():
            break
        retries += 1
    seconds = time.time() - start
    validated, problems, scene_json = False, [], ""
    if code == 0 and img.exists() and scn.exists():
        try:
            scene = json.loads(scn.read_text())
            problems = validate_scene_json(scene, json.loads(state["spec"]))
            validated = not problems
            if validated:
                scene_json = json.dumps(scene, sort_keys=True)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            problems = [f"渲染产物不可读: {e}"]
    rec = {"state_id": sid, "start": start, "end": time.time(), "seconds": seconds,
           "exit_code": code, "retries": retries, "validated": validated,
           "problems": problems, "spec_md5": spec_md5, "render_fp": render_fp,
           "image_bytes": img.stat().st_size if img.exists() else 0}
    if not validated:
        logger.warning(f"[render] {sid} 未通过: code={code} problems={problems[:3]}")
        scene_json = ""
    else:
        logger.info(f"[render] {sid} OK ({seconds:.1f}s, retries={retries})")
    return rec, scene_json if validated else None


def _postprocess_derived(state, root: Path, scene_jsons: dict, pcfg: dict):
    """生成 jpeg / crop 派生对照图（无语义变化，第 3 节）；场景图不变。"""
    from PIL import Image
    sid = state["state_id"]
    src_id = state["source_state_id"]
    src_png = root / "images" / f"{src_id}.png"
    out_png = root / "images" / f"{sid}.png"
    if not src_png.exists():
        return None
    img = Image.open(src_png).convert("RGB")
    if state["postprocess"] == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=pcfg["jpeg_quality"])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    elif state["postprocess"] == "crop":
        w, h = img.size
        s = pcfg["crop_scale"]
        img = img.crop((int(w * (1 - s) / 2), int(h * (1 - s) / 2),
                        int(w * (1 + s) / 2), int(h * (1 + s) / 2)))
    else:
        return None
    img.save(out_png)
    return {"state_id": sid, "image_path": str(out_png),
            "scene_json": scene_jsons.get(src_id, ""), "validated": True,
            "render_seconds": 0.0, "derived": True}


def render_manifest(task: str, manifest: dict, paths: dict, cfg: dict,
                    logger, workers: int = 1, limit=None, force=False) -> dict:
    root = Path(paths["render"]["output_dir"]) / task
    for sub in ("images", "scenes", "specs", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    ledger = root / "ledger.jsonl"
    rcfg = cfg["render"]
    render_fp = _asset_fingerprint(paths, rcfg)
    states = [r for r in manifest["states"].to_dict("records") if r["needs_render"]]
    states.sort(key=lambda r: r["state_id"])
    spec_md5 = {s["state_id"]: _spec_md5(s["spec"]) for s in states}

    def _ledger_matches(rec: dict) -> bool:
        """账本记录是否为当前清单 + 当前规格 / 资产指纹下的有效产物。"""
        sid = rec.get("state_id")
        return bool(rec.get("validated") and sid in spec_md5
                    and rec.get("spec_md5") == spec_md5[sid]
                    and rec.get("render_fp") == render_fp)

    done, stale = set(), 0
    if ledger.exists() and not force:
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("validated"):
                if _ledger_matches(rec):
                    done.add(rec["state_id"])
                else:
                    stale += 1
    todo = [s for s in states if s["state_id"] not in done]
    skipped = len(states) - len(todo)
    if limit is not None:
        todo = todo[:int(limit)]
    logger.info(f"[render] {task}: 待渲染 {len(todo)} 态（账本跳过 {skipped}，"
                f"指纹不符作废 {stale}）")
    gp = _gen_paths(paths)
    ledger_f = ledger.open("a")
    scene_jsons = {}
    t0 = time.time()
    if workers <= 1:
        for st in todo:
            rec, sj = _render_one(st, root, gp, paths, rcfg, logger,
                                  spec_md5[st["state_id"]], render_fp)
            ledger_f.write(json.dumps(rec) + "\n")
            if sj is not None:
                scene_jsons[rec["state_id"]] = sj
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_render_one, st, root, gp, paths, rcfg, logger,
                                 spec_md5[st["state_id"]], render_fp): st
                       for st in todo}
            for fut in futures:
                rec, sj = fut.result()
                ledger_f.write(json.dumps(rec) + "\n")
                if sj is not None:
                    scene_jsons[rec["state_id"]] = sj
    ledger_f.close()
    wall = time.time() - t0

    # 汇总 rendered_states.parquet：只登记当前清单 + 当前指纹下最新一次
    # 校验通过的渲染态（审查问题 6：旧清单 / 旧规格的遗留产物不得混入），
    # 另加派生对照态。同状态多次渲染取账本最后一条匹配记录。
    ok = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if _ledger_matches(rec):
                ok[rec["state_id"]] = rec
    rows, ok_render = [], len(ok)
    for sid in sorted(ok):
        scene = json.loads((root / "scenes" / f"{sid}.json").read_text())
        rows.append({"state_id": sid,
                     "image_path": str(root / "images" / f"{sid}.png"),
                     "scene_json": json.dumps(scene, sort_keys=True),
                     "validated": True,
                     "render_seconds": ok[sid]["seconds"], "derived": False})
    by_id = {r["state_id"]: r["scene_json"] for r in rows}
    n_derived = 0
    for st in manifest["states"].to_dict("records"):
        if st["needs_render"]:
            continue
        d = _postprocess_derived(st, root, by_id, cfg.get("postprocess", {}))
        if d is not None:
            rows.append(d)
            n_derived += 1
    pd.DataFrame(rows).to_parquet(root / "rendered_states.parquet", index=False)
    throughput = len(todo) / wall if wall > 0 else 0.0
    summary = {"task": task, "attempted": len(todo), "skipped_in_ledger": skipped,
               "stale_fingerprint": stale,
               "validated_total": ok_render, "derived": n_derived,
               "wall_seconds": wall, "attempted_per_second": throughput,
               "workers": workers}
    logger.info(f"[render] {task} 完成: {summary}")
    return summary
