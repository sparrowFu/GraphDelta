"""编辑对清单构建（工作方案第 3、4 节；阶段二 README Step 2 的核心逻辑）。

产出三类持久化产物（data/edit_pairs/<task>_pairs.parquet、<task>_states.parquet、
splits.json），全部由 seed 决定、可复现；splits.json 记录清单 md5，清单被改动后
装载直接报错（划分与清单必须配套）。

记录口径：
- pair：一组 before/after 状态引用 + 结构化编辑指令 + 划分；逆向复用同一
  pair_id，不另立记录（逆向相减不增加独立样本数）；
- state：一个唯一场景状态（渲染单位）。同一基础场景的 before 状态被交换对、
  换色对共享；换色 / 平移是独立状态，进渲染账本；jpeg / crop 是后处理派生态
  （needs_render=False，引用 source_state_id），不占账本；
- 划分：家族（绑定的转换 / 关系的类型对）内按基础场景 60/20/20，同场景全部
  版本同组（第 4 节）；顺序取 scene_idx 升序，不引入清单 seed 之外的随机。
"""
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from generation.edits import (
    PlacementError, apply_binding_swap, apply_recolor, apply_relation_flip,
    apply_translate, validate_edit,
)
from generation.scene_spec import (
    BINDING_CONVERSIONS, COLORS, RELATION_TYPE_PAIRS_TRAIN,
    build_binding_base_scene, build_relation_base_scene,
)

EDIT_KINDS = ("swap", "recolor", "flip", "translate", "jpeg", "crop")
_VALIDATE_KIND = {"swap": "binding_swap", "recolor": "recolor",
                  "flip": "relation_flip", "translate": "translate"}
DEFAULT_RATIOS = {"train": 0.6, "val": 0.2, "heldout": 0.2}


def _jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def family_of_binding(conversion) -> str:
    return "|".join(conversion)


def family_of_relation(type_pair) -> str:
    return "|".join("%s:%s" % (s, m) for s, m in type_pair)


def _state_id(base_scene_id: str, tag: str) -> str:
    return f"{base_scene_id}::{tag}"


def split_quotas(n: int, ratios: dict) -> dict:
    """家族内 n 个场景的 60/20/20 配额；n>=3 时保证三组各至少 1 个。"""
    t = max(1, round(n * ratios["train"]))
    v = max(1 if n >= 2 else 0, round(n * ratios["val"]))
    h = n - t - v
    if n >= 3 and h < 1:
        t -= 1 - h  # 从 train 挪给 heldout
        h = 1
    return {"train": t, "val": v, "heldout": max(0, h)}


def assign_scene_splits(scenes_by_family: dict, ratios: dict) -> dict:
    """scene_idx 升序前段 train、中段 val、末段 heldout（确定性划分）。"""
    scene_split = {}
    for fam, scenes in scenes_by_family.items():
        q = split_quotas(len(scenes), ratios)
        counts = {"train": 0, "val": 0, "heldout": 0}
        for spec in scenes:
            if counts["train"] < q["train"]:
                split = "train"
            elif counts["val"] < q["val"]:
                split = "val"
            else:
                split = "heldout"
            counts[split] += 1
            scene_split[spec.base_scene_id] = split
    return scene_split


def _register_state(states: dict, state_id, spec, tag: str, family: str,
                    derived_from=None, postprocess=None):
    if state_id in states:
        return  # 同一 before 状态被多对共享：首登为准（spec 相同）
    states[state_id] = {
        "state_id": state_id,
        "base_scene_id": spec.base_scene_id,
        "task": spec.task,
        "family": family,
        "state_tag": tag,
        "spec": _jdump(spec.as_dict()),
        "needs_render": derived_from is None,
        "source_state_id": derived_from or "",
        "postprocess": postprocess or "",
    }


def _record_pair(pairs: list, task, edit_kind, family, base_scene_id,
                 before, after, instruction, before_tag, after_tag, split,
                 word_set_changed=False):
    """登记一个编辑对并当场校验（语义编辑走 validate_edit；对照要求 spec 不变）。"""
    validate_kind = _VALIDATE_KIND.get(edit_kind)
    if validate_kind is not None:
        validate_edit(before, after, validate_kind)
    elif _jdump(before.as_dict()) != _jdump(after.as_dict()):
        raise ValueError(f"{edit_kind} 对照不得改变场景图: {base_scene_id}")
    pairs.append({
        "pair_id": f"{base_scene_id}_{edit_kind}",
        "task": task,
        "edit_kind": edit_kind,
        "family": family,
        "base_scene_id": base_scene_id,
        "before_state_id": _state_id(base_scene_id, before_tag),
        "after_state_id": _state_id(base_scene_id, after_tag),
        "before_spec": _jdump(before.as_dict()),
        "after_spec": _jdump(after.as_dict()),
        "instruction": _jdump(instruction),
        "word_set_changed": word_set_changed,
        "split": split,
    })


def _sample_binding_scenes(rng, conv, need: int):
    """采一个转换的 need 个基础场景；放置失败重采样（消耗 rng，保持确定性）。"""
    scenes, idx, resamples = [], 0, 0
    while len(scenes) < need:
        try:
            scenes.append(build_binding_base_scene(rng, idx, conv))
        except RuntimeError:
            resamples += 1
        idx += 1
    return scenes, resamples


def _sample_relation_scenes(rng, tp, need: int):
    scenes, idx, resamples = [], 0, 0
    while len(scenes) < need:
        try:
            scenes.append(build_relation_base_scene(rng, idx, tp, a_left=True))
        except RuntimeError:
            resamples += 1
        idx += 1
    return scenes, resamples


def _even_split(n: int, k: int) -> list:
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def _coverage_scenes(scenes: list, scene_split: dict, n_control: int,
                     n_semantic: int) -> list:
    """错误编辑状态（换色 / 平移）的场景选择（审查问题 4）。

    正控制 / 对照取每家族前 n_control 个场景；另为前 n_semantic 个语义
    编辑场景中全部 val/heldout 场景生成同场景错误编辑状态（E4-D 候选池
    第 6 节要求：纳入评价的每个查询都必须有原图、正确编辑图与同场景错误
    编辑图）。train 场景不因候选池额外增加（d_r 估计不进检索评价）。
    """
    control = list(scenes[:n_control])
    extra = [s for s in scenes[:n_semantic]
             if scene_split[s.base_scene_id] != "train"]
    out, seen = [], set()
    for s in control + extra:
        if s.base_scene_id not in seen:
            seen.add(s.base_scene_id)
            out.append(s)
    return out


def build_binding(rng: random.Random, n_swap: int, n_recolor: int,
                  ratios: dict):
    """绑定任务：8 个转换均分场景；换色状态覆盖全部 val/heldout 交换场景。

    换色正控制取前 n_rc 个场景；val/heldout 的交换场景全部生成换色状态
    （E4-D 同场景错误编辑候选，审查问题 4）。换色新颜色取
    COLORS ∖ {c1, c2}（词集变化，word_set_changed=True）。
    返回 (pairs, states, scenes_by_family, scene_split, resamples)。
    """
    pairs, states = [], {}
    resamples = 0
    scenes_by_family = {}
    per_swap = _even_split(n_swap, len(BINDING_CONVERSIONS))
    per_recolor = _even_split(n_recolor, len(BINDING_CONVERSIONS))
    for conv, n_sw, n_rc in zip(BINDING_CONVERSIONS, per_swap, per_recolor):
        scenes, r = _sample_binding_scenes(rng, conv, max(n_sw, n_rc))
        resamples += r
        scenes_by_family[family_of_binding(conv)] = scenes
    scene_split = assign_scene_splits(scenes_by_family, ratios)
    for conv, scenes, n_sw, n_rc in zip(
            BINDING_CONVERSIONS, scenes_by_family.values(), per_swap, per_recolor):
        _, _, c1, c2 = conv
        family = family_of_binding(conv)
        other_colors = sorted(set(COLORS) - {c1, c2})
        for spec in scenes[:n_sw]:
            sid, split = spec.base_scene_id, scene_split[spec.base_scene_id]
            _register_state(states, _state_id(sid, "before"), spec, "before", family)
            after = apply_binding_swap(spec)
            _register_state(states, _state_id(sid, "swap"), after, "swap", family)
            _record_pair(pairs, "binding", "swap", family, sid, spec, after,
                         {"op": "binding_swap", "conversion": list(conv)},
                         "before", "swap", split)
        for spec in _coverage_scenes(scenes, scene_split, n_rc, n_sw):
            sid, split = spec.base_scene_id, scene_split[spec.base_scene_id]
            new_color = rng.choice(other_colors)
            recolored = apply_recolor(spec, "target", new_color)
            _register_state(states, _state_id(sid, "recolor"), recolored, "recolor", family)
            _record_pair(pairs, "binding", "recolor", family, sid, spec, recolored,
                         {"op": "recolor", "obj": "target", "color": new_color},
                         "before", "recolor", split, word_set_changed=True)
    return pairs, states, scenes_by_family, scene_split, resamples


def build_relation(rng: random.Random, n_flip: int, ratios: dict):
    """关系任务：10 个类型对均分场景（统一 a 在 b 左），编辑为左右反转。

    共同平移对照按每 4 个训练场景取 1 个（至少 1）；val/heldout 的全部
    反转场景另生成平移状态（E4-D 同场景错误编辑候选，审查问题 4）。
    平移量在 [-0.6, 0.6]² 采样、越界重采样。
    """
    pairs, states = [], {}
    resamples = 0
    scenes_by_family = {}
    per_flip = dict(zip(RELATION_TYPE_PAIRS_TRAIN, _even_split(n_flip, len(RELATION_TYPE_PAIRS_TRAIN))))
    for tp, need in per_flip.items():
        scenes, r = _sample_relation_scenes(rng, tp, need)
        resamples += r
        scenes_by_family[family_of_relation(tp)] = scenes
    scene_split = assign_scene_splits(scenes_by_family, ratios)
    for tp, scenes in zip(RELATION_TYPE_PAIRS_TRAIN, scenes_by_family.values()):
        family = family_of_relation(tp)
        for spec in scenes:
            sid, split = spec.base_scene_id, scene_split[spec.base_scene_id]
            _register_state(states, _state_id(sid, "before"), spec, "before", family)
            after = apply_relation_flip(spec)
            _register_state(states, _state_id(sid, "flip"), after, "flip", family)
            _record_pair(pairs, "relation", "flip", family, sid, spec, after,
                         {"op": "relation_flip", "mover": "a", "anchor": "b"},
                         "before", "flip", split)
        train_scenes = [s for s in scenes
                        if scene_split[s.base_scene_id] == "train"]
        n_ctrl = max(1, len(train_scenes) // 4)
        for spec in _coverage_scenes(scenes, scene_split, n_ctrl, len(scenes)):
            sid, split = spec.base_scene_id, scene_split[spec.base_scene_id]
            for _try in range(50):
                dx, dy = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)
                try:
                    moved = apply_translate(spec, round(dx, 6), round(dy, 6))
                    break
                except PlacementError:
                    resamples += 1
            else:
                raise RuntimeError(f"平移对照重采样 50 次仍失败: {sid}")
            _register_state(states, _state_id(sid, "translate"), moved, "translate", family)
            _record_pair(pairs, "relation", "translate", family, sid, spec, moved,
                         {"op": "translate", "dx": round(dx, 6), "dy": round(dy, 6)},
                         "before", "translate", split)
    return pairs, states, scenes_by_family, scene_split, resamples


def build_postprocess_controls(rng: random.Random, binding_states: dict,
                               n_control: int, postprocess: dict,
                               scene_split: dict) -> (list, dict):
    """无语义变化对照（第 3 节）：对绑定 swap-after 状态做同图压缩 / 轻微裁剪。

    只登记派生态（spec 与源状态相同、needs_render=False），图片由渲染步按
    postprocess 参数生成；jpeg / crop 各占一半，跨场景轮转选取源状态。
    """
    controls, derived = [], {}
    sources = sorted((st for st in binding_states.values() if st["state_tag"] == "swap"),
                     key=lambda s: s["state_id"])
    if not sources:
        return controls, derived
    kinds = ("jpeg", "crop")
    for i in range(n_control):
        src = sources[i % len(sources)]
        kind = kinds[i % 2]
        state_id = f"{src['state_id']}::{kind}"
        instruction = ({"op": "postprocess", "type": "jpeg", "quality": postprocess["jpeg_quality"]}
                       if kind == "jpeg" else
                       {"op": "postprocess", "type": "crop", "scale": postprocess["crop_scale"]})
        rec = dict(src)  # spec / family 全部继承源状态
        rec.update({"state_id": state_id, "state_tag": kind, "needs_render": False,
                    "source_state_id": src["state_id"], "postprocess": kind})
        derived[state_id] = rec
        controls.append({
            "pair_id": f"{src['base_scene_id']}_pp_{kind}",
            "task": "binding",
            "edit_kind": kind,
            "family": src["family"],
            "base_scene_id": src["base_scene_id"],
            "before_state_id": src["state_id"],
            "after_state_id": state_id,
            "before_spec": src["spec"],
            "after_spec": src["spec"],  # 无语义变化：场景图必须相同
            "instruction": _jdump(instruction),
            "word_set_changed": False,
            "split": scene_split[src["base_scene_id"]],
        })
    return controls, derived


def build_task(task: str, seed: int, n_swap=800, n_recolor=200, n_flip=500,
               n_control=200, ratios=None, postprocess=None):
    """构建一个任务的全部记录（纯函数，落盘由 write_manifest 负责）。"""
    ratios = ratios or dict(DEFAULT_RATIOS)
    postprocess = postprocess or {"jpeg_quality": 75, "crop_scale": 0.9}
    rng = random.Random(seed)
    if task == "binding":
        pairs, states, scenes_by_family, scene_split, resamples = build_binding(
            rng, n_swap, n_recolor, ratios)
        controls, derived = build_postprocess_controls(
            rng, states, n_control, postprocess, scene_split)
        pairs += controls
        states.update(derived)
    elif task == "relation":
        pairs, states, scenes_by_family, scene_split, resamples = build_relation(
            rng, n_flip, ratios)
    else:
        raise ValueError(f"未知任务: {task}")
    pairs.sort(key=lambda p: p["pair_id"])
    states = [states[k] for k in sorted(states)]
    meta = {
        "task": task, "seed": seed, "ratios": ratios,
        "counts": {"pairs": len(pairs), "states": len(states),
                   "render_states": sum(1 for s in states if s["needs_render"]),
                   "resamples": resamples},
        "families": sorted(scenes_by_family),
        "scene_split": scene_split,
        "generator": {
            "binding_conversions": [list(c) for c in BINDING_CONVERSIONS],
            "relation_type_pairs": [[list(a), list(b)] for a, b in RELATION_TYPE_PAIRS_TRAIN],
        },
        "args": {"n_swap": n_swap, "n_recolor": n_recolor, "n_flip": n_flip,
                 "n_control": n_control},
    }
    return {"pairs": pairs, "states": states, "meta": meta}


def manifest_paths(out_dir: Path, task: str):
    out_dir = Path(out_dir)
    return (out_dir / f"{task}_pairs.parquet", out_dir / f"{task}_states.parquet",
            out_dir / f"{task}_splits.json")


def write_manifest(out_dir: Path, task: str, data: dict) -> dict:
    """写三件产物并回填 md5 到 splits.json；返回最终 meta。"""
    pairs_p, states_p, splits_p = manifest_paths(out_dir, task)
    pd.DataFrame(data["pairs"]).to_parquet(pairs_p, index=False)
    pd.DataFrame(data["states"]).to_parquet(states_p, index=False)
    meta = dict(data["meta"])
    meta["pairs_md5"] = hashlib.md5(pairs_p.read_bytes()).hexdigest()
    meta["states_md5"] = hashlib.md5(states_p.read_bytes()).hexdigest()
    splits_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))
    return meta


def load_manifest(out_dir: Path, task: str) -> dict:
    """装载并校验：parquet 当前 md5 必须与 splits.json 记录一致。"""
    pairs_p, states_p, splits_p = manifest_paths(out_dir, task)
    for p in (pairs_p, states_p, splits_p):
        if not p.exists():
            raise FileNotFoundError(f"清单缺失: {p}；先运行 build_edit_pairs.py")
    meta = json.loads(splits_p.read_text())
    for p, key in ((pairs_p, "pairs_md5"), (states_p, "states_md5")):
        if hashlib.md5(p.read_bytes()).hexdigest() != meta[key]:
            raise RuntimeError(
                f"清单与划分不配套: {p} 的 md5 与 {splits_p} 记录不一致。"
                f"清单被改动过，须用原 seed 重建或重新划分。")
    return {"pairs": pd.read_parquet(pairs_p), "states": pd.read_parquet(states_p),
            "meta": meta}
