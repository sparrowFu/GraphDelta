"""PUG: SPAR 官方协议评测（复现 facebookresearch/PUG 的 run_eval_vlms_on_spar.py
CLIP 路径；审查报告 P2-11）。

与自建 2AFC（e1_state_diagnosis 的 spar 部分，结果标注 protocol=2afc_self_built）
相互独立，差异如下：
- 预处理：官方脚本对所有模型统一 Resize(224,224) + ImageNet mean/std
  （非 CLIPProcessor），图像特征缓存 key 单独存放；
- 文本：官方 caption 模板（无句号、无外层模板包装，ensure_period=False）；
- 指标：全候选检索 top1/top5 + hard-negative 二选一。官方对 hard-negative
  随机换序以避免坍缩编码器；argmax 下换序不改变判对条件
  （sim_pos > sim_neg），固定 [pos, neg] 顺序与官方 accuracy 等价
  （平局除外：本实现平局判错，与方案 2AFC 规则一致）；
- 官方脚本的已知 quirk 按原样保留：character1_pos 只有 left/below，
  "bottom/top" 子集为空；actor 列表去 blank（官方用 unique()[1:] 近似实现）；
- 抽样：spar_official.max_per_task 限制每个评测子集的图片数（按 seeds[0] 随机
  抽样，与自建 E1 的 target_per_cell 对齐）。此时偏离官方全量子集，数值不再与
  官方发布结果直接可比；配置缺省（null）保持官方全量；scene 保留官方 10 行。
"""
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from ..data.pug_common import count_rows, iter_image_batches
from ..data.pug_spar import META_COLUMNS, build_metadata
from ..features.cache import ImageFeatureCache, TextCache
from ..utils.io import load_config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TEXTURES = ["Default", "Blue", "Grass"]
POS_PAIRS = [("left", "right"), ("bottom", "top")]


def _env_suffix(world):
    """官方 add_background 版 caption 的环境后缀（world 为 None 时不加）。"""
    return f" in a {world} environment" if world else ""


def _cap_rows(df, cap, rng):
    """子集超过 cap 时按 rng 随机抽样到 cap 行（cap 为 None 时原样返回）。"""
    if cap is None or len(df) <= cap:
        return df
    return df.iloc[rng.permutation(len(df))[:cap]].reset_index(drop=True)


def _fmt(v):
    return f"{v:.4f}" if v == v else "nan"


def _official_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def extract_official_features(model, cache, key, parquet_dir, logger, only_row_ids=None):
    """官方预处理下的 SPAR 图像特征（增量抽取，缓存 id 为 pug_spar_official_%06d）。

    进度日志与 extract.py 统一为 "[extract:key] N/M"（每 2000 张）。
    """
    existing = set()
    if cache.exists(key):
        eids, _idx, _raw, _norm = cache.arrays(key)
        existing = set(eids.tolist())
        logger.info(f"[extract] cache {key}: {len(existing)} ids present")
    tf = _official_transform()
    prefix = "pug_spar_official"
    fmt = lambda rid: f"{prefix}_{rid:06d}"
    if only_row_ids is not None:
        expected_new = sum(1 for r in only_row_ids if fmt(r) not in existing)
    elif existing:
        total_rows = count_rows(parquet_dir)
        only_row_ids = {r for r in range(total_rows) if fmt(r) not in existing}
        expected_new = len(only_row_ids)
        if not only_row_ids:
            logger.info(f"[extract] cache complete, skip: {key}")
            return
    else:
        expected_new = count_rows(parquet_dir)
    ids, raws, norms = [], [], []
    total = 0
    for row_ids, images, _m in iter_image_batches(parquet_dir, META_COLUMNS,
                                                  batch_size=48,
                                                  only_row_ids=only_row_ids):
        todo = [(r, im) for r, im in zip(row_ids, images) if fmt(r) not in existing]
        if not todo:
            continue
        batch = torch.stack([tf(im) for _r, im in todo])
        out = model.encode_image_tensors(batch)
        raws.append(out["raw"])
        norms.append(out["normalized"])
        ids.extend(fmt(r) for r, _im in todo)
        prev, total = total, total + len(todo)
        if total // 2000 > prev // 2000:
            logger.info(f"[extract:{key}] {total}/{expected_new}")
    if not ids:
        logger.info(f"[extract] cache complete, skip: {key}")
        return
    if existing:
        eids, _i, eraw, enorm = cache.arrays(key)
        raws = [eraw.astype(np.float32)] + raws
        norms = [enorm.astype(np.float32)] + norms
        ids = eids.tolist() + ids
    raw = np.concatenate(raws, axis=0)
    norm = np.concatenate(norms, axis=0)
    cache.save(key, ids, raw, norm)
    logger.info(f"[extract] saved {total} new / {len(ids)} total images -> {cache.path(key)}")


# ---------------- 评测原语 ----------------

def _retrieval_eval(F_sub, caps, labels, tcache, model, chunk=256):
    """全候选检索：top1 = argmax 命中率；top5 = 前 5 含正例比例（候选 <5 时 NaN）。"""
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n == 0 or not caps:
        return np.nan, np.nan
    _, txt = tcache.encode(model, caps, ensure_period=False)
    top1 = 0
    top5 = 0
    top5_valid = len(caps) >= 5
    for i0 in range(0, n, chunk):
        sims = F_sub[i0:i0 + chunk] @ txt.T
        top1 += int(np.sum(np.argmax(sims, axis=1) == labels[i0:i0 + chunk]))
        if top5_valid:
            k5 = np.argpartition(-sims, 4, axis=1)[:, :5]
            for r, ki in enumerate(k5):
                top5 += int(labels[i0 + r] in ki)
    return top1 / n, (top5 / n if top5_valid else np.nan)


def _hardneg_eval(F_sub, pos_caps, neg_caps, tcache, model):
    """hard-negative 二选一：sim(pos) > sim(neg) 判对（平局判错，与 2AFC 一致）。

    与全候选检索的 caption 集合无关：正/负 caption 直接编码（tcache 去重），
    不要求负例 caption 出现在检索候选集中（二轮审查问题 7）。
    """
    pos_caps = list(pos_caps)
    neg_caps = list(neg_caps)
    if not pos_caps:
        return np.nan
    _, tp = tcache.encode(model, pos_caps, ensure_period=False)
    _, tn = tcache.encode(model, neg_caps, ensure_period=False)
    s_pos = np.einsum("ij,ij->i", F_sub, tp)
    s_neg = np.einsum("ij,ij->i", F_sub, tn)
    return float(np.mean(s_pos > s_neg))


# ---------------- 官方任务 ----------------

def run(model, paths, logger, smoke=False, seed=None):
    full = load_config("e1_state")
    cfg = full.get("spar_official", {})
    cache = ImageFeatureCache(paths["dirs"]["cache"], model.name)
    tcache = TextCache(paths["dirs"]["cache"], model.name)
    out_tables = paths["dirs"]["outputs"] / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    max_per_task = cfg.get("max_per_task")   # null：官方全量（历史行为）
    rng = np.random.default_rng(full.get("seeds", [0])[0] if seed is None else seed)

    meta = build_metadata(paths["datasets"]["pug_spar"])
    # add_background 候选的 world 展开按全量 metadata（官方脚本用完整 labels.csv，
    # 不随 smoke 子集缩小，二轮审查问题 7 附带核对项）
    worlds = sorted(meta["world_name"].unique().tolist())
    if smoke:
        meta = meta.head(3000).reset_index(drop=True)
        logger.info(f"[E1] spar_official smoke subset: {len(meta)} rows")
    key = "pug_spar_official_smoke" if smoke else "pug_spar_official"
    extract_official_features(model, cache, key, paths["datasets"]["pug_spar"], logger,
                              only_row_ids=set(meta["row_id"]) if smoke else None)
    _ids, index, _raw, feat_all = cache.arrays(key)
    prefix = "pug_spar_official"

    def F_of(df):
        rows = [index[f"{prefix}_{int(r):06d}"] for r in df["row_id"]]
        return feat_all[rows].astype(np.float32)

    results = []

    def add(task, world="all", add_background=False, texture="", pos_pair="",
            hardneg=False, n_images=0, n_captions=0, top1=np.nan, top5=np.nan):
        results.append({"task": task, "world": world, "add_background": add_background,
                        "texture": texture, "pos_pair": pos_pair, "hardneg": hardneg,
                        "n_images": n_images, "n_captions": n_captions,
                        "top1": top1, "top5": top5})
        # 每个结果行一行日志，格式与自建 E1 的 [E1] clevr/… 一致
        tags = [t for t in (texture, pos_pair, "bg" if add_background else "",
                            "hardneg" if hardneg else "") if t]
        if world != "all":
            tags.append(f"w={world}")
        name = task + (f"[{','.join(tags)}]" if tags else "")
        logger.info(f"[E1] spar_official/{name}: n={n_images} caps={n_captions} "
                    f"top1={_fmt(top1)} top5={_fmt(top5)}")

    # ---- 1. scene recognition（官方：blank-blank 前 10 行）----
    sub = meta[(meta["character_name"] == "blank")
               & (meta["character2_name"] == "blank")].head(10)
    sub_worlds = sorted(sub["world_name"].unique().tolist())
    caps = [f"This is a photo in a {w} environment" for w in sub_worlds]
    w2i = {c: i for i, c in enumerate(caps)}
    labels = [w2i[f"This is a photo in a {w} environment"] for w in sub["world_name"]]
    top1, top5 = _retrieval_eval(F_of(sub), caps, labels, tcache, model)
    add("scene_backgrounds", n_images=len(sub), n_captions=len(caps), top1=top1, top5=top5)

    # 单 / 双动物子集（actor 列：blank 侧取另一列）
    single = pd.concat([
        meta[(meta["character_name"] == "blank") & (meta["character2_name"] != "blank")],
        meta[(meta["character2_name"] == "blank") & (meta["character_name"] != "blank")],
    ]).reset_index(drop=True)
    single["actor"] = np.where(single["character_name"] != "blank",
                               single["character_name"], single["character2_name"])
    double = meta[(meta["character_name"] != "blank")
                  & (meta["character2_name"] != "blank")].reset_index(drop=True)

    env_list = [("all", False), ("all", True)]
    if cfg.get("per_world", False):
        env_list += [(w, False) for w in worlds]

    for world, add_back in env_list:
        ss = single if world == "all" else single[single["world_name"] == world]
        sd = double if world == "all" else double[double["world_name"] == world]
        ss = ss.reset_index(drop=True)
        sd = sd.reset_index(drop=True)
        # add_back=True 时候选 caption 覆盖全部 world（官方行为），见下方各任务

        # ---- 2. 单动物识别（按 texture）----
        for texture in TEXTURES:
            st = _cap_rows(ss[ss["character_texture"] == texture].reset_index(drop=True),
                           max_per_task, rng)
            chars = sorted({c for c in st["actor"].unique() if c != "blank"})
            if add_back:
                caps = [f"This is a photo of a {c}" + _env_suffix(w)
                        for c in chars for w in worlds]
            else:
                caps = [f"This is a photo of a {c}" for c in chars]
            cap2i = {c: i for i, c in enumerate(caps)}
            labels = [cap2i[f"This is a photo of a {r['actor']}"
                           + _env_suffix(r["world_name"] if add_back else None)]
                      for r in st.to_dict("records")]
            top1, top5 = _retrieval_eval(F_of(st), caps, labels, tcache, model)
            add("one_object", world=world, add_background=add_back, texture=texture,
                n_images=len(st), n_captions=len(caps), top1=top1, top5=top5)

        # ---- 3. 单动物位置（Default；官方 quirk：bottom/top 子集为空）----
        sr = ss[ss["character_texture"] == "Default"]
        for r1, r2 in POS_PAIRS:
            st = _cap_rows(sr[sr["character1_pos"] == r1].reset_index(drop=True),
                           max_per_task, rng)
            actors = sorted({c for c in st["actor"].unique() if c != "blank"})
            caps = sorted({f"This is a photo of a {a} on the {r} of the picture"
                           + _env_suffix(w)
                           for a in actors for r in (r1, r2)
                           for w in (worlds if add_back else [None])})
            cap2i = {c: i for i, c in enumerate(caps)}
            labels, negs = [], []
            for r in st.to_dict("records"):
                # char1 非空白侧位于 r1 方向；char2 非空白侧位于 r2 方向
                if r["character_name"] != "blank":
                    pos, negpos = r1, r2
                else:
                    pos, negpos = r2, r1
                tail = _env_suffix(r["world_name"] if add_back else None)
                labels.append(cap2i[f"This is a photo of a {r['actor']} on the {pos} of the picture" + tail])
                negs.append(cap2i[f"This is a photo of a {r['actor']} on the {negpos} of the picture" + tail])
            top1, top5 = _retrieval_eval(F_of(st), caps, labels, tcache, model)
            add("one_object_position", world=world, add_background=add_back,
                texture="Default", pos_pair=f"{r1}/{r2}",
                n_images=len(st), n_captions=len(caps), top1=top1, top5=top5)
            pos_caps = [f"This is a photo of a {r['actor']} on the "
                        f"{(r1 if r['character_name'] != 'blank' else r2)} of the picture"
                        + _env_suffix(r["world_name"] if add_back else None)
                        for r in st.to_dict("records")]
            neg_caps = [f"This is a photo of a {r['actor']} on the "
                        f"{(r2 if r['character_name'] != 'blank' else r1)} of the picture"
                        + _env_suffix(r["world_name"] if add_back else None)
                        for r in st.to_dict("records")]
            hn = _hardneg_eval(F_of(st), pos_caps, neg_caps, tcache, model)
            add("one_object_position", world=world, add_background=add_back,
                texture="Default", pos_pair=f"{r1}/{r2}", hardneg=True,
                n_images=len(st), n_captions=2, top1=hn)

        # ---- 4. 双动物识别（按 texture；官方按排序后的动物对）----
        for texture in TEXTURES:
            st = _cap_rows(sd[sd["character_texture"] == texture].reset_index(drop=True),
                           max_per_task, rng)
            pairs = sorted({tuple(sorted((r["character_name"], r["character2_name"])))
                            for r in st.to_dict("records")})
            caps = sorted({f"This is a photo of a {a} and a {b}" + _env_suffix(w)
                           for (a, b) in pairs for w in (worlds if add_back else [None])})
            cap2i = {c: i for i, c in enumerate(caps)}
            labels = [cap2i[f"This is a photo of a {a} and a {b}"
                           + _env_suffix(r["world_name"] if add_back else None)]
                      for r in st.to_dict("records")
                      for a, b in [tuple(sorted((r["character_name"], r["character2_name"])))]]
            top1, top5 = _retrieval_eval(F_of(st), caps, labels, tcache, model)
            add("two_objects", world=world, add_background=add_back, texture=texture,
                n_images=len(st), n_captions=len(caps), top1=top1, top5=top5)

        # ---- 5. 双动物关系（Default；官方过滤同种动物）----
        sr = sd[sd["character_texture"] == "Default"]
        for r1, r2 in POS_PAIRS:
            st = _cap_rows(sr[(sr["character1_pos"] == r1)
                              & (sr["character_name"] != sr["character2_name"])]
                           .reset_index(drop=True), max_per_task, rng)

            def relcap(c1, c2, w):
                return f"This is a photo of a {c1} on the {r1} and a {c2} on the {r2} of the picture" \
                    + _env_suffix(w)

            recs = st.to_dict("records")
            # 全候选检索集合只含真实行的正例 caption（官方行为，二轮审查问题 7）；
            # add_back 时每个正例组合按全部 world 展开（官方 for w in list_worlds，
            # 三轮审查问题 2）；role-swap 负例只用于 hard-negative 二选一，不进入检索集合
            caps = sorted({relcap(r["character_name"], r["character2_name"], w)
                           for r in recs for w in (worlds if add_back else [None])})
            cap2i = {c: i for i, c in enumerate(caps)}
            labels, pos_caps, neg_caps = [], [], []
            for r in recs:
                w = r["world_name"] if add_back else None
                labels.append(cap2i[relcap(r["character_name"], r["character2_name"], w)])
                pos_caps.append(relcap(r["character_name"], r["character2_name"], w))
                neg_caps.append(relcap(r["character2_name"], r["character_name"], w))
            top1, top5 = _retrieval_eval(F_of(st), caps, labels, tcache, model)
            add("two_objects_relation", world=world, add_background=add_back,
                texture="Default", pos_pair=f"{r1}/{r2}",
                n_images=len(st), n_captions=len(caps), top1=top1, top5=top5)
            hn = _hardneg_eval(F_of(st), pos_caps, neg_caps, tcache, model)
            add("two_objects_relation", world=world, add_background=add_back,
                texture="Default", pos_pair=f"{r1}/{r2}", hardneg=True,
                n_images=len(st), n_captions=2, top1=hn)

        # ---- 6. 双动物属性（Blue/Grass；官方过滤同种动物，动物互换纹理不动）----
        for texture in ["Blue", "Grass"]:
            st = _cap_rows(sd[(sd["character_texture"] == texture)
                              & (sd["character_name"] != sd["character2_name"])]
                           .reset_index(drop=True), max_per_task, rng)

            def attrcap(c1, t1, c2, t2, w):
                return f"This is a photo of a {c1} textured with {t1} and a {c2} textured with {t2}" \
                    + _env_suffix(w)

            recs = st.to_dict("records")
            # 检索集合只含真实行正例 caption（官方行为，二轮审查问题 7）；
            # add_back 时每个组合按全部 world 展开（三轮审查问题 2）
            caps = sorted({attrcap(r["character_name"], r["character_texture"],
                                   r["character2_name"], r["character2_texture"], w)
                           for r in recs for w in (worlds if add_back else [None])})
            cap2i = {c: i for i, c in enumerate(caps)}
            labels, pos_caps, neg_caps = [], [], []
            for r in recs:
                w = r["world_name"] if add_back else None
                labels.append(cap2i[attrcap(r["character_name"], r["character_texture"],
                                            r["character2_name"], r["character2_texture"], w)])
                pos_caps.append(attrcap(r["character_name"], r["character_texture"],
                                        r["character2_name"], r["character2_texture"], w))
                neg_caps.append(attrcap(r["character2_name"], r["character_texture"],
                                        r["character_name"], r["character2_texture"], w))
            top1, top5 = _retrieval_eval(F_of(st), caps, labels, tcache, model)
            add("two_objects_attribute", world=world, add_background=add_back,
                texture=texture, n_images=len(st), n_captions=len(caps),
                top1=top1, top5=top5)
            hn = _hardneg_eval(F_of(st), pos_caps, neg_caps, tcache, model)
            add("two_objects_attribute", world=world, add_background=add_back,
                texture=texture, hardneg=True, n_images=len(st), n_captions=2, top1=hn)

    out = pd.DataFrame(results)
    path = out_tables / ("e1_spar_official_smoke.csv" if smoke else "e1_spar_official.csv")
    out.to_csv(path, index=False)
    logger.info(f"[E1] written {path} ({len(out)} rows)")
    return out
