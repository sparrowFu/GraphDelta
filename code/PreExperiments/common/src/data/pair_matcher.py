"""PUG: Animals 单因素配对器（方案 §12 / §59 / §66）。

对每个 factor：
- 固定其余因素（context），只改变该 factor；
- 每个无序值对只保留 canonical 方向上的一个 transition（Δ = target − source）；
- 每个 context 内按"全局已选次数越少的 transition 权重越高"做无放回加权抽样，
  保证 transition 覆盖（审查报告 P2-7）；
- 输出全部 meta_cols 列（变化因素记为 "src->tgt"），供跨因素
  same/different-context 判定使用（审查报告 P2-6）。

重复行的处理（复核 B1 的守卫放宽为确定性去重 + 日志披露）：
- Animals 是组合唯一的完整笛卡尔积，不会触发；
- PUG:ImageNet 实测每个因素有 1,448 行（约 1.6%）与其余行在全部 meta_cols
  上完全相同（同一 context 内同一因素值出现多次）。配对定义只需要每个
  (context, 因素值) 一行，因此保留最小 row_id、丢弃其余，并在日志中
  记录总数——不静默折叠。

键的定义：
- context_key：factor + 其余 metadata 组合（within-factor 的 context 身份）
- cluster_key：除被测因素外的完整 metadata 组合（聚类 bootstrap 的单位，§66）
"""
import numpy as np
import pandas as pd

FIXED_COLS = {
    "environment": ["animal", "texture", "size", "camera"],
    "texture": ["animal", "environment", "size", "camera"],
    "size": ["animal", "environment", "texture", "camera"],
    "camera": ["animal", "environment", "texture", "size"],
}

META_COLS = ["animal", "environment", "texture", "size", "camera"]


def canonical_values(factor, observed, orders=None):
    """返回按 canonical 顺序排列的观测值。顺序只用于统一 Δ 方向，不解释为语义顺序。

    factor 未在 orders 中声明、或声明的值不是列表（如 Animals 配置中
    `environment: alphabetical` 只是说明文字）时，退回按观测值排序
    （数值列按数值序，其余按字典序；PUG:ImageNet 即走此分支）。
    """
    order = orders.get(factor) if orders else None
    if not isinstance(order, (list, tuple)):
        return sorted(observed)
    obs = set(observed)
    return [v for v in order if v in obs]


def build_factor_pairs(df: pd.DataFrame, factor: str, orders: dict,
                       max_pairs_per_context: int, max_total: int, seed: int,
                       meta_cols=None, logger=None) -> pd.DataFrame:
    """meta_cols：该数据集参与配对的语义列（默认 Animals 的 META_COLS）。

    context = meta_cols 中除被测因素外的全部列（审查 A1：PUG:ImageNet 的
    物体类别、其他相机角度、灯光、物体姿态等列也进入固定 context）。
    同一 context 内重复的因素值（全部 meta_cols 相同的行）保留最小 row_id，
    数量经 logger 披露（见模块 docstring）。
    """
    rng = np.random.default_rng(seed)
    meta_cols = list(meta_cols) if meta_cols is not None else META_COLS
    if factor not in meta_cols:
        raise ValueError(f"factor {factor!r} 不在 meta_cols {meta_cols} 中")
    group_cols = [c for c in meta_cols if c != factor]

    # context 顺序随机化 + 在线 transition 计数：同轮中后处理的 context 更倾向
    # 选尚未被覆盖过的 transition，避免 2016 个 environment transition 漏覆盖
    ctx_groups = list(df.groupby(group_cols, sort=False))
    ctx_order = rng.permutation(len(ctx_groups))
    counts = {}          # transition (src, tgt) -> 全局已选次数
    records = []
    dup_rows_total = 0   # 去重的重复行数（跨 context 汇总，供日志披露）
    dup_ctx_total = 0

    for gi in ctx_order:
        ctx_key, grp = ctx_groups[int(gi)]
        if not isinstance(ctx_key, tuple):
            ctx_key = (ctx_key,)
        ctx_dict = dict(zip(group_cols, ctx_key))
        ctx_key_str = "|".join(str(v) for v in ctx_key)

        by_val = {}
        n_dup = 0
        for r in grp.to_dict("records"):
            v = r[factor]
            if v not in by_val:
                by_val[v] = r
            else:
                # 复核 B1：重复行不能被静默折叠——计数汇总后经 logger 披露；
                # 保留最小 row_id，去重结果与输入行序无关（确定性）
                n_dup += 1
                if int(r["row_id"]) < int(by_val[v]["row_id"]):
                    by_val[v] = r
        if n_dup:
            dup_rows_total += n_dup
            dup_ctx_total += 1
        vals = canonical_values(factor, by_val.keys(), orders)
        avail = [(vals[i], vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
        n_take = min(max_pairs_per_context, len(avail))
        chosen = []
        pool = list(avail)
        for _ in range(n_take):
            w = np.array([1.0 / (1 + counts.get(t, 0)) ** 2 for t in pool])
            idx = int(rng.choice(len(pool), p=w / w.sum()))
            chosen.append(pool.pop(idx))
        for t in chosen:
            counts[t] = counts.get(t, 0) + 1

        for src, tgt in chosen:
            ra, rb = by_val[src], by_val[tgt]
            rec = {
                "factor": factor,
                "source_value": src,
                "target_value": tgt,
                "row_id_a": int(ra["row_id"]),
                "row_id_b": int(rb["row_id"]),
                "sample_id_a": ra["sample_id"],
                "sample_id_b": rb["sample_id"],
                "context_key": f"{factor}|{ctx_key_str}",
            }
            for c in meta_cols:
                rec[c] = f"{src}->{tgt}" if c == factor else str(ctx_dict[c])
            rec["cluster_key"] = "|".join(rec[c] for c in meta_cols if c != factor)
            records.append(rec)

    out = pd.DataFrame(records)
    if dup_rows_total:
        # 不静默：去重规模写日志。 Animals 不会触发（完整笛卡尔积）；
        # 若因 context 列配置错误导致大量重复，这里的大数字也能暴露问题。
        msg = (f"factor={factor!r}: 去重 {dup_rows_total} 行重复的 (context, 因素值) 行"
               f"（涉及 {dup_ctx_total} 个 context，每组保留最小 row_id；"
               f"PUG:ImageNet 存在 metadata 完全相同的重复行，Animals 无此情况）")
        if logger is not None:
            logger.warning(f"[pairs] {msg}")
        else:
            print(f"[pairs] WARNING: {msg}")
    if len(out) > max_total:
        keep = rng.choice(len(out), size=max_total, replace=False)
        out = out.iloc[sorted(keep.tolist())].reset_index(drop=True)
    if len(out):
        out.insert(0, "pair_id", [f"{factor}_{i:06d}" for i in range(len(out))])
    return out
