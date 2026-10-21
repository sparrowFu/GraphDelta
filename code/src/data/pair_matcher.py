"""PUG: Animals 单因素配对器（方案 §12 / §59 / §66）。

对每个 factor：
- 固定其余因素（context），只改变该 factor；
- 每个无序值对只保留 canonical 方向上的一个 transition（Δ = target − source）；
- 每个 context 内按"全局已选次数越少的 transition 权重越高"做无放回加权抽样，
  保证 transition 覆盖（审查报告 P2-7）；
- 输出全部 5 个 metadata 列（变化因素记为 "src->tgt"），供跨因素
  same/different-context 判定使用（审查报告 P2-6）。

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


def canonical_values(factor, observed, orders):
    """返回按 canonical 顺序排列的观测值。顺序只用于统一 Δ 方向，不解释为语义顺序。"""
    if factor == "environment":
        return sorted(observed)
    order = orders[factor]
    obs = set(observed)
    return [v for v in order if v in obs]


def build_factor_pairs(df: pd.DataFrame, factor: str, orders: dict,
                       max_pairs_per_context: int, max_total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_cols = FIXED_COLS[factor]

    # context 顺序随机化 + 在线 transition 计数：同轮中后处理的 context 更倾向
    # 选尚未被覆盖过的 transition，避免 2016 个 environment transition 漏覆盖
    ctx_groups = list(df.groupby(group_cols, sort=False))
    ctx_order = rng.permutation(len(ctx_groups))
    counts = {}          # transition (src, tgt) -> 全局已选次数
    records = []

    for gi in ctx_order:
        ctx_key, grp = ctx_groups[int(gi)]
        if not isinstance(ctx_key, tuple):
            ctx_key = (ctx_key,)
        ctx_dict = dict(zip(group_cols, ctx_key))
        ctx_key_str = "|".join(str(v) for v in ctx_key)

        by_val = {r[factor]: r for r in grp.to_dict("records")}
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
            for c in META_COLS:
                rec[c] = f"{src}->{tgt}" if c == factor else str(ctx_dict[c])
            rec["cluster_key"] = "|".join(rec[c] for c in META_COLS if c != factor)
            records.append(rec)

    out = pd.DataFrame(records)
    if len(out) > max_total:
        keep = rng.choice(len(out), size=max_total, replace=False)
        out = out.iloc[sorted(keep.tolist())].reset_index(drop=True)
    if len(out):
        out.insert(0, "pair_id", [f"{factor}_{i:06d}" for i in range(len(out))])
    return out
