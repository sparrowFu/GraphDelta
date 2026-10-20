"""PUG: Animals 单因素配对器（方案 §12 / §59）。

对每个 factor：
- 固定其余因素（context），只改变该 factor；
- 每个无序值对只保留 canonical 方向上的一个 transition（Δ = target − source）；
- 限制每个 context 的 transition 数与该 factor 的总 pair 数。

键的定义：
- context_key：factor + 其余 metadata 组合（within-factor 的 context 身份）
- ae_key：animal|environment。factor=environment 时 environment 本身在变，记为 animal|*
  （跨因素比较与聚类 bootstrap 用，方案 §66）
"""
import numpy as np
import pandas as pd

FIXED_COLS = {
    "environment": ["animal", "texture", "size", "camera"],
    "texture": ["animal", "environment", "size", "camera"],
    "size": ["animal", "environment", "texture", "camera"],
    "camera": ["animal", "environment", "texture", "size"],
}


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
    records = []

    for ctx_key, grp in df.groupby(group_cols, sort=False):
        if not isinstance(ctx_key, tuple):
            ctx_key = (ctx_key,)
        ctx_dict = dict(zip(group_cols, ctx_key))
        ctx_key_str = "|".join(str(v) for v in ctx_key)

        by_val = {r[factor]: r for r in grp.to_dict("records")}
        vals = canonical_values(factor, by_val.keys(), orders)
        cand = [(vals[i], vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
        if len(cand) > max_pairs_per_context:
            keep = rng.choice(len(cand), size=max_pairs_per_context, replace=False)
            cand = [cand[k] for k in sorted(keep.tolist())]

        animal = str(ctx_dict["animal"])
        if factor == "environment":
            env_field, ae_key = "*", f"{animal}|*"
        else:
            env_field = str(ctx_dict["environment"])
            ae_key = f"{animal}|{env_field}"

        for src, tgt in cand:
            ra, rb = by_val[src], by_val[tgt]
            records.append({
                "factor": factor,
                "source_value": src,
                "target_value": tgt,
                "row_id_a": int(ra["row_id"]),
                "row_id_b": int(rb["row_id"]),
                "sample_id_a": ra["sample_id"],
                "sample_id_b": rb["sample_id"],
                "context_key": f"{factor}|{ctx_key_str}",
                "ae_key": ae_key,
                "animal": animal,
                "environment": env_field if factor != "environment" else f"{src}->{tgt}",
            })

    out = pd.DataFrame(records)
    if len(out) > max_total:
        keep = rng.choice(len(out), size=max_total, replace=False)
        out = out.iloc[sorted(keep.tolist())].reset_index(drop=True)
    if len(out):
        out.insert(0, "pair_id", [f"{factor}_{i:06d}" for i in range(len(out))])
    return out
