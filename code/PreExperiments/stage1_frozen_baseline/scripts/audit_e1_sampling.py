"""只读审计 E1 平衡采样输出（2026-11-03 复查用）。

背景（codex 审查 2026-10-28 变更，规范依据：README Step 3 与
src/experiments/common.py::balanced_subsample docstring；方案文档无此条款）：
E1 target 子样本按 pos_value 分层等额抽取，"恒选某个词"的答题策略期望
准确率被压到 50%/1/k；SPAR position 以 50% 概率从 c2 视角出题，正例
方位词覆盖 left/right/below/above。本脚本在重跑后的正式产物上核验该
设计确实生效、汇总表可由 item 级数据重算。不访问模型/图片/特征缓存。

检查项：
  A pos_value 配额平衡（各值计数差 <=1；池不足重分配导致的近似均衡单独标注）
  B SPAR position 视角翻转≈50%（right/above 占比，|z|<=3）+ 方位词真在正例文本中
  C item 级字段完整（item_acc/item_margin 有限、item_acc∈[0,1]）
  D e1_results.csv 可由 items jsonl 重算（accuracy/mean_margin/median/std，容差 1e-9）
  E e1_acc_by_pos_value.csv：分组 n 之和 = n_samples；加权均值 = 总体 accuracy
  F "恒选某词"策略实际期望准确率（平衡因素应≈1/k；正负例词集相同记 immune）
  G run_meta.json 五实验 success、e1_state 产物清单齐全、无 smoke 混入
  H 跨 seed 图片重叠（描述性，不计入通过判定）

用法（任意目录）：python code/scripts/audit_e1_sampling.py [--date YYYY-MM-DD]
输出：code/outputs/analysis/audit_e1_sampling_<date>.json / .md
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "analysis"
MODELS = ("clip_vitb32", "clip_vitl14", "siglip2_so400m")
SEEDS = (0, 1, 2)
EXPECTED_EXPERIMENTS = ("e1_state", "e1_spar_official", "e2_pug_animals",
                        "e2_pug_imagenet", "e3_cogent")
TOL = 1e-9
# median/std 用不同算法重算（numpy 与 statistics 求和顺序不同），
# 数值噪声可达 ~2e-9；真实不一致会在 1e-2 量级，1e-7 足以区分。
TOL_SECONDARY = 1e-7
TOL_BY_METRIC = {"n_samples": TOL, "accuracy": TOL, "mean_margin": TOL,
                 "median_margin": TOL_SECONDARY, "margin_std": TOL_SECONDARY}

# 预期 pos_value 值个数（None = 不预设，只按观测值查均衡）
EXPECTED_K = {("clevr", "object"): 3, ("clevr", "color"): 8, ("clevr", "size"): 2,
              ("clevr", "material"): 2, ("clevr", "relation"): 4,
              ("pug_spar", "position"): 4, ("pug_spar", "attribute"): None,
              ("pug_spar", "relation"): 2}
# 无 pos_value 的因素（正负例词集相同，天然免疫词偏好，抽样回退普通随机）
FALLBACK = {("clevr", "binding"), ("pug_spar", "scene")}
ITEM_FILES = {"clevr": "e1_items_clevr.jsonl", "pug_spar": "e1_items_spar.jsonl"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def word_diff(pos_text, neg_text):
    """正例文本相对负例文本多出的词集（恒选某词策略的判别依据）。"""
    return set(pos_text.lower().split()) - set(neg_text.lower().split())


def check_balance(key, counts):
    """A：pos_value 配额均衡。pass=True/False，None=不适用（回退因素）。"""
    if key in FALLBACK:
        return {"pass": None, "note": "fallback_random（无 pos_value，词集免疫）"}
    vals = sorted(counts)
    n_vals = len(vals)
    cmin, cmax = min(counts.values()), max(counts.values())
    rec = {"k_observed": n_vals, "expected_k": EXPECTED_K.get(key),
           "min": cmin, "max": cmax, "pass": True, "note": ""}
    if EXPECTED_K.get(key) is not None and n_vals != EXPECTED_K[key]:
        rec["pass"] = False
        rec["note"] = f"值个数 {n_vals} != 预期 {EXPECTED_K[key]}"
    elif cmax - cmin <= 1:
        rec["note"] = "等额（±1 余数）"
    elif cmin > 0 and cmax / cmin <= 1.05:
        rec["note"] = f"近似均衡（池不足重分配，max/min={cmax / cmin:.3f}）"
    else:
        rec["pass"] = False
        rec["note"] = f"不均衡：max−min={cmax - cmin}，max/min={cmax / cmin:.3f}"
    return rec


def audit_run(model, seed, sources, images_by_factor):
    """单个 model×seed 的全部检查，返回 (结果dict, 失败清单)。"""
    tables = ROOT / "outputs" / model / f"seed_{seed}" / "tables"
    run_label = f"{model}/seed_{seed}"
    failures = []
    result = {"tables_dir": str(tables.relative_to(ROOT))}

    # ---- G run_meta ----
    meta_path = tables / "run_meta.json"
    sources[str(meta_path.relative_to(ROOT))] = sha256(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    exp_status = {k: meta["experiments"].get(k, {}).get("status") for k in EXPECTED_EXPERIMENTS}
    g_pass = all(v == "success" for v in exp_status.values())
    e1_outputs = meta["experiments"]["e1_state"]["last_success"]["outputs"]
    missing_outputs = [f for f in e1_outputs if not (tables / f).exists()]
    smoke_in_outputs = [f for f in e1_outputs if "smoke" in f]
    if missing_outputs or smoke_in_outputs:
        g_pass = False
        if missing_outputs:
            failures.append(f"G {run_label}: e1_state 产物缺失 {missing_outputs}")
        if smoke_in_outputs:
            failures.append(f"G {run_label}: 正式 success 清单混入 smoke {smoke_in_outputs}")
    if not g_pass:
        failures.append(f"G {run_label}: 实验状态 {exp_status}")
    result["run_meta"] = {"experiments": exp_status, "e1_outputs": e1_outputs,
                          "missing_outputs": missing_outputs, "pass": g_pass}

    # ---- 读 items 与汇总表 ----
    items_by_key = {}
    for ds, fname in ITEM_FILES.items():
        path = tables / fname
        sources[str(path.relative_to(ROOT))] = sha256(path)
        items_by_key[ds] = read_jsonl(path)
    res_rows = read_csv_rows(tables / "e1_results.csv")
    sources[str((tables / "e1_results.csv").relative_to(ROOT))] = sha256(tables / "e1_results.csv")
    pv_path = tables / "e1_acc_by_pos_value.csv"
    pv_rows = read_csv_rows(pv_path)
    sources[str(pv_path.relative_to(ROOT))] = sha256(pv_path)
    res_by_key = {(r["dataset"], r["factor"]): r for r in res_rows}
    pv_by_key = defaultdict(list)
    for r in pv_rows:
        pv_by_key[(r["dataset"], r["factor"])].append(r)

    factors = []
    for ds in ITEM_FILES:
        per_factor = defaultdict(list)
        for it in items_by_key[ds]:
            per_factor[it["factor"]].append(it)
        for f, items in sorted(per_factor.items()):
            key = (ds, f)
            rec = {"n": len(items)}

            # A 配额平衡
            if key in FALLBACK:
                rec["pos_value_counts"] = None
            else:
                pv_counts = Counter(it["pos_value"] for it in items)
                rec["pos_value_counts"] = dict(sorted(pv_counts.items()))
                images_by_factor[(model, ds, f)][seed] = {it["image_id"] for it in items}
            rec["balance"] = check_balance(key, rec["pos_value_counts"] or {})
            if rec["balance"]["pass"] is False:
                failures.append(f"A {run_label} {ds}/{f}: {rec['balance']['note']} "
                                f"{rec['pos_value_counts']}")

            # B SPAR position 翻转比例 + 方位词在正例文本中
            if key == ("pug_spar", "position"):
                flipped = sum(c for v, c in rec["pos_value_counts"].items()
                              if v in ("right", "above"))
                share = flipped / len(items)
                z = (share - 0.5) / math.sqrt(0.25 / len(items))
                viol = sum(1 for it in items
                           if it["pos_value"] not in it["content_pos"].lower().split())
                b_pass = abs(z) <= 3 and viol == 0
                rec["flip"] = {"flip_share": round(share, 4), "z": round(z, 2),
                               "pos_word_missing": viol, "pass": b_pass}
                if not b_pass:
                    failures.append(f"B {run_label}: flip={share:.3f} z={z:.2f} "
                                    f"pos_word_missing={viol}")
            else:
                images_by_factor[(model, ds, f)][seed] = {it["image_id"] for it in items}

            # C item 字段
            miss_acc = sum(1 for it in items
                           if not isinstance(it.get("item_acc"), (int, float))
                           or not math.isfinite(it["item_acc"]))
            miss_margin = sum(1 for it in items
                              if not isinstance(it.get("item_margin"), (int, float))
                              or not math.isfinite(it["item_margin"]))
            bad_range = sum(1 for it in items
                            if isinstance(it.get("item_acc"), (int, float))
                            and math.isfinite(it["item_acc"])
                            and not 0.0 <= it["item_acc"] <= 1.0)
            c_pass = miss_acc == miss_margin == bad_range == 0
            rec["items_fields"] = {"missing_acc": miss_acc, "missing_margin": miss_margin,
                                   "acc_out_of_range": bad_range, "pass": c_pass}
            if not c_pass:
                failures.append(f"C {run_label} {ds}/{f}: {rec['items_fields']}")

            # D 重算 e1_results
            row = res_by_key[key]
            accs = [it["item_acc"] for it in items]
            mgs = [it["item_margin"] for it in items]
            diffs = {
                "n_samples": float(len(items)) - float(row["n_samples"]),
                "accuracy": statistics.fmean(accs) - float(row["accuracy"]),
                "mean_margin": statistics.fmean(mgs) - float(row["mean_margin"]),
                "median_margin": statistics.median(mgs) - float(row["median_margin"]),
                "margin_std": (statistics.stdev(mgs) if len(mgs) > 1 else 0.0)
                              - float(row["margin_std"]),
            }
            d_pass = all(abs(v) <= TOL_BY_METRIC[k] for k, v in diffs.items())
            rec["recompute"] = {k: {"diff": v, "abs_diff": abs(v)}
                                for k, v in diffs.items()}
            rec["recompute"]["pass"] = d_pass
            if not d_pass:
                worst = max(diffs.items(), key=lambda kv: abs(kv[1]))
                failures.append(f"D {run_label} {ds}/{f}: {worst[0]} 偏差 {worst[1]:.3e}"
                                f"（容差 {TOL_BY_METRIC[worst[0]]:g}）")

            # E e1_acc_by_pos_value 一致性
            if key in FALLBACK:
                rec["pv_table"] = {"pass": None,
                                   "note": "无 pos_value，不在逐值表中（预期）"}
            else:
                groups = pv_by_key[key]
                sum_n = sum(int(g["n"]) for g in groups)
                weighted = sum(int(g["n"]) * float(g["accuracy"]) for g in groups) / sum_n
                e_pass = (sum_n == int(row["n_samples"])
                          and abs(weighted - float(row["accuracy"])) <= TOL)
                rec["pv_table"] = {"groups": len(groups), "sum_n": sum_n,
                                   "n_samples": int(row["n_samples"]),
                                   "weighted_acc": weighted,
                                   "accuracy": float(row["accuracy"]), "pass": e_pass}
                if not e_pass:
                    failures.append(f"E {run_label} {ds}/{f}: sum_n={sum_n} "
                                    f"加权acc={weighted:.6f} vs {row['accuracy']}")

            # F 恒选某词策略上界
            strategy = Counter()
            immune = 0
            for it in items:
                diff = word_diff(it["content_pos"], it["content_neg"])
                if not diff:
                    immune += 1
                else:
                    strategy.update(diff)
            if immune == len(items):
                rec["word_strategy"] = {"immune": True, "pass": None,
                                        "note": "正负例词集相同，恒选某词无判别力"}
            else:
                word, cnt = strategy.most_common(1)[0]
                share = cnt / len(items)
                if key in FALLBACK or rec["pos_value_counts"] is None:
                    expected = None
                    f_pass = None      # 回退因素只描述，不判定
                else:
                    expected = 1.0 / len(rec["pos_value_counts"])
                    f_pass = share <= expected + 0.01
                    if not f_pass:
                        failures.append(f"F {run_label} {ds}/{f}: 恒选'{word}' "
                                        f"={share:.4f} > 1/k+0.01={expected + 0.01:.4f}")
                rec["word_strategy"] = {"immune": False, "max_share": round(share, 4),
                                        "argmax_word": word,
                                        "expected_share": (round(expected, 4)
                                                           if expected else None),
                                        "immune_items": immune, "pass": f_pass}
            factors.append({"dataset": ds, "factor": f, **rec})
    result["factors"] = factors
    return result, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    sources = {}
    images_by_factor = defaultdict(dict)   # (model, ds, factor) -> {seed: set(image_id)}
    runs = {}
    all_failures = []
    for model in MODELS:
        for seed in SEEDS:
            res, fails = audit_run(model, seed, sources, images_by_factor)
            runs[f"{model}/seed_{seed}"] = res
            all_failures.extend(fails)

    # ---- H 跨 seed 图片重叠（描述性） ----
    overlap = {}
    for (model, ds, f), by_seed in sorted(images_by_factor.items()):
        if len(by_seed) != len(SEEDS):
            continue
        pair_shares = []
        for i, s1 in enumerate(SEEDS):
            for s2 in SEEDS[i + 1:]:
                inter = len(by_seed[s1] & by_seed[s2])
                pair_shares.append(inter / len(by_seed[s1]))
        overlap[f"{model}|{ds}/{f}"] = {
            "max_pairwise_share": round(max(pair_shares), 4),
            "mean_pairwise_share": round(sum(pair_shares) / len(pair_shares), 4),
            "unique_images_per_seed": [len(by_seed[s]) for s in SEEDS],
        }

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    payload = {
        "date": args.date,
        "git_commit": commit,
        "tolerance": TOL,
        "models": MODELS,
        "seeds": SEEDS,
        "all_checks_passed": not all_failures and len(runs) == len(MODELS) * len(SEEDS),
        "failure_count": len(all_failures),
        "failures": all_failures,
        "cross_seed_image_overlap": overlap,
        "runs": runs,
        "sources_sha256": sources,
    }
    json_path = OUT / f"audit_e1_sampling_{args.date}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")

    # ---- MD 摘要 ----
    lines = [f"# E1 平衡采样审计（{args.date}）", "",
             f"- git commit：`{commit}`；数据源 {len(sources)} 个文件，SHA256 见同名 JSON。",
             f"- 检查 A–G 通过判定：{'全部通过' if not all_failures else f'{len(all_failures)} 项失败（见下）'}；"
             "H 为描述性统计。",
             f"- 消费文件与产物均为正式（非 smoke）表。", ""]
    if all_failures:
        lines += ["## 失败项", ""] + [f"- {f}" for f in all_failures] + [""]
    # 表1：run_meta 状态
    lines += ["## G. run_meta 实验状态", "",
              "| run | e1_state | e1_spar_official | e2_pug_animals | e2_pug_imagenet | e3_cogent |",
              "|---|---|---|---|---|---|"]
    for label, res in runs.items():
        st = res["run_meta"]["experiments"]
        lines.append("| " + label + " | " + " | ".join(
            st[k] or "缺失" for k in EXPECTED_EXPERIMENTS) + " |")
    # 表2：配额平衡（计数相同的压缩显示）
    lines += ["", "## A. pos_value 配额平衡（count: seed0/seed1/seed2）", "",
              "| dataset/factor | " + " | ".join(MODELS) + " |", "|" + "---|" * (len(MODELS) + 1)]
    key_order = [("clevr", f) for f in ("object", "color", "size", "material",
                                        "relation", "binding")] + \
                [("pug_spar", f) for f in ("scene", "position", "attribute", "relation")]
    for ds, f in key_order:
        cells = []
        for model in MODELS:
            per_seed = []
            for seed in SEEDS:
                fac = next(x for x in runs[f"{model}/seed_{seed}"]["factors"]
                           if x["dataset"] == ds and x["factor"] == f)
                cnt = fac["pos_value_counts"]
                per_seed.append("回退" if cnt is None
                                else "/".join(str(c) for c in cnt.values()))
            cells.append("；".join(dict.fromkeys(per_seed)) if any(p != "回退" for p in per_seed)
                         else "回退（无 pos_value）")
        lines.append(f"| {ds}/{f} | " + " | ".join(cells) + " |")
    # 表3：重算最大偏差
    n_checked = sum(len(r["factors"]) for r in runs.values())
    lines += ["", f"## D. e1_results.csv 重算一致性（{n_checked} 个 run×factor 的最大绝对偏差）", ""]
    for metric in ("n_samples", "accuracy", "mean_margin", "median_margin", "margin_std"):
        worst = 0.0
        for res in runs.values():
            for fac in res["factors"]:
                worst = max(worst, fac["recompute"][metric]["abs_diff"])
        lines.append(f"- {metric}: {worst:.3e}（容差 {TOL_BY_METRIC[metric]:g}）")
    # 表4：恒选某词上界
    lines += ["", "## F. “恒选某词”策略期望准确率上界（max_share，argmax 词）", "",
              "| dataset/factor | " + " | ".join(MODELS) + " |", "|" + "---|" * (len(MODELS) + 1)]
    for ds, f in key_order:
        cells = []
        for model in MODELS:
            per_seed = set()
            for seed in SEEDS:
                fac = next(x for x in runs[f"{model}/seed_{seed}"]["factors"]
                           if x["dataset"] == ds and x["factor"] == f)
                ws = fac["word_strategy"]
                per_seed.add("immune" if ws["immune"]
                             else f"{ws['max_share']:.3f}（{ws['argmax_word']}）")
            cells.append("；".join(sorted(per_seed)))
        lines.append(f"| {ds}/{f} | " + " | ".join(cells) + " |")
    # 表5：flip
    lines += ["", "## B. SPAR position 视角翻转比例（right+above 占比，z 检验 |z|<=3）", "",
              "| model | seed0 | seed1 | seed2 |", "|---|---|---|---|"]
    for model in MODELS:
        cells = []
        for seed in SEEDS:
            fac = next(x for x in runs[f"{model}/seed_{seed}"]["factors"]
                       if x["factor"] == "position")
            fl = fac["flip"]
            cells.append(f"{fl['flip_share']:.3f} (z={fl['z']:+.1f})")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    # 表6：跨 seed 图片重叠
    lines += ["", "## H. 跨 seed 图片重叠（同因素三种子两两交集最大占比，描述性）", "",
              "| dataset/factor | " + " | ".join(MODELS) + " |", "|" + "---|" * (len(MODELS) + 1)]
    for ds, f in key_order:
        cells = []
        for model in MODELS:
            o = overlap.get(f"{model}|{ds}/{f}")
            cells.append(f"{o['max_pairwise_share']:.3f}" if o else "—")
        lines.append(f"| {ds}/{f} | " + " | ".join(cells) + " |")
    lines += ["", "## E. e1_acc_by_pos_value.csv 一致性", "",
              f"- 全部 {(sum(len(r['factors']) for r in runs.values()))} 个 run×factor 中，"
              "分组 n 之和与总体 n、加权准确率与总体准确率一致性："
              + ("全部通过（容差 1e-9）" if not any(x.startswith("E ") for x in all_failures)
                 else "存在失败项，见失败清单") + "。"]

    md_path = OUT / f"audit_e1_sampling_{args.date}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[audit] {json_path}")
    print(f"[audit] {md_path}")
    print(f"[audit] failures: {len(all_failures)}")
    for f in all_failures:
        print(f"  - {f}")
    return 0 if not all_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
