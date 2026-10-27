# GraphDelta 代码修改报告

修改日期：2026-10-27
对应审查报告：`GraphDelta_代码审查报告_2026-10-26.md`（补审更新 2026-10-27）
修改基线：`ef329ad`（当前工作区，未提交）
验证：32 项受控检查全部通过（方法见第五节）

## 一、结论

审查报告"三、仍需处理的问题"中的 4 项 P2、补充 A/B 两项 P3、补充 C 数值防护缺口、补充 D 留痕缺口，全部确认属实，全部通过修改代码解决。第五节的"解释和资源边界"各项，报告自己定性为"不是本轮新增程序 BUG"，除其中一条代码建议（披露独特比较数）已实现外，其余不修改，理由见第四节。

修改共涉及 15 个文件，+376 / −50 行。默认配置下的正常运行行为不变：默认数据路径、默认过滤规则、现有结果文件名、旧结果目录的读取兼容都保持原样。

## 二、修改总览

| 审查问题 | 优先级 | 修改位置 | 修改结果 | 验证 |
|---|---|---|---|---|
| 问题 1：失败运行把旧结果标记成新来源 | P2 | io.py + 四个实验入口 | 来源分 attempt / last_success，成功写出后才提交 | 通过 |
| 问题 2：smoke 更新全量结果的来源记录 | P2 | e1 / e1_spar_official / e3 入口 | smoke 与全量使用独立来源键 | 通过 |
| 问题 3：E3 成功重跑后旧单元表仍 passed | P2 | e3 入口 / analyze_e3_units / io.py | 重跑即删除旧单元表并置 stale；分析记录源指纹；消费守卫核对指纹 | 通过 |
| 问题 4：稳定性汇总接受 failed 的探索结果 | P2 | summarize_seed_stability.py | 汇总前逐种子校验；探索模式显式选择并输出独立文件 | 通过 |
| 补充 A：E3 场景/图像路径配置未生效 | P3 | cogent.py / e3 入口 / extract_embeddings.py | 配置路径生效；非默认路径时缓存 key 加指纹 | 通过 |
| 补充 B：E1 两项 SPAR 过滤设置不生效 | P3 | e1 入口 / e1_state.yaml | 不支持的取值运行前显式报错，不再静默忽略 | 通过 |
| 补充 C：异常特征缺少评分前校验 | 条件性 | cache.py / common.py / matching.py | 缓存与评分边界校验形状、行数、图文维度、有限值 | 通过 |
| 补充 D：replaces 同版本重跑时消失 | 次要 | io.py | replaces 改为追加列表，留痕一直保留 | 通过 |
| 第五节建议：披露独特比较数 | 建议 | delta_stats.py / e2 入口 | 结果 CSV 新增 n_cmp_unique_* 列 | 通过 |

## 三、逐项说明

### 问题 1：失败运行会把旧结果标记成新来源 [P2]

**原问题。**`output_tables_dir()` 在运行开始时就把当前代码/缓存身份写进 `experiments/<name>`。运行在数据读取、配对验证或写盘中途失败时，CSV 还是旧内容，来源记录却已指向新的失败尝试。

**修改内容。**

1. `src/utils/io.py:62` `output_tables_dir()`：experiment 参数现在只登记**本次尝试**，写入 `status: "running"`、身份、`started` 时间。不改动 `last_success` 和 `replaces`。
2. `src/utils/io.py:111` 新增 `commit_experiment(tables_dir, experiment, outputs)`：输出文件全部成功写出后调用，把本次身份写入 `last_success`（附带产物文件清单 `outputs`），`status` 置 `success`。
3. 四个实验入口在最后一张表写完后调用提交：
   - `src/experiments/e1_state_diagnosis.py:342`
   - `src/experiments/e1_spar_official.py:395`
   - `src/experiments/e2_change_diagnosis.py:385`
   - `src/experiments/e3_cogent_diagnosis.py:224`
4. 旧格式 run_meta.json（身份写在记录顶层、无 last_success）读取时自动迁移：原顶层身份被视为历史成功来源，不会被新的入口登记清掉。

**修改后行为。** 复现审查报告的场景——目录里已有来源 `old_success` 的 CSV，新运行在配对检查处停止：

```text
CSV 内容：不变（仍是旧结果）
experiments.<name>.status：running
experiments.<name>.last_success：仍指向 old_success
```

失败尝试不再冒充当前来源。运行成功走到最后时，`last_success` 更新为新身份和产物清单。中途崩溃的运行留下 `status: running`，读取端可以区分"上次成功"和"本次未完成"。

### 问题 2：smoke 更新全量结果的来源记录 [P2]

**原问题。** `e1_state`、`e1_spar_official`、`e3_cogent` 三个入口的来源键固定不变。smoke 输出的是 `*_smoke.csv`，运行后全量结果的来源记录却被 smoke 运行更新。

**修改内容。** 三个入口的来源键在 smoke 模式下加 `_smoke` 后缀：

- `src/experiments/e1_state_diagnosis.py:236`：`e1_state_smoke`
- `src/experiments/e1_spar_official.py:167`：`e1_spar_official_smoke`
- `src/experiments/e3_cogent_diagnosis.py:77`：`e3_cogent_smoke`

与 E2 已有的 `e2_<dataset>_smoke` 约定一致。

**修改后行为。** 受控复现 e3：先跑 smoke，run_meta 的 experiments 里只有 `e3_cogent_smoke`，没有 `e3_cogent`；再跑全量后两个键并存，各自记录各自的来源和产物清单。smoke 不再改写全量结果的来源身份。

### 问题 3：E3 成功重跑后，旧单元表仍显示 passed [P2]

**原问题。** `run_e3` 重写主表和 items 后，`e3_unit_gaps.csv` 与其一致性 sidecar 不变。消费守卫只看 sidecar 状态，不核对输入身份，旧单元表带着 passed 标记被继续消费。

**修改内容。**

1. `src/experiments/e3_cogent_diagnosis.py`（run 末尾）：全量运行成功写出全部源文件后，删除旧的 `e3_unit_gaps.csv`，并把 sidecar 写为 `status: "stale"`。失效动作放在全部源文件写出之后：失败的中途运行不会破坏仍然有效的派生表。
2. `scripts/analyze_e3_units.py:113`：sidecar 新增 `source_fingerprint` 字段，取值为 `e3_items_cogent.jsonl` 与 `e3_difficulty_gap.csv` 两个文件内容哈希的组合（`src/utils/io.py:212` `e3_source_fingerprint`）。
3. `src/utils/io.py:229` `verify_e3_consistency`：新增两类拦截——`status == "stale"` 直接拦截；sidecar 有指纹时重算当前源文件指纹，不匹配即拦截。

**修改后行为。** 受控复现完整链条：

```text
全量 v1 → analyze → sidecar passed + 指纹
全量 v2（encoder 输出变化）→ e3_unit_gaps.csv 被删除，sidecar = stale
消费守卫：stale 拦截，报"待重跑 analyze_e3_units"
重跑 analyze → passed + 新指纹，守卫放行
手工改动 e3_items_cogent.jsonl → 指纹不匹配，守卫拦截
```

即使旧单元表被手工恢复，只要源文件内容与 sidecar 指纹不符，绘图（make_figures）和汇总都不会消费它。兼容性：旧 sidecar 没有指纹时不比对（无指纹可查），`not_checked` / `passed` 不拦截，行为与修复前一致。

### 问题 4：稳定性汇总接受 failed 的探索结果 [P2]

**原问题。** 绘图入口已调用 `verify_e3_consistency()`，`summarize_seed_stability.py` 没有相同检查。`analyze_e3_units --no-strict` 产出的 failed sidecar 与探索单元表会进入正式 `seed_stability.csv`。

**修改内容。** `scripts/summarize_seed_stability.py`：

1. 解析完三个种子的目录后，逐种子调用 `verify_e3_consistency`。任何一个种子为 failed / stale / 指纹不匹配，默认停止，不输出正式汇总。
2. 新增 `--allow-exploratory-e3` 参数：显式要求探索对照时继续运行，但输出写入独立的 `seed_stability_exploratory.csv`，不写 `seed_stability.csv`，日志明确说明该输出不构成正式稳定性结论。

**修改后行为。** 受控复现：三个种子目录、seed 1 的 sidecar 为 failed 时，默认运行直接停止且不产生 `seed_stability.csv`；加 `--allow-exploratory-e3` 后正常完成，输出文件为 `seed_stability_exploratory.csv`。探索结果不会静默混入正式文件。

### 补充 A：E3 的场景与图像路径配置未生效 [P3]

**原问题。** 入口读取 `conditions[cond]` 后没有使用其中的 `scene_file`、`image_dir`；loader 固定读取 `scenes/CLEVR_val{A,B}_scenes.json` 和 `images/val{A,B}`。独立抽取脚本同样使用固定路径。配置指向自定义路径时会去开默认文件。

**修改内容。**

1. `src/data/cogent.py:13` `load_condition_scenes`、`:19` `condition_image_dir` 接受路径参数，缺省值保持原固定路径。
2. `src/experiments/e3_cogent_diagnosis.py` 与 `scripts/extract_embeddings.py` 的 cogent 分支统一读取 `configs/e3_cogent.yaml` 的 `conditions.<cond>.scene_file / image_dir` 并传入。
3. `src/data/cogent.py:24` 新增 `condition_cache_key`：任一路径为非默认值时，缓存 key 附加该路径组合的指纹（如 `cogent_valA@efdb5d93664d`）。默认路径的 key 不变，历史缓存继续可用。

**修改后行为。** 受控复现：默认场景文件不存在、配置指向 `custom_scenes/A.json` 与 `custom_imgs/A` 时，e3.run 正常完成（修改前此处抛 `FileNotFoundError`）；缓存文件名为带指纹的 key，不会与默认路径的旧特征混用。审查报告提醒的"更换数据源不能复用同名旧特征"由 key 指纹直接保证。

### 补充 B：E1 的两项 SPAR 过滤设置不生效 [P3]

**原问题。** `same_animal_filter` 从未被读取；`two_object_only` 传入构造器但从未使用。过滤逻辑固定执行，改配置不产生任何效果，用户以为做了消融。

**修改内容。** 按审查报告建议的第一方案（删除或拒绝不支持的选项），采用**拒绝**：

1. `src/experiments/e1_state_diagnosis.py:40` 新增 `_validate_spar_protocol`，在 run 开始时（数据读取前）校验：
   - `same_animal_filter: false` → 立即报错停止。原因写明：双对象同种动物场景的正负文本可能相同或同时为真（P1-2），协议要求过滤，不能用配置关闭。
   - `two_object_only` 与固定协议（`factors` 中的 position/attribute/relation 全集）不一致 → 立即报错停止，报错信息给出协议要求的取值。
2. `build_spar_items` 删除从未使用的 `two_object_only` 参数。
3. `configs/e1_state.yaml` 注释更新：两项是协议的固定描述，入口校验，不支持用配置关闭。

**不做成可生效开关的理由。** 审查报告明确要求：不能为了让开关生效就放开会产生相同或同时为真正负文本的过滤；position/attribute/relation 需要两个非 blank 动物（R3），放开后单动物行会生成无效 item。如需其他协议，需先设计仍保证正负文本有效性的替代方案，这超出本次修复范围。

**修改后行为。** 当前默认配置通过校验，正常运行不变；`same_animal_filter: false` 或修改 `two_object_only` 时运行前报错，不再静默忽略。

### 补充 C：异常特征缺少评分前校验 [条件性数值防护缺口]

**原问题。** 缓存能保存并重新读出含 NaN 的特征；评分前不检查。NaN 相似度的 `s_pos > s_neg` 恒为 False，被当作答错，`eval_2afc_items()` 输出 `accuracy=0` 的确定失败表现。图像特征维度 1、文本维度 2 时，`np.broadcast_to()` 静默扩展，输出有限但指标含义错误（margin 可超过合法上界 2）。

**修改内容。** 三层校验，发现异常明确停止并给出定位信息：

1. `src/features/cache.py:52` `_check_features`（二维形状 + 有限值）、`:69` `_check_ids_rows`（ids/raw/normalized 行数与维度一致）。应用于 `ImageFeatureCache.save / arrays`、`TextCache._load / encode`。报错包含缓存文件路径和首个异常行号。
2. `src/experiments/common.py` `eval_2afc_items` 评分边界：图像与文本特征维度必须严格相等（拦截广播扩展）；两侧有限值检查；报错包含模型名、缓存目录、dataset_key，图像侧异常还给出首条异常的 `image_id`。
3. `src/metrics/matching.py` `per_template_scores` 兜底：img/pos/neg 形状必须一致且为有限值，否则 `ValueError`。

**修改后行为。** 受控复现全部拦截：

```text
save 含 NaN 特征            → RuntimeError（报缓存路径 + 行号）
直写 npz 绕过 save 后加载   → RuntimeError（load 端拦截）
图像维度 1 / 文本维度 2     → SystemExit（报两侧维度与位置，不再广播）
per_template_scores 输入 NaN → ValueError
TextCache 新编码块含 NaN    → RuntimeError
```

NaN 不再被计成答错；正常特征（有限值、维度一致）行为与修改前完全一致。

### 补充 D：replaces 会在同版本重复运行时消失 [次要留痕缺口]

**原问题。** 每次构造新的实验记录，只有身份变化时才写 replaces。A→B 时存在 replaces，随后 B→B 重跑该字段消失。

**修改内容。** `src/utils/io.py` 中 replaces 改为**列表**：`commit_experiment` 提交成功来源时，新身份与上一次 `last_success` 不同则把旧来源**追加**进列表；同版本重跑不追加也不删除。旧格式（单条 dict）读取时自动转为列表。

**修改后行为。** 受控复现：AAA→BBB→BBB→CCC 的提交序列后，`replaces == [AAA, BBB]`，`last_success == CCC`。变更链完整保留，同版本重跑不再丢失留痕。

### 第五节建议：披露独特比较数

审查报告第五节指出：`_distinct_pairs` 保证每对内两索引不同，但比较之间有放回；`n_cmp` 是抽样次数，不是独特比较数；建议另加披露（报告明确不算新 P2）。

**修改内容。** `src/metrics/delta_stats.py` `group_similarities` 的每组记录新增 `n_unique_pairs`（独特 (d1, d2) 组合数）；`src/experiments/e2_change_diagnosis.py` 把它写入结果 CSV 的 `n_cmp_unique_*` 列。

**修改后行为。** 小样本受控数据上，抽样 100 条时各组独特比较数为 62/33/22/12 等，重复程度直接可查。原 `n_cmp_*` 列保留不变。

## 四、不修改项及理由

第五节其余各项，报告原文定性为"以下事项不是本轮新增程序 BUG"，且没有可直接落地的代码修改对象：

| 项 | 不修改的理由 |
|---|---|
| E1/E3 不能证明未提及信息被保留/遗忘 | 研究设计边界，需要新增成对控制实验与指标，不是现有代码的错误 |
| CoGenT A/B 差异可能混有 shape 主效应 | 分析解释边界，需要补充实验设计分离，代码无错可修 |
| CI 含 0.5 是未检出偏离、跨种子一致非充分证据 | 结论措辞问题，属于研究总结文档，不属于代码 |
| crop/JPEG 与 raw embedding 尺度边界 | 实现确定、报告未指出实现错误，无需修改 |
| Norm Gap 两侧聚类单位（context vs 原图） | 统计假设说明或敏感性分析，需要方案层决定衔接方式；按报告建议"不宜只把两侧 ID 改成相同字符串"，未做机械修改 |
| pair 构建的 max_total 在构造后截断 | 资源实测项，报告写明"没有本轮资源实测支持"，无代码错误对象 |
| 数据版本 / goldfish 质量 / checkpoint 身份 / ImageNet 覆盖核实 | 需要服务器真实数据，本地无法执行，属后续验收步骤 |

审查报告建议顺序第 6 条"当前没有依据要求无条件重抽所有模型、所有图像"同样适用于本次：未重抽、未删除任何真实缓存。

## 五、验证方法与边界

**方法。** 自写验证脚本（32 项检查），使用真实代码入口、假 encoder（确定性向量，无 torch）、临时目录、受控场景文件和小尺寸 PNG。覆盖：

- 问题 1 / 补充 D：attempt / last_success / replaces 全生命周期（含失败中断、同版本重跑、身份链 AAA→BBB→CCC、旧格式迁移），共 8 项。
- 问题 2 / 3 / 补充 A：e3 全流程（smoke 键、全量键、自定义路径运行、缓存 key 指纹、analyze 首轮通过、重跑失效、stale 拦截、重算恢复、篡改源文件后指纹拦截），共 11 项。
- 问题 4：三种子汇总默认停止、不写正式 CSV、探索模式写独立文件，共 3 项。
- 补充 B：默认配置通过、两种非法改动被拒，共 3 项。
- 补充 C：正常存取、save/load/编码/评分/指标五层拦截，共 7 项。

另单独验证：`group_similarities` 各组均输出 `n_unique_pairs`；`analyze_e3_units --no-strict` 缺参考主表时输出 `not_checked`、无指纹、守卫放行（与修复前行为一致）。

**边界。**

1. 未加载三个真实 checkpoint、未跑真实数据集端到端。验证覆盖的是来源生命周期、失效链、消费守卫、配置解析和数值校验逻辑；VLM 数值不在本轮验证范围（与审查报告第六节边界一致）。
2. 存量旧 sidecar 没有指纹，无法核对身份；兼容处理为不比对。新跑的 `analyze_e3_units` 起全部带指纹。
3. 实验写多个文件时若在两张表之间崩溃，磁盘上会出现新旧混合状态；此时该实验 `status` 保持 `running`、`last_success` 保持旧值，读取端可据此区分，但未实现多文件原子写入。
4. 修改在工作区未提交；验证产生的临时文件与日志已清理，工作区只有这 15 个修改文件。

## 六、修改文件清单

| 文件 | 修改行数 | 对应问题 |
|---|---|---|
| `code/src/utils/io.py` | +124/−20 | 问题 1、3、4（守卫）、补充 D |
| `code/src/experiments/e1_state_diagnosis.py` | +49/−6 | 问题 1、2、补充 B |
| `code/src/experiments/e3_cogent_diagnosis.py` | +45/−7 | 问题 1、2、3、补充 A |
| `code/src/features/cache.py` | +51/−2 | 补充 C |
| `code/scripts/summarize_seed_stability.py` | +27/−2 | 问题 4 |
| `code/src/data/cogent.py` | +30/−8 | 补充 A |
| `code/src/experiments/common.py` | +27 | 补充 C |
| `code/scripts/extract_embeddings.py` | +17/−4 | 补充 A |
| `code/scripts/analyze_e3_units.py` | +6/−3 | 问题 3 |
| `code/src/experiments/e2_change_diagnosis.py` | +12/−2 | 问题 1、独特比较数披露 |
| `code/src/metrics/delta_stats.py` | +7/−2 | 独特比较数披露 |
| `code/src/metrics/matching.py` | +12 | 补充 C |
| `code/src/experiments/e1_spar_official.py` | +8/−3 | 问题 1、2 |
| `code/configs/e1_state.yaml` | +8/−4 | 补充 B（注释与协议说明） |
| `code/configs/e3_cogent.yaml` | +3 | 补充 A（配置生效说明） |
