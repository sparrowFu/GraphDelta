# GraphDelta 代码修改报告

修改日期：2026-10-27（第一轮）、2026-10-27（复核缺口修复，第二轮）、2026-10-27（逐项确认验证）
审查基线：`ef329ad`
依据：[代码审查报告](GraphDelta_代码审查报告_2026-10-26.md)（第三节八项 + 第五节建议）、[修复复核报告](GraphDelta_修复复核报告_2026-10-27.md)（第二节五个缺口 + 第三节两处说明）
修改构成：第一轮 15 个文件 +376/−50，已提交（含于 `ed00c79`..`463fa35`）；第二轮 10 个文件追加修改，累计相对基线 16 个文件 +528/−58，**第二轮与逐项确认截至本报告更新时在工作区未提交**

## 一、结论

审查报告第三节八项问题（问题 1–4、补充 A–D）、第五节唯一代码建议（披露独特比较数），以及修复复核报告确认的五个缺口和两处说明，全部处理完毕，并已逐项确认。

验证共四组受控检查，全部通过：第一轮 32 项；第二轮缺口专项 22 项 + 回归 8 项；确认轮按复核报告的每个证据场景逐项复现，14 项检查连续两次执行全部通过，另补测两条回归路径（E3 冲突、缺参考主表）。验证方法与未覆盖范围见第六节。

两处按复核意见更正的表述：
1. **CoGenT 缓存 key 不再承诺"默认保持不变"。** key 恒带指纹且包含数据根目录；历史不带后缀的 key 与第一轮生成的 `@` 后缀 key 均不通用，首次使用需重抽一次。
2. **`n_cmp_unique_*` 列为无序比较对口径。** 余弦相似度对称，(a,b) 与 (b,a) 算同一条比较；第一轮实现为有序对，第二轮已改。

## 二、修改总览

| 项 | 来源 | 最终判断 | 处理位置 |
|---|---|---|---|
| 问题 1：失败运行把旧结果标成新来源 | 审查 P2 + 复核缺口 1/2 | 已解决 | io.py、四个实验入口、汇总与绘图入口 |
| 问题 2：smoke 更新全量结果的来源记录 | 审查 P2 | 已解决 | e1 / e1_spar_official / e3 入口 |
| 问题 3：E3 重跑后旧单元表仍 passed | 审查 P2 + 复核缺口 3 | 已解决 | e3 入口、analyze_e3_units、io.py |
| 问题 4：稳定性汇总接受 failed 探索结果 | 审查 P2 | 已解决 | summarize_seed_stability、make_figures |
| 补充 A：E3 路径配置未生效 | 审查 P3 + 复核缺口 4/5 | 已解决 | cogent.py、e3 入口、extract_embeddings.py |
| 补充 B：E1 两项 SPAR 配置不生效 | 审查 P3 | 已解决 | e1 入口、e1_state.yaml |
| 补充 C：异常特征缺少评分前校验 | 审查条件性缺口 | 已解决 | cache.py、common.py、matching.py |
| 补充 D：replaces 同版本重跑消失 | 审查次要缺口 | 已解决 | io.py |
| 第五节建议：披露独特比较数 | 审查建议 + 复核说明 2 | 已实现（无序口径） | delta_stats.py、e2 入口 |
| 说明 1：默认缓存 key 表述更正 | 复核第三节 | 已更正 | e3_cogent.yaml、本报告 |

## 三、逐项说明

### 问题 1：失败运行会把旧结果标记成新来源 [P2]

**原问题。** 实验入口在运行开始时改写 `experiments/<name>` 的当前来源。运行在数据读取、配对验证或写盘中途失败时，CSV 还是旧内容，来源记录已指向失败的尝试。复核进一步确认两个缺口：从未成功过的实验连续失败两次，第一次的失败身份被迁移逻辑误写成 `last_success`；部分写出（如 E2 第一张表已写、第二张表未写）后，正式汇总照常消费。

**修复内容（两轮合并）。**

1. 来源分成尝试与成功两层（`src/utils/io.py:62` `output_tables_dir`、`:116` `commit_experiment`）：入口只登记尝试（`status=running` + 身份 + `started`）；输出文件全部成功写出后由 `commit_experiment` 提交 `last_success`（含产物清单 `outputs`），`status` 置 `success`。四个实验入口在最后一张表写完后调用提交（e1 `:342`、spar_official `:395`、e2 `:385`、e3 `:228`）。
2. 旧 schema 迁移只匹配明确的三轮旧格式（`io.py:98`）：条件为"无 `last_success` **且无 `status`** 且有身份字段"。新格式的 running 记录带 `status`，不再被推定成历史成功来源（复核缺口 1）。
3. 正式消费守卫（`io.py:226` `verify_experiment_committed`，复核缺口 2）：`status=running` 拒绝；`last_success.outputs` 产物清单不含目标文件也拒绝；无 run_meta（旧布局）与旧 schema 记录放行。`summarize_seed_stability.py:119` 对每个种子检查 e1_state / e1_spar_official / e2_pug_animals / e3_cogent 四个实验，该检查无条件执行，`--allow-exploratory-e3` 不能绕过；`make_figures.py:263` 对 e1/e2/e3 做同样检查。

**修改后行为（受控复现）。**

```text
旧 CSV 存在、新运行中途停止：status=running，last_success 仍指旧成功来源，CSV 不变
连续两次失败（从未成功）：不产生任何 last_success
E2 第一张表写出后中断：正式汇总停止，不生成 seed_stability.csv；探索开关也不能绕过
全部实验提交完成：汇总正常生成正式结果
AAA→BBB→BBB→CCC 提交序列：replaces=[AAA, BBB]，last_success=CCC（补充 D 一并覆盖）
旧 schema 目录：身份仍被识别为历史成功来源，迁移有效
```

### 问题 2：smoke 更新全量结果的来源记录 [P2]

**原问题。** 三个入口固定使用 `e1_state` / `e1_spar_official` / `e3_cogent` 作为来源键，smoke 与全量共用。

**修复内容。** smoke 模式加 `_smoke` 后缀：`e1_state_diagnosis.py:236`、`e1_spar_official.py:167`、`e3_cogent_diagnosis.py:78`，与 E2 的 `e2_<dataset>_smoke` 约定一致。

**修改后行为。** e3 受控复现：先跑 smoke，experiments 中只有 `e3_cogent_smoke`（含自己的产物清单）；再跑全量后两个键并存，互不改写。

### 问题 3：E3 成功重跑后，旧单元表仍显示 passed [P2]

**原问题。** `run_e3` 重写主表和 items 后不失效派生单元表；消费守卫只看 sidecar 状态。复核进一步确认：分析流程先写 passed sidecar、后写单元表，分析中断时旧单元表配新 passed sidecar 可被放行。

**修复内容（两轮合并）。**

1. E3 全量成功后主动失效（`e3_cogent_diagnosis.py:215` 起）：先原子写 sidecar `status=stale`，再删除旧 `e3_unit_gaps.csv`。失效放在全部源文件写出之后，失败的中断运行不破坏仍有效的派生表。
2. 分析流程重排（`scripts/analyze_e3_units.py:78` 起，复核缺口 3）：开始先写 `pending` 使旧状态立即失效；单元表用临时文件 + rename 原子写出（`:170`）；成功后才提交最终 sidecar，记录源输入指纹 `source_fingerprint` 与单元表内容哈希 `unit_table_md5`（`:173`）。严格模式检查失败：删除旧表、写 failed sidecar 后停止；缺参考主表：停止且 sidecar 保持 pending（旧 passed 不残留）。
3. 消费守卫（`io.py:281` `verify_e3_consistency`）共拦截五类情况：`failed`、`stale`、`pending`、源文件指纹不匹配、单元表内容哈希不匹配。指纹与哈希函数在 `io.py:273` `md5_file`、`:209` `e3_source_fingerprint`。

**修改后行为（受控复现）。**

```text
正常链：E3 → analyze(passed+双哈希) → 守卫放行
E3 重写：旧单元表删除、sidecar=stale → 守卫拦截 → 重跑 analyze 恢复 passed
单元表被改写/旧表顶替：unit_table_md5 不匹配 → 拦截
分析中断：sidecar 停在 pending → 拦截
源文件（items/主表）变化：指纹不匹配 → 拦截
E3 重跑中断 + 旧无指纹 sidecar：sidecar 层按兼容规则放行，
    但正式汇总被实验来源检查（问题 1 守卫，status=running）拦截
E3 冲突（items 复算与主表不一致）：严格模式停止 + 删旧表 + sidecar=failed；
    --no-strict 输出探索表但守卫拦截
```

### 问题 4：稳定性汇总接受 failed 的探索结果 [P2]

**原问题。** 绘图入口有 E3 一致性检查，汇总没有；`--no-strict` 探索产物可进入正式 `seed_stability.csv`。

**修复内容。** `summarize_seed_stability.py`：逐种子调用 `verify_e3_consistency`，failed / stale / pending / 指纹或哈希不匹配默认停止；新增 `--allow-exploratory-e3`，显式探索模式输出到独立的 `seed_stability_exploratory.csv`，不写正式文件。实验来源完整性由问题 1 的无条件检查覆盖（不能被该开关绕过）。

**修改后行为。** 三种子中 seed 1 sidecar 为 failed 时：默认停止、不写正式 CSV；加开关后完成，输出独立文件。

### 补充 A：E3 的场景与图像路径配置未生效 [P3]

**原问题。** `conditions.<cond>.scene_file / image_dir` 从未被读取，入口与独立抽取脚本都用固定路径。复核确认两个配套缺口：`--force` 删除的是不带指纹的旧 key 文件，实际抽取用的带指纹文件未被删、也未重抽；key 指纹只含相对路径字符串，切换数据根目录后复用原数据特征。

**修复内容（两轮合并）。**

1. 路径生效：`src/data/cogent.py:13` `load_condition_scenes`、`:18` `condition_image_dir` 接受配置参数，缺省保持原固定路径；e3 入口与 `extract_embeddings.py` 统一读取配置传入。
2. 缓存身份（`cogent.py:22` `condition_cache_key`，复核缺口 4/5）：key = `base@md5(数据根目录|scene_file|image_dir)[:12]`。指纹**包含解析后的数据根目录**——只换根目录、保留相对配置与同名 image id 也得到不同 key。key 恒带指纹（复核说明 1 的如实口径），与历史裸 key 及第一轮 `@` 后缀 key 均不通用。
3. `--force` 同 key（`extract_embeddings.py:70` `_cogent_key_cfg`）：生效 key 在 force 之前解析，删除与抽取共用。

**修改后行为（受控复现）。**

```text
默认场景文件不存在、配置指向自定义路径：e3.run 正常完成（修复前 FileNotFoundError）
默认相对路径 + PNG 内容改变后 --force：删除目标即带指纹 NPZ，encoder 对全部图像重新编码（修复前新增编码 0 次）
只换数据根目录（相对配置与文件名不变）：key 变化，encoder 重新编码，旧缓存字节不变
```

### 补充 B：E1 的两项 SPAR 过滤设置不生效 [P3]

**原问题。** `same_animal_filter` 从未被读取，`two_object_only` 传入但未使用，改配置不产生消融效果。

**修复内容。** 按审查报告允许的第一方案（拒绝不支持的选项）：`e1_state_diagnosis.py:40` `_validate_spar_protocol` 在数据读取前校验——`same_animal_filter: false` 或 `two_object_only` 与固定协议不一致时立即报错；`build_spar_items` 删除未使用参数；`configs/e1_state.yaml` 注释注明两项为固定协议描述。未实现为可生效开关：报告明确不能为生效放开会产生相同或同时为真正负文本的过滤。

**修改后行为。** 当前默认配置通过校验，正常运行不变；两种非法改动在运行前被拒。

### 补充 C：异常特征缺少评分前校验 [条件性数值防护缺口]

**原问题。** 缓存可存取含 NaN 的特征；评分前不检查。NaN 的 `s_pos > s_neg` 恒 False 被计成答错；图文维度不一致时 `broadcast_to` 静默扩展，指标含义错误。

**修复内容。** 三层校验，发现异常明确停止并给出定位（模型名、缓存路径、dataset_key、首个异常行号 / image_id）：
1. 缓存层（`src/features/cache.py:52` `_check_features`、`:69` `_check_ids_rows`）：`ImageFeatureCache.save/arrays`、`TextCache._load/encode` 校验二维形状、有限值、ids/行数一致。
2. 评分边界（`src/experiments/common.py` `eval_2afc_items`）：图文维度严格相等（`common.py:53`）、有限值检查。
3. 指标兜底（`src/metrics/matching.py` `per_template_scores`）：三者形状一致 + 有限值，否则 `ValueError`。

**修改后行为。** save 含 NaN、load 含 Inf 的旧缓存、图文维度 1 vs 2、指标函数输入 NaN、TextCache 新编码块含 NaN 五类场景全部拦截；正常特征行为不变。

### 补充 D：replaces 会在同版本重复运行时消失 [次要留痕缺口]

**修复内容。** `io.py` `commit_experiment` 中 `replaces` 为追加列表：身份变化时追加旧来源，同版本重跑不追加也不丢失；旧单条 dict 格式读取时自动转列表。

**修改后行为。** AAA→BBB→BBB→CCC 后 `replaces=[AAA, BBB]`、`last_success=CCC`；旧 dict 格式迁移保留。

### 第五节建议：披露独特比较数（含复核说明 2 更正）

**修复内容。** `src/metrics/delta_stats.py:97`：每组比较记录 `n_unique_pairs`，按**无序**对计数（`tuple(sorted(...))`，余弦对称，(a,b) 与 (b,a) 同一条）；`e2_change_diagnosis.py` 写入结果 CSV 的 `n_cmp_unique_*` 列，注释注明无序口径。

**修改后行为。** 两行数据互相比较的受控用例：抽样 100 条、无序独特对 1。原 `n_cmp_*`（抽样次数）列保留。

## 四、结果消费链守卫总览

正式消费者（`summarize_seed_stability`、`make_figures`）入口处的四层检查：

| 函数 | 位置 | 拦截对象 | 兼容 |
|---|---|---|---|
| `verify_run_meta` | io.py:153 | 目录 model/seed 与请求不符 | 无 run_meta 跳过 |
| `verify_tables_identity` | io.py:180 | 无身份又非已知旧布局的目录 | 已知旧布局按固定属主核对 |
| `verify_experiment_committed` | io.py:226 | status=running 的实验；产物清单外的文件 | 无 run_meta / 旧 schema 放行 |
| `verify_e3_consistency` | io.py:281 | E3 派生表 failed/stale/pending、源指纹或单元表哈希不匹配 | 无 sidecar / 旧 sidecar（无指纹）放行 |

写入端配套：实验产物由 `commit_experiment` 提交清单；sidecar 与单元表均原子写出（`io.py:265` `write_text_atomic`）。

## 五、不修改项及理由

审查报告第五节明确"不是本轮新增程序 BUG"的各项：

| 项 | 理由 |
|---|---|
| E1/E3 不能证明未提及信息被保留/遗忘 | 研究设计边界，需新增成对控制实验与指标，无代码错误对象 |
| CoGenT A/B 差异可能混有 shape 主效应 | 需补充实验设计分离，非代码错误 |
| CI 含 0.5 的解释、跨种子一致的充分性 | 结论措辞问题，属研究总结文档 |
| crop/JPEG、raw embedding 尺度边界 | 实现确定，报告未指出实现错误 |
| Norm Gap 两侧聚类单位衔接 | 统计假设或敏感性分析，需方案层决定；报告明确不宜机械改 ID |
| max_total 截断非峰值上限 | 资源实测项，无代码错误对象 |
| 数据版本 / goldfish / checkpoint 身份核实 | 需服务器真实数据，属后续验收 |

## 六、验证方法与边界

**方法。** 四组自写检查，全部使用真实代码入口、确定性假 encoder（无 torch）、临时目录、受控场景/PNG/CSV，无真实权重与真实数据集：

- 第一轮 32 项：来源生命周期（含失败中断、身份链、旧格式迁移）、e3 全流程（smoke/全量键、自定义路径、缓存 key、stale、指纹）、汇总 failed 拦截、SPAR 配置校验、特征校验五层。
- 第二轮缺口专项 22 项：连续失败不虚构来源、中断实验被汇总拒绝（含开关不可绕过、全提交后正常）、unit_table_md5 与 pending 拦截、缺参考主表保持 pending、stale 链回归、force 删对目标并重抽、换根目录不复用、无序独特对。
- 第二轮回归 8 项：身份链、失败不改写 last_success、自定义/默认路径、SPAR 校验、特征校验、failed sidecar 拦截与探索文件。
- 确认轮（对照复核报告逐个证据场景）：14 项检查连续两次执行全部通过，覆盖五个缺口各自的复现路径与两处说明；其中 `interrupted_e3_old_sidecar` 场景（E3 重跑中断 + 旧无指纹 sidecar，sidecar 层放行）确认为由实验来源检查在汇总入口拦截。另补测 E3 冲突路径（严格停止 + 删表 + failed；探索输出但守卫拦截）与缺参考主表路径（严格停止且保持 pending）。

**复核第四节六项验收回归的口径。** 第二轮重排了 `analyze_e3_units` 流程，与之直接相关的两项（E3 冲突、缺参考主表）已按上一条重新实测通过；其余四项（ImageNet smoke、按实验更新来源正常路径、全因素为空、非对象 JSON）相关代码两轮零改动（`cache.py` 协议部分、E2 smoke 筛选部分），以复核报告在 `ed00c79` 上的回归结果为准。

**边界。**

1. 未加载真实 checkpoint、未跑真实数据端到端；验证覆盖来源生命周期、失效链、消费守卫、配置解析与数值校验逻辑，不覆盖 VLM 数值（与审查/复核报告边界一致）。
2. CoGenT 缓存 key 计算两轮共变化两次（加路径指纹、再加数据根目录）：历史裸 key 与第一轮 `@` key 均需重抽，这是把数据身份纳入 key 的必要代价。
3. 实验写多个文件仍非原子：两张表之间崩溃会留下新旧混合状态，此时该实验 `status=running`，正式消费被第四节守卫拒绝；未实现暂存目录式整体提交。
4. 存量旧 sidecar 无指纹/哈希，无法核对内容身份（守卫放行）；新跑的 `analyze_e3_units` 起全部带双哈希。

## 七、修改文件清单

相对审查基线 `ef329ad` 累计 16 个文件 +528/−58。第一轮（已提交至 `463fa35`）15 个文件 +376/−50；第二轮（工作区未提交）在 10 个文件上追加：

| 文件 | 累计行数 | 对应问题 | 第二轮是否追加 |
|---|---|---|---|
| `code/src/utils/io.py` | +196/−24 | 问题 1、3、4；补充 D | 是（迁移条件、消费守卫、pending/哈希拦截、原子写） |
| `code/src/experiments/e3_cogent_diagnosis.py` | +49/−10 | 问题 1、2、3；补充 A | 是（stale 原子写与顺序、key 含根目录） |
| `code/scripts/analyze_e3_units.py` | +41/−8 | 问题 3 | 是（pending 先行、原子写、unit_table_md5） |
| `code/scripts/summarize_seed_stability.py` | +42/−6 | 问题 1、4 | 是（实验来源无条件检查） |
| `code/scripts/extract_embeddings.py` | +32/−7 | 补充 A | 是（force 用同一有效 key） |
| `code/src/data/cogent.py` | +37/−11 | 补充 A | 是（key 含数据根目录、恒带指纹） |
| `code/scripts/make_figures.py` | +7 | 问题 1 | 是（消费守卫，新文件） |
| `code/src/experiments/e1_state_diagnosis.py` | +49/−6 | 问题 1、2；补充 B | 否 |
| `code/src/features/cache.py` | +51/−2 | 补充 C | 否 |
| `code/src/experiments/common.py` | +27 | 补充 C | 否 |
| `code/src/metrics/matching.py` | +12 | 补充 C | 否 |
| `code/src/experiments/e2_change_diagnosis.py` | +13/−2 | 问题 1；独特比较数 | 是（无序口径注释） |
| `code/src/metrics/delta_stats.py` | +9/−3 | 独特比较数 | 是（无序计数） |
| `code/src/experiments/e1_spar_official.py` | +8/−3 | 问题 1、2 | 否 |
| `code/configs/e1_state.yaml` | +8/−4 | 补充 B | 否 |
| `code/configs/e3_cogent.yaml` | +5 | 补充 A | 是（key 身份说明更正） |
