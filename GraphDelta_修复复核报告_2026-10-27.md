# GraphDelta 修复复核报告

复核日期：2026-10-27  
复核代码：`ed00c792d051ff0195c4ad21286806deb3f56778`（`ed00c79`）  
依据：[代码审查报告](GraphDelta_代码审查报告_2026-10-26.md)、[代码修改报告](GraphDelta_修改报告_2026-10-27.md)、当前 `code/` 源码及独立受控测试。

## 一、结论

**不能确认所有问题均已解决。** 修改确实修复了多数原始触发，但修改报告“全部通过修改代码解决”的结论过强。

按审查报告第三节的八项逐项判断：**五项原问题已解决，三项部分解决、仍不宜关闭**。问题 1 的失败运行生命周期、问题 3 的派生结果提交顺序仍有漏洞；补充 A 的路径读取已修复，但缓存身份和 `--force` 的配套修改不完整。

这些是源码检查及临时数据上的复现结果，不表明服务器上的真实结果已经受到影响。

| 原审查项 | 本次判断 | 核查结果 |
|---|---|---|
| 问题 1：失败运行把旧结果标成新来源 | **部分解决** | 单次失败保留已提交的 `last_success`；但连续失败会虚构成功来源，多文件部分写出后仍能进入正式汇总 |
| 问题 2：smoke 覆盖全量来源键 | **已解决原问题** | E1、SPAR 官方、E3 均区分 smoke/full 键；E3 两种模式实际写出各自成功记录与产物清单 |
| 问题 3：E3 重跑后旧单元表仍 passed | **部分解决** | 正常成功重跑会删除旧表、置 stale，指纹校验有效；分析在单元表写出前提交 passed，失败恢复仍可能放行旧表 |
| 问题 4：正式稳定性汇总接受 failed 探索表 | **已解决原问题** | failed/stale 默认停止；显式探索输出独立 CSV。上游错误 passed 属问题 3 的剩余缺口 |
| 补充 A：E3 配置路径未生效 | **路径读取已修复，缓存配套未完整解决** | 自定义场景和图像路径实际生效；缓存 key 与 force 删除目标不一致，且未纳入数据根目录 |
| 补充 B：E1 两项 SPAR 配置静默无效 | **已解决原问题** | 默认配置通过；不支持的 false/空列表取值在读取数据前明确拒绝，符合原报告允许的处理方式 |
| 补充 C：异常特征缺少校验 | **已解决原问题** | NaN/Inf、二维形状、行数、raw/normalized 形状、图文维度错配均有拦截；原始 NaN 计错和广播用例不再成立 |
| 补充 D：同版本重跑丢失 replaces | **已解决原问题** | AAA→BBB→BBB→CCC 后保留 `[AAA, BBB]`；旧单条 dict 迁移到列表也通过 |

## 二、仍需处理的具体缺口

### 1. 连续失败运行被误迁移成历史成功来源 [P2，问题 1]

位置：[io.py:99](code/src/utils/io.py#L99)。

旧格式迁移的判断只有“没有 `last_success` 且存在身份字段”，没有区分旧格式记录与新格式 `status=running` 记录。第一次从未成功的运行失败后，正好满足这个条件。

使用实际 `e2.run()`，连续两次在 0 pair 检查处停止，复现如下：

```text
第一次：身份 failed_A，status=running，无 last_success
第二次：身份 failed_B，status=running，last_success.git_commit=failed_A
目录内 CSV 数量：0
```

从未产出成功结果，却出现成功来源。它不是“失败状态留为 running”的单纯措辞问题，而是迁移逻辑写出了错误的成功记录。

建议迁移仅匹配明确的旧 schema；新格式未完成记录不能推定成功，成功来源必须由产物提交产生。

证据：[provenance/results.json](code/outputs/audit_recheck_2026-10-27/provenance/results.json) 中 `P1_repeated_failed_run_fabricates_success`。这里的 `P1` 是脚本对“问题 1”的编号，严重级别仍为 P2。

### 2. 部分写出的新结果仍被正式汇总消费 [P2，问题 1]

位置：[E2 第一张表写出](code/src/experiments/e2_change_diagnosis.py#L361)、[E2 第二张表写出及提交](code/src/experiments/e2_change_diagnosis.py#L380)、[目录校验](code/src/utils/io.py#L148)、[汇总读取 E2](code/scripts/summarize_seed_stability.py#L212)。

修改报告已披露多文件写出不原子，并称读取端“可据此区分”。实际读取端只校验模型/种子，未据实验 `status` 阻止正式消费。

本次先用真实 E2 流程完成一次成功运行；第二次受控改变 delta，在第一张 CSV 写出后、第二张表计算前抛异常：

```text
e2_delta_stats.csv：新内容
e2_transition_stats.csv：旧内容
实验 status：running
last_success：仍指向上次成功
实际 summary.main：成功生成正式 seed_stability.csv
正式汇总中的 E2 数值：来自本次未完成运行
```

保留旧 `last_success` 本身有效，但不足以证明磁盘上现有文件属于该成功运行。建议采用暂存后提交，或按运行目录/产物清单组织；正式消费者必须拒绝未完成、产物身份无法对应的实验。

证据：同一结果文件的 `P1_partial_output_consumed_by_official_summary`。测试对 delta 乘 3 仅用来区分两次输出，不涉及真实模型结论。

### 3. E3 分析先发布 passed，后写单元表 [P2，问题 3 的中断恢复缺口]

位置：[analyze_e3_units.py:109](code/scripts/analyze_e3_units.py#L109)、[单元表写出 :158](code/scripts/analyze_e3_units.py#L158)。

分析在计算和写出单元表之前，已经把 sidecar 更新为 `passed`，并绑定当前源文件指纹。该指纹证明的是源文件身份，不能证明旧单元表由这些源文件生成。

受控模拟 E3 源结果更新后旧单元表删除失败，再模拟分析写单元表失败，得到：

```text
当前主表 Gap：1
旧单元表 Gap：0，字节未变化
sidecar：passed，源指纹与当前源文件匹配
verify_e3_consistency：放行
实际 summary.main：正式文件生成，旧 Gap=0 单元被判 stable=True
```

这是需要“旧表仍存在且分析随后失败”的条件性缺口；**正常成功重跑的原始用例已通过**，不能将上述复现描述为每次 E3 成功重跑仍会残留旧表。

建议分析开始时发布 pending/stale，在新单元表安全写出后再提交 passed，并将单元表本身与该次分析绑定。旧 sidecar 无指纹的兼容路径也不会检查实验 running 状态，E3 中途失败时仍可放行旧表，应一并纳入来源检查。

证据：[e3/results.json](code/outputs/audit_recheck_2026-10-27/e3/results.json) 中 `analysis_failure_publishes_passed_for_old_units`、`summary_consumes_stale_units_after_analysis_failure` 和 `interrupted_e3_old_sidecar`。

### 4. CoGenT 的 --force 删除错误缓存键，可能完全未重抽 [P2，补充 A 配套回归]

位置：[extract_embeddings.py:83](code/scripts/extract_embeddings.py#L83)、[删除目标 :59](code/scripts/extract_embeddings.py#L59)、[实际抽取键 :100](code/scripts/extract_embeddings.py#L100)。

force 仍把裸 `args.dataset`（如 `cogent_valA`）交给 reset；实际抽取已使用 `cogent_valA@<hash>`。在协议一致时，reset 删除的是不带指纹的 NPZ，实际使用的文件仍在。

使用真实抽取脚本 main 分支、PNG 和确定性 encoder，先缓存图片、改变原图内容，再执行 force：

```text
实际缓存：cogent_valA@09271e9668a8_images.npz
force 删除目标：cogent_valA_images.npz
force 后 encoder 新增图像调用：0
实际缓存 NPZ 字节：完全未变
```

默认 YAML 明确写了非空路径，也会生成带指纹键，因此不只自定义路径受影响。协议损坏或不匹配时整目录删除仍有效，这是另一条分支，不能用其通过证明同协议 force 正确。

建议在 reset 前解析最终有效缓存键，由删除和抽取共同使用。

证据：[config/config_results.json](code/outputs/audit_recheck_2026-10-27/config/config_results.json) 中 `A_force_fails_to_reextract`。

### 5. CoGenT 缓存身份未包含数据根目录 [P2，条件性来源缺口]

位置：[cogent.py:24](code/src/data/cogent.py#L24)。

key 只由相对 `scene_file|image_dir` 字符串决定，没有包含 `paths.datasets.cogent`。在同一缓存目录下切换到另一套数据根目录、保留相对配置和同名 image IDs，会复用原数据特征。

本次用内容不同的两套 PNG/场景目录验证：只切换根目录后，encoder 新增调用为 0，缓存字节不变。因此修改报告“更换数据源不能复用同名旧特征由 key 指纹直接保证”的说法不成立。路径配置确实传到了 loader，但缓存命中会跳过新图像编码。

建议缓存身份至少纳入解析后的数据根目录和有效路径，并明确数据版本/内容变更的失效策略。

证据：同一配置结果文件的 `A_dataset_root_not_in_cache_identity`。

## 三、两处次要说明不准确

1. **默认缓存键没有保持不变。** `condition_cache_key` 只有两个参数均空时才保留旧键，但默认 YAML 的两个字段均非空。默认 A 的实际键是 `cogent_valA@9d14557edd2e`，不是 `cogent_valA`。修改报告的兼容承诺不符；这本身主要意味着缓存不能直接复用、可能额外重抽，不证明得分错误。
2. **独特比较数已披露，但为有序端点对数。** [delta_stats.py:95](code/src/metrics/delta_stats.py#L95) 将 `(a,b)` 与 `(b,a)` 分别计数；受控三次比较的有序数为 3，无序余弦比较数为 2。代码注释确实写了“独特 `(d1,d2)` 组合”，本次不另算 P2；建议列说明明确有序口径，或按需要改为无序比较覆盖。

另外，修改报告中的“当前工作区未提交”是历史描述。复核开始时工作区干净，报告及这批代码已在 `ed00c79` 提交；这不影响上述代码结论。

## 四、原六项验收的回归情况

| 原验收项 | 本次结果 |
|---|---|
| ImageNet smoke | 实际 E2 运行：192 行筛至 96 行，四因素各 48 pairs |
| E3 冲突时旧表与消费守卫 | 严格模式停止、旧单元表删除、sidecar failed、守卫拒绝 |
| 按实验更新来源 | 正常成功提交有效，独立实验键存在；失败及中断边界仍见问题 1 |
| 全因素为空 | allow-empty 下明确 SystemExit，没有空 concat |
| 非对象 JSON 恢复 | `[]`、`null`、字符串、损坏 JSON 读取拒绝，force 对临时未知来源目录重建有效 |
| 缺参考主表状态 | 严格模式停止；no-strict 为 not_checked、all_ok=false、空指纹 |

额外使用保存的官方 ImageNet 元数据样本回归：18 个实例、36 个 camera pairs、跨实例配对 0；重复元数据被拒绝。该样本回归不等于全量 ImageNet 覆盖验收。

## 五、证据、复跑与边界

本轮独立脚本和结果保存在新目录，未覆盖旧审计证据：

| 范围 | 脚本 | 结果 |
|---|---|---|
| 来源生命周期、历史、部分写出 | [verify_provenance.py](code/outputs/audit_recheck_2026-10-27/provenance/verify_provenance.py) | [results.json](code/outputs/audit_recheck_2026-10-27/provenance/results.json) |
| E3 失效、分析、消费、旧修复 | [verify_e3.py](code/outputs/audit_recheck_2026-10-27/e3/verify_e3.py) | [results.json](code/outputs/audit_recheck_2026-10-27/e3/results.json) |
| 配置、模式键、缓存衔接 | [verify_config.py](code/outputs/audit_recheck_2026-10-27/config/verify_config.py) | [config_results.json](code/outputs/audit_recheck_2026-10-27/config/config_results.json) |
| 数值防护、smoke、配对、JSON | [verify_numeric.py](code/outputs/audit_recheck_2026-10-27/numeric/verify_numeric.py) | [results.json](code/outputs/audit_recheck_2026-10-27/numeric/results.json) |

在仓库根目录运行：

```powershell
& D:\conda\python.exe -B code/outputs/audit_recheck_2026-10-27/provenance/verify_provenance.py
& D:\conda\python.exe -B code/outputs/audit_recheck_2026-10-27/e3/verify_e3.py
& D:\conda\python.exe -B code/outputs/audit_recheck_2026-10-27/config/verify_config.py
& D:\conda\python.exe -B code/outputs/audit_recheck_2026-10-27/numeric/verify_numeric.py
```

四份脚本均已执行完成。**完成包含缺口复现，不能把检查数量解释成修复通过数量。** 另对 41 个实验/脚本 Python 文件做了语法解析，结果正常；结果文件记录源码哈希。

测试使用真实入口、临时 parquet/CSV/JSON/PNG、确定性替代 encoder，以及指定阶段的异常注入。没有修改实验源码、配置、真实缓存或旧报告，也没有下载或加载真实权重。临时场景已清理，脚本和机器可读证据保留。

原审查报告第五节的研究解释、统计假设、构建峰值内存、数据版本、goldfish 敏感性和 checkpoint 验收边界仍然存在；它们不混算为第三节八项修复失败，也没有因本次代码测试而自动解决。尤其尚未完成五数据集、三模型的真实端到端验收。

建议先修复来源迁移与正式消费守卫、E3 单元表/sidecar 提交顺序、CoGenT 有效缓存键与数据根目录身份，再做真实数据 smoke 和必要结果重算。
