# GraphDelta 代码审查报告

审查日期：2026-10-26  
补审更新：2026-10-27  
代码基线：`ef329ad`  
范围：`code/` 中的数据筛选、配对、模型入口、缓存恢复、实验输出、来源记录、E3 一致性检查、稳定性汇总与绘图；补审配置传递、异常特征输入和统计假设。

## 一、结论

**本次六项修复的原始触发用例均已通过对应验证，但不能据此认定所有问题已经关闭。**

ImageNet smoke、全部因素为空、非对象 JSON 恢复、E3 冲突时删除旧表、缺主表状态区分，以及按实验更新来源记录的正常路径，都已生效。仍有四条 P2 级别的运行生命周期或消费链缺口：

1. 实验失败前已改写结果来源，旧 CSV 却未变化。
2. E1/E3 的 smoke 与全量结果共用来源身份键。
3. E3 成功重跑后，旧单元表和 passed 状态没有自动失效。
4. 稳定性汇总仍会消费明确标记为 failed 的探索单元表。

**不能将上述四项理解为“代码只剩四个问题”。**扩大检查还确认两项 P3 配置缺陷：E3 配置的数据路径未生效、E1 部分过滤开关未生效；另复现了异常特征可生成有效外观准确率的条件性防护缺口。此前发现的 `replaces` 历史丢失也仍存在。它们的触发范围和优先级不同，不能都算作默认实验必然错算。

这些结论基于源码及受控流程复现，不表示服务器当前真实结果已经受影响。P2/P3 是本报告对处理优先级的判断，实际触发条件分别列明。审查和有限测试不能证明不存在其他 BUG，尤其尚未完成真实数据、真实权重的端到端验收。

## 二、六项修复验收

| 修复项 | 本轮验证结果 | 判断 |
|---|---|---|
| ImageNet smoke 筛选 | 192 行反序实例/类别完整网格，筛后 96 行；四因素各 48 pairs，实际 run 完成 | 原筛空触发已解决 |
| E3 检查失败后的旧表与绘图 | 冲突使严格模式停止、删除旧单元表，failed 守卫拒绝消费 | 原 failed 路径已解决；其他生命周期见问题 3/4 |
| 按实验更新来源 | A/v1→B/v2 更新 experiments，保留一次 replaces，不同实验可独立记录 | 正常完成路径已改善；与文件产出关联仍有问题 1/2 |
| 全部因素跳空 | allow-empty 下明确 SystemExit，不再进入空 pd.concat | 已解决 |
| 非对象 JSON | []、null、字符串被读取端拒绝，force 按未知来源重建 | 已解决 |
| 缺参考主表的状态 | 严格模式明确停止；探索模式为 not_checked、all_ok=false | 已解决 |

“原始触发通过”和“整个功能在所有运行阶段正确”不同。尤其来源记录不能只验证 helper 更新，还需要核对它是否对应实际成功写出的文件。

## 三、仍需处理的问题

### 问题 1：失败运行会把旧结果标记成新来源 [P2]

位置：[e2_change_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:143)、[io.py](D:/code/GraphDelta/code/src/utils/io.py:84)。其他实验入口也采用运行开始时登记来源的方式。

**事实。**`output_tables_dir()` 在数据读取、配对验证和结果写出前更新 `experiments/<name>`。记录中没有区分尝试、失败和成功，也没有与输出文件提交绑定。

实际 `e2.run()` 受控复现：

```text
预置 CSV 来源：old_success
新运行：failed_attempt，在 0 pair 检查处停止
CSV：字节完全未变，仍为旧结果
experiments.e2_pug_imagenet：已记录 failed_attempt
```

因此，正常 helper 更新的修复并未保证“当前来源”代表成功产出的结果。该问题在已有结果目录中重跑失败时触发；不等于所有成功运行都被误标。

**建议。**分别记录 attempt 和 last_success；仅在关联输出成功写出后更新成功来源。若输出过程可能部分完成，应使用 pending/failed 状态阻止正式消费，或以独立运行目录/产物清单提交。不能只在入口覆盖当前来源。

### 问题 2：smoke 更新全量结果的来源记录 [P2]

位置：[e1_state_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e1_state_diagnosis.py:206)、[e1_spar_official.py](D:/code/GraphDelta/code/src/experiments/e1_spar_official.py:168)、[e3_cogent_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e3_cogent_diagnosis.py:76)。

**事实。**三个入口分别固定使用 `e1_state`、`e1_spar_official`、`e3_cogent` 作为实验键；但 smoke 与全量输出是不同文件。E2 已区分 `_smoke`，这三个入口没有区分。

实际 `e3.run(smoke=True)` 受控复现：

```text
全量 e3_cogent.csv：保持 full_B 的内容，未改写
smoke CSV：成功生成
experiments.e3_cogent：被更新为 smoke_C
独立 smoke 来源键：不存在
```

E3 的流程已实际执行；E1 两入口的同名键模式由源码确认。不能把一次 smoke 成功等同于全量结果重新生成。

**建议。**统一区分模式，例如 `e1_state_smoke`、`e1_spar_official_smoke`、`e3_cogent_smoke`；更稳妥的是直接给每组输出文件绑定来源。目录首次身份可以保留，但不能作为所有产物的统一当前身份。

### 问题 3：E3 成功重跑后，旧单元表仍显示 passed [P2]

位置：[e3_cogent_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e3_cogent_diagnosis.py:133)、[io.py](D:/code/GraphDelta/code/src/utils/io.py:154)。

**事实。**`run_e3` 改写主表和 items 时，没有失效 `e3_unit_gaps.csv` 或其一致性状态；消费守卫只看 sidecar 的状态，不核对当前输入身份。

使用实际 E3 写出流程、替代加载/抽取/评分，复现：

1. 第一次运行产生主 Gap=1，随后分析生成单元 Gap=1、状态 passed。
2. 第二次成功运行产生主 Gap=0，items 也对应 0。
3. 旧单元表字节未变，非缺失 Gap 仍为 1；sidecar 仍是 passed，守卫放行。

这里的 0/1 是受控评分，不是真实模型性能。它证明正常重跑会保留过期派生状态。若每次都重新执行完整分析，可能不会触发最终错用；不能因此把状态守卫视为已经验证了新输入。

**建议。**E3 源结果更新时，将派生结果标记为 pending/失效；分析通过后记录源输入指纹，绘图和汇总核对指纹。仅凭一个历史 passed 标记，不能证明当前文件一致。

### 问题 4：稳定性汇总接受 failed 的探索结果 [P2]

位置：[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:224)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:252)。

**事实。**绘图入口已调用 `verify_e3_consistency()`，稳定性汇总没有相同检查。

本轮实际运行三个种子的 `analyze_e3_units --no-strict`，生成 failed sidecar 与探索单元表；随后执行实际 summary main：

```text
三个种子 sidecar：failed / failed / failed
正式 seed_stability.csv：仍生成
对应单元 stable：True / True
```

这不是说符号计算本身出错，而是已知无效的输入仍进入正式结论。正常严格模式会删除失败表，原触发有效；该缺口出现在受支持的探索模式与另一个消费者之间。

**建议。**正式汇总也验证状态和输入身份；如允许探索性比较，应显式选择模式、记录状态并输出独立文件，不能静默混入正式稳定性结果。

### 补充 A：E3 的场景与图像路径配置未生效 [P3]

位置：[e3_cogent.yaml](D:/code/GraphDelta/code/configs/e3_cogent.yaml:2)、[e3_cogent_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e3_cogent_diagnosis.py:88)、[cogent.py](D:/code/GraphDelta/code/src/data/cogent.py:5)。

**事实。**入口读取 `conditions[cond]` 后没有使用其中的 `scene_file`、`image_dir`；loader 固定读取 `scenes/CLEVR_val{A,B}_scenes.json` 和 `images/val{A,B}`。独立抽取脚本也使用这组固定路径。

实际调用 `e3.run()`：配置指向已存在的 `custom/A.json`，仍尝试打开默认场景文件并抛 `FileNotFoundError`；补上默认场景后，捕获到传给抽取函数的目录仍是 `images/valA`，不是配置的 `custom_images/A`。

**影响边界。**默认配置与固定路径一致，未证明默认官方目录结果错误；改用自定义路径会失败，或在默认目录也存在时读取非预期文件。建议统一实验入口和抽取脚本的路径解析，或删除不可配置的字段并明确限制。更换数据源时还需核对缓存身份，不能只改路径后复用同名旧特征。

### 补充 B：E1 的两项 SPAR 过滤设置不生效 [P3]

位置：[e1_state.yaml](D:/code/GraphDelta/code/configs/e1_state.yaml:18)、[e1_state_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e1_state_diagnosis.py:146)、[e1_state_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e1_state_diagnosis.py:260)。

**事实。**`same_animal_filter` 从未读取，`two_object_only` 虽传入构造器却未使用；过滤逻辑固定执行。`attribute_require_diff_texture` 则确实参与逻辑，不能把所有配置都说成无效。

实际执行两次 `e1.run()`，仅把 `same_animal_filter: true` 改成 `false`、把 `two_object_only` 改成空列表；真实生成的 SPAR item JSONL 和过滤统计完全相同，同种动物仍过滤，单动物仍只进入 scene。

**影响边界。**默认规则符合当前有效负例约束，未证实默认结果错误，但配置修改不能产生用户预期的消融。建议删除或拒绝不支持的选项，或设计明确且仍保证正负文本有效的替代协议；不能为了让开关生效就直接放开会产生相同或同时为真的正负文本的过滤。

### 补充 C：异常特征缺少评分前校验 [条件性数值防护缺口]

位置：[cache.py](D:/code/GraphDelta/code/src/features/cache.py:66)、[common.py](D:/code/GraphDelta/code/src/experiments/common.py:36)、[matching.py](D:/code/GraphDelta/code/src/metrics/matching.py:13)。

**已复现的处理行为。**缓存能保存并重新读出含 NaN 的特征，评分前未拒绝：NaN 相似度的 `s_pos > s_neg` 比较返回 False，随后被当作答错。实际 `eval_2afc_items()` 输出 `accuracy=0`、准确率 CI 为 `[0,0]`，而 margin 为 NaN。无效观测因而可能被表现为确定的失败表现。

另一个用例中，图像特征维度为 1、文本维度为 2，`np.broadcast_to()` 自动扩展图像末维，流程仍输出有限准确率，没有报告图文特征维度不匹配。

该用例的原始图文输入各自为单位向量，广播后输出 margin 约为 2.8284，超过合法单位向量余弦差的上界 2，说明维度错配确实改变了指标含义。

**尚未证实的触发来源。**测试使用人工异常特征；没有证据表明三个真实 checkpoint 已输出 NaN、错误维度，或真实缓存已经含此类数据。它与正常生命周期的四项 P2 分开列示，不能据此断言现有模型得分已污染。

**建议。**缓存写入/加载和评分边界检查二维形状、ID/行数一致性、图文维度严格相等及有限值；发现异常时明确停止，记录模型、缓存和样本位置。不要把 NaN 自动计为答错，也不要通过静默删样本改变评价集合。

### 补充 D：replaces 会在同版本重复运行时消失 [次要留痕缺口]

位置：[io.py](D:/code/GraphDelta/code/src/utils/io.py:86)。

每次构造新的 rec，只有身份变化时才写 replaces。已复现：A→B 时存在 replaces，随后 B→B 重跑该字段消失。它不改变当前评分，但“变更留痕一直保留”的表述不完整。

如果需要可审计历史，建议将运行记录追加到历史列表或独立记录，而不是每次重建当前 rec。这里不要求保存无限历史，也不将其另算作主要评分 BUG。

## 四、数据、模型和主流程状态

| 数据集 | 对应流程 | 当前状态 |
|---|---|---|
| CLEVR_v1.0 | E1 val 状态/属性/关系诊断 | 入口保留；来源生命周期问题需要处理 |
| CLEVR_CoGenT_v1.0 | E3 valA/valB、sphere 对照 | 主表写出和严格检查可运行；派生状态与消费链仍有缺口 |
| PUG_Animals | E2 单因素响应、nuisance | 空因素处理有效；失败运行的来源记录仍有问题 |
| PUG_ImageNet | E2 实例固定与辅助分析 | smoke 和全空保护有效；有限真实元数据样本实例约束未回归 |
| PUG_SPAR | E1 自建 2AFC、官方协议 | world、NA、符号相关既有实现本轮未改变 |

三个指定模型的配置与工厂仍支持：

| 模型 | CLI 名称 |
|---|---|
| siglip2-so400m-patch14-384 | `siglip2_so400m` |
| clip-vit-base-patch32 | `clip_vitb32` |
| clip-vit-large-patch14 | `clip_vitl14` |

模型加载代码本轮未改。本次未复现三个模型在正常特征输入下的新增主评分公式错误；补充 C 证明的是异常特征输入的处理缺口。没有加载真实权重，不能把入口和构造流程成功写成三模型真实端到端验收通过。

## 五、解释和资源边界

以下事项不是本轮新增程序 BUG：

- 当前 E1/E3 仍不能直接证明未提及信息被保留或遗忘；它们没有完整/不完整描述成对控制及相应指标。
- CoGenT 单颜色 A/B 差异仍可能混有 shape 主效应和候选构成；不能只凭该差异归因为组合绑定能力。
- 区间包含 0.5 是未检出偏离随机水平，不是等效证明；跨种子符号一致也不是新数据稳健性的充分证据。
- 图像 crop/JPEG 的实现确定不代表绝对无语义改变，raw embedding 尺度也不能跨模型直接排序。
- nuisance 分批有效，但 pair 的 max_total 仍在全部记录构造后截断，不是峰值内存上限。“约 12 万行可承受”没有本轮资源实测支持；截断后的规模不能代表构建峰值。
- E2 Norm Gap 的因素端按 context、nuisance 端按原图聚类，两侧标签不同，未联动重采样重合图像的依赖。方案未明确两种聚类单位如何衔接；这是需要说明或做敏感性分析的统计假设，不能直接断言 CI 必然偏窄，也不宜只把两侧 ID 改成相同字符串。
- `_distinct_pairs` 保证每对两个索引不同，但比较之间有放回；`n_cmp` 表示抽样次数，不保证相同数量的独特比较。对称 cosine 的均值估计在均匀抽样下并不因此有偏；小样本重复会降低有效覆盖，建议另披露独特比较数，不将此直接算成新 P2。
- 真实数据版本、goldfish 数据质量敏感性、checkpoint/配置身份及完整 ImageNet 覆盖尚未在服务器上核实。它们是待验证条件，不应无证据认定现有结果已污染。

研究总结措辞未改写、数据敏感性未实施，不应混算为上述六项代码修复失败；同样不能因为代码修复成功，就自动升级研究结论的可信度。

## 六、可重放证据

- [本轮验证脚本](D:/code/GraphDelta/code/outputs/audit_ef329ad_2026-10-26/verify_current.py)
- [机器可读结果](D:/code/GraphDelta/code/outputs/audit_ef329ad_2026-10-26/verification_results.json)
- [配置补审脚本](D:/code/GraphDelta/code/outputs/audit_ef329ad_2026-10-26/verify_config_supplement.py)及[配置复现结果](D:/code/GraphDelta/code/outputs/audit_ef329ad_2026-10-26/config_supplement_results.json)
- [异常特征补审脚本](D:/code/GraphDelta/code/outputs/audit_ef329ad_2026-10-26/verify_numeric_supplement.py)及[数值复现结果](D:/code/GraphDelta/code/outputs/audit_ef329ad_2026-10-26/numeric_supplement_results.json)
- [保存的官方元数据样本](D:/code/GraphDelta/code/outputs/audit_2026-10-26/imagenet_metadata_sample.json)

运行：

```powershell
& D:\conda\python.exe -W ignore::RuntimeWarning D:\code\GraphDelta\code\outputs\audit_ef329ad_2026-10-26\verify_current.py
& D:\conda\python.exe D:\code\GraphDelta\code\outputs\audit_ef329ad_2026-10-26\verify_config_supplement.py
& D:\conda\python.exe D:\code\GraphDelta\code\outputs\audit_ef329ad_2026-10-26\verify_numeric_supplement.py
```

脚本记录 41 个实验 Python 文件的源码哈希及 12 类证据，包含原触发修复验证和剩余问题复现。**12 类不是“12 项缺陷均修复”或真实模型测试数量。**命令中的 warning 过滤仅用于减少受控稀疏单元的空均值提示，不更改计算；也可不加该参数运行。

2026-10-27 已重新运行两份补审脚本：配置路径/开关复现和异常特征复现均与本报告一致。补审脚本成功是复现检查完成，不表示所发现的缺陷已修复。

验证采用真实代码入口、临时 parquet/CSV、保存的元数据和替代 encoder/评分。E3 生命周期用例替代加载、抽取及评分，只验证写出、来源和失效流程，不验证 VLM 数值。

本轮为独立用例复核，不把修复说明所称的 21 项、66 项等计数相加作为本轮通过数量。当前环境有 NumPy/Pandas/PyArrow/Pillow/YAML/Matplotlib，无 torch/transformers 与服务器全量数据/权重。

本轮未修改实验源码或真实缓存；测试文件在 workspace 临时目录中创建并清理。没有完成五数据集、三模型的真实 tokenizer/encoder 对齐和全量端到端验收。

## 七、建议处理顺序

1. 将输出来源分成运行尝试与成功产物，成功写出后再提交 last_success，失败状态不能冒充成功结果来源。
2. 给 E1/SPAR/E3 的 smoke 与全量分配独立来源键或产物记录。
3. 在 E3 源结果更新时失效派生状态，分析通过后绑定输入指纹。
4. 对稳定性汇总加入同样的失败/来源检查，将探索输出与正式输出分开。
5. 修正或移除不生效的配置，在评分前加入特征有效性校验，明确统计近似与独特比较覆盖。
6. 然后执行真实数据 smoke、模型预处理对齐和必要结果重算。

当前没有依据要求无条件重抽所有模型、所有图像。优先修正来源与消费链；数据身份稳定时，可复用特征并重做必要的配对、统计或派生分析。
