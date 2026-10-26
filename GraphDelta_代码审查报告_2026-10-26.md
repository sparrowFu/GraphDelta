# GraphDelta 代码审查报告

审查日期：2026-10-26  
代码基线：`7b68232`  
审查范围：`code/` 的数据读取、单因素配对、模型分发、特征缓存、E1/E2/E3 实验、统计汇总与绘图流程。

## 一、审查结论

**当前代码已具备五个数据集、三个模型的实验入口，主要诊断流程能够在已覆盖的构造用例中运行，但仍存在三项 P2 问题和三项 P3 边界问题，尚不适合直接认定为完整验收通过。**

优先处理的问题是：

1. ImageNet 的 smoke 筛选可能将具有有效配对的数据错误筛空。
2. E3 一致性检查失败后，旧单元表仍可被绘图消费。
3. 复用结果目录时，来源元数据仍可能保留首次运行的代码和协议身份。

本次没有发现新的、已证实的主评分公式错误。这里的“未发现”仅限已检查源码与用例，不是无 BUG 保证；真实权重和全量图像的端到端验收尚未完成。

优先级为审查判断：P2 表示会影响受支持流程或正式结果来源，建议在正式实验前处理；P3 表示输入边界、恢复能力或状态表达问题。触发条件与实际影响在各项中单独说明。

## 二、数据集和模型支持情况

### 2.1 数据集

| 数据集 | 当前实现 | 验收边界 |
|---|---|---|
| CLEVR_v1.0 | E1；读取 val；对象、属性、关系与 binding-style 文本负例 | 已接入，未完成当前环境的真实图像端到端验收 |
| CLEVR_CoGenT_v1.0 | E3；读取 valA/valB；A/B 匹配难度与 sphere 对照 | 已接入；一致性检查到绘图的失效链仍有问题 |
| PUG_Animals | E2；单因素配对、表示变化、视觉/文本 nuisance | 已接入；局部诊断的全部因素为空边界仍需处理 |
| PUG_ImageNet | E2；实例级 context、额外姿态/灯光固定字段、独立输出 | 实例配对已验证；smoke 筛选有回归 |
| PUG_SPAR | E1 自建 2AFC 与独立官方协议入口 | 描述性汇总、world 匹配和不适用指标处理已通过对应验证 |

不同数据集承担不同诊断任务，不需要在每个数据集上都运行 E1/E2/E3。入口存在与真实数据、真实模型验收通过是两个层次。

### 2.2 模型

| 指定模型 | CLI 配置名 | 当前分发 |
|---|---|---|
| siglip2-so400m-patch14-384 | `siglip2_so400m` | `SigLIP2Encoder`，FixRes 加载路径 |
| clip-vit-base-patch32 | `clip_vitb32` | `CLIPEncoder` |
| clip-vit-large-patch14 | `clip_vitl14` | `CLIPEncoder` |

配置与权重路径由 [paths.yaml](D:/code/GraphDelta/code/configs/paths.yaml:9) 和[模型工厂](D:/code/GraphDelta/code/src/models/__init__.py:11)分发。三个模型均按冻结 baseline 使用；当前阶段不训练 GraphDelta 模块，符合阶段一方案。

上述结论是配置和实现层面的支持确认，不表示本轮已加载三个真实 checkpoint 或完成官方输出数值对齐。

## 三、已确认的有效行为

### 3.1 ImageNet 实例约束与重复项检查

`character_name` 已参与固定 context，类别与具体物体实例不再混为同一身份。对保存的官方连续 100 行元数据样本，实际 loader/matcher 输出：

| 指标 | 结果 |
|---|---:|
| 配对数 | 66 |
| context 数 | 18 |
| 配对端点覆盖的实例数 | 18 |
| 跨实例配对数 | 0 |

重复 `(context,factor_value)` 输入会明确报错，不再静默覆盖。这是有限样本验证，不能外推为全量数据没有重复项或所有因素覆盖充分。

### 3.2 汇总中的 NA、符号与环境维度

- SPAR 描述行的 `stable` 为 NA，不再混入肯定的稳定数量。
- hard-negative 未定义的 top5 不再输出为有效指标行。
- E3 使用原始 Gap 判断符号，再对显示值舍入；`[+1e-5,-1e-5,+1e-5]` 正确判定为方向不一致。
- SPAR 按 world 独立匹配；测试输入 desert 的 `[0.1,0.9,0.1]` 正确保留，最大差为 0.8，重复键被拒绝。

### 3.3 局部诊断、缓存与 nuisance 分批

- 某因素无配对时，严格模式明确停止；允许局部诊断时，具有有效配对的因素可以继续运行。
- 已全部命中的 PUG 图像请求会直接返回，不调用图像迭代器；部分命中只向迭代器传入缺失行。
- 截断的协议 JSON 可以通过 force 的未知来源分支处理，越界缓存路径被拒绝。
- nuisance 采用分批生成、编码和释放。200 图、模型批大小 37 的替代 encoder 验证中，实际批最大为 37，图像 ID 顺序及 raw/normalized delta 与一次性参考一致。

替代 encoder 的数值一致不等于真实 fp16 GPU 输出逐位一致；当前证据支持控制流程、增强种子和批处理顺序的正确性。

## 四、需要处理的问题

### 问题 1：ImageNet smoke 筛选可能错误清空有效数据 [P2]

位置：[e2_change_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:146)、[pug_imagenet.py](D:/code/GraphDelta/code/src/data/pug_imagenet.py:31)。

**原因。**smoke 通过 `spec['meta_cols'][:2]` 决定筛选字段。ImageNet 的前两列现在是实例哈希与类别；分别取字典序前六个值后再 AND，两个相关集合可能没有交集。字段顺序变化也使该逻辑不再限制 environment。

**已复现事实。**构造 12 个实例和 12 个类别，使实例排序与类别排序相反；每实例具有完整的 2 environments × 2 textures × 2 sizes × 2 cameras，共 192 行。使用实际 parquet loader 和 run：

```text
完整输入：四个因素各有 96 对有效配对
smoke 筛选后：0 行
运行结果：SystemExit，提示 environment 0 对配对
```

这是筛选程序造成的数据丢失，不是源数据本来没有因素覆盖。尚未证明用户完整真实数据的默认 smoke 必然触发；`smoke=False` 不执行该筛选分支。

**建议。**按数据集显式定义 smoke 字段，避免依赖 `META_COLS` 的位置。多个相关字段应基于已选子集逐层筛选，或按实例采样后保留必要的因素变化覆盖；筛选后区分“筛选失败”和“源数据无配对”。

### 问题 2：E3 严格检查失败后，旧派生表仍可被绘图使用 [P2]

位置：[analyze_e3_units.py](D:/code/GraphDelta/code/scripts/analyze_e3_units.py:102)、[make_figures.py](D:/code/GraphDelta/code/scripts/make_figures.py:166)。

**原因。**严格检查只阻止写入新单元表，没有失效已经存在的 `e3_unit_gaps.csv`；绘图函数不核对 `e3_unit_consistency.json`。

**已复现事实。**先用一致输入成功生成单元表，再将主表改为与 items 冲突，严格模式正确退出并写 `all_ok=false`，但旧单元 CSV 原样保留。独立实际绘图验证仍能读取旧表并生成图像。

**影响边界。**仅在输入冲突、部分重跑或旧派生文件存在时触发，不代表正常一致输入的计算错误。

**建议。**检查失败时失效旧派生表，或要求下游强制校验失败状态与输入身份。探索性输出应使用独立名称或显式标记，避免正式绘图误用。

### 问题 3：复用目录时，来源元数据可能记录旧代码/协议 [P2]

位置：[io.py](D:/code/GraphDelta/code/src/utils/io.py:70)。

**原因。**`output_tables_dir` 只在 `run_meta.json` 不存在时写入。旧格式文件不会补齐新增字段；相同目录重跑也不会更新或拒绝不同的来源身份。

**已复现事实。**对实际 helper 传入两次受控身份：

```text
第一次：commit=A，cache_protocol=v1
第二次：commit=B，cache_protocol=v2
第二次磁盘记录仍为：commit=A，cache_protocol=v1
```

若随后覆盖同目录的 CSV，记录的来源可能仍是首次版本。没有证据说明服务器现有真实结果已误标，也不直接影响评分公式。

**建议。**按实验/运行保存来源，或在身份不同时拒绝复用目录。不能简单覆盖一个共享 meta，因为目录中未重跑的其他实验文件可能仍属于旧版本。代码 commit 和 `git_dirty` 也不能独立替代配置、数据和 checkpoint 身份。

### 问题 4：全部因素跳空后，局部诊断仍抛内部异常 [P3]

位置：[e2_change_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:178)、[delta_stats.py](D:/code/GraphDelta/code/src/metrics/delta_stats.py:38)。

`allow_empty_factors=True` 且全部因素无有效配对时，空字典继续进入 `build_delta_frame`，实际抛 `ValueError: No objects to concatenate`。

默认严格模式和存在有效因素的局部模式可以正常处理。建议在没有任何有效因素时，给出明确的停止消息，不进入后续统计。

### 问题 5：协议 JSON 为非对象时，缓存恢复仍失败 [P3]

位置：[extract_embeddings.py](D:/code/GraphDelta/code/scripts/extract_embeddings.py:47)、[cache.py](D:/code/GraphDelta/code/src/features/cache.py:27)。

`[]`、`null`、字符串都是可解析 JSON，但不是协议对象。直接调用 `.get()` 会抛 AttributeError，当前异常处理没有覆盖；实际 force 因此失败，旧文件保留。

这是非法协议结构的恢复边界，不是正常协议下错误复用特征。建议先验证解析结果是 dict，再检查 protocol 字段，其余结构按来源未知处理。

### 问题 6：缺少参考主表时，检查状态被记录为通过 [P3]

位置：[analyze_e3_units.py](D:/code/GraphDelta/code/scripts/analyze_e3_units.py:77)、[analyze_e3_units.py](D:/code/GraphDelta/code/scripts/analyze_e3_units.py:99)。

主表不存在时没有进行比较，但实际输出 `all_ok=true, checks=[]` 并生成单元表。应区分 `passed`、`failed`、`not_checked`；严格模式可要求参考主表存在。

完整绘图目前会因主表缺失另行失败，因此不能将本问题描述为“缺表也能生成正式图”。问题在于检查状态表意不正确。

## 五、研究解释、数据质量和成本边界

以下事项不作为新增程序 BUG 计数，但会限制结果解释。

1. **CoGenT 的 shape 混杂。**同颜色在 A/B 中对应不同的 cube/cylinder。单元差异可能来自 shape 基础识别难度，不能直接归因为颜色—形状绑定能力；sphere 对照不能完全排除该解释。当前主指标按题型等权，也没有强制颜色/shape 单元等权。
2. **未显著不等于等效。**区间包含 0.5 只能说明未检出偏离随机水平；符号跨种子一致也不等于效应显著或对新数据稳健。
3. **信息保留尚无直接检验。**当前 E1/E3 是真假 caption 匹配，E2 文本 nuisance 是同内容换模板；没有完整/不完整描述成对控制及未提及信息保留指标，不能据此声称已验证信息保留或遗忘。这是研究覆盖边界，不违反现阶段冻结 baseline 方案。
4. **nuisance 强度需要解释。**面积/JPEG 参数实现一致不保证 crop 无语义影响；Factor/Nuisance Gap 对扰动强度敏感。不同模型的 raw projection 尺度也不同，不应直接按 raw delta 给模型排语义敏感性。
5. **资源评估需看构建过程。**nuisance 全量视图驻留已被分批降低，但 pair 的 `max_total` 在记录构建完成后才截断，不是峰值内存上限。应测量全量 metadata 配对耗时与峰值，不能由 smoke 成功推断全量资源充足。
6. **数据质量和版本仍需核实。**官方 Animals 提示部分 goldfish 图像渲染异常；这不证明用户当前版本仍包含这些图，也不能仅凭整体比例小就认定各因素无影响。[官方数据质量提示](https://raw.githubusercontent.com/facebookresearch/PUG/main/PUG_Animals/README.md) 数据 revision、分片顺序、checkpoint 和配置身份尚未被完整校验；没有证据时应列为条件性风险，而非认定实际结果已污染。

## 六、证据和验证范围

### 6.1 可重放材料

- [验收脚本](D:/code/GraphDelta/code/outputs/audit_7b68232_2026-10-26/verify_acceptance.py)
- [机器可读结果](D:/code/GraphDelta/code/outputs/audit_7b68232_2026-10-26/acceptance_checks.json)
- [保存的官方元数据样本](D:/code/GraphDelta/code/outputs/audit_2026-10-26/imagenet_metadata_sample.json)

重放命令：

```powershell
& D:\conda\python.exe D:\code\GraphDelta\code\outputs\audit_7b68232_2026-10-26\verify_acceptance.py
```

结果包含 41 个实验 Python 文件的源码哈希和 AST 检查，以及 14 类证据。**14 类证据不是“14 项问题均已修复”或“14 个真实模型测试通过”**，其中包含已修行为验证和残留问题复现。

官方元数据样本为连续 100 行，API offset=10000、length=100，不是随机或全量代表性样本。文件 SHA256：

```text
9bd0c94e5fd2c0affe176a61a4b12e1ea61af8bcd660fe74eadd4ec32f81d1c7
```

实际绘图、路径越界守卫、部分缓存增量等另做了独立临时验证，没有全部纳入该脚本。报告不将不同来源的检查数量相加作为通过数量。

### 6.2 本次文档生成时的核对

本次重新生成报告时，已确认：

- 当前代码基线仍为 `7b68232`。
- 41 个实验源码文件与保存的验收结果哈希一致。
- 验收脚本哈希与结果记录一致。

因此本报告采用已保存的验收证据，没有声称本次又重复执行了全部测试。

### 6.3 未覆盖范围

当前环境无 torch/transformers，缺少服务器上的真实权重、全量图像和历史结果。构造流程使用替代 encoder；没有完成真实 tokenizer/encoder 数值对齐，也没有完成五数据集、三个模型的真实端到端验收。

实验源码本轮未修改。报告中的“已确认有效”仅针对相应路径和用例。

## 七、处理顺序建议

1. 修正 ImageNet smoke 的显式筛选规则，检查筛选后的因素覆盖。
2. 闭合 E3 检查失败到绘图的失效链，避免使用旧派生表。
3. 为重复运行保存准确的实验级来源，不用共享目录 meta 冒充所有文件的统一版本。
4. 补齐全部因素跳空、JSON 结构、未检查状态三个低成本边界。
5. 再进行真实数据小样本、模型预处理对齐及必要的结果重算。

这些问题不要求无条件重抽所有模型、所有数据集。数据版本和行号映射稳定时，可优先复用原图特征，重做筛选、配对、统计和来源记录。
