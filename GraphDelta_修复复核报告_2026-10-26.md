# GraphDelta 修复复核报告（2026-10-26，独立核实修订版）

基线：当前实验源码提交 `3cc6dae`。本次重新检查代码，也检查上一版审查结论的前提、证据强度和优先级。修改范围为审查文档及审查证据；没有修改实验实现，没有运行真实模型或删除实际缓存。

**结论：原 A1–A7 的明确触发路径基本已有修复，不能把新增保护需求一概算成原问题没有修。当前仍有需要修复的代码问题，其中 ImageNet 实例标识丢失已得到真实元数据样本支持；另发现 SPAR 开启逐环境输出时汇总取错行。现有证据不足以认定全部实验结果错误，也不足以认定五数据集、三模型已经通过真实端到端验收。**

## 1. 对前版报告的纠正

| 前版表述或隐含前提 | 本次独立判断 |
|---|---|
| A4、A6 一律记为“部分解决” | 过于笼统。原 NaN-CI 判定、已知 new/legacy 混用触发已修复；其他汇总缺陷和来源风险分别记录 |
| 构造跨实例 pair 可代表真实数据已被污染 | 推论过强。构造证明算法允许该错误；真实连续样本证明实例覆盖压缩，但本次真实样本跨实例 pair 为 0 |
| HF Viewer 显示一个值，故完整 ImageNet 因素只有一个值 | 不成立。局部展示不能外推全量；官方说明明确有多个背景、尺度、纹理 |
| E3 舍入符号变化属于主要实验错误 | 优先级偏高。确有描述性符号 BUG，但只涉及舍入到零附近的值，不证明显著效应方向改变；降为 P3 |
| 手填不同 protocol 字段仍被接受，证明现有结果混用 | 不成立。只证明不检查该字段；真实结果是否混用无证据，转列条件性来源风险 |
| CI 包含 0.5 就说明“与随机无差别” | 不成立。只能写“未检出偏离 0.5”；等效结论需要预设有实际意义的容差及相应分析 |
| 同 encoder 语义意味着 force 重抽图像必须清空全部文本缓存 | 不成立。语义改变/来源未知时整套失效有依据；同协议的图像重抽不必导致文本重抽 |
| 合成数据加替代 encoder 跑通证明流程与逻辑整体正确 | 范围过宽。只证明覆盖用例可运行，不能代替真实字段分布、同类多实例、真实预处理和全量资源验收 |

本报告区分：**源码/运行事实、条件性风险、研究解释限制、建议或优先级判断**。P1/P2/P3 是审查判断，不是测量值：P1 优先处理评估样本语义；P2 处理受支持功能或正式汇总的错误；P3 处理边界恢复、低幅度描述及性能问题。使用条件会影响优先级。

## 2. 原 A1–A7 的修复状态

| 原编号 | 原触发条件复核 | 尚需区分的事项 |
|---|---|---|
| A1：ImageNet 无抽取/实验入口 | 已接通 CLI、注册表、字段与独立输出 | 入口修复不等于严格配对验收，B1 是数据语义问题 |
| A2：无协议旧 NPZ 被认证；force 校验前失败 | 已修复：旧 NPZ 被拒绝、不补写协议；force 已前置 | B7 是损坏 JSON 的不同边界；同协议删文本是保守策略和成本 |
| A3：面积比例按边长实现 | 已修复：sqrt(s)、向上取整 | 仅证明参数实现一致，不证明 crop 保留语义；新旧 nuisance 不可混用 |
| A4：NaN CI 记为 ns/contains 且稳定 | 已修复：invalid 不判稳定 | SPAR 描述行和 E3 点符号是另两条路径，见 B3/B4 |
| A5：显式无身份目录冒充其他模型 | 原触发已修复；错误属主/未知无元数据目录被拒绝 | 不证明文件内容、数据版本或全部协议一致 |
| A6：已知新旧布局混用仍正式输出 | 已修复：默认拒绝，实际 main 验证通过 | 同 new 布局的版本追溯不足是条件性风险 R1 |
| A7：相似度比较超配置上限 | 已修复：构造输入四因素各组均不超过上限 100 | 不证明抽样量足够或全量资源充足 |

不能简单用“七项全未修”或“所有问题均已解决”概括。修复说明中“每对只有目标因素变化”仍缺少实例身份约束，这是需要优先纠正的具体结论。

## 3. 仍存和新增的代码问题

### B1 [P1，真实元数据样本已证实] ImageNet 配对丢弃实例身份，压缩实例覆盖

位置：[pug_imagenet.py](D:/code/GraphDelta/code/src/data/pug_imagenet.py:28)、[pug_imagenet.py](D:/code/GraphDelta/code/src/data/pug_imagenet.py:44)、[pair_matcher.py](D:/code/GraphDelta/code/src/data/pair_matcher.py:70)。

**事实：**`character_name` 读取后没有保留到返回 metadata 或固定 context；只固定 `character_label`。`by_val={factor_value:row}` 覆盖同一 context 内相同因素值的重复行。官方 ImageNet 为 88,328 图、724 个 assets、151 个类别，类别不是实例身份。[官方说明](https://raw.githubusercontent.com/facebookresearch/PUG/main/PUG_ImageNet/README.md)

本轮获取官方 API 连续 100 行，仅保存非 image 元数据：offset=10000、length=100，日期 2026-10-26。包含 18 个 assets、5 个类别；不是随机或全量代表性样本。[官方样本 API](https://datasets-server.huggingface.co/rows?dataset=facebook%2FPUG_ImageNet&config=default&split=train&length=100&offset=10000)

实际 loader/matcher 结果（camera、seed=0、每 context 上限 4）：

| 实现 | pairs | contexts | 端点覆盖 assets | 跨实例 pairs |
|---|---:|---:|---:|---:|
| 当前源码 | 17 | 5 | 5 | 0 |
| 仅在内存中追加 character_name 并固定实例 | 66 | 18 | 18 | 0 |

94 行处于重复的 `(context,camera)` 单元。可以确认该真实样本实例覆盖被压缩。**不能把 66−17 直接称为漏掉 49 个完整合法 pair**：加入实例改变了 context 预算单位，两边均受每 context 上限影响；也不能估计全量比例。

另用构造数据复现跨实例 pair：同类 asset_A/camera0 与 asset_B/camera90，其余字段相同，当前实现会配对。**跨实例污染是算法可能性，本次真实样本未观测到。**

建议：保留并固定 `character_name`；显式检查 `(context,factor_value)` 唯一性；披露类别、实例、transition 覆盖。修复后检查权重和统计，不能只加列就认定协议不变。若数据版本及行号映射不变，可复用原图特征，优先重配 pair 和统计，不必无条件重抽 GPU 特征。

附带事实修正：`character_label` 是 `African_crocodile` 等字符串，不是修复说明所称无法用于文本的编号。[官方 HF 样本](https://huggingface.co/datasets/facebook/PUG_ImageNet) 当前 [caption](D:/code/GraphDelta/code/src/data/captions.py:125) 省略类别属于内容覆盖选择，可以改进，但不能据此推断视觉核心 Gap 算错。

### B2 [P3，条件性输入边界] 无有效 pair 的因素使 E2 抛 KeyError

位置：[pair_matcher.py](D:/code/GraphDelta/code/src/data/pair_matcher.py:100)、[e2_change_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:145)。

**事实：**空 records 生成无列 DataFrame；run 的日志访问 `p['context_key']` 失败。用合法字段类型、仅 camera 可变的两行临时 parquet，实际 loader + `run(smoke=True,dataset='pug_imagenet')` 在抽图前抛 `KeyError('context_key')`。

**未证实：**官方完整数据或默认 smoke 一定触发。前版 P2 定性偏强；只运行完整固定数据且全部因素可用时，它可能不影响主流程。

建议明确模式：严格协议可解释原因后停止；局部诊断可跳过空因素并记录缺失，不能把部分结果标为四因素完整完成。空表补 schema 不是完整修复，零长度抽样和缺失 Gap 也要处理。

### B3 [P2，汇总/计数] SPAR 不适用指标计入 stable=True

位置：[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:150)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:162)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:227)。

SPAR 分支明确只描述最大差、无稳定阈值，却固定 `add(...,True)`。hard-negative 的 top5 未定义，例如 [协议实现](D:/code/GraphDelta/code/src/experiments/e1_spar_official.py:291) 只输出 top1；实际 summary main 对三种子 top5 全 NaN 仍输出 stable=True，计入稳定总数。

**边界：**污染“多少项稳定”的汇总，不等于模型 top1 或 bootstrap 算错。建议不适用项跳过或记 NA，必需但缺失的项记 invalid；描述性最大差不计入肯定的 stable 数量。历史 115/109 等数值在原 CSV 缺失时不能重新确认，也不能沿用为修复后结论。

### B4 [P3，由 P2 下调] E3 先舍入再取符号

位置：[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:208)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:211)。

实际 main 输入 `[+1e-5,-1e-5,+1e-5]`，先舍入为 `[0,-0,0]`，再报告 stable=True、符号一致。确有精确符号与显示精度混用。

**不能推出：**显著方向被翻转，或全部 E3 结论失效。反例只涉及近零描述值。建议原值判定、最后显示时舍入；若需实质差异容差，应预先定义近零类别，而非隐含采用四位小数为科学阈值。

### B5（旧编号保留）转为条件性来源风险 R1

不再把“未验证自定义 protocol 字段”写为现有结果混用的确认事实。原 A6 的 new/legacy 路径已修复，扩展来源需求见第 4 节。

### B6 [P2，前提是输入已冲突] E3 强校验仅记录差异

位置：[analyze_e3_units.py](D:/code/GraphDelta/code/scripts/analyze_e3_units.py:65)、[analyze_e3_units.py](D:/code/GraphDelta/code/scripts/analyze_e3_units.py:77)。

受控输入：items 重算总 Gap=1，主表总 Gap=0。实际 main 记录差异后仍输出单元表。下游 Figure 5 同读主表和单元表，可能展示冲突来源。

这证明冲突未被阻止，**不证明正常同版本输出会算错**。建议检查有限性及容差；严格分析模式停止输出，探索模式则把未通过一致性检查写入结果而非仅日志。

### B7 [P3，恢复边界] 损坏协议 JSON 时 force 也失败

位置：[extract_embeddings.py](D:/code/GraphDelta/code/scripts/extract_embeddings.py:38)。

实际函数配合截断 JSON `{"protocol":`，force 分支抛 JSONDecodeError，旧 NPZ 未删除。只证明损坏缓存恢复不足，不证明正常运行复用错误或数据已损坏。

建议将读取/解析失败视为来源未知，提供明确恢复路径；可移入隔离目录后重建以保留追溯。任何删除前均应校验绝对目标路径。

### B8 [P2，新发现] SPAR 逐环境汇总遗漏 world

位置：[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:149)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:155)；逐环境输出由 [e1_spar_official.py](D:/code/GraphDelta/code/src/experiments/e1_spar_official.py:223) 支持。

**触发条件：**`spar_official.per_world=true`。key 和匹配均无 world，`.iloc[0]` 反复选第一行。

实际 summary main 最小 CSV：all 三种子 `[0.5,0.5,0.5]`，desert `[0.1,0.9,0.1]`。输出两条同 key，均为 `[0.5,0.5,0.5]`、maxdiff=0，desert 实际最大差 0.8 被丢失。

默认 per_world=false 不触发。建议将 world 纳入唯一键和匹配，验证每种子恰好一行，不能由 `.iloc[0]` 掩盖重复。

### B9 [P3，性能] 已缓存的 smoke 行仍重复扫描和解码

位置：[extract.py](D:/code/GraphDelta/code/src/features/extract.py:89)、[pug_common.py](D:/code/GraphDelta/code/src/data/pug_common.py:48)。

`only_row_ids` 分支在 expected_new=0 时不返回；迭代器读取 image 列并转换所有批次后才筛行。实际 1,000 行临时 parquet，请求 3 行且全已缓存：读取 image 列覆盖 1,000 行、解码 3 行、encoder 调用 0 次。

这是 I/O/解码开销，不是评分错误。先对请求集合减去已缓存行、空集立即返回是低成本修复；nuisance 随机抽样整库扫描需按 shard/row-group 定位。不解码其余图片不等于不读取压缩字节。

## 4. 条件性风险、遗漏变量与成本

### R1. 结果和缓存的来源身份不完整

**事实：**run_meta 仅写 model/seed/created，未检查协议/配置身份。A3/A7 修改后仍写同一 new 目录；缓存依赖别名、静态协议、按分片行序生成的 ID。

**条件风险：**不同版本分种子运行、替换 checkpoint、数据 revision 或分片顺序变化时，身份可能失效。本轮没有服务器缓存/真实 CSV，不能认定已触发。前版手填 protocol 的测试只证明不检查该字段。

低成本方案：每个实验记录配置/代码/预处理身份，数据记录 revision 与分片清单摘要；先保存清单，不必立即反复哈希全部大图。协议未知/改变时整套失效；同协议仅重抽图像不必清空模型全部文本。若历史错误逻辑已给旧 NPZ 写入当前 marker，新校验不能只凭该 marker 追溯生成方式；该服务器情形未经核实。此时匹配协议的单数据集 force 也不能证明其他数据集已重建，应先隔离完整受影响目录再逐项重建。

两个 TextCache 内存快照覆盖彼此新增项已复现，但仅支持增量缓存丢失/重复编码；未证明当前评分错误或真实 NPZ 损坏。串行可规避；需要同模型并行时再加写锁和原子合并。

### R2. smoke 成功不能预测全量资源可行

`_nuisance_visual` 先保留两组全部增强 PIL 图像，再分别编码。[代码](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:85)

配置 n_images=2000；**若原图为 512×512**，保留面积 85%–100%，仅三通道 RGB 逻辑像素载荷估计：

```text
2 × 2000 × 512 × 512 × 3 × [0.85,1] / 2^30
≈ 2.49–2.93 GiB
```

这是公式估计，不是实测 RAM 或 OOM 结论；未计 Pillow 内部存储、模型、cache/delta。按批生成/编码/释放可降低峰值。

max_total 在全部 pair records 建完后才截断，不是峰值内存上限。ImageNet 稀疏设计不能按 Animals 完整笛卡尔积预测覆盖/成本。先测全量 metadata 配对耗时和峰值内存，再安排模型；本报告不给未经测量的运行时长承诺。

### R3. 数据质量与样本权重

官方 Animals 提示约 192 张 goldfish 图像渲染异常。[官方提示](https://raw.githubusercontent.com/facebookresearch/PUG/main/PUG_Animals/README.md) 这不证明用户当前 HF/本地版本仍含全部异常。

整体少于 0.1% 不保证每个条件/transition 无影响。按方案完整组合，单 animal 有 64×3×4×4=3072 行；若 192 张集中于该 animal，内部比例为 6.25%。这是条件算术，不是图像缺陷实测。建议先核实版本，再做包含/排除的敏感性，不无记录删类。

固定实例后，max_total/每 context 上限改变类别、实例与 transition 权重，应报告覆盖。面向新物体外推是否按 asset 或分层 bootstrap 是协议选择，不能直接判当前 context bootstrap 为 BUG。三个种子通常复用图片，也不是三份独立数据。

## 5. 研究解释边界：运行正确不等于推论成立

### 5.1 CoGenT 单颜色差异不能排除 shape 主效应

[阶段总结](D:/code/GraphDelta/code/outputs/STAGE1_SUMMARY.md:54) 据单颜色 A/B 差异推出组合难度，依据不足。同一颜色在两条件对应不同 cube/cylinder，也改变了被测 shape。[配置](D:/code/GraphDelta/code/configs/e3_cogent.yaml:12)

反例：设完全不依赖颜色/A/B 的模型，cube 准确率 0.9，cylinder 0.3。当前调色板下，四色单元 Gap=+0.6，另四色 −0.6；理想等颜色平衡时平均 0。**这些数字是假设反例，不是真实性能。**它足以说明绑定困难不是唯一解释；sphere 对照不能排除 cube/cylinder 主效应。

实际主指标只按题型等权，没有强制颜色/shape 单元等权，故 pooled≈0 也不是必然的四对四算术抵消。

建议收窄为“按颜色分组的 A/B 匹配表现不同，与 shape/候选构成混杂”。进一步区分需 shape 基线、匹配候选、明确交互检验或受控对照。这是解释识别性不足，不是 Gap 加减法错误。

### 5.2 未显著、方向稳定、等效不同

[总结](D:/code/GraphDelta/code/outputs/STAGE1_SUMMARY.md:13) 的“与随机无差别”应改为“当前区间未检出偏离随机水平”。预设实际容差后才适合讨论等效。

16 单元已注明只描述，因此不能把未校正本身再列代码 BUG；但不能把多个描述性 CI 当作已控制整体误报的联合结论。三种子符号一致不等于显著，也不等于新数据/物体/模型稳健。

### 5.3 尚未直接检验未提及信息的保留

E1/E3 是已有图像与真假 caption 匹配；E2 文本 nuisance 是同内容换模板。[caption](D:/code/GraphDelta/code/src/data/captions.py:28)、[公共评估](D:/code/GraphDelta/code/src/experiments/common.py:29)

没有发现完整/不完整描述成对控制、明确遗漏机制和未提及信息保留指标，不能声称现有代码已经验证信息保留/遗忘。描述只覆盖部分图像内容，不等于开展了“完整 vs 不完整描述”实验。

这不违反当前阶段一方案，是长期任务/月度文字与现有证据间的覆盖缺口。现有实验也不是跨平台事件数据验证，不能直接推出事件匹配/回溯有效。

### 5.4 nuisance 与跨模型尺度并非天然受控

面积/JPEG 实现一致不保证 crop 无语义改变，仍可能影响对象可见范围。Gap 对 nuisance 强度敏感，建议预先固定或披露若干合理强度敏感性，避免看结果后选最有利基线。

不同模型 raw projection 尺度不同，不能直接按 raw Δ 给三个模型排语义敏感性。优先解释各模型内部相对 nuisance 的变化并同时报告 normalized 空间。冻结 baseline 的 CoGenT A/B 也不能称为本研究 seen/unseen；方案对此限制正确。

## 6. 可重放证据和边界

- [离线复核脚本](D:/code/GraphDelta/code/outputs/audit_2026-10-26/verify_review.py)：不下载图像/模型、不改实验源码。
- [机器可读结果](D:/code/GraphDelta/code/outputs/audit_2026-10-26/audit_checks.json)：记录 41 个实验 Python 文件的 SHA256/AST、运行库版本、19 项证据记录。**不是“19 个产品测试均通过”**，包含缺陷复现、反例和资源估计。
- [官方元数据样本](D:/code/GraphDelta/code/outputs/audit_2026-10-26/imagenet_metadata_sample.json)、[配对摘要](D:/code/GraphDelta/code/outputs/audit_2026-10-26/imagenet_matcher_summary.json)：来源 URL、区间、API 行号、日期、参数及非随机说明；没有图像字节或图像 URL。摘要明确区分规范化 rows 的哈希与实际保存文件的哈希，不能将任一值当作完整原始 API 响应的校验值。

本机重放：

```powershell
& D:\conda\python.exe D:\code\GraphDelta\code\outputs\audit_2026-10-26\verify_review.py
```

B9 I/O/解码计数、双缓存写入为独立临时验证，没有全部纳入脚本。修复说明所称历史 39 项检查未按其原脚本原样重放，不能与本次计数相加。

本机无 torch/transformers，服务器实际 checkpoint/数据/历史 CSV 不可用，未完成真实 tokenizer/encoder 对齐或五数据集×三模型端到端验收。三模型配置和分发已接通、本轮未变；指定 SigLIP2 FixRes 的 384/patch14/siglip 设置与[官方配置](https://huggingface.co/google/siglip2-so400m-patch14-384/raw/main/config.json)相符，但仍不等于权重测试。

## 7. 优先顺序（建议）

1. 修 ImageNet 实例 context，检查重复及覆盖；确认缓存身份后复用原图特征，优先重配 pair 和统计。
2. 修 SPAR world 键、NA/invalid 和 stable 总数；为 E3 冲突输入设置停止/标记规则。
3. 明确空因素的严格/局部模式；修符号舍入和损坏 JSON 恢复。
4. 收窄组合效应、随机等效、完整/不完整描述已验证等过强推论，再决定追加对照。
5. 补来源清单、数据质量敏感性和全量 metadata/内存预算，再安排真实模型小样本及必要全量重跑。

不建议把多跑种子或重抽全部特征当作通用补救：它们不能消除实例混淆、world 键遗漏或解释混杂，却会增加成本。
