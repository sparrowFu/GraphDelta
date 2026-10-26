# GraphDelta 复核问题核实与修复报告（2026-10-26）

对照《GraphDelta_修复复核报告_2026-10-26》（基线 `76f1bda`）逐条核实当前代码。
属实的问题已修复并用测试验证；条件性风险和解释类问题按性质分别处理，未修
改的部分给出原因。

**结论：复核报告的 B1–B9 共 9 条代码问题全部属实，已全部修复。R1 完成低
成本部分（run_meta 记录来源身份），R2 已修（nuisance 分批编码），R3 与
第 5 节的解释问题属数据/研究决定，未改代码，给出了待确认的具体建议。
本轮修改 15 个文件；新增 27 项检查全部通过，上一轮 39 项检查与 E2 端到端
测试无回归。**

## 1. 逐条核实与修复

### B1 [P1] ImageNet 配对丢弃实例身份 —— 属实，已修复

核实：`pug_imagenet.py` 的 `META_COLS` 不含 `character_name`，固定 context 只到
`character_label`（类别）。官方 README 证实：724 个 assets 分属 151 个类，
`character_name` 是实例身份；复核报告的 HF 真实样本（连续 100 行、18 assets、
5 类）显示当前实现 17 对/5 context/覆盖 5 实例，固定实例后 66 对/18 context/
覆盖 18 实例。`pair_matcher.py` 的 `by_val` 字典还会静默折叠同 context 同因素
值的重复行。

修复：

- `pug_imagenet.py`：`character_name` 加入 `META_COLS` 首位（实例先于类别固定），
  配对、固定 context、返回表全部保留实例身份；E2 日志增加 `instances=N` 披露。
- `pair_matcher.py`：构建 `by_val` 时显式检查 (context, 因素值) 唯一性，重复
  行抛 `ValueError`（说明原因），不再静默折叠。
- `captions.py`：修正上一轮的错误注释——`character_label` 是类别名字符串
  （官方数据如 `African_crocodile`），不是编号；caption 改为
  `a {类别名} in the {environment}`（下划线转空格）。

验证：同类两实例同 context 数据下实例分开配对（2 对/2 context）、端点同实例
（无跨实例污染）；完全重复行触发明确报错；caption 输出
`a African crocodile in the forest`。

**协议影响：ImageNet 的 context 定义变为实例级（更细），每 context 配对上限
的预算单位随之变化。当前没有 ImageNet 历史结果，无作废问题；首次真实运行
时需看日志确认各因素的 instances/contexts 数（真实数据若在同一实例 context
内仍有重复因素值行，会触发新增的唯一性报错，届时再定去重规则）。**

### B2 [P3] 空因素抛 KeyError —— 属实，已修复

核实：`build_factor_pairs` 空结果返回无列 DataFrame，`e2_change_diagnosis.run`
的日志访问 `p['context_key']` 抛 KeyError（上一轮集成测试实际撞到过同一行）。

修复：某因素 0 对配对时默认 `SystemExit` 并说明原因；`--allow-empty-factors`
（`run_e2.py` 新增，`run()` 新参数）跳过该因素并警告"本次结果不构成完整因素
集"；后续统计只运行实际有配对的因素。

验证：单值 size 数据默认停止（消息含"0 对配对"）；加 flag 后完成，CSV 只含
3 个因素。

### B3 [P2] SPAR 不适用指标计入 stable=True —— 属实，已修复

核实：`e1_spar_official.py` 的 hardneg 行只写 top1、top5 恒为 NaN；汇总脚本
对该分支固定 `add(..., True)`，三种子全 NaN 的行也输出 stable=True 并计入
稳定总数。

修复：三种子均未定义的指标行直接跳过（不计行）；保留的描述行 stable 记 NA
（CSV 中为空），日志输出 `stable=N, NA(描述)=M, UNSTABLE=K` 三类计数；
NA 安全的统计替换 `~out.stable`（此前遇 None 会抛 TypeError）。

验证：合成三种子表全流程运行——hardneg 的 top5 行不存在于输出，SPAR 行
stable 全为 NA 且不计入稳定总数。

### B4 [P3] E3 先舍入再取符号 —— 属实，已修复

核实：e3_unit 分支先 `round(gap, 4)` 再取符号，`[+1e-5,-1e-5,+1e-5]` 舍入成
`[0,-0,0]` 后误判"符号一致"。

修复：符号判定用原始值，`round` 只用于显示列。

验证：合成 ±1e-5 模式输出 stable=False（旧实现为 True）。

### B6 [P2] E3 强校验只记录不拦截 —— 属实，已修复

核实：`analyze_e3_units.py` 复算总 gap 后只打一行日志（含 |diff|），无容差
判断，冲突时仍写单元表，Figure 5 同读两表可能展示冲突来源。

修复：复算与主表按容差比较（默认 atol=1e-8，`--check-atol` 可调；双侧 NaN
视为一致）；检查结果始终写 `e3_unit_consistency.json`（含逐组 recomputed/
report/abs_diff/ok）；不一致时默认 `SystemExit`（不写单元表），`--no-strict`
为探索模式（仍输出单元表供排查）。

验证：构造 items 与主表冲突的输入——默认停止且未写单元表、sidecar 记录
`all_ok=false`；`--no-strict` 输出单元表。

### B7 [P3] 损坏协议 JSON 使 force 失败 —— 属实，已修复

核实：`_force_reset` 与 `_check_protocol` 都直接 `json.loads`，截断 JSON 抛
`JSONDecodeError`，force 分支删不掉旧 NPZ。

修复：协议文件损坏/不可读一律按"来源未知"处理——`_check_protocol` 报错并
指引 `--force`；`_force_reset` 走整目录删除。删除前校验目标路径必须位于
cache_root 下一级，异常路径直接拒绝。

验证：截断 JSON + force → 目录被删除且无异常；`_force_reset(".../outside")`
被路径校验拒绝。

### B8 [P2] SPAR 逐环境汇总遗漏 world —— 属实，已修复

核实：`e1_spar_official.py` 的 per_world 行带 world 列；汇总脚本的 key 和
匹配都不含 world，`.iloc[0]` 反复取同 key 第一行（复核实测：desert 三种子
最大差 0.8 被丢失）。

修复：`world` 列存在时计入 key（`|w=...` 后缀）与匹配条件；每种子每个 key
必须恰好命中 1 行，否则 `SystemExit`（不允许 `.iloc[0]` 掩盖重复/缺失）。

验证：合成 per_world 表全流程运行——`w=desert` 独立成行且 maxdiff=0.8 正确
输出。

### B9 [P3] 已缓存行仍被扫描解码 —— 属实，已修复（解码部分）

核实：`extract_pug_images` 的 `only_row_ids` 分支算出 `expected_new` 后不提前
返回；迭代器对 only 集合内的行全部解码，已缓存的行解码后才被丢弃。

修复：先把请求行减去已缓存行，空集直接返回"cache complete, skip"；非空时
只把缺失行传给迭代器（不再解码已缓存行）。

**未修改部分及原因：迭代器仍按批读取 image 列的压缩字节（读取≠解码）。
复核报告指出彻底解决需按 shard/row-group 定位，那要改变 parquet 遍历结构；
当前修复已消除解码与 encoder 调用的浪费，压缩字节的顺序读开销留给需要时
再做。**

验证：全部已缓存时迭代器零调用；部分缓存时迭代器收到的 only 集合只含缺失行。

### R1 来源身份 —— 完成低成本部分

已做：`run_meta.json` 除 model/seed/created 外，记录 `git_commit`、`git_dirty`
（当前 commit 与是否有未提交改动）和 `cache_protocol`（四个实验入口统一传入）。

**未做部分及原因：数据 revision 与分片清单摘要未记录——需要先定义清单格式
和校验方式（哪些文件、什么哈希、何时核对），属于协议决定；当前服务器数据
版本本身未变，先记录代码/预处理身份已覆盖最主要的混淆源。**

### R2 资源风险 —— 分批编码已修

已做：`_nuisance_visual` 改为分批生成/编码/释放（批大小 ≤128 或模型 bs_img
取小者），不再同时保留全部 2000×2 张增强图像；每图增强种子固定，分批不改变
任何数值（合成测试：批大小 ≤128、多批、结果列数与行为不变）。

**未做部分及原因：max_total 截断发生在全部 pair records 建完后（不是峰值
内存上限）——完整 Animals 约 12 万行 pair 记录在内存中可承受；ImageNet 全量
metadata（88,328 行）的配对耗时与峰值需在服务器实测后再决定是否流式化，
现在改属于无测量依据的预先优化。**

### R3 数据质量（goldfish 异常渲染）—— 未修改，原因如下

这是数据协议决定：是否排除该子集、按什么口径做敏感性分析、以及固定实例后
的 bootstrap 单位（asset 还是 context），都会改变统计口径。复核报告自己也把
它列为"先核实版本，再做敏感性"的建议而非缺陷。需要你确认处理方式后再实现
（建议顺序：先在服务器核实本地数据版本 → 决定 goldfish 处理 → 首跑时观察
覆盖日志）。

## 2. 第 5 节解释类问题：建议的文档修改（未改，待你确认）

以下是对 `outputs/STAGE1_SUMMARY.md` 措辞的收窄建议。它们是研究结论的表述
问题，不是代码缺陷，我不直接改研究总结：

1. §5.1：把"组合难度不对称"的解释收窄为"按颜色分组的 A/B 匹配表现不同，
   与 shape/候选构成混杂"；如需区分，补 shape 基线或受控对照。
2. §5.2：把"与随机无差别"改为"当前区间未检出偏离随机水平"；16 单元注明
   多重比较只做描述；三种子符号一致不写成"稳健"。
3. §5.3：删除或改写"验证了信息保留/遗忘"类表述为"现有实验未直接检验未提及
   信息的保留"；完整/不完整描述对照属后续任务。
4. §5.4：跨模型比较注明 raw 空间尺度不可直接比语义敏感性，以 normalized
   空间和各模型内部相对变化为主。

你确认后我可以按上述要点修改 STAGE1_SUMMARY.md。

## 3. 修改文件清单（15 个）

| 文件 | 对应问题 |
|---|---|
| src/data/pug_imagenet.py | B1 |
| src/data/pair_matcher.py | B1 |
| src/data/captions.py | B1 |
| src/experiments/e2_change_diagnosis.py | B1 披露、B2、R2 |
| scripts/run_e2.py | B2 |
| scripts/summarize_seed_stability.py | B3、B4、B8 |
| scripts/analyze_e3_units.py | B6 |
| scripts/extract_embeddings.py | B7 |
| src/features/cache.py | B7 |
| src/features/extract.py | B9 |
| src/utils/io.py | R1 |
| src/experiments/e1_state_diagnosis.py | R1 |
| src/experiments/e1_spar_official.py | R1 |
| src/experiments/e3_cogent_diagnosis.py | R1 |
| code/README.md | 文档同步 |

## 4. 验证方式与边界

已做：

- 全部 Python 文件 `py_compile` 通过。
- 新增 27 项检查（`B1` 实例配对/端点同实例/重复报错、`B2` 停止与跳过、
  `B3` NA 与跳行、`B4` 原始值判号、`B6` 严格/探索两模式与 sidecar、
  `B7` 损坏 JSON 三情形与路径校验、`B8` world 入 key 且 maxdiff 不丢、
  `B9` 零迭代/只迭代缺失行、`R1` run_meta 字段、`R2` 分批且批≤128）。
- `summarize_seed_stability.main()` 与 `analyze_e3_units.main()` 首次以合成
  三种子/合成 items 输入完整运行（此前两脚本只做过代码级验证）。
- 上一轮 39 项检查与两个数据集的 E2 端到端测试全部无回归。

边界（与复核报告第 6 节一致）：

- 本机无 torch/transformers 与真实数据；全部测试基于合成 parquet + 假 encoder，
  证明修复逻辑正确，不等于真实 checkpoint 数值验收。
- B1 的实例修复效果基于复核报告的真实 API 样本与本轮合成数据；完整 ImageNet
  （88,328 行）的实例分布、配对耗时、覆盖比例需服务器首跑时用日志核实。
- STAGE1_SUMMARY 的历史数值仍无法核实（CSV/缓存已清理）；按复核建议，历史
  115/109 等计数不能沿用为修复后结论。

## 5. 后续动作（结合复核报告第 7 节）

1. 服务器首跑 ImageNet smoke：看各因素 `pairs/contexts/instances` 日志与
   唯一性检查是否通过；通过后再跑三模型小样本端到端。
2. 旧缓存按 `--force` 入口重建（A2/B7 修复后该入口覆盖协议不匹配、无协议、
   损坏 JSON 三种情形）。
3. Animals 的 E2 结果按面积比例 nuisance 语义重算（上轮 A3），ImageNet 按
   本轮实例级协议首跑。
4. 补齐同版本三种子后运行 `summarize_seed_stability.py`（stable 列现在含 NA
   语义，per_world 结果可正确汇总）。
5. 你确认第 2 节的 STAGE1_SUMMARY 修改建议后，我改文档；R3 的 goldfish
   敏感性口径也待你定。
