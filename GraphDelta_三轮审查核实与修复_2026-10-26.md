# GraphDelta 三轮审查核实与修复报告（2026-10-26）

对照《GraphDelta_代码审查报告》（基线 `7b68232`，本轮工作区 HEAD `1460f5e`）
逐条核实当前代码。属实的问题已修复并用测试验证；不需要修改的部分给出原因。

**结论：报告的 6 个问题（3 项 P2、3 项 P3）全部属实，已全部修复。第 5 节的
解释与数据边界事项不是代码缺陷，其中 STAGE1_SUMMARY 措辞收窄仍待你确认
（与上一轮报告第 2 节相同）。本轮修改 10 个文件；新增 21 项检查全部通过，
前两轮 66 项检查与 E2 端到端测试无回归。**

## 1. 逐条核实与修复

### 问题 1 [P2] ImageNet smoke 筛选可能筛空有效数据 —— 属实，已修复

核实：`e2_change_diagnosis.py` 的 smoke 用 `spec['meta_cols'][:2]` 取前两列。
B1 修复把 ImageNet 前两列变成（实例、类别）后，两列各自取字典序前 6 再求交，
排序反相关时交集为空；且不再限制 environment。按报告构造（12 实例×12 类反序、
每实例完整 2×2×2×2 组合、192 行）复现：旧逻辑筛后 0 行。

修复：筛选字段改为按数据集显式声明（注册表 `smoke_cols`：Animals
[animal, environment]，ImageNet [character_name, environment]），并**逐层**
筛选——后一列在已选子集内取前 6 个值，不再独立求交；筛选后 0 行属于配置
问题，显式报错。空因素消息在 smoke 模式追加提示"smoke 子集可能筛掉了该
因素的覆盖，完整运行无此筛选"，区分筛选失败与源数据无配对。

验证：反序构造数据下旧逻辑等价复算筛空（0 行）；修复后 smoke 完成、子集
96 行（6 实例×16 行）、四因素齐全。

### 问题 2 [P2] E3 检查失败后旧单元表仍可被绘图消费 —— 属实，已修复

核实：严格模式只阻止写入新单元表，目录里已存在的 `e3_unit_gaps.csv` 原样
保留；`make_figures` 不读 `e3_unit_consistency.json`。

修复：

- `analyze_e3_units.py`：一致性检查失败时**删除旧派生单元表**（日志说明），
  再按严格/探索模式决定停止或继续写出新表。
- `io.py` 新增 `verify_e3_consistency()`：sidecar 状态为 failed 的目录，
  下游消费其派生表前直接 `SystemExit`；`make_figures` 在目录身份校验后
  调用。兼容只有 `all_ok` 字段的旧 sidecar。

验证：预置旧单元表 + 冲突输入 → 严格模式停止且旧表被删；sidecar 为
failed（含旧格式 all_ok=false）时下游守卫拦截；not_checked 与无 sidecar
不拦截。

### 问题 3 [P2] 复用目录时来源元数据保留首次身份 —— 属实，已修复

核实：`output_tables_dir` 只在 `run_meta.json` 不存在时写入；重跑不更新、
不拒绝；旧格式文件不补新字段。按报告复现：两次传入不同 commit/protocol，
磁盘仍记录第一次的身份。

修复（按报告建议采用"按实验保存来源"方案）：`run_meta.json` 顶层保留首次
身份（代表目录创建时的来源），新增 `experiments` 字典按实验名
（e1_state / e1_spar_official / e2_<dataset>[_smoke] / e3_cogent）记录各自
当前来源（git_commit、git_dirty、cache_protocol、updated）；同一实验身份
变化时旧身份存入 `replaces`，不覆盖。目录里未重跑的其他实验文件仍对应其
各自记录，共享顶层不再冒充全部文件的统一版本。

验证：同实验 protocol v1→v2 更新记录且 `replaces` 保留 v1、顶层不变；
不同实验名各自独立记录。

未选"身份不同时拒绝复用"的原因：开发流程中提交代码后重跑同目录很常见，
一律拒绝会阻塞正常迭代；按实验记录 + 变更留痕已满足来源可追溯，是否升级
为硬拒绝可在正式实验阶段再定。

### 问题 4 [P3] 全部因素跳空仍抛内部异常 —— 属实，已修复

核实：`allow_empty_factors=True` 且全部因素无配对时，空 dict 进入
`build_delta_frame`，`pd.concat([])` 抛 `ValueError: No objects to concatenate`。

修复：配对循环后若 `pairs_by_factor` 为空，`SystemExit` 明确说明"全部因素
均无有效配对，--allow-empty-factors 只跳过个别空因素"。

验证：全单值因素数据 + allow_empty_factors=True → 得到明确停止消息而非
ValueError。

### 问题 5 [P3] 协议 JSON 为非对象时恢复失败 —— 属实，已修复

核实：`[]`、`null`、`"x"` 都是合法 JSON，`.get()` 对其抛 AttributeError，
两处异常捕获（JSONDecodeError/OSError/ValueError）都不覆盖；force 因此
失败、旧文件保留。

修复：`cache.py` 与 `extract_embeddings.py` 解析后先验证 `isinstance(dict)`，
不是对象一律按"来源未知"处理——读取端报错指引 `--force`，force 端走整目录
删除。

验证：三种非对象 JSON 读取端全部被拒（RuntimeError）；force + `[]` 整目录
删除成功。

### 问题 6 [P3] 缺主表时状态记录为通过 —— 属实，已修复

核实：参考主表不存在时不做比较，sidecar 写 `all_ok=true, checks=[]` 并生成
单元表，"未检查"被表达成"通过"。

修复：sidecar 增加 `status` 字段（passed / failed / not_checked，`all_ok`
仅在 passed 时为 true）；严格模式要求参考主表存在，缺失时停止并说明
（`--no-strict` 探索模式继续，状态如实记 not_checked）。下游守卫只拦截
failed，不拦截 not_checked（完整绘图缺主表时自身会失败，不存在"缺表还出
正式图"的路径）。

验证：缺主表严格模式停止；探索模式完成且 status=not_checked、单元表照常
输出、下游守卫放行。

## 2. 不需要修改的部分及原因

- **第 2/3 节（数据集与模型支持、已确认有效行为）**：本轮复核与报告描述
  一致，未改动。
- **第 5 节解释与数据边界（6 项）**：不是代码缺陷。其中 1–4 的
  STAGE1_SUMMARY 措辞收窄建议已在上一轮报告第 2 节列明，仍待你确认后修改
  文档；5（全量资源实测）与 6（数据版本/goldfish 核实）需要服务器真实数据，
  属运行阶段事项。
- **max_total 截断时机**（第 5 节第 5 条）：与上轮结论相同，完整 Animals 约
  12 万行 pair 记录内存可承受，ImageNet 全量需实测后再决定是否流式化，现在
  改属无测量依据的预先优化。

## 3. 修改文件清单（10 个）

| 文件 | 对应问题 |
|---|---|
| src/experiments/e2_change_diagnosis.py | 问题 1、4 |
| scripts/analyze_e3_units.py | 问题 2、6 |
| scripts/make_figures.py | 问题 2 |
| src/utils/io.py | 问题 2（守卫）、3 |
| src/features/cache.py | 问题 5 |
| scripts/extract_embeddings.py | 问题 5 |
| src/experiments/e1_state_diagnosis.py | 问题 3 |
| src/experiments/e1_spar_official.py | 问题 3 |
| src/experiments/e3_cogent_diagnosis.py | 问题 3 |
| code/README.md | 文档同步 |

## 4. 验证方式与边界

已做：

- 全部 Python 文件 `py_compile` 通过。
- 新增 21 项检查：问题 1 的复现条件（旧逻辑筛空）与修复后行为（96 行、
  四因素）、问题 2 的旧表删除/下游守卫/旧格式兼容、问题 3 的 per-experiment
  记录与 replaces、问题 4 的明确停止、问题 5 的三种非对象 JSON 与 force、
  问题 6 的严格停止与 not_checked 状态。
- 前两轮 66 项检查与两个数据集 E2 端到端测试全部无回归。

边界（与报告第 6 节一致）：

- 本机无 torch/真实数据；全部验证基于合成 parquet + 假 encoder，证明修复
  逻辑正确，不等于真实 checkpoint 数值验收。
- ImageNet 真实数据的实例/环境分布在 smoke 逐层筛选下的覆盖仍需服务器首跑
  日志确认（空因素会得到带 smoke 提示的明确消息）。
- 历史 CSV/缓存已清理，历史数字不可核实，不能沿用为修复后结论。

## 5. 后续动作

1. 服务器首跑 ImageNet smoke（看 smoke 子集行数、instances/contexts、
   唯一性检查），随后三模型小样本端到端。
2. 你确认 STAGE1_SUMMARY 措辞收窄后我改文档（清单见上轮报告第 2 节）。
3. 其余按本报告与上轮报告的顺序建议执行；本轮无新的重算要求（修复均为
   流程/边界问题，未改任何统计口径——唯一影响运行行为的是 ImageNet smoke
   子集的选取方式，而 smoke 输出本就不作为正式结果）。
