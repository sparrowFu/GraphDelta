# 代码风格统一后的行为一致性审查

审查日期：2026-10-21。上次通过的版本为 `5642dfb`；中间版本为 `6483f10`；本次当前版本为 `8ebdd3f`（统一代码风格）。

**结论：没有发现本次重构改变实验评分、文本候选、配对或统计核心逻辑；但本次提交并非只有格式调整，并确认一处结果目录初始化回归。**

## [P2] 删除旧输出后，部分实验独立首次运行会在保存 CSV 时失败

位置：[e3_cogent_diagnosis.py:129](D:/code/GraphDelta/code/src/experiments/e3_cogent_diagnosis.py:129)，相同问题涉及 [e2_change_diagnosis.py:246](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:246) 和 [e1_spar_official.py:341](D:/code/GraphDelta/code/src/experiments/e1_spar_official.py:341)。

`8ebdd3f` 删除了 `code/outputs/tables` 下全部 13 个已跟踪输出文件。Git 不保留空目录，因此 fresh clone 不再具备 `outputs/tables`。`load_paths()` 只创建配置里的 outputs 根，日志只创建 outputs/logs；上述三个实验只拼接 `out_tables`，没有创建目录便调用 `to_csv`。

在隔离临时目录中使用真实 `load_paths/get_logger`，确认 outputs 根和 logs 已创建、tables 未创建；运行原 E3 编排及统计，只以合成场景与特征替换真实推理，真实 CSV 写入报：

```text
OSError: Cannot save file into a non-existent directory: '...\outputs\tables'
```

影响：独立首跑 E2、E3 或官方 SPAR，或清空输出目录后运行，可能在完成数据处理和推理后丢失本次结果。自建 E1 的 `write_jsonl()` 会创建父目录，因此按 README 先运行 E1 会掩盖问题。

建议每个实验在初始化 `out_tables` 后执行 `out_tables.mkdir(parents=True, exist_ok=True)`，或在统一路径初始化中创建 tables。无需通过保留旧结果文件来保证运行。

## 本次实际改变了什么

`5642dfb → 6483f10` 只删除三份旧审查报告，没有实验源码变更。`6483f10 → 8ebdd3f` 除等价重构外，还包含：

| 改动 | 对现有行为的影响 |
|---|---|
| 新增 `--dataset pug_spar_official` 特征抽取入口 | 新增功能，不是纯格式变更；使用已有官方预处理和缓存 key |
| 新增 parquet footer 行数统计 | 用于进度日志分母 |
| 已缓存 PUG 图片提前筛选 | 减少图像解码；缓存内容和 ID 顺序未变 |
| 环境后缀抽为 `_env_suffix`，Counter 直接导入 | 在数据集有效输入上与原实现等价 |
| 统一抽取进度日志，移除 encoder 内部计数日志 | 日志行为改变，数值计算路径不变 |
| 删除未使用的 `FACTOR_COLUMNS` 和 `utils/seed.py` | 仓库内无残留调用，不影响现有流水线 |
| 删除旧 smoke 输出 | 清理产物本身合理，但暴露上述目录初始化问题 |

## 验证证据

- E2、bootstrap、delta_stats、pair_matcher、E3、common 六个核心模块在三个版本间 AST 一致。
- 50 个合成 CLEVR 场景生成的 3,056 条候选及过滤统计一致；108 行 SPAR metadata 生成的 198 条候选及统计一致。
- 官方 SPAR 覆盖 `per_world × smoke` 四种组合，候选、标签、hard-negative 文本、特征对应及输出表逐项一致。
- 官方图像提取覆盖七种缓存状态，最终 ID、raw / normalized 数组一致。
- 实际小型 parquet 的 PUG 提取差分：部分缓存补齐前后，ID 顺序和两套特征逐元素相同；新代码的补齐解码数由 5 降至 3，完整缓存由 5 降至 0，符合优化目的。
- 既有 18 项核心回归全部通过；增加 AST、真实 parquet 缓存差分及独立首次运行检查后，**共 21 项，20 项通过，1 项复现目录问题**。
- 当前 src/scripts 共 37 个 Python 文件语法通过；比上次少一个文件是删除未使用的 seed 工具所致。

复现入口：[preexperiment_style_checks.py](D:/code/GraphDelta/code/review/preexperiment_style_checks.py)。结果：[preexperiment_style_results.json](D:/code/GraphDelta/code/review/preexperiment_style_results.json)。

```powershell
& 'D:\conda\python.exe' -B 'D:\code\GraphDelta\code\review\preexperiment_style_checks.py'
```

发现目录回归时退出码为 1，属预期；历史审查结果没有被覆盖，已删除的 smoke 输出没有恢复。

## 与预实验方案的关系及验证边界

本次没有更改模型冻结、官方 split、文本真假检查、canonical 配对方向、双端点聚类、题型等权或 sphere 对照，未发现新增的核心方案偏差。此前明确的统计汇总/复现覆盖缺项未因本次格式重构得到补齐，也没有将其重新计作新 bug。

本轮未修改实验源码、配置或用户清理的输出文件，仅新增本检查材料。真实 CLIP 推理、显存/性能及真实全量数据没有在本机验证；等价性结论来自源码差分、合成场景、合成特征和真实小型 parquet 的回归检查。
