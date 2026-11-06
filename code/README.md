# GraphDelta 预实验说明

对应《GraphDelta_研究框架_统一修订版》《GraphDelta_完整预实验方案_统一修订版》与
《[下一步工作方案：受控结构编辑与迁移验证](../下一步工作方案_受控结构编辑验证_2026-11-03.md)》
（2026-11-03 定稿，下称"工作方案"）。本文件是预实验的总说明：每个环节讲清
做什么、为什么做、怎么做、怎么执行；操作细节见各阶段 README。

## 1. 背景与两阶段结构

研究问题（研究框架）：在相同数据、可训练容量和训练预算下，图编辑指导的
上下文相关变化学习，能否提高模型对未见属性组合与关系结构的图文匹配能力。

回答它之前必须先知道两件事，否则训练无从设计：

1. 冻结的预训练模型在这些结构因素上到底哪里弱、弱多少——决定主实验的动机
   和数据构造重点；
2. 受控的结构编辑在冻结表示里留下多大可读出的信号——决定训练路线（全局
   适配 / 局部表示 / 骨干更新）是否有依据。

预实验因此分两阶段，之后直接进入主实验，不再有第三轮：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段一 | 冻结 baseline 诊断（E1–E3），公开数据 + 固定协议，只评估不训练 | 已完成，结果冻结 |
| 阶段二 | 受控结构编辑验证（E4），自建受控数据 + 诊断，骨干冻结、只训线性探针 | 代码完成；三轮代码审查修复完毕，76 个单元测试 + 全流程驱动验证通过，待正式运行 |

代码全部在 [`PreExperiments/`](PreExperiments/README.md)（共用模块 `common/`，
阶段目录 `stage1_frozen_baseline/`、`stage2_controlled_edit/`）；配置在
`configs/`，数据在 `data/`，结果在 `outputs/`（位置都不随代码目录变动）。

## 2. 环境与前提

```bash
cd GraphDelta/code
conda run -n CudaVersion128Fuxp python <script>
# python 3.10 / torch 2.7.0+cu128 / transformers 5.x / pandas / pyarrow
```

| 项 | 位置 / 说明 |
|---|---|
| 五个公开数据集 | `configs/paths.yaml` 的 `datasets`（CLEVR、CoGenT、PUG:SPAR / Animals / ImageNet，绝对路径） |
| 三个冻结模型 | 同文件 `models`（CLIP ViT-L/14 首选、ViT-B/32 对照、SigLIP2 so400m 对照） |
| 渲染环境（仅阶段二） | 同文件 `render`：Blender 2.78c + clevr-dataset-gen `f0ce2c8`，输出目录在仓库外 |

脚本全部以 `__file__` 锚定路径，任意目录可运行；不配置齐数据 / 模型路径时，
`resolve_models` / `load_paths` 会直接报错并列出可用值。

## 3. 阶段一：冻结 baseline 诊断（E1–E3，已完成）

### 3.0 特征抽取（Step 2 前置）

- **做什么**：把各数据集全部图像编码成特征，写入 `data/cache/<model>/`，
  raw 与 normalized 双份。
- **为什么**：特征只抽一次，后续所有实验读缓存，把 GPU 时间与实验时间分开；
  缓存目录带 `protocol.json` 指纹，预处理语义变更后旧缓存直接报错，防止静默
  串用（阶段一 §3.8 审查教训）。
- **怎么执行**：`PreExperiments/stage1_frozen_baseline/scripts/extract_embeddings.py
  --dataset all --model all`；实验脚本发现缓存缺失也会自动补抽。

### 3.1 E1 状态级结构诊断（问题一：能力是否不均衡）

- **做什么**：对 CLEVR 构造六类文本 hard negative 2AFC（object / color /
  size / material / relation / binding），对 SPAR 构造四类（scene / position /
  attribute / relation），报告 2AFC 准确率、margin、模板方差与按正例取值分层
  的准确率。
- **为什么**：这是主实验动机的直接来源。恒选某个词的答题策略会伪装成能力，
  所以 target 子样本按正例取值分层等额抽取（该策略期望准确率恰 50%），
  逐侧准确率之差就是词偏好的直接证据。
- **怎么做**：图像不变只换文本候选，不重新渲染；每条样本过"正例为真、
  负例为假"检查；三种子（0/1/2）重复采样检验判定稳定性。
- **怎么执行**：`run_e1.py --model all`（SPAR 官方协议复现另跑
  `run_e1_spar_official.py`，两套结果分开报告）。

### 3.2 E2 单因素表示响应（问题二：表示变化有无稳定规律）

- **做什么**：在 PUG:Animals 的完整笛卡尔积里固定其余因素、只变一个因素
  （environment / texture / size / camera），计算视觉 Δ = after − before 的
  范数与分层相似度，主指标 Transition Similarity Gap。
- **为什么**：若单因素变化没有稳定方向，任何"变化向量可迁移"的设计都失去
  依据。nuisance 基线（同图两次确定性增强之差、同内容不同模板文本之差）
  给出"无语义扰动"的水平线，因素响应必须超出它才算信号。
- **怎么执行**：`run_e2.py --dataset all --model all`。

### 3.3 E3 组合难度不对称（问题三：CoGenT A/B 难度差）

- **做什么**：CoGenT valA / valB 上构造换色 / 换形负例的 2AFC，报告 A−B
  Difficulty Gap；sphere 在两组均可取全 8 色，作颜色词难度对照组；另做逐
  (shape, color) 单元分析，检查效应是否被平均抵消。
- **为什么**："未见组合是否更难"是主实验组合留出设计的依据；sphere 对照把
  "颜色词难度"与"组合难度"两种效应分开，避免把前者误读成后者。
- **怎么执行**：`run_e3.py --model all`，逐单元分析 `analyze_e3_units.py
  --model all`（写出前用全部 items 复算总 gap，不一致默认停止）。

### 3.4 汇总与阶段判定

- **做什么**：三种子关键判定稳定性检验（`summarize_seed_stability.py`）、
  五张诊断图（`make_figures.py`）、阶段总结（`outputs/STAGE1_SUMMARY.md`）与
  分析报告（`outputs/analysis/预实验分析报告_平衡采样_2026-11-03.md`）。
- **为什么**：判定是否进入第二阶段必须抗采样噪声；结论只从正式产物计算，
  报告可由 `render_analysis_report_html.py` 复渲染。
- **怎么执行**：`summarize_seed_stability.py --model all` → `make_figures.py
  --model all --seed 0`。

阶段一结论入口：E1 词偏好与因素分类、E2 单因素响应、E3 组合难度、
情况 A/B/C/D 对照，见 `outputs/STAGE1_SUMMARY.md` 与分析报告。

## 4. 阶段二：受控结构编辑验证（E4）

阶段二回答"受控结构编辑在冻结表示里留下多大信号"，五个环节（Step 1–5）
全部已实现；正式运行前须按冻结时点表完成相应验收（工作方案开篇待办）。

### 4.1 Step 1 题目审计

- **做什么**：导出 SPAR 两类稳定低于 50% 的题型（attribute / relation）与
  CLEVR binding 对照的逐题审查清单和原图，供人工核验正句为真、负句为假、
  对象可见、指代无歧义。
- **为什么**：低分可能来自真值错误而非模型缺陷；先用抽查把数据问题排除掉，
  后续诊断才有解释力。审计对象按低分题型选取，结论限定在抽查范围内。
- **怎么做**：三模型三种子逐题成绩聚合成"一致错 / 分歧 / 一致对"三层，
  层内按正例取值轮转抽样；种子固定可复现。
- **怎么执行**：`PreExperiments/stage2_controlled_edit/scripts/
  export_audit_items.py`（全量约 1 分钟，输出 `outputs/audit/`，251 题）。

### 4.2 Step 2 生成编辑对清单

- **做什么**：构造受控编辑对清单——800 对颜色归属交换（主任务，8 个转换 ×
  约 100 个互不相交的基础场景）、200 对目标换色（正控制）、500–1,000 对左右
  关系反转、约 200 对无语义变化扰动对照（jpeg / crop 后处理、共同平移）；
  家族内 60/20/20 划分训练 / 验证 / 留出机制集。
- **为什么**：公开数据测不了"同一场景只改一处结构"的因果问题，必须自建；
  划分先于任何估计固定，避免把重复场景当成泛化（工作方案第 4 节）。
- **怎么做**：`src/generation/` 纯逻辑采样场景规格（放置约束与官方
  clevr-dataset-gen 一致），编辑算子改规格，逐对 `validate_edit` 校验"场景图
  差异恰好等于预期编辑集合"；全部由 seed 决定，逆向复用同一 pair_id。
- **怎么执行**：`build_edit_pairs.py --task all`（参数与产物见阶段二 README
  §4；清单落 `data/edit_pairs/`，`splits.json` 记录 md5，改动即报错）。

### 4.3 Step 3 渲染

- **做什么**：把每个唯一场景状态交给 Blender 渲染成图，回填 3d 坐标与像素
  坐标，复核渲染结果与规格一致，登记渲染账本。
- **为什么**：官方脚本无随机种子、场景不可复现，正式数据必须由带种子规格
  驱动；账本按唯一状态记账，吞吐含校验与重试，预算不重复计费（工作方案
  第 8 节）。
- **怎么做**：`render/blend_render_driver.py` 在 Blender 2.78 自带 Python 3.5
  内运行（规格经 JSON 传入，不 import 本仓库代码），可见性检查与关系计算
  直接 import 官方实现；Python 侧逐态复核字段 / 位置 / 方向轴 / 关系矩阵，
  可见性失败是确定性失败、登记剔除不重试。
- **怎么执行**：`render_edit_pairs.py --task binding --limit 8`（小批试运行）
  或全量；账本跳过已完成状态，`--force` 重渲染。复用必须同时匹配规格指纹、
  资产指纹，且磁盘产物内容摘要与成功记录一致——被后续尝试或另一清单的
  同名状态覆盖过的文件自动作废重渲染，不把旧成功记录套到已变化的产物上。

### 4.4 Step 4 E4 评测

- **做什么**：对每个编辑对计算完整 2×2 图文相似度矩阵，执行四个子实验：
  E4-A 配对结构判断（两图选文本 + 两文本选图，Pair Accuracy 平局记错）；
  E4-B 视觉变化诊断（Δv 方向一致性、近零比例、留出场景线性探针 + 颜色 /
  像素 / 单端点参照）；E4-C 转换级跨模态诊断（Δt 对照、C 的符号翻转检验、
  C ≡ Δv·Δt 代数自检）；E4-D 跨场景方向迁移（训练集方向 + 验证集全局
  非负幅度，固定候选池检索真实 after 图，对照含 α=0 / 错误转换 / 随机 /
  反向 / 文本差）。
- **为什么**：这四个问题层层递进——分得开（A）才有"可辨识"；读得出（B）
  才谈得上信息存在；对得上（C）才谈跨模态几何；迁得动（D）才支持"固定
  方向"这一最朴素的方案。允许阴性；每条结论区分"证据不足"与"无效"。
- **怎么做**：骨干完全冻结；特征走独立 float32 命名空间缓存（键带清单指纹，
  清单重建自动换键）；统计口径冻结在 `configs/e4_protocol.yaml`（1,000 次置换、
  α=0.05、允许重排不足时全枚举）；产物按阶段一 run_meta 机制登记（提交时逐
  产物记录内容摘要 `outputs_md5`，供汇总端核对磁盘文件确属该次提交），全部
  CSV/JSON 原子写出（中断不留半写文件），smoke 带 `_smoke` 后缀不写正式结果。
- **怎么执行**：`run_e4.py --model clip_vitl14 --stage all`（先跑核心骨干，
  SigLIP2 复核；`--smoke` 用小批数据查流程）。

### 4.5 Step 5 汇总与阶段判断

- **做什么**：汇总各模型 E4 产物，输出逐转换效应、区间与 p 值，并按预先
  写死的规则给三轴判定：结构可辨识（A 的置换 p）、跨模态对应（C 的转换级
  p）、方向迁移（D 的 correct 是否高于全部对照），标注"有支持证据 / 证据
  不足 / 数据不足"。
- **为什么**：预实验的结束条件是"交付结论与主实验协议"，不是"方法有效"；
  判定规则随产物落盘，防止事后换口径挑显著结果（工作方案第 7 节）。
- **怎么做**：完整性是判定的前置条件——预先确定的任务（binding+relation）、
  规定转换（绑定 8 个 / 关系 10 个类型对）、规定对照（E4-D 四个）与冻结清单
  规定的留出场景覆盖，任一缺失对应轴记"数据不足"，不凭已出现的行判支持；
  确认性检验族规模冻结（A/C 每任务 1 个 + D 每任务×4 对照 = 12 个），缺数据
  的检验按 p=1 计入族内，不因缺数据缩小族、放宽其余检验的校正；共享输出
  路径的文件须与某条成功提交的内容摘要一致才被消费，被中断运行覆盖过即拒绝。
- **怎么执行**：`summarize_e4.py --model clip_vitl14 --seed 0`，产物
  `e4_summary.csv` + `e4_stage_decision.json` 写入该模型 tables 目录。

## 5. 数据与结果位置

| 位置 | 内容 |
|---|---|
| `configs/` | `paths.yaml`（数据 / 模型 / 渲染路径）、各实验配置；`e4_protocol.yaml` 为 E4 冻结配置 |
| `data/manifests/`、`data/cache/` | PUG manifests；特征缓存（阶段一 float16、阶段二 `f32/` 命名空间） |
| `data/edit_pairs/` | 阶段二编辑对清单 + 划分（seed 可再生） |
| `<render.output_dir>` | 渲染图、回填场景 JSON、账本（仓库外） |
| `outputs/<model>/seed_<k>/tables/` | 各实验结果表 + `run_meta.json`（含 git commit 与缓存协议指纹） |
| `outputs/analysis/`、`outputs/audit/`、`outputs/figures/` | 分析报告、审查清单、诊断图 |

## 6. 命令速查（从 code/ 目录，可直接复制）

以下命令都从 `code/` 目录发起（脚本自身锚定路径，任意目录也可运行）。

### 6.1 阶段一（已完成，重跑用于复现；详见各步参数表）

```bash
# 特征：3 模型 × 7 数据集一次抽完（模型外层、数据集内层；首次全量约数小时 GPU）
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/extract_embeddings.py --dataset all --model all
# 四个实验（各模型结果写 outputs/<model>/seed_0/tables/）
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/run_e1.py --model all
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/run_e1_spar_official.py --model all
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/run_e2.py --dataset all --model all
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/run_e3.py --model all
# 三种子：上面四条各再跑两遍，--seed 1、--seed 2（结果自动进 seed_1/ seed_2/）
# 汇总 / 逐单元分析 / 绘图（不加载模型）
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/analyze_e3_units.py --model all
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/summarize_seed_stability.py --model all
conda run -n CudaVersion128Fuxp python PreExperiments/stage1_frozen_baseline/scripts/make_figures.py --model all --seed 0
```

改代码后的快速自检：各实验脚本加 `--smoke`（输出带 `_smoke` 后缀，分钟级，
不写正式结果；`run_e2.py --smoke` 需配 `--allow-empty-factors`）。

### 6.2 阶段二·冒烟链路（分钟级，改代码后先跑这个）

```bash
# 0) 单元测试：76 个用例，不依赖 Blender / 模型 / 数据集，秒级
conda run -n CudaVersion128Fuxp python -m pytest PreExperiments/stage2_controlled_edit/tests/
# 1) 小批清单（8 个转换 × 每转换 3 场景；会覆盖 data/edit_pairs/ 下的同名清单）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/build_edit_pairs.py --task binding --binding-n 24 --recolor-n 24 --control-n 8
# 2) 小批渲染（约 30–39 秒/态，单态失败可重跑，账本跳过已完成）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/render_edit_pairs.py --task binding --limit 18
# 3) E4 冒烟（输出带 _smoke 后缀，不写正式结果）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/run_e4.py --model clip_vitl14 --stage all --smoke
# 4) 冒烟汇总（只记录流程是否跑通，不给机制标签）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/summarize_e4.py --model clip_vitl14 --smoke
```

冒烟通过后，正式运行前须用默认参数重建清单（`build_edit_pairs.py --task all`，
不带 `--binding-n` 等小批参数）。

### 6.3 阶段二·正式全量（渲染最耗时，详见阶段二 README §6.3 的核对清单）

```bash
# 1) 题目审计清单（一次性，约 1 分钟）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/export_audit_items.py
# 2) 默认配额清单（800 交换 + 200 换色 + 500 反转 + 200 扰动；秒级）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/build_edit_pairs.py --task all
# 3) 全量渲染（约 3,400 个唯一状态；单进程 30–39 秒/态 ≈ 1.5 天，--workers 并行，
#    加速比以账本实测为准；可中断重跑，账本 + 内容摘要核对自动续）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/render_edit_pairs.py --task all --workers 8
# 4) E4 评测（GPU，分钟级；核心骨干先跑，SigLIP2 复核）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/run_e4.py --model clip_vitl14 --stage all
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/run_e4.py --model siglip2_so400m --stage all
# 5) 汇总与阶段判断（产物 e4_summary.csv + e4_stage_decision.json）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/summarize_e4.py --model clip_vitl14 --seed 0
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/summarize_e4.py --model siglip2_so400m --seed 0
```
