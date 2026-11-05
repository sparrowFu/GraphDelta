# GraphDelta 第二阶段：受控结构编辑验证（E4）说明

对应《[下一步工作方案：受控结构编辑与迁移验证（执行定稿）](../../../outputs/analysis/下一步工作方案_受控结构编辑验证_2026-11-03.md)》
（2026-11-03 定稿，下称"工作方案"）。本目录
（`code/PreExperiments/stage2_controlled_edit/`）是阶段二的全部代码：受控数据
生成、规格驱动渲染、E4 诊断与汇总。共用模块在 `../common/`（import 名
`src.*`），配置在 `code/configs/`（`e4_protocol.yaml` 是 E4 冻结配置），数据
与结果仍在 `code/data/`、`code/outputs/`（位置不变）。总览见
`code/PreExperiments/README.md`。

阶段定位（工作方案开篇）：预实验只有 E1–E3 和本阶段的 E4 两轮，之后直接进入
主实验；E4 全程骨干冻结，只训练线性探针、估计转换方向；方向迁移和原始跨模态
对齐允许阴性或证据不足，不设预实验门槛。

代码布局：`scripts/` 五个入口（Step 1–5）；`src/generation/` 场景规格与编辑
算子（纯逻辑，无 Blender 依赖）；`src/e4/` 清单构建、渲染调度、评测与统计；
`render/blend_render_driver.py` 在 Blender 2.78 自带 Python 3.5 内运行，不
import 本仓库代码，规格经 JSON 传入。

---

## 1. 运行环境

Python 环境与命令前缀同阶段一（`conda run -n CudaVersion128Fuxp`，见
`../stage1_frozen_baseline/README.md` §1）。渲染环境（2026-11-03 试通通过，
[渲染试通记录](../../../outputs/analysis/render_trial_2026-11-03.md)）配置在
`configs/paths.yaml` 的 `render` 段：

| 项 | 值 |
|---|---|
| Blender | 2.78 (sub 0)，`render.blender` |
| 生成器 | clevr-dataset-gen，commit `f0ce2c8`（`render.clevr_gen`） |
| 渲染参数 | 480×320、512 采样、tile 256（`e4_protocol.yaml` 的 `render` 段） |
| 输出目录 | `render.output_dir`（仓库外；图片 / 场景 JSON / 账本都在此） |

- 官方脚本无随机种子、场景不可复现：正式数据必须走本目录的规格驱动渲染
  （`build_edit_pairs.py` 带种子出规格 → `render_edit_pairs.py` 调 Blender）；
- 单进程实测约 30–39 秒 / 态（两对象场景偏快）；多进程加速比待实测写入账本，
  不按进程数线性外推；
- 特征缓存：本阶段用独立 float32 命名空间（`data/cache/f32/<model>/`），与
  阶段一的 float16 缓存互不影响；缓存键带清单指纹与图片内容指纹，清单重建
  或任一图片重渲染/被替换都会自动换键，不串用旧特征。

## 2. 受控数据（工作方案第 3、4 节）

语义编辑对合计约 1,500–2,000 个（一个编辑对 = 一组 before/after 图片，逆向
复用同一 `pair_id`，不增加独立样本数），另加无语义变化的扰动对照：

| 编辑 | 数量 | 必须保持或核验的条件 |
|---|---:|---|
| 颜色归属交换（主任务） | 800 对 | 对象身份和形状、位置、大小、材质、相机、光照、背景不变；颜色集合不变 |
| 目标对象换色（正控制） | 200 对 + val/heldout 覆盖 | 只改 target 颜色；新色取其余 6 色（词集变化，`word_set_changed` 单列）；val/heldout 的全部交换场景另生成换色状态（E4-D 同场景错误编辑候选） |
| 左右关系反转 | 500–1,000 对 | 仅移动 a 从 b 左侧到右侧（点反射）；间距与深度不变；核验可见性 |
| 无语义变化（对照） | 约 200 对 + val/heldout 覆盖 | 绑定侧同图压缩 / 轻微裁剪（后处理，不占渲染账本）；关系侧共同平移（每 4 个训练场景取 1，val/heldout 全覆盖） |

绑定任务用 8 个默认转换（`src/generation/scene_spec.py` 的
`BINDING_CONVERSIONS`），每转换约 100 个互不相交的基础场景；关系任务从 10
个合法训练类型对（`RELATION_TYPE_PAIRS_TRAIN`）等额取场景，统一 left→right。

每个编辑对保存：`pair_id`、`base_scene_id`、任务与编辑类型、家族、前后状态
引用、前后场景图（spec JSON）、结构化编辑指令、`word_set_changed`、划分。
每个状态保存：`state_id`、spec、是否需要渲染、派生来源。构建时逐对
`validate_edit`（"场景图差异恰好等于预期编辑集合"）；渲染回填后再用
`validate_scene_json` 复核（字段、位置、高度、方向轴、关系矩阵、可见性）。

划分：家族（绑定的转换 / 关系的类型对）内按基础场景 60/20/20（`scene_idx`
升序，随清单 seed 冻结），同场景全部版本同组；`splits.json` 记录清单 md5，
清单被改动后装载直接报错。约 1,000 对是先导规模，报告效应量和不确定性。

人工核验（首批核验前落实流程）：自动检查覆盖全部样本；按转换分层抽至少
10% 完整图片对人工核验，单人另抽约 5% 间隔复核；系统性错误整批暂停、修复、
重验，不只删抽查中已发现的错误。

## 3. 实验矩阵 E4（工作方案第 5–7 节）

对每个图片对及对应文本计算完整 2×2 相似度矩阵 `S_ij = z_i · t_j`（归一化
点积；t 为模板，i∈{0=before,1=after} 图，j∈{0,1} 文本）：

| 子实验 | 内容 | 主要输出 |
|---|---|---|
| E4-A 配对结构判断 | 两图分别在同一对文本中选描述，两句文本分别在一对图中选图 | I2T / T2I / Pair Accuracy（平局记错；先模板内两图均对，再跨模板平均）；场景内匹配置换参考分布（与报告口径同一计分）；换色正控制的同套配对评分（`e4a_control_results`） |
| E4-B 视觉变化诊断 | Δv = after − before（z 与 raw 双空间）；d_r 只用训练集 | 留出场景与 d_r 方向相似度、‖Δv‖、近零比例（训练分位数）+ 与随机/错误方向参照；方向可靠性（训练内对半划分子集比较 + 按场景重采样波动）；对照编辑（换色/jpeg/crop/平移）的 Δv 同口径诊断；线性探针（状态二分类 / 转换身份）+ 颜色直方图 / 像素差 / 单端点参照 |
| E4-C 转换级跨模态诊断 | Δt = after 文 − before 文；转换级条件 = 绑定 8 转换 + 关系 10 类型对 | 逐转换 C 的均值 + 区间 + 探索性 p（不进判定）；任务级 `__overall__` 主统计（C 均值 / 区间 / 符号翻转 p / 方向）；跨颜色组对照；C ≡ Δv·Δt 代数自检 |
| E4-D 跨场景方向迁移 | d_r 固定符号，验证集选全局 α≥0，留出场景预测 after 表示 | 固定候选池 after 检索 top1（主终点，pool 清单随行落盘）；对照：α=0、错误转换、随机、反向、文本差等幅；打乱配对代数自检（与 d_r 同批训练场景）；正式运行缺规定候选直接报错 |

统计口径（`configs/e4_protocol.yaml`，正式 E4 运行前冻结）：默认 1,000 次
置换、α=0.05，允许重排不足时全枚举；平局记错；配对分数 C 与 `Δv·Δt` 代数
相等，不作两份独立证据；Pair Accuracy 参考分布用场景内匹配置换（与报告
口径同一计分函数），不按 25% 处理；E4-D 不能用测试图片的真实差向量当预测
方向；确认性检验族（A 每任务 1 个 + C 每任务 1 个 + D 每任务×规定对照各
1 个）族内 Holm 校正，方向必须为正。

---

## 4. 完整流程（Step 1–5，全部已实现）

```text
Step 1  题目审计：导出 E1 低分题型人工审查清单
Step 2  生成编辑对清单：场景采样 + 编辑算子 + 校验 + 60/20/20 划分
Step 3  渲染：规格驱动 Blender 渲染、场景图复核、账本、后处理对照
Step 4  E4 评测：2×2 配对矩阵、Δv/Δt 诊断、探针、方向迁移
Step 5  汇总与阶段判断：冻结统计程序汇总，对照结束条件
```

记号约定同阶段一 README §5；命令运行时加 §1 前缀。

### Step 1 题目审计（开篇待办 1）

命令（全部参数）：

```bash
python PreExperiments/stage2_controlled_edit/scripts/export_audit_items.py --spar-per-class [100(默认), 任意正整数] --clevr-n [50(默认), 任意正整数]
```

作用：读取三个模型 × 三个种子的 E1 逐题结果，导出低分题型的人工审查清单和
对应图片（工作方案第 2 节）。对象：SPAR attribute（B/32 约 39.9%）与
relation（L/14 约 45.6%），CLEVR binding 作对照；分层：一致错 / 分歧 /
一致对 = 40/40/20，层内按正例取值轮转抽样；种子固定（20261103），全量约
1 分钟。输出 `outputs/audit/`（251 行清单 + 汇总 JSON + 图片），默认参数已
运行一次，审查列尚未填写。该脚本无 smoke 模式，小配额运行会覆盖
`outputs/audit/`。

参数（共 2 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--spar-per-class` | 任意正整数 | `100` | SPAR 每类低分题型的抽查题数；层内配额按 40/40/20 等比缩放 |
| `--clevr-n` | 任意正整数 | `50` | CLEVR binding 对照题数，三层均分（实际 51 = 3×17） |

### Step 2 生成编辑对清单

命令（全部参数）：

```bash
python PreExperiments/stage2_controlled_edit/scripts/build_edit_pairs.py --task [all(默认), binding, relation] --binding-n [800(默认), 任意正整数] --relation-n [500(默认), 任意正整数] --recolor-n [200(默认), 任意正整数] --control-n [200(默认), 任意正整数] --seed [20261103(默认), 任意整数]
```

作用：按 §2 的构成调用 `src/generation/` 采样基础场景、应用编辑算子并逐对
`validate_edit`，按家族 60/20/20 划分，写出清单与划分哈希。放置失败自动
重采样（计数写日志）；换色对复用同转换前段场景的 before 状态；逆向复用同
一 `pair_id`。正式编码前须冻结转换清单与逐类型配额（工作方案开篇待办 3）。

参数（共 6 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--task` | `all` / `binding` / `relation` | `all` | 生成哪个任务的编辑对 |
| `--binding-n` | 任意正整数 | `800` | 颜色归属交换对数（8 个转换均分） |
| `--relation-n` | 任意正整数 | `500` | 左右关系反转对数（10 个类型对均分；上限 1,000） |
| `--recolor-n` | 任意正整数 | `200` | 换色正控制对数（复用绑定场景） |
| `--control-n` | 任意正整数 | `200` | jpeg / crop 扰动对照对数（各半，不占渲染账本） |
| `--seed` | 任意整数 | `20261103` | 场景采样与划分种子；换种子须分别报告 |

### Step 3 渲染

命令（全部参数）：

```bash
python PreExperiments/stage2_controlled_edit/scripts/render_edit_pairs.py --task [all(默认), binding, relation] --workers [1(默认), 任意正整数] --limit [不带(默认), 任意正整数] --force
```

作用：读清单 → 展开唯一渲染状态 → 子进程调 Blender（`render/
blend_render_driver.py`，可见性与关系计算直接 import 官方实现）→ 渲染回填
的场景图逐项复核（字段 / 位置 / 高度 / 方向轴 / 关系矩阵）→ 回填
`rendered_states.parquet`（只登记当前清单对应的有效产物）→ 账本登记
（`ledger.jsonl`：状态、秒、重试、字节、规格与资产指纹；有效吞吐含校验与
重试）→ 生成 jpeg / crop 后处理对照图。可见性检查失败是确定性失败（退出码
3，不重试），登记后该状态剔除。账本复用必须同时匹配场景规格（`spec_md5`）
与渲染设置/资产指纹（`render_fp`）；换 seed、改规格或改设置后旧记录作废、
重新渲染。

参数（共 4 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--task` | `all` / `binding` / `relation` | `all` | 渲染哪个任务的清单 |
| `--workers` | 任意正整数 | `1` | 并行 Blender 进程数；加速比须实测写入账本 |
| `--limit` | 任意正整数 | 不带 | 只渲染前 N 个唯一状态（小批试运行 / 冒烟） |
| `--force` | 开关 | 不带 | 重渲染已完成状态；默认跳过账本中已登记的状态 |

### Step 4 E4 评测

命令（全部参数）：

```bash
python PreExperiments/stage2_controlled_edit/scripts/run_e4.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --stage [all(默认), a, b, c, d] --smoke --seed [0(默认), 1, 2, …]
```

作用：装载清单与渲染产物（不完整的任务自动跳过并警告）→ float32 缓存抽取
→ 按 `--stage` 执行 E4-A/B/C/D → 按阶段一 run_meta 机制登记与提交
（experiment = `e4_<stage>`，smoke 加 `_smoke` 后缀，产物文件同步加后缀）。
某阶段数据不足（如某家族 split 不全）时警告并继续，不中断其余阶段。

参数（共 4 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 评估的冻结 baseline；核心诊断用 ViT-L/14，SigLIP2 复核主要发现 |
| `--stage` | `all` / `a` / `b` / `c` / `d` | `all` | 运行哪个子实验 |
| `--smoke` | 开关 | 不带 | 小样本自检（配合小批清单 / `--limit` 渲染），输出带 `_smoke` 后缀，不写正式结果 |
| `--seed` | 任意整数 | `0` | 置换 / bootstrap / 探针种子（默认取配置 `seeds[0]`） |

### Step 5 汇总与阶段判断

命令（全部参数）：

```bash
python PreExperiments/stage2_controlled_edit/scripts/summarize_e4.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --seed [0(默认), 任意整数] --smoke
```

作用：读 `outputs/<model>/seed_<seed>/tables/` 的 e4 产物，写
`e4_summary.csv` 与 `e4_stage_decision.json`。正式汇总只消费 run_meta 中
**成功提交**的正式产物（`commit_experiment` 记录核验），不回退 smoke 文件；
检查预先确定的任务（binding+relation）与转换数是否齐全，缺失逐项列出。
判定规则随产物落盘（避免事后换口径）：确认性检验族 = E4-A 每任务 1 个 +
E4-C 每任务 1 个（任务级 `__overall__` C，方向须为正）+ E4-D 每任务×规定
对照各 1 个（同场景配对差，均值为正），族内 Holm 校正后 p < α 才判
"有支持证据"；对照缺覆盖或任务缺失记"数据不足"。`--smoke` 只读
`*_smoke.csv`，写独立的 `e4_summary_smoke.csv` / `e4_stage_decision_smoke.json`，
仅记录流程是否跑通，不给出机制支持标签。不加载模型。

参数（共 3 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 汇总的模型 |
| `--seed` | 任意整数 | `0` | 结果所属种子（定位 tables 目录） |
| `--smoke` | 开关 | 不带 | 只汇总 smoke 产物（独立文件名，仅记录流程是否跑通） |

### 冻结时点（工作方案开篇待办表）

| 时点 | 必须完成 |
|---|---|
| Step 2 正式编码前 | 8 个转换的正式清单、逐类型场景配额、编辑对元数据表结构 |
| Step 4 正式运行前 | E4 协议配置冻结；划分与哈希；基础场景互不相交自动检查；方向可靠性程序 |
| Step 3 首批人工核验前 | 核验清单、约 5% 间隔复核、系统性错误整批处理与补样流程 |
| 主实验批量生成前 | 合法组合表、编辑边表、泄漏检查；渲染账本与实测有效吞吐 |
| 主实验正式训练前 | 第 0 步联调通过；普通适配 / 困难负例强化、C-only 与范数控制配置 |

资源不足时的裁剪顺序：先减尚未运行的关系样本量和第二骨干复核范围，保留
所主张任务的基本诊断；已看过效果的任务不因结果不利而删除。

---

## 5. 输出结果

| 文件 / 目录 | 内容 | 产生 |
|---|---|---|
| `data/edit_pairs/<task>_pairs.parquet` | 编辑对清单（§2 的全部字段 + 划分） | Step 2 |
| `data/edit_pairs/<task>_states.parquet` | 唯一状态表（spec、是否渲染、派生来源） | Step 2 |
| `data/edit_pairs/<task>_splits.json` | 家族→划分清单、配额、md5、生成器版本 | Step 2 |
| `<render.output_dir>/<task>/images/`、`scenes/` | 渲染图（`<state_id>.png`）与回填场景 JSON | Step 3 |
| `<render.output_dir>/<task>/ledger.jsonl` | 渲染账本：状态、秒、重试、字节数、校验问题 | Step 3 |
| `<render.output_dir>/<task>/rendered_states.parquet` | 已通过校验的状态（图路径 + 场景 JSON）+ 派生态 | Step 3 |
| `outputs/audit/` | 题目审查清单 + 汇总 + 图片（251 题） | Step 1 |
| `outputs/<model>/seed_<k>/tables/e4a_results.csv` + `e4a_items.jsonl` | E4-A 逐任务 × 家族 × split 的 Pair Accuracy + 置换参考分布；逐对 2×2 矩阵 | Step 4 |
| `outputs/<model>/seed_<k>/tables/e4a_control_results.csv` | 换色正控制的同套配对评分（pair_acc + 区间 + 置换参考） | Step 4 |
| `outputs/<model>/seed_<k>/tables/e4b_delta_stats.csv` + `e4b_items.jsonl` + `e4b_probe.csv` | E4-B 方向相似度 / 范数 / 近零比例（含留出 vs 随机/错误方向参照）；探针与参照 | Step 4 |
| `outputs/<model>/seed_<k>/tables/e4b_control_delta.csv` | 对照编辑（换色/jpeg/crop/平移）的 Δv 长度与方向诊断 | Step 4 |
| `outputs/<model>/seed_<k>/tables/e4b_reliability.csv` | 方向可靠性：训练内对半划分子集比较 + 按场景重采样波动 | Step 4 |
| `outputs/<model>/seed_<k>/tables/e4c_conversion_stats.csv` + `e4c_identity.json` | E4-C 逐转换 C（探索性）+ 任务级 `__overall__` 主统计 + 跨组对照；代数自检 | Step 4 |
| `outputs/<model>/seed_<k>/tables/e4d_retrieval.csv` + `e4d_meta.json` | E4-D 各方法 top1 / 秩 / 相似度（pool 清单随行落盘）；α 与打乱配对自检 | Step 4 |
| `outputs/<model>/seed_<k>/tables/e4_summary.csv` + `e4_stage_decision.json` | 三轴判定（有支持证据 / 证据不足 / 数据不足；来源核验 + 任务完整性 + Holm 校正） | Step 5 |
| `outputs/<model>/seed_<k>/tables/e4_summary_smoke.csv` + `e4_stage_decision_smoke.json` | smoke 汇总（仅流程是否跑通，无机制标签） | Step 5（`--smoke`） |

## 6. 快速自检（小样本，分钟级）

```bash
# 1) 生成与统计模块单元测试（不依赖 Blender / 模型）
conda run -n CudaVersion128Fuxp python -m pytest PreExperiments/stage2_controlled_edit/tests/
# 2) 小批清单（2 个转换 × 3 场景，split 1/1/1）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/build_edit_pairs.py --task binding --binding-n 24 --recolor-n 24 --control-n 8
# 3) 小批渲染（约 30 秒/态）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/render_edit_pairs.py --task binding --limit 18
# 4) E4 冒烟（输出带 _smoke 后缀，不写正式结果）
conda run -n CudaVersion128Fuxp python PreExperiments/stage2_controlled_edit/scripts/run_e4.py --model clip_vitl14 --stage all --smoke
```

## 7. 主实验接口（工作方案第 9 节，随第二阶段一并交付）

本节全部训练代码未实现，属第二阶段交付物；第 0 步联调是其验收步骤。

- 组合划分：绑定面板八色固定为 `G0={red,blue}`、`G1={green,yellow}`、
  `G2={gray,brown}`、`G3={purple,cyan}`；cube 训练 G0∪G1 / OOD 验证 G2 /
  OOD 测试 G3，cylinder 对调，sphere 八色均可见作锚点；按无向边计 12 个
  合法训练转换、2 个 OOD 验证转换、2 个 OOD 测试转换。关系面板对象类型为
  (shape, material) 六类共 12 个无序类型对：OOD 验证 `{cube_metal,
  cylinder_metal}`，OOD 测试 `{cube_rubber, cylinder_metal}`，其余 10 类用于
  训练及 IID。排除规则对联合训练的全部入口生效，由程序审计泄漏；
- 主指标：组合 OOD 测试集上的 I2T Pair Accuracy 宏平均（任务等权 → 转换
  等权 → 场景内七模板平均；平局记错）；至少 3 个训练种子，按基础场景家族
  bootstrap；验证选模用 OOD 验证集同口径指标；
- 基线表：冻结原模型、普通状态匹配适配、hard-negative 强化、相同 2×2 样本
  的 hinge 配对排序目标（四个为判断主线必需）；另有 C-only 与范数项控制、
  状态级图结构监督、无上下文编辑变换、平坦编码对照、完整 GraphDelta 及
  去对齐 / 去上下文消融；
- 第 0 步联调：开发集约 50–200 对、几十至几百步，验收损失有限、预期参数
  有梯度、stop-gradient 位置、保存加载一致、identity 适配层与原评分一致、
  普通 / 困难负例版本候选集合一致等；不检验统计提升，不接触封存测试；
- 预算：按唯一场景状态登记渲染账本，实测有效吞吐、GPU 小时与磁盘量；尚无
  实测值的成本字段标记"待试运行测量"，不预填数字。
