# GraphDelta 第一阶段：冻结 baseline 诊断流水线说明

对应《GraphDelta_完整预实验方案_统一修订版》。本目录是阶段一（问题验证）的评估代码：
在固定的公开数据和固定的评估协议上，对冻结的预训练模型做零样本评估和表示分析。

**本目录只做评估，不做任何训练**：代码中没有 optimizer / loss / backward / LoRA /
fine-tuning / projection training，也没有 GraphDelta 的任何方法模块（方案 §55 / §83 / §84）。

阶段一回答三个问题（方案 §3）：

| 问题 | 内容 | 对应实验 |
|---|---|---|
| 问题一 | 冻结 baseline 对不同类型结构因素的图文区分能力是否不均衡 | E1 |
| 问题二 | 单因素变化引起的表示变化是否有稳定、可重复的规律 | E2 |
| 问题三 | CoGenT A/B 两组颜色—形状搭配的匹配难度是否存在不对称 | E3 |

---

## 1. 环境

```bash
cd GraphDelta/code
conda run -n CudaVersion128Fuxp python <script>
# python 3.10 / torch 2.7.0+cu128 / torchvision 0.22 / transformers 5.x / pandas / pyarrow
```

运行前提（`configs/paths.yaml`）：

- 五个数据集路径已配置且存在：CLEVR v1.0、CLEVR-CoGenT v1.0、PUG:SPAR、PUG:Animals、PUG:ImageNet；
- 三个模型的本地权重路径已配置且存在（见下表）。

---

## 2. 支持的模型（`--model` 参数的全部取值）

所有脚本的 `--model` 参数都有四个取值：

| `--model` 取值 | 模型 | 角色 | 输入预处理 |
|---|---|---|---|
| `all` | —（不是一个模型） | 依次运行下面全部已配置模型 | 按各模型自己的预处理 |
| `clip_vitl14` | CLIP ViT-L/14 | 首选 baseline（方案 §4 第一轮必做），**所有脚本的默认值** | 224×224，官方 preprocess（短边 resize + center crop） |
| `clip_vitb32` | CLIP ViT-B/32 | 冻结对照 baseline（方案 §4 允许增加） | 224×224，官方 preprocess |
| `siglip2_so400m` | SigLIP2 so400m-patch14-384 | 冻结对照 baseline（方案 §4 第二轮对照） | 官方 384×384 resize；文本编码前显式小写 |

`all` 的执行方式：

- **加载模型的 5 个脚本**（`extract_embeddings.py` / `run_e1.py` / `run_e1_spar_official.py` /
  `run_e2.py` / `run_e3.py`）：串行依次评估，**一次只加载一个模型**——上一个模型跑完、
  释放显存后再加载下一个；
- **只读结果的 3 个脚本**（`summarize_seed_stability.py` / `make_figures.py` /
  `analyze_e3_units.py`）：串行处理各模型的结果目录，不加载模型；
- 顺序固定为 `clip_vitl14 → clip_vitb32 → siglip2_so400m`（`paths.yaml` 中的配置顺序）；
- 某个模型中途失败即停止，后面的模型不再运行；
- 特征缓存与结果目录本来就按模型隔离，`all` 与逐个运行 `--model <name>` 的产物完全相同。

可用模型列表 = `configs/paths.yaml` 的 `models` 段与 `configs/<name>.yaml` 同时存在的名字；
传未配置的名字会直接报错并列出可用值。

规定（方案 §61 / §84-22）：

- 三个模型都使用原始预训练权重，`eval()` + `requires_grad=False`，完全冻结；
- 每个模型使用自己的官方预处理和 tokenizer，不共用 CLIP 的输入设置；
- 不同模型特征维度不同，不跨模型计算 cosine；
- 特征缓存按模型分目录（`data/cache/<model>/`），结果目录按模型 + 种子隔离
  （`outputs/<model>/seed_<k>/tables/`），换 `--model` 不会覆盖已有结果。

推荐执行顺序（方案 §4）：先跑 `clip_vitl14`；现象明确后再跑 `siglip2_so400m`
（或 `clip_vitb32`）复现关键趋势；两者趋势一致时再考虑第三个 baseline。

---

## 3. 支持的数据集（`--dataset` 参数的全部取值）

`--dataset` 出现在三个脚本里，各自允许的取值不同：

```text
scripts/build_manifests.py    --dataset ∈ { all, pug_spar, pug_animals, pug_imagenet }
scripts/extract_embeddings.py --dataset ∈ { all, clevr_val, cogent_valA, cogent_valB,
                                             pug_spar, pug_spar_official,
                                             pug_animals, pug_imagenet }（必填）
scripts/run_e2.py             --dataset ∈ { all, pug_animals, pug_imagenet }
```

各取值对应的数据与用途：

| 取值 | 数据集与 split | 数量 | 用途 |
|---|---|---:|---|
| `all`（build_manifests） | — | — | 依次生成下面三个 PUG 数据集的 manifest |
| `all`（extract_embeddings） | 全部 7 个数据集 | — | 串行依次抽取，顺序：`clevr_val → pug_spar → pug_spar_official → pug_animals → cogent_valA → cogent_valB → pug_imagenet`（含辅助集 pug_imagenet；方案 §14 允许暂缓，暂缓时不用 `all`，逐个运行其余数据集，缓存增量、之后可补抽） |
| `all`（run_e2） | pug_animals + pug_imagenet | — | 两个数据集串行跑 E2，核心数据集在前 |
| `clevr_val` | CLEVR v1.0 val split | 15,000 | E1 CLEVR 部分的图像特征（只用 val；test 无场景标注，方案 §16a） |
| `cogent_valA` | CLEVR-CoGenT val_A | 15,000 | E3 Condition A 的图像特征（cube∈{gray,blue,brown,yellow}，方案 §24） |
| `cogent_valB` | CLEVR-CoGenT val_B | 15,000 | E3 Condition B 的图像特征（cube∈{red,green,purple,cyan}，方案 §24） |
| `pug_spar` | PUG:SPAR 全量（纯评测集，无训练集） | 43,560 | E1 自建 2AFC 的 SPAR 图像特征 |
| `pug_spar_official` | PUG:SPAR 全量（同一批图，另一套预处理） | 43,560 | SPAR 官方协议专用特征：Resize 224×224 + ImageNet mean/std，缓存 key 单独，不与 `pug_spar` 混用 |
| `pug_animals` | PUG:Animals 全量 | 215,040 | E2 核心数据（70 动物 × 64 环境 × 3 大小 × 4 纹理 × 4 相机，每个组合一张图，方案 §11） |
| `pug_imagenet` | PUG:ImageNet 全量 | 88,328 | E2 辅助验证（方案 §14：检查规律是否只在动物场景成立，可暂时不跑） |

注意：CLEVR 和 CoGenT 不需要 manifest，各实验直接读官方 scenes JSON；
`--dataset` 一次只接受一个值（三个脚本的 `all` 除外）。

---

## 4. 实验矩阵（方案 §6 / §26 / §52）

| 实验 | 用途 | 数据集 | 输出目录 |
|---|---|---|---|
| E1 State-level Structural Diagnosis | 不同结构因素的 2AFC 图文区分能力 | PUG:SPAR + CLEVR v1.0 | `outputs/<model>/seed_<k>/tables/` |
| E2 Single-Factor Representation Response | 单因素变化的表示响应 | PUG:Animals（核心）；PUG:ImageNet（辅助） | 同上 |
| E3 Composition Difficulty Diagnosis | 属性—形状组合难度不对称 + sphere 对照 | CLEVR-CoGenT | 同上 |

E1 的因素（`configs/e1_state.yaml`）：

- CLEVR：`object` / `color` / `size` / `material` / `relation` / `binding`（六类 hard negative）；
- SPAR：`scene` / `position` / `attribute` / `relation`。

E2 的因素（`configs/e2_change.yaml`）：`environment` / `texture` / `size` / `camera`。

---

## 5. 完整流程（按方案 §82 的 Step 1–7）

```text
Step 1  整理数据 manifest（PUG:SPAR / Animals / ImageNet）
Step 2  特征抽取（每个数据集 × 每个模型一次）
Step 3  PUG:SPAR 结构诊断（自建 2AFC + 官方协议）
Step 4  CLEVR hard-negative 结构诊断
Step 5  PUG:Animals 单因素 Delta 分析（可选扩展 PUG:ImageNet）
Step 6  CLEVR-CoGenT 难度不对称诊断（含 sphere 对照 + 逐单元分析）
Step 7  汇总：多种子稳定性 + 五张图，判断是否进入第二阶段
```

以下每一步的命令都展示该脚本的**全部支持参数**。记号约定：

- `--xxx [a(默认), b, c]`：参数的可选取值列表，`(默认)` 标默认值；必填参数在列表后
  标 `（必填，无默认）`；
- 自由取值的参数同样用列表写法：`--seed [0(默认), 1, 2, …]`（`…` 表示还接受任意
  其他整数）、`--tables-dir [路径(默认), 任意目录路径]`；
- 开关参数（如 `--smoke`）直接列出：带上即生效，不带即关闭（默认不带）；
- 除标注（必填）外，所有参数都可省略，省略时用默认值；
- 这些是记号形式，不能直接复制运行；**可直接复制运行的命令（参数取具体值）见 §6**，
  运行时按 §1 加 `conda run -n CudaVersion128Fuxp` 前缀。

以下每一步给出完整参数命令、作用、参数说明表。

### Step 1 整理数据 manifest（方案 §57 / §58）

命令（可选；`--dataset all` 一条即可覆盖三个 PUG 数据集）：

```bash
python scripts/build_manifests.py --dataset [all(默认), pug_spar, pug_animals, pug_imagenet]
```

作用：读取 PUG 三个数据集的已有 metadata，统一字段，生成
`data/manifests/<dataset>_meta.parquet` 和 `data/manifests/inventory.json`
（记录行数和每列取值数）。只整理已有数据，不生成、不修改任何图像。

参数（共 1 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--dataset` | `all` / `pug_spar` / `pug_animals` / `pug_imagenet` | `all` | 生成哪个数据集的 manifest。`all` = 依次生成 `pug_spar`、`pug_animals`、`pug_imagenet` 全部三个；其余取值只生成对应一个 |

### Step 2 特征抽取（方案 §61 / §62）

命令（全部参数；`--dataset all` = 一次抽全部 7 个数据集，`--model all` = 三个模型各抽一套，
两者可组合成 3 模型 × 7 数据集全交叉，具体写法见 §6）：

```bash
python scripts/extract_embeddings.py --dataset [clevr_val, cogent_valA, cogent_valB, pug_spar, pug_spar_official, pug_animals, pug_imagenet, all]（必填，无默认） --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --force
```

作用：把指定数据集的全部图像编码成特征，写入 `data/cache/<model>/`。每个样本同时保存
raw（归一化前）和 normalized（归一化后）两种 embedding（方案 §37 / §62-9）。
**所有特征只抽一次，后续实验全部读缓存。**

说明：

- 不先运行本脚本也可以：run_e1 / run_e2 / run_e3 发现缓存缺失时会自动抽取；
  先单独抽取可以把 GPU 时间和实验时间分开。
- `--dataset all` 与 `--model all` 同时使用时的嵌套顺序：**模型在外层、数据集在内层**，
  同一模型连抽全部数据集后再换下一个模型；某个（模型， 数据集）组合失败即停止。
- 缓存按样本 id 增量补齐：已抽过的跳过，缺的补抽后合并写回。
- `pug_spar_official` 是 SPAR 官方协议专用的另一套特征，与自建 2AFC 用的
  `pug_spar` 缓存分开。
- CoGenT 的缓存 key 带数据根目录和路径配置的指纹；换数据根目录或
  `configs/e3_cogent.yaml` 的路径后得到新 key，需要重抽一次。
- 每个 `<model>` 缓存目录下有 `protocol.json` 记录该 encoder 的预处理语义指纹；
  预处理变更后旧缓存会直接报错，要求删缓存重抽。

参数（共 3 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--dataset` | `all` / `clevr_val` / `cogent_valA` / `cogent_valB` / `pug_spar` / `pug_spar_official` / `pug_animals` / `pug_imagenet`（八个取值见 §3 的表） | 无（必填） | 要抽取特征的数据集；`all` = 全部 7 个串行依次抽取 |
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m`（四个取值见 §2 的表） | `clip_vitl14` | 用哪个（哪些）冻结 baseline 抽特征；`all` = 三个模型依次各抽一套 |
| `--force` | 开关（带上即生效，无值） | 不带 | 删除已有缓存重新抽取。协议匹配时只删该数据集图像缓存 + 文本缓存；协议不匹配或无 `protocol.json` 时删整个模型缓存目录 |

### Step 3 PUG:SPAR 结构诊断（方案 §8–§10 / §27–§33）

命令一：自建 2AFC（E1 的 SPAR 部分）：

```bash
python scripts/run_e1.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
```

命令二：SPAR 官方协议复现（方案 §9"官方已有协议优先复现"）：

```bash
python scripts/run_e1_spar_official.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
```

作用：

- `run_e1.py`：对 SPAR 构造自建 2AFC 样本（scene / position / attribute / relation
  四个因素，每个因素正例 caption + 负例 caption，图像不变只换文本），报告 2AFC
  accuracy、matching margin、bootstrap CI、模板方差。重点看 relation 是否明显弱于
  attribute / scene（方案 §10）。
- `run_e1_spar_official.py`：复现 PUG 官方 `run_eval_vlms_on_spar.py` 的 CLIP 评测
  路径（全候选检索 top1/top5 + hard-negative 二选一，无背景 / caption 带背景两种），
  用于验证自建 pipeline 是否与已发表协议对齐。与自建 2AFC 结果分开报告。

注意：`run_e1.py` 一次运行同时覆盖 SPAR 和 CLEVR（Step 3 + Step 4），
输出在 `e1_results.csv` 里用 `dataset` 列区分，不需要分开跑两遍。

参数（`run_e1.py`，共 3 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 评估的冻结 baseline；`all` = 三个模型串行依次评估 |
| `--smoke` | 开关 | 不带 | 小样本快速自检，输出文件带 `_smoke` 后缀，不写正式结果 |
| `--seed` | 任意整数（方案 §66 固定用 `0` / `1` / `2`） | 取配置 `seeds[0]`（=0） | 候选采样种子。检验多次采样稳定性时分别用 `--seed 1`、`--seed 2` 重跑，结果自动写入 `seed_1` / `seed_2` 目录 |

参数（`run_e1_spar_official.py`，共 3 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 评估的冻结 baseline；`all` = 三个模型串行依次评估 |
| `--smoke` | 开关 | 不带 | 小样本快速自检（只取前 3000 行，部分任务样本少或为空，全量才有可比数字） |
| `--seed` | 任意整数（方案 §66 固定用 `0` / `1` / `2`） | 取配置 `seeds[0]`（=0） | 查询图片抽样种子 |

### Step 4 CLEVR hard-negative 结构诊断（方案 §15–§22 / §27–§33）

命令（与 Step 3 是同一个脚本，一次运行同时完成；全部参数与 Step 3 相同）：

```bash
python scripts/run_e1.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
```

作用：读取 CLEVR val 的官方 scene annotation，确定性生成 caption，构造六类
hard negative 2AFC（object / color / size / material / relation / binding-style）。
每条样本通过"正例为真、负例为假"检查，无效样本丢弃并记录数量（方案 §20a）。
图像不变，只修改文本候选，不重新渲染（方案 §16）。

重点比较（方案 §32）：object / color / size / material 与 relation / binding-style
之间是否存在明显能力差，并与 SPAR 的趋势交叉验证（方案 §9）。

参数（`run_e1.py`，共 3 个，与 Step 3 完全相同）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 评估的冻结 baseline；`all` = 三个模型串行依次评估 |
| `--smoke` | 开关 | 不带 | 小样本快速自检，输出带 `_smoke` 后缀 |
| `--seed` | 任意整数（方案固定用 `0` / `1` / `2`） | 取配置 `seeds[0]`（=0） | 候选采样种子 |

### Step 5 PUG:Animals 单因素 Delta 分析（方案 §11–§14 / §34–§45 / §59）

命令（全部参数；`--dataset all` = 核心数据集 + 辅助数据集串行，pug_animals 在前）：

```bash
python scripts/run_e2.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --dataset [pug_animals(默认), pug_imagenet, all] --smoke --seed [0(默认), 1, 2, …] --allow-empty-factors
```

作用：按已有 metadata 配对（方案 §12 / §59：固定其余因素、只变一个因素），
计算视觉 embedding 的 Δ = h_after − h_before，回答问题二。具体输出：

- 各因素（environment / texture / size / camera）的 Δ 范数（raw / normalized 两空间）；
- nuisance 基线（方案 §39a，必做）：同一张图的两个确定性增强视图之差
  （crop 0.85–1.0 + JPEG 90/75）+ 同内容不同模板的文本之差，
  用于判断因素响应是否超出无语义扰动水平；
- 分层相似度（方案 §40 / §41）：same transition（不同 context）/
  same factor different transition / different factor（分同 / 异 context）/ random；
- 主指标 Transition Similarity Gap = Sim(same transition, diff context) −
  Sim(diff factor, diff context)，带按 context 聚类的 bootstrap CI（方案 §66）；
- 逐 transition 表（environment 有 64×63÷2 = 2,016 个 transition，逐个汇总，方案 §64）。

配对方向（方案 §38）：size 按 small<medium<large；camera 按数值序；texture /
environment 按固定列表 / 字母序。顺序只用于统一 Δ 的符号，不解释为语义顺序。

PUG:ImageNet 的重复行处理：该数据集实测每个因素有 1,448 行（约 1.6%）与其他行
在全部 metadata 列上完全相同（同一 context 内同一因素值出现多次；Animals 是每个
组合恰好一张图的完整笛卡尔积，没有此情况）。配对时每个 (context, 因素值) 只保留
最小 `row_id` 的一行，其余确定性丢弃，数量以 `[pairs] ... 去重 ...` 警告写入日志，
不静默折叠。

参数（共 5 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 评估的冻结 baseline；`all` = 三个模型串行依次评估（模型在外层，数据集在内层） |
| `--dataset` | `all` / `pug_animals` / `pug_imagenet`（三个取值见 §3 的表） | `pug_animals` | E2 数据集。`pug_animals`=核心（全量 215,040）；`pug_imagenet`=辅助验证（方案 §14），输出文件名为 `e2_delta_stats_pug_imagenet.csv`；`all` = 两个数据集串行 |
| `--smoke` | 开关 | 不带 | 小样本快速自检（按 smoke_cols 逐层筛选小子集），输出带 `_smoke` 后缀 |
| `--seed` | 任意整数（方案固定用 `0` / `1` / `2`） | 取配置 `seeds[0]`（=0） | 配对与 nuisance 采样种子 |
| `--allow-empty-factors` | 开关 | 不带 | 某因素 0 对有效配对时跳过该因素而不是停止。默认停止是为了保证完整因素集；开启后结果不构成完整因素集。smoke 下 pug_imagenet 子集覆盖不全，自检时需要带上此参数 |

### Step 6 CLEVR-CoGenT 难度不对称诊断（方案 §23–§25 / §46–§51）

命令一：E3 主实验：

```bash
python scripts/run_e3.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
```

命令二：逐单元分析（方案 §49 附加分析 2 / §71 情况 C 排查）：

```bash
python scripts/analyze_e3_units.py --model [clip_vitl14, clip_vitb32, siglip2_so400m, all]（默认不带） --seed [0(默认), 1, 2, …] --tables-dir [任意目录路径]（默认不带） --no-strict --check-atol [1e-8(默认), 任意浮点数]
```

`analyze_e3_units.py` 的目录定位：`--tables-dir` 优先；否则用 `--model` + `--seed` 定位
`outputs/<model>/seed_<seed>/tables`（`--model all` = 依次分析全部模型，串行、不加载模型）；
两者都不带时默认 `clip_vitl14` / seed 0，目录不存在则回退旧布局 `outputs/tables`。

作用：

- `run_e3.py`：在 CoGenT valA（Condition A）/ valB（Condition B）上构造
  正确 color-shape 描述 vs 错误 color-shape 描述的 2AFC（换色负例 + 换形负例），
  报告 A / B 各自 accuracy 与 margin、Difficulty Gap = A − B（只限 cube / cylinder），
  **sphere 作为对照组**（sphere 在 A / B 都可取全部 8 色，用于分离"颜色词难度"和
  "组合难度"两种效应，方案 §49）。另输出逐 (color, shape) 组合表和逐颜色表，
  检查是否被少数颜色词拉平。解释边界（方案 §51）：冻结模型没有"见过 / 没见过"
  的定义，只能下"两组搭配难度不对称且方向与 A/B 划分一致"的结论。
- `analyze_e3_units.py`：读 E3 的 item 级结果，按 (shape_group, color) 共 16 个
  单元分别计算 A−B gap + 图片聚类 bootstrap CI，检验"逐组合效应方向不一致、
  被平均抵消"的情况。写出前先用全部 items 复算总 gap 并与主表对照，不一致默认停止。

参数（`run_e3.py`，共 3 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 评估的冻结 baseline；`all` = 三个模型串行依次评估 |
| `--smoke` | 开关 | 不带 | 小样本快速自检，输出带 `_smoke` 后缀 |
| `--seed` | 任意整数（方案固定用 `0` / `1` / `2`） | 取配置 `seeds[0]`（=0） | 候选采样种子 |

参数（`analyze_e3_units.py`，共 5 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | 不带（沿用默认 clip_vitl14/seed_0 布局） | 结果所属模型：定位 `outputs/<model>/seed_<seed>/tables`；`all` = 依次分析全部模型（串行，不加载模型）。不能与 `--tables-dir` 同用 |
| `--seed` | 任意整数 | `0` | 结果所属种子，与 `--model` 配合定位 tables 目录 |
| `--tables-dir` | 目录路径 | 不带 | 直接指定 tables 目录（优先于 `--model`）；默认不带时：`outputs/clip_vitl14/seed_0/tables`，不存在则回退旧布局 `outputs/tables` |
| `--no-strict` | 开关 | 不带 | 探索模式：总 gap 复算与主表不一致时仍输出单元表（检查结果始终写入 `e3_unit_consistency.json`） |
| `--check-atol` | 浮点数 | `1e-8` | 总 gap 复算与主表对照的绝对容差 |

### Step 7 汇总与阶段判断（方案 §66–§72）

先补齐三个种子（方案 §66 固定 seeds 0 / 1 / 2，只用于 sampling / bootstrap /
negative / 扰动，不是训练）：Step 3–6 的四个实验脚本各再跑两遍，`--seed` 分别取
`1` 和 `2`（其余参数不变，结果自动写入 `seed_1` / `seed_2` 目录）：

```bash
python scripts/run_e1.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
python scripts/run_e1_spar_official.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
python scripts/run_e2.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --dataset [pug_animals(默认), pug_imagenet, all] --smoke --seed [0(默认), 1, 2, …] --allow-empty-factors
python scripts/run_e3.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
# 每个种子跑完后对其逐单元分析（供稳定性汇总消费；--seed 取该种子，三个种子各跑一次）：
python scripts/analyze_e3_units.py --model [clip_vitl14, clip_vitb32, siglip2_so400m, all]（默认不带） --seed [0(默认), 1, 2, …] --tables-dir [任意目录路径]（默认不带） --no-strict --check-atol [1e-8(默认), 任意浮点数]
```

然后汇总（全部参数形式）：

```bash
# 三种子判定稳定性汇总（需 seeds 0/1/2 的全量结果都已跑完）
python scripts/summarize_seed_stability.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --allow-mixed-legacy --allow-exploratory-e3
# 生成方案 §67 的五张图
python scripts/make_figures.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --seed [0(默认), 1, 2, …] --tables-dir [outputs/<model>/seed_<seed>/tables(默认), 任意目录路径]
```

作用：

- `summarize_seed_stability.py`：逐项检验三种子下关键判定是否翻转——E1 accuracy
  CI 与 0.5 的关系、E1 factor gap CI 符号、SPAR 官方三种子最大差、E2 Norm Gap /
  Transition Gap CI 符号、E3 总 gap 与逐单元 gap 符号。输出 `seed_stability.csv`。
- `make_figures.py`：生成 §67 五图——fig1 2AFC accuracy、fig2 matching margin、
  fig3 Δ 范数 vs nuisance、fig4 相似性阶梯、fig5 CoGenT A/B + 逐单元 gap
  （PNG 300dpi + 合并 PDF）。

阶段判断（方案 §68–§72）：按 E1 / E2 / E3 的结果对照方案中的情况 A / B / C / D，
决定是否进入第二阶段 Controlled Edit 验证。旧结果快照与阶段总结见
`outputs/STAGE1_SUMMARY.md`。

参数（`summarize_seed_stability.py`，共 3 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 汇总的模型（读 `outputs/<model>/seed_{0,1,2}/tables`）。该模型三种子缺全量结果时直接报错并停止。`all` = 依次汇总全部模型（串行，不加载模型） |
| `--allow-mixed-legacy` | 开关 | 不带 | 三种子混用新旧目录布局时仍输出，仅作版本临时对照（默认停止，因为新旧数据口径不可比） |
| `--allow-exploratory-e3` | 开关 | 不带 | E3 派生单元表状态异常（failed / stale / 指纹不匹配）时仍汇总，但输出写入独立的 `seed_stability_exploratory.csv`，不冒充正式结果 |

参数（`make_figures.py`，共 3 个）：

| 参数 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `--model` | `all` / `clip_vitl14` / `clip_vitb32` / `siglip2_so400m` | `clip_vitl14` | 结果所属模型：定位 tables 目录与图标题。`all` = 依次给全部模型绘图（串行，不加载模型；某模型缺全量结果即停止） |
| `--seed` | 任意整数 | `0` | 结果所属种子：定位 tables 目录与图标题 |
| `--tables-dir` | 目录路径 | `outputs/<model>/seed_<seed>/tables`（`clip_vitl14` + seed 0 可回退旧布局 `outputs/tables`） | 直接指定 tables 目录；只对应一个模型，不能与 `--model all` 同用。其他模型 / 种子缺少新结果时不会回退，直接报错 |

图输出目录：所有模型与种子统一写 `outputs/figures/<model>_seed<k>/`，
互不覆盖（`clip_vitl14` + seed 0 不再特判写顶层 `outputs/figures/`）。

---

## 6. 一条命令跑全部（`--model all` / `--dataset all`）

本节命令是**可直接复制运行**的具体形式（§5 记号形式的具体取值，前缀见 §1；
均省略了可省参数 `--smoke` / `--seed` / `--force` / `--allow-empty-factors` 等，
全部参数见 §5 各步）。三个实验、特征抽取、汇总绘图全部支持 `all`，完整流程只需 8 条命令：

```bash
# 1) 特征：3 模型 × 7 数据集 = 21 组一次抽完（串行，模型外层、数据集内层）
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset all --model all
# 2) 四个实验（各模型结果写 outputs/<model>/seed_0/tables/）
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --model all
conda run -n CudaVersion128Fuxp python scripts/run_e1_spar_official.py --model all
conda run -n CudaVersion128Fuxp python scripts/run_e2.py --dataset all --model all
conda run -n CudaVersion128Fuxp python scripts/run_e3.py --model all
# 3) 逐单元分析 / 汇总 / 绘图（不加载模型，同样支持 --model all）
conda run -n CudaVersion128Fuxp python scripts/analyze_e3_units.py --model all
conda run -n CudaVersion128Fuxp python scripts/summarize_seed_stability.py --model all
conda run -n CudaVersion128Fuxp python scripts/make_figures.py --model all --seed 0
```

也可以只指定单个模型或单个数据集，例如只跑 SigLIP2 对照：把 `--model all` 换成
`--model siglip2_so400m`；只抽某个数据集：把 `--dataset all` 换成具体数据集名。

SigLIP2 注意事项：当前实现是 FixRes checkpoint 路径（固定 384×384），文本编码前
显式小写。SPAR 官方协议对其使用自身 384 预处理，因此 SigLIP2 的官方协议数字
不复现任何官方发布值，只作 backbone 对照。

---

## 7. 输出结果

结果目录：`outputs/<model>/seed_<k>/tables/`（按模型 + 种子隔离，互不覆盖）。
目录内 `run_meta.json` 记录模型、种子、git commit 与特征缓存协议指纹；每个实验
启动时登记 `running`，全部文件成功写出后才标记 `success`——运行中途失败时旧
结果不会被失败运行冒充。smoke 运行的文件名带 `_smoke` 后缀，不更新正式结果。

| 文件 | 内容 | 方案 |
|---|---|---|
| `e1_results.csv` | 按 dataset × factor：2AFC accuracy + CI、margin 均值 / 中位数 / 标准差、模板方差、n | §63 |
| `e1_factor_gaps.csv` | 同数据集内各因素两两 accuracy gap + bootstrap CI | §66 |
| `e1_clevr_filter_stats.json` / `e1_spar_filter_stats.json` | 各过滤原因丢弃的样本计数 | §20a |
| `e1_items_clevr.jsonl` / `e1_items_spar.jsonl` | 每条 2AFC 样本（正 / 负文本、图片、因素），可复查 | §60 |
| `e1_spar_official.csv` | SPAR 官方协议：六个任务 × 检索 top1/top5 + hard-negative × 无背景 / 带背景 | §9 |
| `e2_delta_stats.csv`（ImageNet 为 `e2_delta_stats_pug_imagenet.csv`） | 按因素：Δ 范数（raw/z）、nuisance 范数、分层相似度、Transition Gap + CI、Norm Gap + CI | §64 |
| `e2_transition_stats.csv` | 逐 (factor, source, target)：n_contexts、相似度、Δ 范数 | §64 |
| `e3_cogent.csv` | 按 condition × shape_group × negative_type 的 accuracy + CI + margin | §65 |
| `e3_difficulty_gap.csv` | A−B 难度差（题型等权主指标 + 汇总口径对照；cube/cylinder 主指标 + sphere 对照） | §50 |
| `e3_cogent_by_combo.csv` / `e3_cogent_by_color.csv` | 逐 (color, shape) / 逐颜色 accuracy 与 margin | §49 |
| `e3_items_cogent.jsonl` | E3 每条 item 级结果，供逐单元分析复查 | §49 |
| `e3_unit_gaps.csv` + `e3_unit_consistency.json` | 逐 (shape_group, color) 单元 A−B gap + CI；sidecar 记录一致性检查状态与输入指纹 | §49 |
| `seed_stability.csv` | seeds 0/1/2 三种子关键判定稳定性汇总 | §66 |
| `figures/<model>_seed<k>/` | §67 五图（PNG 300dpi + 合并 PDF） | §67 |

历史说明：`outputs/tables`、`outputs/tables_seed{1,2}` 是 2026-10-22 clip_vitl14
三种子运行在目录隔离修复前的旧布局快照，只读保留；读取端只允许 clip_vitl14
（绘图另限 seed 0）回退，其他模型缺少新结果时直接报错。

---

## 8. 快速自检（小样本，分钟级）

改代码后先跑 smoke 确认流程通畅，输出带 `_smoke` 后缀、不写正式结果：

```bash
python scripts/run_e1.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
python scripts/run_e1_spar_official.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
python scripts/run_e2.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --dataset [pug_animals(默认), pug_imagenet, all] --smoke --seed [0(默认), 1, 2, …] --allow-empty-factors
python scripts/run_e3.py --model [clip_vitl14(默认), clip_vitb32, siglip2_so400m, all] --smoke --seed [0(默认), 1, 2, …]
```

`--model all` / `--dataset all` 在 smoke 下同样可用（串行自检全部模型 / 数据集）。
`run_e2.py` 的 smoke 自检要带 `--allow-empty-factors`：pug_imagenet 的 smoke 子集
覆盖不全，texture 因素 0 对配对会默认停止（全量运行无此问题）。

注意：SPAR 官方协议的 smoke 模式只取前 3000 行，部分任务样本很少或为空，
全量运行才有可与论文对齐的数字。

---

## 9. 许可

PUG 系列（Animals / SPAR / ImageNet）为 CC-BY-NC，附加条款禁止用于训练生成式
模型。本流水线只做评估，符合许可；开源代码或写论文时注明数据来源和许可。
CLEVR / CLEVR-CoGenT 无此限制（方案 §62a）。
