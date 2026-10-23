# GraphDelta 代码审查报告

- 初次审查及修复复核日期：2026-10-23（Asia/Shanghai）
- 审查目录：`code/`
- 初次审查版本：`main`，提交 `fac3cda5d6ff593947387a4916d720c6e33096cf`
- 本轮复核版本：`main`，提交 `3fc46b1f851f4fe49986a7e22ec7726ad2d32786`
- 对照文档：[完整预实验方案](GraphDelta_完整预实验方案_统一修订版.md)、[研究框架](GraphDelta_研究框架_统一修订版.md)、[代码 README](code/README.md)
- 状态：修复复核完成；5 项源码问题已修正，R2、R7 部分解决；缓存重建和全量结果更新尚未完成验收。本次仅更新报告，未修改源代码。

## 1. 审查结论

初次审查确认 3 项 P1、4 项 P2。对更新后的 `3fc46b1` 逐项复核，**R1、R3、R4、R5、R6 的源码问题已修正；R2、R7 仍为部分解决**。源码修正不等于真实权重推理、缓存迁移和全部实验结果均已验收。

当前已包含 `clip_vitl14`、`clip_vitb32`、`siglip2_so400m` 三个模型配置；B/32 的配置和工厂分发已接通。

| 编号 | 原优先级 | 原问题 | 本轮状态 | 尚需完成 |
|---|---|---|---|---|
| R1 | P1 | SigLIP2 FixRes 被按 NaFlex 加载和预处理 | 实现已修复 | 确认旧缓存已移出当前命名空间，完成真实模型对齐及重跑 |
| R2 | P1 | 实验输出未按模型、种子隔离 | 部分解决：写入已隔离，读取仍可能混用历史结果 | 修复汇总和绘图的无条件旧布局回退，校验模型与种子身份 |
| R3 | P1 | SPAR 只检查一侧 `blank` | 源码及最小验证通过 | 重新生成全量 SPAR items、评估与相关汇总 |
| R4 | P2 | SigLIP2 文本缺少小写化 | 实现及模拟调用验证通过 | 重建旧文本缓存，完成真实 tokenizer/特征对齐 |
| R5 | P2 | CLIP-B-32 缺少配置与权重路径 | 配置及工厂分发验证通过 | 用实际 checkpoint 完成小样本端到端验证 |
| R6 | P2 | SPAR 官方单动物检索候选池与原脚本不同 | 源码及模拟候选构造验证通过 | 重跑受影响任务，核对全量候选和新指标 |
| R7 | P2 | 图标题写死 CLIP ViT-L/14、seed 0 | 部分解决：标题已参数化，仍可能标错回退数据 | 从结果元数据取得标签，或校验参数与实际输入一致 |

当前主要残留是：**后处理仍会把历史 CLIP-L/14 结果用于其他模型；SigLIP2 旧缓存未自动失效；仓库内全量结果仍为修复前版本，远端是否已重跑待确认**。应先阻止错误的数据回退，再核验缓存重建与结果重算。

优先级沿用初次审查：P1 应在继续正式实验前处理；P2 应在结果汇总、对外报告或声明协议复现前处理。优先级表示实验影响与处理顺序，不表示代码运行一定会报错。

## 2. 审查范围与验证边界

重点检查模型工厂、CLIP/SigLIP2 encoder、checkpoint 转换、图文特征缓存、PUG:SPAR 元数据与 caption、SPAR 官方评估，以及 E1/E2/E3 的输出和后处理。

初次审查已完成：

- 41 个 Python 文件的 AST 语法解析。
- 不加载真实模型的 E1 输出覆盖复现。
- `build_spar_items` 的 `blank` 输入最小复现。
- 已有 seed 0/1/2 SPAR item 文件的逐条统计。
- 对所选 SigLIP2 checkpoint、Transformers v5.13.0 实现及 PUG 官方评估脚本的核对。

本轮复核新增验证：

- 更新后 41 个 Python 文件的 AST 语法解析。
- 实际 E1 `run` 配合模拟数据和评分，验证两模型 × 两种子生成四套互不覆盖的结果。
- 实际 SPAR 元数据和 item 函数的双侧 `blank`、纯背景、同种双动物与正常双动物最小验证。
- 实际 SPAR 官方 `run` 配合模拟特征和评分，验证 32 动物 × 10 背景 × 3 纹理的单动物候选数为 33／330。
- 实际模型工厂及 SigLIP2 类配合替代构造器、tensor/model/tokenizer，验证名称与路径传递、小写化、句号开关、长度 64 和图像入口路由。
- 执行绘图及稳定性汇总入口中的目录选择逻辑，验证缺少目标模型新结果时的历史数据回退；不运行实际绘图或全量指标汇总。
- 模拟旧 NPZ 配合实际缓存、抽取函数，验证旧特征仍会命中并跳过新 encoder。
- 检查当前仓库内旧全量 SPAR 输出和新 smoke 输出的状态。

未完成真实模型全量推理或重新计算全部指标：当前可用 Python 环境没有 `torch`、`transformers`，配置中的数据与权重路径指向 Linux 目录。模拟验证用于检查控制流、参数传递和缓存行为，不能代替真实 checkpoint、tokenizer 或官方特征对齐。报告不量化修复后的真实准确率变化，也不将语法或模拟检查视为端到端运行通过。

## 3. 本轮修复复核记录（3fc46b1）

### 3.1 R1：FixRes 实现已改正，缓存重建与真实推理待验

[siglip_encoder.py](code/src/models/siglip_encoder.py) 第 61、68 行已改为 `SiglipModel` 与 `SiglipImageProcessor`；普通图像和官方协议入口均使用对应 processor。[paths.yaml](code/configs/paths.yaml) 第 13 行指向原始 checkpoint 目录。转换脚本已标注废弃，默认执行路径不再依赖 `-v5` 产物。

实现已符合本项修复方向。仍需用真实权重核验与官方调用的特征一致性，并确认旧 NaFlex 路径的缓存已移出当前命名空间。缓存风险见 §3.8。

### 3.2 R2：实验写入隔离已通过，汇总读取仍有残留

四个实验入口均已调用 [output_tables_dir](code/src/utils/io.py)（第 39 行），输出到 `outputs/<model>/seed_<seed>/tables/`。调用位置为 E1 第 204 行、E2 第 93 行、E3 第 74 行、SPAR 官方评估第 166 行。

实际 E1 `run` 的模拟验证得到四份独立文件：

```text
outputs/clip_vitl14/seed_0/tables/e1_results.csv
outputs/clip_vitl14/seed_1/tables/e1_results.csv
outputs/siglip2_so400m/seed_0/tables/e1_results.csv
outputs/siglip2_so400m/seed_1/tables/e1_results.csv
```

**残留问题**：[summarize_seed_stability.py](code/scripts/summarize_seed_stability.py) 第 55–63 行对任意模型逐种子回退旧目录，没有验证这些旧数据属于哪个模型。当前运行 `--model siglip2_so400m` 或 `--model clip_vitb32` 的目录选择逻辑，均会选择历史 CLIP-L/14 的 `tables`、`tables_seed1`、`tables_seed2`。若仅部分种子存在新结果，还可能混合两个模型。第 179 行将汇总写回所选 seed 0 目录，错误回退还可能覆盖旧 `seed_stability.csv`。

建议仅在明确确认历史数据身份后允许兼容读取；缺少所请求模型的结果时应报错，不能仅因旧文件存在就回退。运行元数据应包含模型、种子和配置标识，供各后处理入口共同校验。目前这些身份主要体现在目录路径，CSV 和运行 manifest 尚未提供相应校验。

### 3.3 R3：SPAR 样本生成已修正，仓库内全量结果仍为旧版

[pug_spar.py](code/src/data/pug_spar.py) 第 43 行已要求两侧均非 `blank`；[e1_state_diagnosis.py](code/src/experiments/e1_state_diagnosis.py) 第 160–166 行跳过纯背景并正确判定双对象，第 178–180 行阻止单动物进入双对象任务；[captions.py](code/src/data/captions.py) 第 65–73 行从实际非空一侧取单动物名称。

实际函数最小验证通过：两侧单动物均只生成正确 scene；纯背景、同种双动物按规则跳过；双对象任务只接受有效双动物行。

仓库中的 `outputs/tables`、`tables_seed0/1/2` 仍保留修复前的全量 SPAR items 和统计，远端是否已重跑待确认。新增 `outputs/clip_vitl14/seed_0/tables/e1_items_spar_smoke.jsonl` 共 800 条，未发现 `blank` 正例；但 smoke 子集不覆盖原问题所在的尾部样本，不能替代全量重跑。初次审查的无效 item 计数保留在 §4 的 R3 记录中，表示历史输出状态。

### 3.4 R4：显式小写化已加入，旧文本缓存仍需失效

[siglip_encoder.py](code/src/models/siglip_encoder.py) 第 112 行已增加 `batch = [t.lower() for t in batch]`，覆盖开启和关闭补句号的两条路径。

实际 encoder 类配合替代 tokenizer/model 的验证通过：`This is an Elephant` 在开启补句号时传入 tokenizer 为 `this is an elephant.`，关闭后为 `this is an elephant`，padding/truncation 长度保持 64。该验证不等于真实 tokenizer 的 token ID 对齐通过。

旧文本缓存仍使用相同模型目录和输入文本键，会绕过修复后的编码路径。必须重建或版本化文本缓存，详见 §3.8。

### 3.5 R5：B/32 配置和工厂分发已接通

新增 [clip_vitb32.yaml](code/configs/clip_vitb32.yaml)，`model_type: clip`；[paths.yaml](code/configs/paths.yaml) 第 11 行增加原始 B/32 checkpoint 路径。当前规范名称是 **`clip_vitb32`**，README 已给出相应 `--model` 示例。

实际 `build_encoder` 配合替代构造器，验证三个模型的名称、权重路径和 batch size 正确传递。原先缺配置的问题已解决；尚未在当前环境执行 B/32 真实权重加载、图文编码或评估。

### 3.6 R6：官方单动物候选中的 blank 已补回

[e1_spar_official.py](code/src/experiments/e1_spar_official.py) 第 242 行已将 `blank` 加回单动物候选池。实际 `run` 配合模拟特征、评分和文件写入，在 32 动物 × 10 背景 × 3 纹理的构造中，三种纹理均得到无背景 33、带背景 330 个候选，通过断言。

原先遗漏干扰候选的问题已修正。旧全量 CSV 仍记录 32／320 个候选，需要重跑受影响任务；模拟构造不能代替实际全量候选、标签及指标的一致性验收。

### 3.7 R7：标题已参数化，历史数据回退仍可造成错误标注

[make_figures.py](code/scripts/make_figures.py) 第 87–88 行已使用 `model_label` 和 `seed`，不再固定写 CLIP-L/14、seed 0；非默认模型/种子的图像输出目录也已区分。

**残留问题**：第 247–248 行对任何模型、种子都可能回退 `outputs/tables`，但第 252 行仍按请求模型加载标签，第 263–264 行按请求种子绘图。执行当前目录选择逻辑可复现：

```text
请求：--model siglip2_so400m --seed 1
实际 tables：outputs/tables（历史 CLIP-L/14 seed0）
标题标签：SigLIP2 so400m-p14-384, seed 1

请求：--model clip_vitb32 --seed 1
实际 tables：outputs/tables（历史 CLIP-L/14 seed0）
标题标签：CLIP ViT-B/32, seed 1
```

显式传入 `--tables-dir` 时也没有校验该目录的真实模型与种子。应修复无条件回退，并从结果元数据取得标签，或验证 CLI 参数与输入一致。仅将固定标题改为参数化标题不足以关闭本项。

### 3.8 R1/R4 共同待验事项：旧 SigLIP2 缓存不会自动失效

[cache.py](code/src/features/cache.py) 第 11、48 行继续使用原有 `siglip2_so400m` 命名空间，第 31、60 行直接读取旧 NPZ，没有 checkpoint、processor 或预处理版本校验。[extract.py](code/src/features/extract.py) 第 43–49 行按已有图像 ID 跳过编码；文本键仍基于调用前的原始大小写字符串。

模拟旧 NPZ 配合实际缓存、抽取函数得到：

```text
旧图像缓存命中：encoder_calls=0，返回旧向量 [1, 0]
旧文本缓存命中：encoder_calls=0，返回旧向量 [1, 0]
```

模拟新 encoder 应返回 `[0, 1]`，但未被调用。这证明仅替换源码不能自动更新已存在的旧缓存；并不表示服务器上目前一定存在旧缓存。本机没有相应缓存目录，无法确认远端是否已按 README 要求清理重抽。

建议将旧缓存移至历史命名空间，为新缓存记录 checkpoint/processor/文本预处理指纹。至少应在重跑前确认该模型全部图像与文本缓存已失效。`extract_embeddings.py --force` 只删除指定图像缓存，不会删除 `text_cache.npz`。

## 4. 初次审查问题详情（fac3cda，保留为历史依据）

本节记录初次审查时的问题、证据和建议；除另有说明，以下位置与行号对应 `fac3cda`，其中“当前”等表述均指初次审查版本。最新修复状态以 §1 和 §3 为准，不能将历史缺陷描述理解为更新后源码仍保持相同实现。

### R1 [P1] SigLIP2 FixRes 被按 NaFlex 加载，转换不保持完整模型语义

**位置**：

- [siglip_encoder.py](code/src/models/siglip_encoder.py)：52、59 行。
- [convert_siglip2_checkpoint.py](code/scripts/convert_siglip2_checkpoint.py)：54–68、70–87 行。
- [paths.yaml](code/configs/paths.yaml)：11 行，默认指向转换后的 `-v5` 目录。

**问题**：所选 `google/siglip2-so400m-patch14-384` 是 FixRes checkpoint。官方配置的 `model_type` 为 `siglip`，图像 processor 为 `SiglipImageProcessor`，输入固定为 384×384。Transformers v5.13.0 的 `SiglipModel` 仍保留对应的 Conv2d patch embedding，不能将这里的结构差异解释为必须进行的版本升级转换。[官方模型配置](https://huggingface.co/google/siglip2-so400m-patch14-384/raw/main/config.json)、[官方预处理配置](https://huggingface.co/google/siglip2-so400m-patch14-384/blob/main/preprocessor_config.json)、[v5.13.0 SiglipModel 实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/siglip/modeling_siglip.py)。

本地将卷积权重展平后，强制使用 `Siglip2Model` 和 `Siglip2ImageProcessor`。后者按 patch 数量动态缩放并保留图像纵横比，属于 NaFlex 路径，传入旧配置的 `size` 不能使它恢复固定 384×384 的处理。[FixRes/NaFlex 官方说明](https://huggingface.co/docs/transformers/en/model_doc/siglip2)、[v5.13.0 Siglip2ImageProcessor 实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/siglip2/image_processing_siglip2.py)。

**验证证据**：按官方动态尺寸算法，在 `patch_size=14`、`max_num_patches=729` 下复现得到：

| 原图尺寸（高×宽） | FixRes 输入 | 当前转换路径输入 | 当前实际 patch 网格 |
|---|---|---|---|
| 512×512 | 384×384 | 378×378 | 27×27 |
| 320×480 | 384×384 | 308×462 | 22×33 |

转换脚本的断言只验证同一输入张量上 Conv 与 Linear 的 patch 映射等价，没有比较两套 processor 和完整 encoder 的输出。它不能覆盖图像缩放、patch 网格和位置编码处理的变化。

**实验影响**：不符合方案 §4、§61、§84-22 的原始 baseline 与官方预处理要求。由该路径产生的 SigLIP2 结果不能直接视为指定 FixRes 模型的标准结果。

**建议修复与验收**：使用原始 checkpoint，通过 `AutoModel`/`SiglipModel` 与对应官方 processor 加载；移除该 FixRes 模型对转换目录的依赖。以同图、同文本对照官方调用得到的特征，确认完整路径一致后，重建该模型相关缓存并重跑评估。

### R2 [P1] 多模型、多种子运行共用同名结果文件

**位置**：

- [e1_state_diagnosis.py](code/src/experiments/e1_state_diagnosis.py)：195 行。
- [e2_change_diagnosis.py](code/src/experiments/e2_change_diagnosis.py)：92 行。
- [e3_cogent_diagnosis.py](code/src/experiments/e3_cogent_diagnosis.py)：73 行。
- [e1_spar_official.py](code/src/experiments/e1_spar_official.py)：162 行。

**问题**：所有入口均设置 `out_tables = paths["dirs"]["outputs"] / "tables"`，随后写入固定文件名。表格缺少模型及运行种子标识，切换 `--model` 或 `--seed` 不会自动改变输出位置。特征缓存虽然按模型隔离，但结果文件没有沿用这一隔离规则。

**最小复现**：在临时目录中调用实际 E1 `run`，将数据读取、特征提取和模型评分替换为小型模拟输入。先后使用两种模型名称与不同评分，得到：

```text
after clip_vitl14:    e1_results.csv, accuracy=[0.25, 0.25], model_column=False
after siglip2_so400m: e1_results.csv, accuracy=[0.75, 0.75], model_column=False
result CSV count: 1
```

该复现只验证输出路由与覆盖行为，不验证模型评分。

**实验影响**：完整重跑会覆盖前一模型结果；只重跑部分实验时，`outputs/tables` 可能同时包含不同模型或种子的结果，后续绘图和稳定性汇总无法可靠识别。

**建议修复与验收**：统一采用例如 `outputs/<model>/seed_<seed>/tables/` 的目录结构，保存 `model`、`seed`、协议与配置标识，并让分析和绘图脚本消费同一份运行信息。连续运行两个模型、两个种子后，应保留四套可追溯且互不覆盖的结果。

### R3 [P1] SPAR 单侧 `blank` 判断污染自建 2AFC

**位置**：

- [e1_state_diagnosis.py](code/src/experiments/e1_state_diagnosis.py)：157 行。
- [pug_spar.py](code/src/data/pug_spar.py)：40 行。
- [captions.py](code/src/data/captions.py)：65–72 行。

**问题**：代码以 `character2_name != "blank"` 判断双对象，但数据允许 `character_name` 为空。于是主对象为空、第二对象为动物的场景被纳入双对象任务；scene 模板同样把 `blank` 当成动物。双侧均为空的纯背景场景也会生成动物存在描述。

**最小复现**：向实际 `build_spar_items` 传入主对象为 `blank`、第二对象为 `Elephant` 的一行元数据，可生成：

```text
position:  the blank is to the left of the Elephant
attribute: a blue blank and a red Elephant
relation:  the blank is to the left of the Elephant
```

已有输出还包含 `a blank and a Elephant in the desert` 和 `a blank in the boxing ring`。见 [e1_items_spar.jsonl](code/outputs/tables/e1_items_spar.jsonl) 的 1937、1966 行等样例。

**已有结果统计**：下表统计正例内容含独立单词 `blank` 的 item 数，各单元总样本数均为 2,000。

| 种子 | scene | position | attribute | relation |
|---|---:|---:|---:|---:|
| 0 | 64 | 62 | 68 | 62 |
| 1 | 66 | 52 | 76 | 57 |
| 2 | 57 | 82 | 63 | 58 |

seed 0 读取 `outputs/tables`；seed 1/2 分别读取 `outputs/tables_seed1` 和 `outputs/tables_seed2`。seed 0 的 scene 中还包含 5 条双侧均为空的纯背景样本。

**实验影响**：这些 item 不满足正例为真的基本条件。无效正例占各单元约 2.6%–4.1%，可能影响准确率、margin 与 factor gap；不能从这些计数直接推断修复后的指标变化幅度。

**建议修复与验收**：双对象必须同时满足两列非空；单动物 caption 应读取实际非空的一侧；纯背景场景应使用背景描述，或从当前动物描述任务中排除。同步修复元数据标记和 caption 生成，补充过滤计数，重新抽样并评估。该问题本身不要求重抽图像 embedding。

### R4 [P2] SigLIP2 绕过 processor 后缺少文本小写化

**位置**：[siglip_encoder.py](code/src/models/siglip_encoder.py)：60、101–116 行。

**问题**：代码直接使用 `AutoTokenizer`，只设置补句号、padding 和 truncation，没有显式小写化。原 checkpoint 的 `tokenizer_class` 为 `GemmaTokenizer`；对应 v5.13.0 实现的 normalizer 不负责 lowercase，不能仅凭配置中的 `do_lower_case` 就认为处理已经生效。[官方 tokenizer 配置](https://huggingface.co/google/siglip2-so400m-patch14-384/raw/main/tokenizer_config.json)、[Gemma tokenizer 实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/gemma/tokenization_gemma.py)。

SigLIP2 官方要求与训练一致的小写文本预处理；独立编码文本时推荐使用具备该处理的 `Siglip2Tokenizer`。[官方文本预处理说明](https://huggingface.co/docs/transformers/en/model_doc/siglip2#text-embeddings-and-retrieval)。

**触发场景与影响**：SPAR 动物名保留大写，官方模板还包含 `This` 等文本。自建及官方 SPAR 路径均会受影响，偏离方案 §84-22。此项通过配置和源码核对确认，未在当前环境运行真实 tokenizer 或量化指标变化。

**建议修复与验收**：显式进行模型要求的小写化，或使用匹配的官方 processor/tokenizer。固定含大写动物名的文本，与官方调用对照 token 序列及特征；修复后重建 SigLIP2 文本缓存。

### R5 [P2] 当前工作区缺少 CLIP-B-32 的可用配置

**位置**：[models/__init__.py](code/src/models/__init__.py)：10–18 行；[paths.yaml](code/configs/paths.yaml)：9–11 行；`code/configs/`。

**问题**：模型工厂按名称读取 `configs/<model>.yaml`，但当前只有 `clip_vitl14.yaml` 和 `siglip2_so400m.yaml` 两份模型配置，权重路径也只有这两项，未发现 B-32 的配置或名称映射。

**验证证据**：直接调用配置加载函数，传入 `clip-b-32` 等名称会触发 `FileNotFoundError`。因此，`--model` 参数的存在不代表 B-32 已完成接入。

**建议修复与验收**：明确 B-32 的统一配置名与原始 checkpoint，补齐 YAML、`paths.yaml` 和 README 示例。现有 `CLIPEncoder` 可以复用，无需新增重复 encoder 类；验收时应确认缓存与输出使用独立模型名称，并完成小样本图文编码和评估。

### R6 [P2] SPAR 官方单动物检索候选池与原脚本不一致

**位置**：[e1_spar_official.py](code/src/experiments/e1_spar_official.py)：238–243 行。

**问题**：本地通过 `actor` 合并非空侧，并去除 `blank` 后构造候选；[PUG 官方评估脚本](https://github.com/facebookresearch/PUG/blob/main/PUG_SPAR/run_eval_vlms_on_spar.py) 的 `one_object` 任务直接使用 `subset_char_only_by_texture['character_name'].unique().tolist()`，没有在该任务中移除 `blank`，并会为其生成 caption。

**实验影响**：候选集合发生变化，进而可能改变 top1/top5。即使将 `max_per_task` 设为 `null` 恢复全量查询，本地实现仍不能称为严格复现该官方任务。

**建议修复与验收**：严格复现模式应恢复官方候选集合；如保留清理后的版本，应使用独立协议名称并明确差异。验收应直接对比候选 caption 集合、标签和查询样本，不能只对比样本数量或最终准确率。

该项应与自建 2AFC 区分：官方检索中的 `blank` 是原脚本的干扰候选；R3 中的 `blank` 被写进了声明为真的正例，两者的语义与修复要求不同。

### R7 [P2] 绘图标题写死模型与种子

**位置**：[make_figures.py](code/scripts/make_figures.py)：87 行。

**问题**：Figure 1/2 的标题固定包含 `CLIP ViT-L/14, seed 0`。脚本允许通过 `--tables-dir` 读取其他目录，但标题不会据此改变。

**实验影响**：读取 SigLIP2 或其他种子的结果时，会将图表错误标为 CLIP ViT-L/14、seed 0。修复 R2 的输出隔离后，此问题仍需单独处理。

**建议修复与验收**：从运行元数据读取模型和种子，或要求用户传入明确参数。使用不同模型、不同种子的结果分别绘图，检查标题、文件名与输入结果一致。

## 5. 已检查且未发现明确主链问题的部分

- CLIP 与 SigLIP2 均调用 `eval()`，并逐参数设置 `requires_grad_(False)`；已审查路径保持冻结、纯评估边界。
- 特征缓存按模型名称分目录，没有发现已注册模型之间直接串用缓存或跨模型计算 cosine 的明确问题；同一模型修复前后版本的旧缓存复用风险见 §3.8。
- E2 的 canonical transition、同 transition 跨 context 比较及双端点 context bootstrap，与方案 §38、§41、§66 的定义基本一致。
- E3 的真假负例过滤、sphere 对照、图片聚类和题型等权 Difficulty Gap，未发现足够确定的新问题。
- SPAR 官方任务中的 `bottom/top` 空子集、双对象过滤、背景 world 展开等行为，经核对与原脚本一致，未将这些已有 quirk 另列为缺陷。

上述结论表示本次审查未找到可证实问题，不等同于所有模型、数据与运行环境均已通过端到端验证。

## 6. 剩余工作与结果重算范围

后续顺序：先修复 R2/R7 的读取回退和身份校验，再确认旧 SigLIP2 缓存已失效，完成真实 checkpoint 小样本对齐，随后重跑受影响的全量结果、稳定性汇总和图表。R1/R3/R4/R5/R6 已修正的源码无需按原建议重复改写。

| 修复项 | 缓存处理 | 需要重跑的结果 |
|---|---|---|
| R1：FixRes 实现已改正 | 确认旧缓存已失效，重建受影响图像缓存；结合 R4 重建文本缓存 | 该模型已运行的 E1、SPAR 官方评估、E2、E3 |
| R2：写入已隔离，读取待修正 | 无须因此重抽 embedding | 修复回退后核对各输入身份，重新生成稳定性汇总；来源无法确认的旧结果需重跑 |
| R3：SPAR 样本生成已改正 | 图像缓存可复用 | 自建 E1 SPAR 的全量 items、过滤统计、指标和 factor gap；相关图表与稳定性汇总 |
| R4：小写化实现已改正 | 重建或版本化该模型文本缓存 | 该模型依赖文本特征的结果，包括 E1、官方 SPAR、E3 和 E2 文本 nuisance |
| R5：B-32 配置已接通 | 建立独立模型缓存 | 先验证真实权重加载，再按选定实验矩阵执行 B-32 评估 |
| R6：blank 候选已补回 | 可复用图像缓存；文本缓存可增量补齐正确候选 | 受影响的官方单动物检索任务与相关汇总 |
| R7：标题已参数化，数据身份待校验 | 无须重建特征缓存 | 修复读取和标签校验后重新生成对应图表 |

在重算前应保留旧结果及对应代码版本，确保修复前后能够追溯。正式验收除真实 checkpoint 加载、官方预处理对齐、小样本端到端执行和输出互不覆盖外，还应检查：缺少目标模型/种子结果时拒绝错误回退；部分种子缺失时不混用其他模型；旧缓存不能在预处理版本变更后静默命中；全量 SPAR 输出不再包含无效 blank 正例。
