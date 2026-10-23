# GraphDelta 代码审查报告

- 审查日期：2026-10-23（Asia/Shanghai）
- 审查目录：`code/`
- 审查版本：`main`，提交 `fac3cda5d6ff593947387a4916d720c6e33096cf`
- 对照文档：[完整预实验方案](GraphDelta_完整预实验方案_统一修订版.md)、[研究框架](GraphDelta_研究框架_统一修订版.md)、[代码 README](code/README.md)
- 状态：审查完成，尚未修复代码；本文件仅记录审查结论与修复建议。

## 1. 审查结论

确认 **3 项 P1 问题、4 项 P2 问题**。当前 SigLIP2 接入偏离所选 checkpoint 的官方实现；SPAR 自建 2AFC 已有输出包含无效正例；多模型结果缺少隔离，继续运行可能覆盖或混用已有结果。建议先修复这三项，再进行正式的跨模型对照。

当前工作区只有 `clip_vitl14` 和 `siglip2_so400m` 两个模型配置，**尚未发现 CLIP-B-32 的完整接入**。本报告按实际检出的代码作出判断。

| 编号 | 优先级 | 问题 | 主要影响 | 验证方式 |
|---|---|---|---|---|
| R1 | P1 | SigLIP2 FixRes 被按 NaFlex 加载和预处理 | 不再等价于指定的原始预训练 baseline | 官方配置与源码核对、尺寸算法复现 |
| R2 | P1 | 实验输出未按模型、种子隔离 | 结果覆盖，汇总可能混用不同运行的数据 | 实际 E1 流程的模拟 encoder 复现 |
| R3 | P1 | SPAR 只检查一侧 `blank` | 四类自建 2AFC 出现无效正例 | 实际函数最小复现、已有三种子输出统计 |
| R4 | P2 | SigLIP2 文本缺少小写化 | 偏离训练时文本预处理 | tokenizer 配置、官方实现与调用链核对 |
| R5 | P2 | CLIP-B-32 缺少配置与权重路径 | 模型选择无法使用 | 配置清单检查、配置加载复现 |
| R6 | P2 | SPAR 官方单动物检索候选池与原脚本不同 | 不能严格复现官方 top1/top5 | 官方脚本与本地候选构造对照 |
| R7 | P2 | 图标题写死 CLIP ViT-L/14、seed 0 | 模型和种子被错误标注 | 绘图源码检查 |

优先级含义：P1 应在继续正式实验前处理；P2 应在结果汇总、对外报告或声明协议复现前处理。优先级表示实验影响与处理顺序，不表示代码运行一定会报错。

## 2. 审查范围与验证边界

重点检查模型工厂、CLIP/SigLIP2 encoder、checkpoint 转换、图文特征缓存、PUG:SPAR 元数据与 caption、SPAR 官方评估，以及 E1/E2/E3 的输出和后处理。

已完成：

- 41 个 Python 文件的 AST 语法解析。
- 不加载真实模型的 E1 输出覆盖复现。
- `build_spar_items` 的 `blank` 输入最小复现。
- 已有 seed 0/1/2 SPAR item 文件的逐条统计。
- 对所选 SigLIP2 checkpoint、Transformers v5.13.0 实现及 PUG 官方评估脚本的核对。

未完成真实模型全量推理或重新计算全部指标：当前可用 Python 环境没有 `torch`、`transformers`，配置中的数据与权重路径指向 Linux 目录。因此，本报告没有量化各问题修复后准确率或表示统计会变化多少，也不将语法检查视为端到端运行通过。

## 3. 问题详情

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

## 4. 已检查且未发现明确主链问题的部分

- CLIP 与 SigLIP2 均调用 `eval()`，并逐参数设置 `requires_grad_(False)`；已审查路径保持冻结、纯评估边界。
- 特征缓存按模型名称分目录，没有发现已注册模型之间直接串用缓存或跨模型计算 cosine 的明确问题。
- E2 的 canonical transition、同 transition 跨 context 比较及双端点 context bootstrap，与方案 §38、§41、§66 的定义基本一致。
- E3 的真假负例过滤、sphere 对照、图片聚类和题型等权 Difficulty Gap，未发现足够确定的新问题。
- SPAR 官方任务中的 `bottom/top` 空子集、双对象过滤、背景 world 展开等行为，经核对与原脚本一致，未将这些已有 quirk 另列为缺陷。

上述结论表示本次审查未找到可证实问题，不等同于所有模型、数据与运行环境均已通过端到端验证。

## 5. 修复顺序与结果重算范围

建议先统一模型和种子的输出隔离，再修复 SigLIP2 模型与文本预处理、SPAR 样本有效性，随后完成 B-32 接入及官方协议和图表标注修正。

| 修复项 | 缓存处理 | 需要重跑的结果 |
|---|---|---|
| R1：恢复 SigLIP2 FixRes 路径 | 重建该模型受影响的图像缓存；结合 R4 重建文本缓存 | 该模型已运行的 E1、SPAR 官方评估、E2、E3 |
| R2：输出隔离 | 无须因此重抽 embedding | 检查旧表来源；来源无法确认或已被覆盖的运行需重跑 |
| R3：SPAR 有效样本修正 | 图像缓存可复用 | 自建 E1 SPAR 的 items、过滤统计、指标和 factor gap；相关图表与稳定性汇总 |
| R4：SigLIP2 文本预处理 | 重建该模型文本缓存 | 该模型依赖文本特征的结果，包括 E1、官方 SPAR、E3 和 E2 文本 nuisance |
| R5：B-32 接入 | 建立独立模型缓存 | 按选定实验矩阵执行 B-32 评估 |
| R6：官方候选池修正 | 可复用图像缓存；文本缓存可增量补齐正确候选 | 受影响的官方单动物检索任务与相关汇总 |
| R7：图标题修正 | 无须重建特征缓存 | 重新生成对应图表 |

在重算前应保留旧结果及对应代码版本，确保修复前后能够追溯。对正式实验的最终验收，应包含真实 checkpoint 加载、官方预处理对齐、小样本端到端执行，以及多模型、多种子输出互不覆盖的检查。
