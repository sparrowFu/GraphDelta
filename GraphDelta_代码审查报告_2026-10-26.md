# GraphDelta 当前代码审查报告（2026-10-26）

审查基线：`deb6d3a`，工作区 `D:/code/GraphDelta/code/`。对照《GraphDelta_完整预实验方案_统一修订版》和《GraphDelta_研究框架_统一修订版》，并按本次要求检查五个数据集、三个指定模型。此前审查报告仅作为线索，下面的结论以当前源码和本轮验证为准。本轮未修改实验源码。

**结论：当前代码的阶段一实验框架基本符合方案，但尚未完整满足“五数据集、三模型”的要求，也仍有会影响结果可信度的缺陷。** 三个模型均有配置和工厂分发；四个数据集有实验主链；PUG_ImageNet 尚未接入特征抽取与评估。优先解决旧缓存被错误认证、ImageNet 接入和 E2 nuisance 参数语义，再重算受影响结果。

## 1. 数据集和模型支持范围

“有实现”表示当前代码具有入口及相应流程，不等于真实数据和权重已经通过端到端验收。

| 要求的数据集 | 当前入口与用途 | 审查结论 |
|---|---|---|
| CLEVR_v1.0 | E1；`clevr_val`；Object/Color/Size/Material/Relation/Binding | 已实现；默认只用 val，符合方案 §16a |
| CLEVR_CoGenT_v1.0 | E3；`cogent_valA`、`cogent_valB` | 已实现；A/B 难度比较、sphere 对照、题型等权 Gap |
| PUG_Animals | E2；`pug_animals`；四因素单因素配对 | 已实现；读取全量 metadata，不进行训练/测试随机拆分 |
| PUG_SPAR | E1 自建 2AFC；另有官方协议入口 | 已实现；单动物/双动物和 blank 过滤已修复 |
| PUG_ImageNet | metadata 模块和 manifest builder | **未完成：没有特征抽取 CLI 和实验主链** |

方案 §14 允许暂缓运行 PUG_ImageNet，但本次用户明确要求支持该数据集，因此不能据此把缺少实现视为满足要求。五个数据集按方案分别承担不同任务，无须在每个数据集上都运行 E1/E2/E3。

| 要求的模型 | `--model` 参数值 | 当前加载路径 |
|---|---|---|
| siglip2-so400m-patch14-384 | `siglip2_so400m` | `SigLIP2Encoder`，原始 checkpoint；`SiglipModel` + `SiglipImageProcessor` |
| clip-vit-base-patch32 | `clip_vitb32` | `CLIPEncoder` + 对应 checkpoint 的 `AutoProcessor` |
| clip-vit-large-patch14 | `clip_vitl14` | `CLIPEncoder` + 对应 checkpoint 的 `AutoProcessor` |

三个工厂分支的模型名称、权重路径、图像/文本 batch size 已用替代构造器核对。三个 encoder 都执行 `eval()`、参数冻结和 `no_grad()`，没有发现训练 backbone、adapter 或 projection 的代码。

指定 SigLIP2 checkpoint 的官方配置确为 FixRes：`model_type=siglip`、384 输入、patch14；当前实现的固定 384 图像处理、显式文本小写化、固定 padding/truncation 到 64 与该路径相符。来源：[官方模型配置](https://huggingface.co/google/siglip2-so400m-patch14-384/raw/main/config.json)、[官方预处理配置](https://huggingface.co/google/siglip2-so400m-patch14-384/blob/main/preprocessor_config.json)、[官方文本处理说明](https://huggingface.co/docs/transformers/en/model_doc/siglip2#text-embeddings-and-retrieval)。这不代替实际 checkpoint 输出对齐测试。

## 2. 确认的问题

### A1 [P1] PUG_ImageNet 没有接入可执行实验流程

位置：[extract_embeddings.py](D:/code/GraphDelta/code/scripts/extract_embeddings.py:27)、[e2_change_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:101)、[pug_imagenet.py](D:/code/GraphDelta/code/src/data/pug_imagenet.py:10)。

`extract_embeddings.py` 的 `--dataset` 列表及后续分发没有 `pug_imagenet`。E2 的 metadata loader、特征提取、视觉 nuisance 和结果中的 dataset 字段都固定为 `pug_animals`；`run_e2.py` 也没有数据集选择参数。全仓的 ImageNet 引用只接通路径、metadata 和 manifest，未接通表示变化分析。

此外，ImageNet 的 `META_COLUMNS` 未保留 `character_rotation_yaw`、`character_rotation_roll`、`character_rotation_pitch`。这些字段确实存在于[官方数据集 schema](https://huggingface.co/datasets/facebook/PUG_ImageNet)。后续配对必须保留并固定物体姿态、其他相机角度和灯光等非目标因素，不能直接替换 Animals 的路径后沿用其五字段配对规则。当前没有 ImageNet 实验结果，因此这里不是声称已有 ImageNet 指标受到污染。

建议：补齐 ImageNet 抽取分支、适配后的 metadata、严格单因素配对和独立评估入口，并让 nuisance、dataset 标签及输出文件按实际数据集选择。先验证每对只有目标因素发生变化，再执行辅助实验。

### A2 [P1] 没有协议文件的旧缓存会被直接认证为当前新缓存

位置：[cache.py](D:/code/GraphDelta/code/src/features/cache.py:26)。

`_check_protocol()` 在发现已有 `protocol.json` 时会校验；但文件不存在时直接写入当前协议，未检查目录里是否已有图像或文本 NPZ。恰好是尚未版本化的历史缓存没有该文件，因而会被错误地标记成新版协议。

本轮用实际 `ImageFeatureCache`、`TextCache` 和 `extract_clevr_images` 复现：预置没有协议文件的旧特征 `[1,0]`，新 encoder 应返回 `[0,1]`，初始化当前 SigLIP2 协议后得到：

```text
image: [[1.0, 0.0]]
text: [[1.0, 0.0]]
new_encoder_calls: 0
new_marker: {'protocol': 'siglip2-fixres-lowercase-v1'}
```

影响：如果服务器保留修复前的 NaFlex 或未小写化特征，即使当前 encoder 已修正，实验仍可能静默使用旧特征。不能用新生成的 `protocol.json` 证明旧特征确实按新语义抽取。当前本机没有实际缓存，无法断言服务器正在触发此问题。

建议：仅允许空目录首次登记协议；无协议但已有 NPZ 的目录应拒绝复用或移入历史命名空间。更完整的指纹还应包含 checkpoint/processor 身份，避免同一模型别名换权重后继续命中缓存。

附带的恢复流程缺陷：[extract_embeddings.py](D:/code/GraphDelta/code/scripts/extract_embeddings.py:37) 先构造并校验缓存，再在 39–40 行处理 `--force`。协议不匹配时会先抛异常，`--force` 删除分支不可达。本轮复现确认旧 NPZ 保持原样。应提供明确的整套缓存迁移/重建入口；重建不能只覆盖图像缓存而保留旧文本缓存。

### A3 [P2] nuisance 的裁剪比例实现与其面积定义不一致

位置：[delta.py](D:/code/GraphDelta/code/src/features/delta.py:13)、[delta.py](D:/code/GraphDelta/code/src/features/delta.py:17)。

函数声明 `crop_scale` 是“面积比例区间”，实际同时计算 `nw=w*s`、`nh=h*s`，保留面积为 `s²`。配置 `[0.85,1.0]` 允许保留约 72.25% 面积，并非所声明的至少 85%。

实际调用：

```python
augment_view(Image.new('RGB', (1000, 1000)), 0, .85, .850000001, 90)
# 返回 850×850，保留面积比例 0.7225
```

影响：相对面积比例的定义，扰动更强，会改变 E2 visual nuisance 的范数和相似度，以及 Factor/Nuisance Norm Gap。不能由此直接量化已有结论会变化多少。方案 §39a 要求固定轻量扰动，但未规定 85% 面积；本项是代码内部参数语义错误，不是声称违反了方案中不存在的数值规定。

建议：若采用面积比例，两边使用 `sqrt(s)`；若本意是边长比例，应修正文档并明确确认这组 nuisance 参数。改变实际裁剪策略后，重算 E2 视觉 nuisance、相关 Gap 和图，不必因此重抽原始图像特征。

### A4 [P2] 缺失置信区间被判为“不显著且稳定”

位置：[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:28)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:37)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:96)。

`ci_vs_ref()` 和 `ci_sign()` 不检查有限性。`NaN` 与参考值的比较均为 False，分别落入 `contains` 和 `ns`；随后三个种子的分类相同就被写成 `stable=True`。上游 bootstrap 在有效重采样不足、E3 缺整类题型等情况下明确返回 NaN，因此该情况不是只存在于不合法输入。

用实际汇总脚本、三种子最小 CSV 运行，所有 CI 均设为 NaN，得到：

```text
e1_acc  stable=True  contains/contains/contains
e1_gap  stable=True  ns/ns/ns
e2      stable=True  ns/ns/ns
e3_gap  stable=True  ns/ns/ns
```

影响：“无法判断”会被写成“稳定地不显著”，掩盖统计计算失败或样本不足，影响方案 §82 的重复采样稳定性结论。

建议：先检查点估计和 CI 的 `isfinite`，将缺失/失败单独标为 `invalid` 或 `insufficient_data`；任何种子无效时不产生肯定的稳定性结论。

### A5 [P2] 显式结果目录仍能绕过模型/种子身份校验

位置：[io.py](D:/code/GraphDelta/code/src/utils/io.py:66)、[make_figures.py](D:/code/GraphDelta/code/scripts/make_figures.py:246)、[make_figures.py](D:/code/GraphDelta/code/scripts/make_figures.py:257)。

自动选择目录时，当前代码已经阻止 SigLIP2/B32 错误回退旧 CLIP 结果；这部分旧问题已修复。但 `--tables-dir` 可以指向没有 `run_meta.json` 的历史目录，`verify_run_meta()` 直接放行，图标题仍从请求的 `--model/--seed` 取得。

用实际 `make_figures.main()` 测试目录选择和标签生成，替代绘图函数以避免生成无意义图像：指向模拟的旧 `outputs/tables`（内容标记为 CLIP-L/14 seed0），同时请求 `--model siglip2_so400m --seed 1`，流程接受并传给绘图函数：

```text
source=tables
label=SigLIP2 so400m-p14-384
seed=1
```

这是输入路由与标签生成的复现，没有用模拟数据验证实际模型指标。影响范围是无运行元数据的输入目录；有正确 `run_meta.json` 的模型/种子不匹配会被拒绝。

建议：显式目录也需要可验证身份；已知历史目录应绑定其实际模型和种子，其余无元数据的目录不能直接被标为某个模型的正式结果。运行信息还应携带协议/配置版本，供下项检查使用。

### A6 [P2] 明知新旧协议不可比，仍输出跨协议的种子稳定性

位置：[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:60)、[summarize_seed_stability.py](D:/code/GraphDelta/code/scripts/summarize_seed_stability.py:74)。

对 CLIP-L/14，脚本逐种子独立选择新目录或旧目录；可以形成 seed0 使用修复后结果、seed1/2 使用修复前结果的组合。74–76 行已明确识别并警告“两套口径不可直接比较”，但仍继续计算 `stable` 并输出正式 CSV。该 CSV 中没有记录各行的协议来源。

影响：SPAR blank 过滤、官方候选集合等实现变更会与随机种子的变化混在一起，所得“稳定/翻转”不能解释为固定协议下的抽样稳定性。不代表当前本机存在这套混合结果；当前仓库没有相应 CSV。

建议：已知协议不一致时停止正式稳定性汇总；补齐同版本三种子后再计算。若保留历史对照，应输出独立的版本比较结果，并记录代码、协议、配置及缓存版本。

### A7 [P3] E2 两个相似度分支不遵守比较次数上限

位置：[delta_stats.py](D:/code/GraphDelta/code/src/metrics/delta_stats.py:82)、[delta_stats.py](D:/code/GraphDelta/code/src/metrics/delta_stats.py:101)。

`same_transition_dc` 和 `same_factor_dt_sc` 使用 `per=max(32,ceil(max_comparisons/分组数))`，拼接后没有施加全局上限。实际配对器构建的 384 行完整小型 metadata 组合，在 `max_comparisons=100` 时得到：

```text
same_transition_dc: 192
same_factor_dt_sc:  576
same_factor_dt_dc:  100
diff_factor_sc:       9
diff_factor_dc:     100
random:            100
```

影响主要是配置预算与实际计算量不符，增加相似度和 bootstrap 工作量；本项本身不足以证明点估计方向错误。建议分配严格的分层预算，并输出实际比较数。

## 3. 需要注意的统计质量风险

E2 的 `diff_factor_sc` 是先从所有其他因素联合抽样，再筛选共同固定字段相同者；只尝试 `3*max_comparisons` 次。完整 Animals 中的同 context 事件很稀少，例如 texture 与 size 共同固定 animal/environment/camera，匹配概率约为 `1/(70*64*4)`。

按官方维度构造、每 context 最多四对且每因素最多 30,000 对的合法合成 delta 表，设置上限 20,000 时，四因素的该层级仅生成 74/21/11/15 个比较；这些是合成验证结果，不是实际数据集重跑计数。结果 CSV 只输出均值，没有各组实际比较量；日志中的来源计数也只针对 `diff_factor_dc`（[e2_change_diagnosis.py](D:/code/GraphDelta/code/src/experiments/e2_change_diagnosis.py:192)）。

这里不应把条件筛选后因素比例不同直接称为错误偏差：当前代码声明的是对其他 delta 联合均匀抽样，其条件分布本来可能不均匀。真正的问题是该辅助层级的样本量可能不足，却缺少数量披露和不足标记。建议直接索引可匹配的 shared context，明确抽样分布及权重，并输出每组数量/来源。核心 Transition Gap 使用的是 different-context 对照，本风险不能被夸大为核心 Gap 已经计算错误。

## 4. 与方案一致、已完成验证的部分

- 阶段一保持冻结 baseline 和固定公开图片，不进行 GraphDelta 训练，不新增 Controlled Edit 图片；不能因没有第二/三阶段训练模块就判断当前阶段一不完整。
- CLEVR 只读 val，CoGenT 只读 valA/valB；E3 不把 A/B 直接解释为 seen/unseen。
- CLEVR relation 从官方 relationships 取方向；负例按整幅 scene 判假。binding 唯一性按 caption 实际涉及的 color/shape 判断。
- SPAR 当前正确处理任一侧 blank、双侧 blank、同种双动物和相同纹理的属性交换过滤。
- 2AFC 采用 `s+ > s-`、平局判错；跨模板聚合并报告模板间方差。
- E2 使用 canonical transition、`target-source`、raw/normalized 两空间，以及同 transition 跨 context 的分析。
- E2 核心相似度 Gap 使用双端点、同步 context bootstrap；E3 Gap 按题型等权、按图片聚类，并有 sphere 对照。
- 三模型结果写入按模型和种子隔离；普通自动读取路径对历史结果的模型归属限制已经加入。

本轮检查与构造测试：

| 检查 | 结果与边界 |
|---|---|
| 全部 Python 文件 AST 解析 | 41 个文件通过；仅证明无语法错误 |
| 三模型工厂 | 替代构造器核对名称/路径/batch size 通过；未加载真实权重 |
| SPAR item builder | 7 行边界样例；scene=5、position=2、attribute=1、relation=2；无 blank 正例、无正负相同文本 |
| Animals metadata + pair matcher | 192 行完整小型笛卡尔积；四因素逐对检查只有目标字段变化，通过 |
| CLEVR item builder | 100 个构造场景，独立验证 10,063 条样本；正例真、负例假全部通过 |
| 缓存迁移 | 复现 A2：旧 NPZ 被新协议认证，encoder 未被调用 |
| nuisance crop | 复现 A3：85% 被当作边长比例，面积约 72.25% |
| 稳定性汇总 | 实际 main 配合最小 CSV，复现 A4 |
| 图表来源 | 实际 main 的目录/标签路径配合替代绘图函数，复现 A5 |
| E2 比较预算 | 实际 matcher + similarity 函数，复现 A7 |

未再列为当前缺陷的旧问题：SigLIP2 FixRes 加载、小写化、B/32 配置、SPAR blank 正例、结果写入目录隔离、普通自动目录回退的跨模型误标。这些已有修复；A2/A5/A6 是仍未覆盖的触发路径。

## 5. 验证边界和后续处理范围

当前可用 Python 为 `D:/conda/python.exe`（3.13），有 NumPy/Pandas/PyArrow/Pillow/YAML，但没有 torch/transformers。`paths.yaml` 指向 Linux 服务器的 `/mnt/disk1/...` 数据及权重，本机不存在这些实际输入。未下载数据或模型，未运行真实 checkpoint 的小样本/全量实验，也未验证真实 tokenizer token ID 或与官方完整 encoder 的数值对齐。

当前仓库 `code/outputs/` 只保留总结和分析 Markdown，未保留其引用的 CSV、图和特征缓存；因此不能核实历史数字对应的源码/缓存版本，也不能确认服务器已按 README 所述清理和重抽。上述复现不会证明历史结果一定已受影响，应以运行来源和缓存来源核实为准。

建议处理顺序和重算范围：

1. 修复无协议缓存迁移；核实服务器缓存来源。若来源未知或确为旧 SigLIP2 特征，重建该模型的图像/文本缓存，并重跑受影响的 E1、SPAR 官方协议、E2、E3。
2. 补齐 PUG_ImageNet 的严格单因素评估路径；确认三个模型都能在五数据集各自任务上完成小样本端到端运行。
3. 明确 nuisance crop 参数语义。若改为面积比例，重算所有已运行模型的 E2 视觉 nuisance、Norm Gap 和相关汇总/图；原图 embedding 可复用。
4. 修复 NaN 分类、目录身份校验和跨协议种子混用，再生成稳定性 CSV 与图表；无需因此重抽 embedding。
5. 约束 E2 相似度预算并公开实际样本量；若调整抽样策略，固定协议后重算该部分统计和三种子比较。

正式验收还应保留每次运行的代码版本、checkpoint/processor 身份、数据版本、协议/配置和种子。上述缺口关闭并通过真实端到端运行后，才能确认本次要求完整满足。
