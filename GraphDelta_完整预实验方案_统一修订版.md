# GraphDelta 预实验完整方案
## ——冻结 baseline 模型与固定公开数据下的问题验证，以及后续机制验证指南

> 本文档用于指导 GraphDelta 预实验的数据准备、代码生成、实验执行和阶段性判断。  
> 当前首选 baseline：**CLIP ViT-L/14**；第一阶段允许增加 **SigLIP2、其他 OpenCLIP / VLM 预训练模型**作为冻结 baseline 对照，但它们都不是 GraphDelta 的方法模块。  
> 当前固定公开数据：**PUG: Animals、PUG: ImageNet、PUG: SPAR、CLEVR v1.0、CLEVR-CoGenT v1.0**。  
> 第一阶段固定数据与评估协议；所有 baseline 模型保持原始预训练参数，只做零样本评估与表示分析，不重新生成训练数据、不训练 GraphDelta。  
> 只有在第一阶段验证研究问题成立后，才进入第二阶段的 Controlled Edit 机制验证。  
> 修订记录（2026-10-20）：统一两个 GraphDelta 文档的阶段定义；明确第一阶段可增加冻结 baseline 模型，但不允许训练或加入 GraphDelta 模块；修正 CoGenT 解释边界，并统一 E1/E2/E3 的验证目标。

---

# 1. 预实验的总体原则

GraphDelta 当前还处于研究问题验证阶段。

因此第一阶段的目标不是：

```text
训练一个更好的模型
```

也不是：

```text
提前证明 GraphDelta 有效
```

而是：

> **在固定的预训练模型和固定的公开数据集上，确认现有视觉—语言表示是否真的存在结构理解缺陷，以及这些缺陷是否与“结构变化响应不足”有关。**

因此第一阶段必须遵循以下原则：

```text
固定公开数据与评估协议
+
允许增加冻结的预训练 VLM baseline
+
所有 baseline 保持原始参数
+
不训练 backbone / adapter / projection
+
不加入 GraphDelta 模块
+
不重新渲染数据
+
不额外构造新的训练数据
+
只做诊断和验证
```

这里“允许增加模型”只表示增加**外部 baseline**。例如 CLIP、SigLIP2 可以分别独立评估，用于判断现象是否具有跨模型一致性；不能在第一阶段训练这些模型，也不能把任何新增模块视为 GraphDelta 的一部分。

---

# 2. 整个预实验分为三个阶段

整个预实验和后续方法验证建议明确拆成三个阶段。

```text
阶段一：问题验证
Fixed Model + Fixed Public Datasets
        ↓
验证现有模型是否存在结构缺陷
        ↓
验证表示对不同结构变化是否具有不同响应

阶段二：机制验证
Controlled Edit Pair
        ↓
严格研究 Structural Delta
        ↓
验证跨模态变化规律和未见组合迁移

阶段三：方法验证
GraphDelta Training
        ↓
加入结构变化监督
        ↓
验证是否真正改善 unseen composition
```

当前首先完成：

> **阶段一。**

阶段二和阶段三只作为后续路线保留。

---

# 3. 阶段一到底要验证什么？

阶段一只回答三个问题。

## 问题一

> **冻结的预训练视觉—语言 baseline 对不同类型的细粒度语义结构是否具有明显不同的区分能力？**

重点关注：

```text
Object
Attribute
Position
Relation
Composition
```

如果模型对：

```text
Object / Color
```

比较敏感，

但对：

```text
Relation / Attribute Binding / Composition
```

明显更弱，

说明现有表示存在结构性不均衡。

---

## 问题二

> **在已有公开数据中，当某一个因素发生变化时，不同冻结 baseline 的表示变化是否具有可分析、可重复的规律？**

例如：

```text
same animal
same background
same camera

texture A
→
texture B
```

或者：

```text
same animal
same texture
same camera

small
→
large
```

研究：

```text
Δ = h_after - h_before
```

是否对不同变化类型表现出不同模式。

这一阶段不要求证明完整的 Structural Delta Geometry。

只需要回答：

> 表示变化是否包含值得进一步研究的结构信号。

---

## 问题三

> **在 CLEVR-CoGenT 的两种颜色—形状搭配条件下，冻结 baseline 的图文匹配难度是否存在系统性不对称？**

先明确一个边界：

> CoGenT 的 A/B 划分是为“在 Condition A 上训练、在 Condition B 上测试”的模型设计的。
> 第一阶段使用冻结的通用预训练 baseline，它们都没有在 CoGenT Condition A 上进行本研究中的训练，因此“见过 / 没见过”不能由 A/B 划分直接定义；第一阶段只能把 A/B 当作两组受控属性—形状组合条件进行难度比较。

因此第一阶段只回答：

```text
不同属性—形状搭配的匹配难度是否不同，
且难度差异的方向是否与 A/B 划分一致
```

如果难度不对称明显存在，说明 GraphDelta 面向组合问题的研究目标有现实依据。

直接验证“primitive 见过、组合没见过导致下降”属于第三阶段（train-on-A、test-on-B，在有训练的设置下进行）。第一阶段 E3 的结果作为研究动机的间接证据使用。

---

# 4. 第一阶段 baseline 模型设置

第一阶段的模型角色必须区分清楚：

```text
Baseline Model
≠
GraphDelta Method
```

第一阶段首选 baseline：

```text
CLIP ViT-L/14
```

为了判断观察到的现象是否只属于某一个模型，可以增加少量冻结对照模型，例如：

```text
SigLIP2
其他容易获得的 OpenCLIP / VLM 预训练模型
```

推荐执行顺序：

```text
第一轮：
CLIP ViT-L/14

如果第一轮现象明确：
再加入 SigLIP2 复现关键实验

如果两个模型趋势一致：
再考虑是否需要第三个 baseline
```

所有 baseline 必须满足：

```text
使用原始预训练权重
encoder.eval()
requires_grad = False
不做 fine-tuning
不做 LoRA
不训练 projection
不训练 adapter
不加入 GraphDelta 模块
```

不同模型使用各自官方：

```text
input resolution
image preprocess
tokenizer
text preprocess
```

因此第一阶段的结论应表述为：

> **现有预训练视觉—语言 baseline 的原始表示中是否存在结构性差异。**

如果只运行 CLIP，结论限定为 CLIP；如果 CLIP 与 SigLIP2 得到一致趋势，则可以更有把握地说明该现象并非单一 backbone 特例。

# 5. 第一阶段固定数据集

第一阶段固定使用现有数据。

```text
PUG: SPAR
PUG: Animals
PUG: ImageNet
CLEVR v1.0
CLEVR-CoGenT v1.0
```

但不同数据集承担不同任务。

---

# 6. 数据集总体分工

| 数据集 | 第一阶段作用 | 核心程度 |
|---|---|---:|
| PUG: SPAR | Attribute / Position / Relation 结构理解诊断 | 核心 |
| PUG: Animals | 单因素表示变化与 nuisance control | 核心 |
| CLEVR v1.0 | Object / Attribute / Relation 可控状态诊断 | 核心 |
| CLEVR-CoGenT | 属性—形状组合难度不对称诊断；严格 unseen composition 留到有训练的后续阶段 | 核心 |
| PUG: ImageNet | 大规模视觉单因素 robustness 辅助验证 | 辅助 |

第一阶段不新增：

```text
EditCLEVR
ARO
SugarCrepe
What'sUp
其他数据集
```

这些全部放到后续阶段。

注意：冻结 baseline 模型可以按 §4 增加，这不属于“新增数据”，两条规则不冲突。

---

# 7. 第一阶段允许“配对”，但不允许“改数据”

这里需要明确一个边界。

第一阶段可以：

> 根据数据集中已有 metadata，把现有样本整理成实验 pair。

例如 PUG: Animals 中：

```text
same animal
same texture
same size
different background
```

可以形成一组：

```text
Background Change Pair
```

这只是对已有样本进行筛选和匹配。

不属于修改数据集。

---

第一阶段不允许：

```text
重新渲染 CLEVR
重新生成图像
人为修改 scene
创建新的 Binding Swap 图像
生成新的训练样本
```

这些属于第二阶段 Controlled Edit。

---

# 8. PUG: SPAR 的作用

PUG: SPAR 是第一阶段最重要的数据之一。

它用于验证：

```text
Scene
Position
Attribute
Relation
```

这些不同类型的组合语义是否具有明显不同的难度。

它特别适合回答：

> 首选 baseline（CLIP）以及其他冻结 baseline 是否对 Relation 和 Attribute Structure 表现出一致的能力差异？

---

# 9. PUG: SPAR 的实验任务

第一阶段只做：

```text
Zero-shot Evaluation
```

不训练。

如果官方已有 evaluation protocol：

> 优先按照官方设定复现。

PUG 原论文（arXiv:2308.03977）已经在 SPAR 上报告过 CLIP 的 Scene / Position / Attribute / Relation 结果。因此 E1 的 SPAR 部分在很大程度上是复现已知结论，它的作用是：

```text
1. 验证自建 pipeline 是否正确（与已发表数字对齐）
2. 与 CLEVR 的结果交叉验证趋势
```

真正的新证据在 CLEVR Binding-style 和 CoGenT 部分。

另外统一计算：

```text
Accuracy
2AFC Accuracy
Matching Margin
```

如果数据格式允许。

---

# 10. PUG: SPAR 的重点结果

重点比较：

```text
Attribute
Position
Relation
```

是否存在明显性能差异。

特别关注：

```text
Relation
```

如果 Relation 明显弱于 Attribute/Object-level 判断，

说明：

> 模型对“概念之间如何关联”的编码可能弱于对“场景中有哪些概念”的编码。

这直接支持 GraphDelta 的研究动机。

---

# 10a. PUG: SPAR 的数据字段与 Caption 生成

PUG: SPAR 共 43,560 张图，纯评测集，无训练集。

HuggingFace 版本（facebook/PUG_SPAR）的字段：

```text
image
world_name          # 环境，10 类
character_name      # 主体动物，11 类
character2_name     # 客体动物，33 类，含 "blank"
character1_pos      # left / right
character2_pos      # left / right
character_texture   # 3 类
character2_texture  # 3 类
```

两点必须处理：

```text
1. 数据表没有现成 caption 列，
   caption 需要像 CLEVR 一样从 metadata 确定性生成。

2. character2_name = "blank" 表示单动物场景，
   Position / Relation 任务必须先过滤出双对象样本。
```

Caption 模板（示例，实现时固定并写入 configs）：

```text
Scene:
a photo of a {animal1} and a {animal2} in the {environment}

Position / Relation:
the {animal1} is to the {pos} of the {animal2}

Attribute:
a {texture1} {animal1} and a {texture2} {animal2}
```

---

# 11. PUG: Animals 的作用

PUG: Animals 第一阶段主要承担：

> **单因素变化与表示响应诊断。**

它本身包含多个可控变化因素：

```text
Animal（70 种）
Environment（64 种）
Texture（4 种）
Size（3 种）
Camera Orientation（4 种）
```

总计 215,040 张 = 70 × 64 × 3 × 4 × 4，每个 metadata 组合恰好一张图。

两个直接后果：

```text
1. 可以直接利用现有 metadata 配对。

2. 数据集内不存在“同 metadata、不同渲染”的样本，
   nuisance 基线必须用图像增强实现（见 39a）。
```

---

# 12. PUG: Animals 的 Pair 原则

只使用：

```text
已有图片
```

构造 matched pair。

例如：

## Background Change

固定：

```text
animal
texture
size
camera
```

只改变：

```text
environment
```

---

## Texture Change

固定：

```text
animal
environment
size
camera
```

只改变：

```text
texture
```

---

## Size Change

固定：

```text
animal
environment
texture
camera
```

只改变：

```text
size
```

---

## Camera Change

固定：

```text
animal
environment
texture
size
```

只改变：

```text
camera orientation
```

---

# 13. PUG: Animals 的核心问题

研究：

```text
不同单因素变化
```

在冻结 baseline 的 visual embedding 中引起多大的变化。

例如：

```text
Δ_background
Δ_texture
Δ_size
Δ_camera
```

比较这些变化之间的差异。

目的不是立即证明：

```text
某一个 Delta 是语义方向
```

而是先判断：

> 冻结 baseline 的原始表示对不同变化因素是否具有稳定、可重复的响应差异。

---

# 14. PUG: ImageNet 的作用

PUG: ImageNet 只作为辅助实验。

它可以验证：

> PUG: Animals 中观察到的单因素变化规律，是否只在动物场景中成立。

重点可以选择：

```text
background
size
texture
camera
```

不需要全部运行。

如果第一阶段时间有限：

> PUG: ImageNet 可以暂时不跑。

---

# 15. CLEVR v1.0 的作用

CLEVR 第一阶段主要承担：

> **可控语义状态诊断。**

它提供：

```text
Object
Color
Size
Material
Spatial Relation
Scene Annotation
```

因此可以用来验证：

```text
Object / Attribute / Relation
```

等不同语义因素对图文匹配的影响。

---

# 16. 第一阶段 CLEVR 不重新渲染

这是一个重要原则。

第一阶段：

```text
不使用 CLEVR generator 重新生成 before / after scene
```

因为阶段一只是在固定公开数据上做诊断。

可以做的事情是：

```text
读取已有 scene metadata
筛选合适的样本
生成 deterministic caption
构造 2AFC text candidates
```

不生成新的图片。

---

# 16a. Split 的硬性规定

```text
CLEVR v1.0：
只使用 val split（15,000 张）。
test split 没有公开场景标注，不能使用。
train split 不需要。

CLEVR-CoGenT：
只使用 val_A（15,000 张）和 val_B（15,000 张），两者都有场景标注。

PUG: Animals：
E2 配对使用全量 215,040 张。
E2 是纯评估，不存在训练/测试泄漏；
随机 train/test 划分会把配对需要的组合拆开。

PUG: SPAR：
全部 43,560 张（该数据只有测试集）。
```

---

# 17. CLEVR Caption 生成

CLEVR 原始任务主要是 VQA。

但 GraphDelta 需要：

```text
Image ↔ Text
```

状态匹配。

因此文本只根据已有 Scene Annotation 确定性生成。

第一阶段允许这样做，因为：

> 只是在把已有 scene annotation 转换成文本描述，不修改原始图像和 scene。

---

# 18. 单对象 Caption

模板：

```text
A {size} {color} {material} {shape}.
```

例如：

```text
A small red metal cube.
```

---

# 19. Relation Caption

模板：

```text
A {attr1} {obj1} is to the left of a {attr2} {obj2}.
```

统一关系：

```text
left_of
right_of
front_of
behind
```

---

# 20. 第一阶段 CLEVR 主要做什么？

主要做：

```text
Object Hard Negative
Color Hard Negative
Size Hard Negative
Material Hard Negative
Relation Hard Negative
```

例如：

```text
Positive:
A red cube is to the left of a blue sphere.

Negative:
A red cube is to the right of a blue sphere.
```

图像不变。

只修改文本候选。

这仍然属于：

> Fixed Dataset Evaluation。

---

# 20a. CLEVR 负例有效性规则（全部类型通用）

每一条 2AFC 样本必须同时满足：

```text
1. 正例为真：
   图中确实存在满足描述的对象或对象对。

2. 负例为假：
   图中不存在任何满足负例描述的对象或对象对。
   relation 负例尤其注意：
   图中存在多个 red cube 和多个 blue sphere 时，
   “left of” 和 “right of” 可能同时为真，
   此时负例无效，整条样本丢弃。

3. 正例唯一（建议）：
   满足描述的对象或对象对只有一组。
   不唯一时任务有歧义，丢弃或单独标记。
```

该规则对 Object / Color / Size / Material / Relation / Binding-style 全部六类生效，不只对 Binding-style。

检查基于完整 scene annotation，在生成 caption 时完成。丢弃的样本数量要记录并写入输出。

---

# 21. 为什么第一阶段不直接做 Binding Swap 图像？

Binding Swap：

```text
red cube + blue sphere
→
blue cube + red sphere
```

如果要严格验证：

```text
before image
after image
```

必须重新生成受控图像。

但这已经属于第二阶段。

因此第一阶段：

> **不强行把 Binding Delta 做进去。**

Binding 问题可以先通过：

```text
文本 hard negative
```

做静态 matching diagnosis。

---

# 22. CLEVR Binding-style Diagnosis

对于已有 CLEVR 图像：

如果图中存在：

```text
red cube
blue sphere
```

生成：

```text
Positive:
A red cube and a blue sphere.

Negative:
A blue cube and a red sphere.
```

如果 negative 中对应的组合不符合图像：

则构成：

```text
Binding-style Hard Negative
```

这里没有修改图像。

只是测试：

> 冻结 baseline 是否能够正确利用已有图像中的属性—对象绑定信息。

这非常适合作为第一阶段验证。

---

# 23. CLEVR-CoGenT 的作用

CLEVR-CoGenT 第一阶段只负责：

> **属性—形状组合难度不对称诊断（Composition Difficulty Diagnosis）。**

第一阶段不重新设计 split，直接使用官方：

```text
Condition A
Condition B
```

但需要明确：

```text
第一阶段：
冻结通用预训练 baseline
→ 比较 A / B 两组组合条件是否存在系统性难度差异

后续有训练的阶段：
train on A
→ test on B
→ 才能严格讨论 CoGenT 意义上的 unseen composition transfer
```

因此第一阶段的 CoGenT 结果是 GraphDelta 研究动机的辅助证据，不单独承担“证明 unseen composition 泛化失败”的任务。

# 24. CoGenT 的核心逻辑

Condition A 和 Condition B 的精确定义：

```text
Condition A：
cube     ∈ {gray, blue, brown, yellow}
cylinder ∈ {red, green, purple, cyan}

Condition B：
cube     ∈ {red, green, purple, cyan}
cylinder ∈ {gray, blue, brown, yellow}

sphere：
A、B 两个条件下都可取全部 8 种颜色，
不受划分限制。
```

A→B 的设计通过对调 cube 和 cylinder 的颜色集合形成标准 compositional split。

注意：只有在“Condition A 上训练、Condition B 上测试”的设置中，才能严格解释为 Seen / Unseen composition。第一阶段使用冻结通用 baseline，因此这里只把 A / B 作为两组受控组合条件进行难度比较，解释边界见 §3 问题三和 §51。

可用数据：

```text
val_A：15,000 张，有场景标注
val_B：15,000 张，有场景标注
```

---

# 25. CoGenT 的第一阶段任务

只做：

```text
Zero-shot Image-Text Matching
2AFC
Matching Margin
```

例如构造：

```text
正确 Color-Shape 描述
vs
错误 Color-Shape 描述
```

比较：

```text
Condition A
Condition B
```

是否出现明显性能差距。

---

# 26. 第一阶段实验整体结构

阶段一建议只做三个实验。

```text
E1 State-level Structural Diagnosis

E2 Representation-change Diagnosis

E3 Composition Difficulty Diagnosis
（CoGenT 难度不对称 + sphere 对照）
```

不做：

```text
Projection Training
GraphDelta Training
Controlled Re-rendering
Complex Delta Classification
```

---

# 27. E1：State-level Structural Diagnosis

## 目的

回答：

> **冻结 baseline（首选 CLIP ViT-L/14）对不同结构因素的图文区分能力是否不均衡？**

---

# 28. E1 使用数据

```text
PUG: SPAR
CLEVR
```

CLEVR-CoGenT 只归 E3 使用，不进 E1，避免重复。

---

# 29. E1-A：2AFC Matching

给定：

```text
Image I
```

两个文本：

```text
T+
T-
```

计算：

```text
s+ = cosine(I, T+)
s- = cosine(I, T-)
```

预测正确条件：

```text
s+ > s-
```

---

# 29a. E1 只做 image → text 方向

第一阶段 2AFC 只做：

```text
给定图像，在 T+ 和 T- 之间选择
```

不做 T2I（给定文本，在两张图之间选择）。原因：

> T2I 的硬负例需要“只差一个结构因素”的两张图，必须重新渲染，属于第二阶段 Controlled Edit。

研究框架文档中的 I2T / T2I / Pair Accuracy 完整指标集，在第二阶段之后才补齐。这不是漏做。

---

# 30. E1 Hard Negative 类型

CLEVR：

```text
Object
Color
Size
Material
Relation
Binding-style
```

PUG: SPAR：

```text
Attribute
Position
Relation
```

---

# 31. E1 指标

每类分别报告：

```text
2AFC Accuracy
Mean Matching Margin
Median Matching Margin
Margin Std
95% Bootstrap CI
样本数 n
```

其中：

```text
Matching Margin = s+ - s-
```

统一规定：

```text
1. Prompt 模板：
   CLIP 零样本结果对模板高度敏感。
   固定 5–7 个外层 prompt 模板
   （如 “a photo of {}” / “a picture showing {}” 等），
   报告所有模板的平均结果，并报告模板间方差。
   CLEVR / SPAR 的内容模板之外必须再包一层外层 prompt。

2. 平局规则：
   s+ == s- 一律记为错误，全文档统一。

3. 样本量：
   每个 dataset × factor 组合至少 1000 条有效样本；
   超出时随机下采样，固定种子，写入 manifest。
```

---

# 32. E1 最重要的比较

重点看：

```text
Object / Attribute
vs
Relation / Binding-style
```

如果：

```text
Relation / Binding-style
```

明显更差，

这是 GraphDelta 最直接的问题证据。

---

# 33. E1 预期输出

表格：

| Dataset | Factor | 2AFC Acc | Mean Margin | Median Margin | n |
|---|---|---:|---:|---:|---:|
| CLEVR | Object | | | | |
| CLEVR | Color | | | | |
| CLEVR | Size | | | | |
| CLEVR | Material | | | | |
| CLEVR | Relation | | | | |
| CLEVR | Binding-style | | | | |
| PUG SPAR | Scene | | | | |
| PUG SPAR | Position | | | | |
| PUG SPAR | Attribute | | | | |
| PUG SPAR | Relation | | | | |

---

# 34. E2：Single-Factor Representation Response Diagnosis

## 目的

回答：

> **在固定公开数据中，当某个单因素发生变化时，冻结 baseline 的表示响应是否具有稳定、可重复的规律？**

第一阶段只研究：

```text
已有 paired variation
```

不生成新的编辑图片。

---

# 35. E2 主要使用数据

核心：

```text
PUG: Animals
```

辅助：

```text
PUG: ImageNet
```

CLEVR 第一阶段不强制做 Delta。

因为普通 CLEVR 没有严格配好的 before / after edit pair。

---

# 36. E2 Pair 类型

PUG: Animals：

```text
Background Change
Texture Change
Size Change
Camera Change
```

只从已有图像中匹配。

---

# 37. E2 Feature

同时保存：

```text
h
```

和：

```text
z = h / ||h||
```

其中：

```text
h
```

表示 normalization 前的 projected feature。

```text
z
```

表示 normalized embedding。

---

# 38. E2 Delta

计算：

```text
Δ_h = h_after - h_before
```

辅助：

```text
Δ_z = z_after - z_before
```

方向约定必须固定，但要区分**有序因素**和**离散类别因素**。

### 有序因素

例如 size：

```text
small < medium < large
```

可以统一定义：

```text
Δ = 后一个等级 − 前一个等级
```

### 离散类别因素

例如：

```text
texture
environment
camera orientation
```

不存在天然的语义大小顺序。

因此主分析单位定义为：

```text
transition_key
=
(factor, source_value, target_value)
```

例如：

```text
(texture, texture_1, texture_2)
(camera, front, left)
```

相同 transition 在不同 context 上的 Delta 才作为最直接的“可重复变化”证据。

为了避免正反方向相消，同一 transition 只保留一个 canonical orientation；canonical orientation 可以由官方 metadata index 决定，但这个 index **只用于统一符号，不解释为语义顺序**。

因此 E2 后续区分：

```text
Same Transition:
相同 factor + 相同 source→target，不同 context

Same Factor:
factor 相同，但 source→target 不同

Different Factor:
factor 不同
```

其中最核心的观察量是：

```text
Same Transition + Different Context
```

是否明显高于：

```text
Different Factor + Different Context
```

第一阶段不要把 Delta 解释成真正的 Structural Delta，只把它视为：

> **原始预训练表示对已有单因素变化的 response vector。**

# 39. E2-A：Delta Magnitude

比较：

```text
||Δ_background||
||Δ_texture||
||Δ_size||
||Δ_camera||
```

以及 cosine distance。

回答：

> 冻结 baseline 对哪些因素最敏感？

---

# 39a. E2-C：Nuisance 基线（必做）

研究框架要求比较“因素变化 vs 无语义扰动”。第一阶段允许使用**evaluation-only、非持久化的输入扰动**建立参考基线；这不等于修改公开数据集，也不会生成新的训练样本：

```text
视觉 nuisance 基线：
同一张图的两个轻量增强视图（例如轻微 crop / resize / JPEG perturbation；参数固定在 config 中）
Δ_nuisance = h_aug2 - h_aug1

文本 nuisance 基线：
同一内容、不同外层模板的文本
Δ_template = h_t2 - h_t1
```

解释规则：

```text
||Δ_factor|| 明显大于 ||Δ_nuisance||
→ 该因素的响应超出 nuisance 水平

||Δ_factor|| 与 ||Δ_nuisance|| 接近
→ 只能说“画面变化引起了响应”，
  不能解释为“模型对该语义因素敏感”
```

为什么必须用增强而不是 render seed：

> PUG: Animals 的 215,040 张图恰好覆盖全部 metadata 组合，每个组合只有一张图，不存在“同 metadata、不同渲染”的样本。

---

# 40. E2-B：Same-transition / Same-factor / Different-factor

对 Delta 做分层相似度分析。

最重要的是：

### Same Transition

例如：

```text
dog:      texture_1 → texture_2
elephant: texture_1 → texture_2
```

变化的 source / target 完全一致，只是 context 不同。

### Same Factor, Different Transition

例如：

```text
texture_1 → texture_2
vs
texture_3 → texture_4
```

二者都属于 texture，但不是同一个具体变化。

### Different Factor

例如：

```text
texture_1 → texture_2
vs
camera_front → camera_left
```

计算：

```text
cos(Δ_a, Δ_b)
```

这种分层可以避免把“同一个因素”误当成“同一个具体变化”。

# 41. E2 第一阶段的核心比较

E2 按两个维度组织：

```text
变化维度：
Same Transition
Same Factor / Different Transition
Different Factor

context 维度：
Same Context
Different Context
```

真正支持后续 Structural Delta 研究的核心观察是：

```text
Same Transition + Different Context
>
Different Factor + Different Context
```

其次观察：

```text
Same Factor / Different Transition
```

是否仍保留较弱的共同结构。

Random Pair 作为参照。

同时必须处理配对独立性：

> 同一个 context（除被测因素外的其余 metadata 组合）会产生大量共享 context 的 pair，这些 pair 彼此相关。bootstrap 必须按 context 聚类抽样（cluster bootstrap），或限制每个 context 的配对数上限，否则置信区间会偏窄。注意 context 的定义随因素变化：测 texture 时 context = animal + environment + size + camera；测 environment 时 context = animal + texture + size + camera。

第一阶段不要求：

```text
同一 factor 的所有变化都形成固定全局方向
```

只需要验证：

> **至少对于相同具体 transition，表示变化是否能跨 context 保留可重复性。**

# 42. E2 不训练 Linear Probe

为了保持第一阶段完全纯验证：

> **第一阶段不训练 Linear Classifier。**

只使用：

```text
distance
similarity
distribution statistics
```

如果 E2 结果支持，

第二阶段再加入：

```text
Linear Edit Probe
```

---

# 43. E2 指标

报告：

```text
Delta Norm
Delta Cosine Similarity
Same-transition Similarity（重点报告 Different Context）
Same-factor / Different-transition Similarity
Different-factor Similarity（分 Same / Different Context）
Random Similarity
Nuisance Norm / Nuisance Similarity（视觉增强、文本模板两种）
```

分别在：

```text
raw feature
normalized feature
```

上计算。

指标冗余说明：

> z 是单位向量时，||Δ_z||² = 2 − 2·cos(z_before, z_after)。归一化特征上 Delta Norm 和 cosine 距离是同一信息的两种写法，只报一个；raw feature 上两者不同，都报。

---

# 44. E2 推荐图

```text
Figure 1:
Delta Norm by Factor

Figure 2:
Same-transition / Same-factor / Different-factor / Random Similarity

Figure 3:
Factor-wise Similarity Matrix
```

---

# 45. E2 的结果怎么解释？

如果：

```text
Same Transition (Different Context)
>
Different Factor (Different Context)
```

并且 Same Factor / Different Transition 至少不与 Random 完全重合，

并且：

```text
||Δ_factor||
>
||Δ_nuisance||
```

说明：

> 冻结 baseline 的表示变化中存在超出 nuisance 水平的、可重复的 transition / factor-specific regularity。

这支持后续继续研究 Structural Delta。

---

如果：

```text
Same Transition (Different Context)
≈
Different Factor (Different Context)
```

或者：

```text
||Δ_factor|| ≈ ||Δ_nuisance||
```

说明：

> 原始 global embedding 中的变化结构较弱，或响应主要来自画面变化而不是语义因素。

第二阶段就需要更严格的 Controlled Edit。

---

# 46. E3：Composition Difficulty Diagnosis

## 目的

回答：

> **不同颜色—形状搭配的匹配难度是否存在系统性不对称，且方向与 CoGenT 的 A/B 划分一致？**

解释边界见 §3 问题三：冻结模型没有“见过 / 没见过”的定义。

---

# 47. E3 数据

只使用：

```text
CLEVR-CoGenT
```

---

# 48. E3 不训练模型

直接使用：

```text
Frozen baseline VLM
```

在：

```text
Condition A
Condition B
```

上评价。

---

# 49. E3 任务

构造：

```text
正确 Color-Shape Description
vs
错误 Color-Shape Description
```

例如：

```text
Image:
red cube

T+:
a red cube

T-:
a blue cube
```

或者：

```text
T-:
a red cylinder
```

两个必做的附加分析：

```text
1. sphere 对照组：
   sphere 在 Condition A 和 B 都可取全部 8 种颜色，
   不受 A/B 划分限制。
   在 sphere 上跑同样的 2AFC，
   得到“不受划分影响”的难度基线，
   用来分离两种效应：
   - 颜色词难度（cyan / purple 等词在文本侧的熟悉度）
   - 组合难度（颜色 × 形状搭配本身）

2. 逐组合报告：
   按 (color, shape) 全部组合分别报告准确率，
   不只报 A / B 两个总数；
   同时按颜色逐项报告，检查颜色词混淆。
```

---

# 50. E3 指标

分别报告：

```text
Condition A Accuracy（cube / cylinder 样本）
Condition B Accuracy（cube / cylinder 样本）
Condition A / B Margin
sphere 对照 Accuracy（A、B 分别报告）
逐 (color, shape) 组合 Accuracy
逐颜色 Accuracy
```

并计算：

```text
Difficulty Gap
=
Condition A
-
Condition B
（只限 cube / cylinder 样本）
```

命名说明：

> 不再叫 OOD Drop。第一阶段的冻结通用预训练 baseline 没有在本研究中以 Condition A 作为训练集，因此“OOD Drop”会产生过度解释。这个量只表示两种搭配条件下的难度差。

---

# 51. E3 最重要的结论

如果：

```text
Condition B
```

明显低于：

```text
Condition A
```

第一阶段能下的结论只有：

> 两组颜色—形状搭配的匹配难度存在系统性不对称，且方向与 CoGenT 的 A/B 划分一致。

不能下的结论：

> “组合没见过所以下降”。
> 第一阶段的冻结通用预训练 baseline 没有在本研究中以 Condition A 进行训练，因此不能由 A/B 划分直接定义“见过 / 没见过”；这里只能解释为两组受控组合条件之间的难度差。

两个辅助判据（提高可信度）：

```text
1. cube / cylinder 上出现 A−B 差距，
   sphere 上不出现或明显更小
   → 差距更可能来自搭配，而不是颜色词

2. 差距集中在少数颜色词上
   → 更可能是颜色词混淆，不是组合问题
```

直接的 unseen composition 验证属于第三阶段（train-on-A、test-on-B）。E3 的结果作为 GraphDelta 研究动机的间接证据使用。

---

# 52. 第一阶段最终实验矩阵

| 实验 | PUG SPAR | PUG Animals | PUG ImageNet | CLEVR | CLEVR-CoGenT |
|---|---:|---:|---:|---:|---:|
| 2AFC | ✓ | — | — | ✓ | ✓（E3） |
| Matching Margin | ✓ | — | — | ✓ | ✓（E3） |
| Nuisance 基线 |  | ✓ | ✓ |  |  |
| Delta Magnitude |  | ✓ | ✓ |  |  |
| Same/Different-factor Delta（2×2） |  | ✓ | ✓ |  |  |
| 组合难度不对称 + sphere 对照 |  |  |  |  | ✓ |

说明：

```text
1. PUG: Animals / ImageNet 第一阶段不做 2AFC，只做 E2。
2. CoGenT 只出现在 E3，不进 E1。
3. 框架文档中的 T2I / Pair Accuracy 推迟到第二阶段之后。
```

---

# 53. 第一阶段最小可行实验

如果希望最快得到结果，

只做：

```text
PUG: SPAR
+
PUG: Animals
+
CLEVR
+
CLEVR-CoGenT
```

PUG: ImageNet 暂时跳过。

---

# 54. 最小实验组合

## Model

```text
第一轮必做：
CLIP ViT-L/14（Frozen）

建议对照：
SigLIP2（Frozen）

可选：
其他预训练 VLM baseline（Frozen）
```

新增 baseline 只用于复现趋势，不改变第一阶段“纯评估、不训练”的性质。

## Dataset

```text
PUG: SPAR
PUG: Animals
CLEVR
CLEVR-CoGenT
```

## Experiment

```text
1. Structural 2AFC
2. Matching Margin
3. Single-factor Delta Analysis
4. Same-transition / Same-factor vs Different-factor Delta
5. CoGenT Difficulty Asymmetry Diagnosis（含 sphere 对照）
```

---

# 55. 第一阶段代码原则

第一阶段代码应该是：

```text
Evaluation Pipeline
```

而不是：

```text
Training Pipeline
```

代码中不应该出现：

```text
optimizer
scheduler
backward()
loss.backward()
LoRA
fine-tuning
```

---

# 56. 第一阶段代码目录

建议：

```text
graphdelta_stage1/
│
├── configs/
│   ├── paths.yaml
│   ├── clip_vitl14.yaml
│   ├── e1_state.yaml
│   ├── e2_change.yaml
│   └── e3_cogent.yaml
│
├── data/
│   ├── raw/
│   ├── manifests/
│   └── cache/
│
├── src/
│   ├── data/
│   │   ├── clevr.py
│   │   ├── cogent.py
│   │   ├── pug_spar.py
│   │   ├── pug_animals.py
│   │   ├── pug_imagenet.py
│   │   ├── captions.py
│   │   └── pair_matcher.py
│   │
│   ├── models/
│   │   ├── vlm_encoder.py
│   │   ├── clip_encoder.py
│   │   └── siglip_encoder.py
│   │
│   ├── features/
│   │   ├── extract.py
│   │   ├── cache.py
│   │   └── delta.py
│   │
│   ├── metrics/
│   │   ├── matching.py
│   │   ├── margin.py
│   │   ├── delta_stats.py
│   │   └── bootstrap.py
│   │
│   ├── experiments/
│   │   ├── e1_state_diagnosis.py
│   │   ├── e2_change_diagnosis.py
│   │   └── e3_cogent_diagnosis.py
│   │
│   └── utils/
│       ├── io.py
│       ├── seed.py
│       └── logging.py
│
├── scripts/
│   ├── build_manifests.py
│   ├── extract_embeddings.py
│   ├── run_e1.py
│   ├── run_e2.py
│   └── run_e3.py
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── logs/
│
└── README.md
```

---

# 57. build_manifests.py

职责：

```text
读取已有数据 metadata
↓
统一字段
↓
生成 manifest
```

只整理已有数据。

不能：

```text
重新渲染
修改图片
创建新 scene
```

---

# 58. 第一阶段统一 Manifest

建议：

```json
{
  "sample_id": "xxx",
  "dataset": "pug_animals",
  "image_path": "...",
  "caption": "...",
  "metadata": {
    "animal": "dog",
    "texture": "default",
    "environment": "desert",
    "size": "small",
    "camera": "front"
  }
}
```

对于 pair：

```json
{
  "pair_id": "xxx",
  "factor": "texture",
  "sample_a": "id_a",
  "sample_b": "id_b"
}
```

---

# 59. PUG Pair Matcher

实现：

```python
match_pairs(metadata, factor)
```

例如：

```text
factor = texture
```

要求：

```text
animal same
environment same
size same
camera same
texture different
```

---

# 60. CLEVR Hard Negative Builder

只生成：

```text
text candidate
```

不改图片。

例如：

```text
positive_caption
negative_caption
factor
```

---

# 61. Baseline VLM Feature Extraction

所有 baseline 使用统一接口：

```python
encode_image(...)
encode_text(...)
```

返回：

```python
{
    "raw": feature,
    "normalized": normalized_feature
}
```

所有 baseline 必须：

```text
eval
frozen
```

不同 baseline 的特征维度不要求一致；同一模型内部的图像 / 文本比较按其原始共享空间进行，不跨模型直接计算 cosine。

预处理统一规定：

```text
CLIP ViT-L/14：224×224，官方 preprocess（短边 resize + center crop）
SigLIP2：按其官方分辨率和 preprocess
推理精度 fp16，批大小写入 configs
```

---

# 62. 第一阶段 Feature Cache

所有 feature 只提取一次。

保存：

```text
image raw embedding
image normalized embedding
text raw embedding
text normalized embedding
```

后续实验全部读取 cache。

---

# 62a. 数据许可与算力估计

许可：

```text
PUG 系列（Animals / SPAR / ImageNet）：CC-BY-NC，
附加条款禁止用于训练生成式模型。
第一阶段只做评估，符合许可；
开源代码和论文中注明来源和许可。
CLEVR / CLEVR-CoGenT：无此限制。
```

算力与存储（数量级估计）：

```text
需编码图像约 30 万张：
PUG Animals            215,040
PUG SPAR                43,560
CLEVR val               15,000
CoGenT val_A + val_B    30,000

ViT-L/14 @224 单卡 1 小时内完成。
特征存储约 2 GB（float32，raw + normalized；fp16 可减半）。
文本编码量远小于图像。
```

---

# 63. E1 输出

输出：

```text
e1_results.csv
```

字段：

```text
dataset
factor
n_samples
accuracy
mean_margin
median_margin
ci_low
ci_high
template_variance
```

---

# 64. E2 输出

输出：

```text
e2_delta_stats.csv
```

字段：

```text
dataset
factor
n_pairs
n_contexts
mean_delta_norm
median_delta_norm
nuisance_norm_visual
nuisance_norm_text
same_factor_sim_same_context
same_factor_sim_diff_context
different_factor_sim_same_context
different_factor_sim_diff_context
random_similarity
same_transition_sim_diff_context
same_factor_diff_transition_sim
transition_similarity_gap
```

另输出逐 transition 表：

```text
e2_transition_stats.csv
字段：dataset, factor, source_value, target_value,
     n_contexts, mean_sim, mean_delta_norm
```

> environment 因素有 2,016 个 transition（64×63÷2，每个无序对取一个方向），必须逐 transition 汇总，不能只按 factor 汇总。

---

# 65. E3 输出

输出：

```text
e3_cogent.csv
```

字段：

```text
condition
shape_group（cube_cylinder / sphere）
accuracy
mean_margin
median_margin
```

以及：

```text
Difficulty Gap（A − B，只限 cube / cylinder）
```

另输出逐组合表：

```text
e3_cogent_by_combo.csv
字段：color, shape, n, accuracy, mean_margin
```

---

# 66. 第一阶段统计要求

第一阶段不训练模型，因此重点是估计差异的稳定性。

统一报告：

```text
Mean
Median
Std
95% Bootstrap CI
```

固定随机种子：

```text
0
1
2
```

这些 seed 只用于：

```text
sampling
bootstrap
random negative
evaluation-only perturbation
```

不是模型训练。

### E1 / E3：Accuracy 与 Margin

对于准确率比较：

```text
Accuracy Gap
=
Acc_A - Acc_B
```

使用 bootstrap CI。

只有当：

```text
95% CI 不含 0
```

并且差距具有实际意义时，才写“明显更弱 / 明显更困难”。

`5 percentage points` 可以作为一个预设的实践意义参考阈值，但不作为所有实验通用的统计显著性标准。

Matching Margin 单独报告其 bootstrap CI。

### E2：Representation Response

E2 不使用 accuracy 阈值。

核心差值定义为：

```text
Transition Similarity Gap
=
Sim(Same Transition, Different Context)
-
Sim(Different Factor, Different Context)
```

如果：

```text
95% bootstrap CI > 0
```

说明相同具体 transition 在不同 context 下具有更稳定的表示响应。

同时报告：

```text
Factor / Nuisance Norm Gap
```

### 聚类 bootstrap

E2 必须按 context 聚类抽样：

```text
cluster = 除被测因素外的其余 metadata 组合
（测 texture 时：animal + environment + size + camera
 测 environment 时：animal + texture + size + camera
 以此类推）
```

不能把高度相关的 pair 当作完全独立样本。

# 67. 第一阶段最重要的图

## Figure 1

```text
2AFC Accuracy by Structural Factor
```

---

## Figure 2

```text
Matching Margin by Structural Factor
```

---

## Figure 3

```text
Delta Magnitude by PUG Factor
```

---

## Figure 4

```text
Same-transition
vs
Same-factor / Different-transition
vs
Different-factor
vs
Random Delta Similarity
```

---

## Figure 5

```text
CLEVR-CoGenT
Condition A vs Condition B
```

---

# 68. 第一阶段成功判定

阶段一的目的不是得到某个固定数值。

而是判断 GraphDelta 的研究问题是否成立。

---

# 69. 情况 A：最支持 GraphDelta

如果发现：

```text
1. Relation / Binding-style
   明显弱于 Object / Attribute

2. PUG: SPAR 和 CLEVR
   都出现类似趋势

3. PUG: Animals 中
   Same Transition + Different Context
   比 Different Factor + Different Context 更一致，
   且响应超出 nuisance 水平

4. CoGenT cube / cylinder
   出现难度不对称且方向与 A/B 划分一致，
   sphere 对照不出现同样趋势
```

说明：

> 现有 baseline 存在结构理解不足，CoGenT 受控组合条件存在难度不对称，同时单因素表示响应中可能保留一定可重复规律；这些结果共同支持进入第二阶段做严格 Controlled Edit 验证。

此时非常适合进入第二阶段。

---

# 70. 情况 B：静态结构问题存在，但 Delta 没规律

如果：

```text
Relation / Binding-style 很弱
```

但是：

```text
Same Transition (Different Context)
≈
Different Factor (Different Context)
```

说明：

> 原始 global embedding 中没有稳定的变化结构。

第二阶段重点就应该转向：

```text
Controlled Structural Edit
```

验证更严格的 Structural Delta。

---

# 71. 情况 C：CoGenT 没明显下降

如果：

```text
Condition A
≈
Condition B
```

需要检查：

```text
caption 是否过于简单
任务是否真的依赖组合
negative 是否过于容易
sphere 对照是否本来就接近上限（整体任务太简单）
逐组合结果是否被少数颜色词拉平
```

如果排除实验设计问题后仍然如此，

说明：

> 当前 CoGenT 任务不足以体现 GraphDelta 关注的组合问题。

注意：这只削弱“组合难度不对称”这一条证据，不否定 GraphDelta 的动机；Relation / Binding 证据（E1）与此相互独立。

第二阶段需要更严格的 Binding / Relation Controlled Edit。

---

# 72. 情况 D：所有结构任务都很好

如果：

```text
Relation
Binding-style
CoGenT
```

都表现非常好，

首先检查：

```text
数据泄漏
caption shortcut
negative 太简单
```

如果排除问题后仍然很好，

则说明：

> 当前研究问题在当前 baseline 模型 + 当前 benchmark 上证据不足。

不应该立即进入复杂 GraphDelta。

---

# 73. 第一阶段结束后再决定第二阶段

只有阶段一结果支持，

才进入：

```text
阶段二：Controlled Edit Mechanism Validation
```

这时才允许：

```text
重新生成受控 scene
自建 Binding Swap
自建 Relation Edit
加入 EditCLEVR
计算严格 Before / After Delta
训练 Linear Probe
```

---

# 74. 第二阶段的目标

第二阶段回答：

> **在严格受控的结构编辑下，Structural Delta 是否真的存在，并且是否可以跨模态、跨 context 迁移？**

这时才真正研究：

```text
Δ_v
Δ_t
```

---

# 75. 第二阶段可以加入的数据

推荐：

```text
EditCLEVR
```

用于：

```text
Color
Shape
Size
Material
```

Controlled Edit。

---

# 76. 第二阶段自建 CLEVR Pair

这时才使用 CLEVR generator 构造：

```text
Relation Change
Binding Swap
```

例如：

```text
red cube + blue sphere
→
blue cube + red sphere
```

严格保持：

```text
camera
position
object set
color set
lighting
```

不变。

---

# 77. 第二阶段可以加入 Linear Probe

这时才训练：

```text
Linear Edit Classifier
```

输入：

```text
Δ
```

预测：

```text
Color
Size
Object
Relation
Binding
```

并测试：

```text
Held-out Context
```

---

# 78. 第二阶段可以加入 Cross-modal Delta

研究：

```text
Δ_v
vs
Δ_t
```

比较：

```text
Matched
Same Edit
Different Edit
Random
```

---

# 79. 第二阶段结束后再决定 GraphDelta

只有第二阶段证明：

```text
Structural Delta
```

确实具有：

```text
可识别性
跨模态一致性
跨 context 迁移能力
```

才进入完整 GraphDelta 方法。

---

# 80. 第三阶段：GraphDelta 方法验证

第三阶段才允许：

```text
Edit Encoder
Context-conditioned Transformation
Cross-modal Delta Alignment
Fine-tuning
LoRA
Projection
```

---

# 81. 三阶段关系

最终逻辑必须保持：

```text
阶段一
证明问题存在

        ↓

阶段二
证明 Structural Delta 值得建模

        ↓

阶段三
提出 GraphDelta 改善这个问题
```

而不是：

```text
先设计 GraphDelta
再想办法找实验支持
```

---

# 82. 第一阶段当前最推荐执行顺序

```text
Step 1
整理 PUG: SPAR / Animals / CLEVR / CoGenT

Step 2
固定 CLIP ViT-L/14 Feature Extraction

Step 3
PUG: SPAR 结构诊断

Step 4
CLEVR Hard-negative 结构诊断

Step 5
PUG: Animals 单因素 Delta 分析

Step 6
CLEVR-CoGenT 难度不对称诊断（含 sphere 对照）

Step 7
汇总结果，判断是否进入第二阶段
```

---

# 83. 第一阶段不应该出现的代码

如果第一阶段代码中出现：

```text
optimizer
loss
backward
scheduler
train_epoch
LoRA
adapter
projection training
```

说明阶段划分出了问题。

第一阶段应该全部是：

```text
load
encode
match
compare
analyze
visualize
```

---

# 84. 第一阶段代码生成总要求

后续让代码模型生成第一阶段代码时，必须遵守：

```text
1. CLIP ViT-L/14 作为首选 baseline，固定并冻结。

2. 允许增加 SigLIP2 / OpenCLIP 等预训练 VLM 作为额外 baseline，
   但新增模型同样必须完全冻结，只用于独立评估。

3. 不训练任何 baseline 参数，不训练 adapter / projection / probe。

4. 不加入 GraphDelta 的任何方法模块。

5. 不重新生成或修改公开数据图像；
   只允许 evaluation-only、非持久化的轻量输入扰动用于 nuisance 参考。

6. 第一阶段固定使用已有 PUG、CLEVR、CLEVR-CoGenT 及其官方 split。

7. CLEVR 只允许基于已有 scene annotation 生成文本描述和 hard negative。

8. PUG 只允许根据已有 metadata 配对。

9. 所有 baseline embedding 分模型缓存；
   raw projected feature 和 normalized embedding 同时保存（模型支持时）。

10. 第一阶段不使用 Linear Probe。

11. 第一阶段不使用 Projection Training。

12. 第一阶段不使用 EditCLEVR。

13. 第一阶段不自建新的 Binding / Relation 图像。

14. 所有实验均为 evaluation / analysis。

15. 最终目标是决定是否值得进入 Controlled Edit 机制验证。

16. Split 固定：CLEVR 用 val；CoGenT 用 val_A / val_B；
    PUG: Animals 用全量；SPAR 全量。

17. PUG: SPAR 的 caption 从 metadata 确定性生成；
    Position / Relation 任务先过滤双对象样本。

18. 所有 2AFC 样本必须通过“正例为真、负例为假”检查。

19. E2 的主比较单位为 transition_key=(factor, source, target)；
    对离散因素，metadata 顺序只用于统一方向，不解释为语义顺序。

20. E2 的 bootstrap 按 context 聚类。

21. 外层 prompt 模板固定 5–7 个，报告模板间方差。

22. 新增 baseline 必须使用各自官方 preprocess / tokenizer，
    不能强制所有模型共用 CLIP 的输入预处理。
```

# 85. 第一阶段一句话概括

> **GraphDelta 第一阶段不是训练方法，而是在固定公开数据和统一评估协议上，对一个或多个冻结预训练 VLM baseline 做小规模诊断，验证 Relation / Binding / Composition 等结构问题是否存在，以及原始表示对已有单因素变化是否表现出可分析、可重复的响应规律。**

---

# 86. 整个预实验一句话概括

> **先用固定公开数据和一个或多个冻结 baseline 证明“问题值得研究”，再通过 Controlled Edit 验证“Structural Delta 是否值得建模”，最后才训练 GraphDelta；三个阶段必须严格分开。**
