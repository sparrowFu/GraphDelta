# GraphDelta：面向未见组合的结构变化感知视觉—语言表示学习

## 1. 工作定位

GraphDelta 关注视觉—语言统一表示中的一个细粒度问题：

> 现有视觉—语言模型已经能够较好地识别图像和文本中的对象、属性和关系，但这些语义信息未必在统一表示空间中被正确组织。尤其当对象—属性绑定、对象间关系或角色发生变化时，全局图文表示可能仍然保持较高相似度，从而导致模型难以区分“概念相同但结构不同”的语义状态。

例如：

```text
场景 A：
red cube + blue sphere

场景 B：
blue cube + red sphere
```

两个场景包含完全相同的原子概念：

```text
red
blue
cube
sphere
```

区别只在于：

```text
场景 A：
red  → cube
blue → sphere

场景 B：
blue → cube
red  → sphere
```

也就是说，**概念集合没有变化，但概念之间的绑定关系发生了变化**。

现有全局视觉—语言表示更擅长建模“场景中有什么”，但未必能够稳定表示“这些概念是如何组合和关联的”。

因此，本工作进一步研究：

> **当对象、属性、关系或绑定结构发生变化时，视觉表示和文本表示是否能够产生一致、可识别并且可以迁移的结构变化响应。**

在此基础上，利用 Scene Graph Edit 作为结构监督，学习一种面向未见组合的结构变化感知统一表示。

暂定方法名称：

**GraphDelta**

可用英文标题：

> **GraphDelta: Structure-Change-Aware Vision-Language Representation Learning for Unseen Compositions**

> **阶段说明：** 本文档描述完整研究框架；第一阶段的具体实验协议与《GraphDelta_完整预实验方案》保持一致。第一阶段允许加入多个冻结预训练 VLM 作为 baseline，但所有 baseline 仅用于评估，不属于 GraphDelta 方法模块。

---

# 2. 最核心的研究问题

GraphDelta 最核心的问题可以用一句话概括：

> **如何让视觉—语言模型不仅知道“当前场景是什么”，还知道“当前场景是怎么变成另一个场景的”，并将这种结构变化规律迁移到训练中没有见过的新组合上？**

传统视觉—语言学习主要解决：

```text
Image ↔ Text
```

GraphDelta 进一步研究：

```text
原始场景
   ↓
发生一个结构变化
   ↓
新场景
```

以及：

```text
原始表示 z
   ↓
发生相应的表示变化
   ↓
新表示 z'
```

因此，本工作真正关注的不是单独的表示 `z`，而是：

```text
z → z'
```

也就是：

> **语义结构发生变化时，统一表示空间应该如何响应。**

---

# 3. 为什么这个问题值得研究？

## 3.1 “看到了什么”不等于“理解了怎么组合”

视觉—语言模型可能已经知道：

```text
red
blue
cube
sphere
```

但它仍然可能无法稳定区分：

```text
red cube + blue sphere
```

和：

```text
blue cube + red sphere
```

原因是两者的大部分全局语义是相同的。

真正决定它们不同的，不是“有哪些概念”，而是：

```text
red 到底属于哪个对象？
blue 到底属于哪个对象？
```

因此：

> **全局语义一致，不代表内部结构一致。**

---

## 3.2 现有结构方法更多关注“结构状态”

已有 Scene Graph、局部对齐和结构增强方法已经说明：

- 对象；
- 属性；
- 关系；
- 局部实体；
- subject-object role；

这些结构信息对细粒度视觉—语言理解十分重要。

但多数方法研究的是：

```text
当前图像是什么结构？
当前文本是什么结构？
图像结构与文本结构是否一致？
```

即：

```text
Graph State ↔ Graph State
```

GraphDelta 进一步关注：

```text
Graph State A
     ↓
Structural Edit
     ↓
Graph State B
```

也就是：

> **结构从 A 变成 B 时，表示空间发生了什么变化？**

研究对象由：

```text
Semantic State
```

进一步扩展为：

```text
Semantic Structural Transformation
```

---

# 4. 为什么要从“状态”转向“变化”？

最终目标是解决：

> **未见组合（Unseen Composition）**

例如训练阶段出现：

```text
red sphere
blue sphere

green cube
yellow cube
```

但没有出现：

```text
red cube
blue cube
```

测试时要求模型理解：

```text
red cube
→
blue cube
```

这里：

- `red` 见过；
- `blue` 见过；
- `cube` 见过；
- 颜色变化也见过；
- 但是 `red + cube` 和 `blue + cube` 这两个具体组合没见过。

如果模型只是记忆完整组合，它很难处理这种情况。

如果模型真正学到：

> “red → blue”是一种可以作用在不同对象上的颜色变化规律，

那么它就更可能将这个规律迁移到新的对象组合上。

因此，本工作的核心目标不是让模型记住更多 tuple，而是让模型学会：

> **对象、属性、关系和绑定发生变化时，存在什么可以迁移的结构变化规律。**

---

# 5. 整个 GraphDelta 只需要记住五步

整个研究框架可以压缩成：

```text
现有 VLM
只能较好建模“当前场景是什么”
        ↓
发现问题
结构或绑定发生变化后，
embedding 可能变化不足
        ↓
Scene Graph Edit
明确告诉模型“哪里变了、怎么变了”
        ↓
Structural Delta Learning
学习结构变化对应的表示变化规律
        ↓
Cross-modal Delta Alignment
让图像和文本对同一种变化形成一致响应
        ↓
Unseen Composition
验证这种变化规律能否迁移到未见组合
```

后续所有方法设计和实验设计都围绕这五步展开。

---

# 6. Scene Graph 在这里到底做什么？

Scene Graph 在 GraphDelta 中不是最终推理依赖，而是**训练阶段的结构监督工具**。

它只解决一个问题：

> **明确描述两个语义状态之间到底发生了什么变化。**

例如：

```text
red cube
→
blue cube
```

可以写成：

```text
TYPE: COLOR_CHANGE
ENTITY: cube
SOURCE: red
TARGET: blue
```

对于：

```text
cube left of sphere
→
cube right of sphere
```

可以写成：

```text
TYPE: RELATION_CHANGE
SUBJECT: cube
OBJECT: sphere
SOURCE: left_of
TARGET: right_of
```

对于：

```text
red cube + blue sphere
→
blue cube + red sphere
```

可以写成：

```text
TYPE: BINDING_SWAP

BEFORE:
red  → cube
blue → sphere

AFTER:
blue → cube
red  → sphere
```

因此 Scene Graph 的核心作用不是：

```text
把整个图谱输入模型
```

而是：

```text
告诉模型：
哪个实体、属性、关系或绑定发生了什么变化
```

第一阶段不需要复杂 GNN，也不需要额外知识图谱。

---

# 7. 完整研究关注的结构变化类型

完整 GraphDelta 重点研究以下五类结构变化。第一阶段不会为这五类变化都重新生成 before / after 图像；第一阶段只用固定公开数据做问题诊断，严格 Controlled Edit 留到第二阶段。

## 7.1 Object Change

```text
red cube
→
red sphere
```

只改变对象类别。

---

## 7.2 Color Change

```text
red cube
→
blue cube
```

保持对象不变，只改变颜色。

---

## 7.3 Size Change

```text
small cube
→
large cube
```

保持其他因素不变，只改变大小。

---

## 7.4 Relation Change

```text
cube left of sphere
→
cube right of sphere
```

保持实体和属性不变，只改变对象之间的关系。

---

## 7.5 Binding Change

```text
red cube + blue sphere
→
blue cube + red sphere
```

这里原子概念完全不变，只改变属性与对象之间的绑定。

这一类变化尤其重要，因为它最直接反映：

> **模型是否真正理解概念之间的组合关系。**

---

# 8. GraphDelta 的核心方法

## 8.1 第一步：获得变化前后的视觉和文本表示

给定一对语义状态：

```text
Before
After
```

视觉编码：

```text
Image Before → h_v
Image After  → h_v'
```

文本编码：

```text
Text Before → h_t
Text After  → h_t'
```

分别得到视觉变化：

```text
Δ_v = h_v' - h_v
```

和文本变化：

```text
Δ_t = h_t' - h_t
```

可以把 `Δ` 直观理解为：

> **模型认为这两个状态之间“发生了什么”。**

---

## 8.2 第二步：编码 Scene Graph Edit

对：

```text
G → G'
```

提取结构变化信息：

```text
q_e = EditEncoder(G, G')
```

这个表示不需要重新编码完整场景，而只需要描述：

```text
变化类型是什么？
谁发生了变化？
从什么变成什么？
涉及哪些实体和关系？
```

例如：

```text
COLOR_CHANGE
cube
red → blue
```

或者：

```text
BINDING_SWAP
red/cube
blue/sphere
```

---

## 8.3 第三步：学习上下文相关的结构变化

一个重要假设是：

> 同一种结构变化具有共同规律，但在不同场景中具体的 embedding 变化不一定完全相同。

例如：

```text
red cube → blue cube
```

和：

```text
red car → blue car
```

都属于 Color Change。

但它们具体的表示变化未必完全相同。

因此 GraphDelta 不简单假设：

```text
COLOR_CHANGE = 一个固定向量
```

而是建模：

```text
Δ = F(Edit, Context)
```

例如：

```text
d_v = F_v(h_v, q_e)
d_t = F_t(h_t, q_e)
```

其中：

- `h_v / h_t`：当前场景的语义上下文；
- `q_e`：结构编辑信息；
- `d_v / d_t`：模型预测的表示变化。

模型需要学习：

```text
d_v ≈ h_v' - h_v
```

以及：

```text
d_t ≈ h_t' - h_t
```

这就是：

> **Context-conditioned Structural Transformation**

---

## 8.4 第四步：让视觉和文本的变化保持一致

假设：

```text
Image:
red cube → blue cube

Text:
"red cube" → "blue cube"
```

两者表达的是同一个结构变化。

因此希望视觉变化：

```text
Δ_v
```

与文本变化：

```text
Δ_t
```

在共享变化空间中具有一致性。

可以加入轻量投影：

```text
P_v(Δ_v)
P_t(Δ_t)
```

然后约束：

```text
P_v(Δ_v) ≈ P_t(Δ_t)
```

最终模型同时学习两种一致性：

```text
状态一致性：
Image ↔ Text

变化一致性：
Image Change ↔ Text Change
```

这就是 GraphDelta 与普通视觉—语言对齐最本质的区别。

---

# 9. GraphDelta 最终想让模型学会什么？

不是让模型死记：

```text
red cube
blue sphere
green cylinder
```

而是让模型逐渐形成：

```text
颜色变化规律
对象变化规律
大小变化规律
关系变化规律
绑定变化规律
```

最终希望做到：

```text
训练中：
red sphere → blue sphere

测试中：
red cube → blue cube
```

即使测试组合没有出现过，模型仍然能够：

```text
识别这是同一种变化规律
```

这就是：

> **Structural Transformation Transfer**

也是 GraphDelta 最核心的能力。

---

# 10. 为什么必须分阶段验证？

完整 GraphDelta 最终研究的是：

```text
结构发生变化
        ↓
表示如何变化
        ↓
这种变化规律能否跨模态、跨组合迁移
```

但不能一开始就构造复杂方法并训练。

整个验证流程统一分成三个核心阶段：

```text
阶段一：问题验证
冻结 baseline + 固定公开数据
        ↓
确认结构理解问题是否存在
确认原始表示对已有单因素变化是否有可重复响应

阶段二：机制验证
Controlled Edit Pair
        ↓
严格研究 Structural Delta
验证跨模态一致性和跨 context 迁移

阶段三：方法验证
GraphDelta Training
        ↓
显式学习结构变化规律
验证 unseen composition transfer
```

后续再增加真实数据和更多 backbone 作为扩展验证。

---

# 11. 第一阶段：固定公开数据上的问题验证

第一阶段的目标不是证明 GraphDelta 有效，而是回答：

> **研究问题是否真实存在，并且是否值得进入 Controlled Edit 机制验证？**

第一阶段允许使用多个预训练视觉—语言模型作为 baseline，例如：

```text
CLIP ViT-L/14
SigLIP2
其他容易获得的 OpenCLIP / VLM
```

但所有 baseline 必须：

```text
使用原始预训练权重
完全冻结
不 fine-tune
不训练 projection / adapter / probe
不加入 GraphDelta 模块
```

推荐先用 CLIP ViT-L/14 跑通全部实验；如果现象明确，再用 SigLIP2 复现关键结果。

第一阶段固定公开数据：

```text
PUG: SPAR
PUG: Animals
PUG: ImageNet（辅助）
CLEVR v1.0
CLEVR-CoGenT v1.0
```

不重新渲染 CLEVR，不创建新的 Binding / Relation 图像。

---

# 12. 第一阶段 E1：State-level Structural Diagnosis

E1 回答：

> **现有冻结 baseline 对不同结构因素的图文区分能力是否不均衡？**

使用：

```text
PUG: SPAR
CLEVR val
```

CLEVR 中基于已有 scene annotation 生成确定性文本，并构造：

```text
Object Hard Negative
Color Hard Negative
Size Hard Negative
Material Hard Negative
Relation Hard Negative
Binding-style Hard Negative
```

例如：

```text
Positive:
A red cube and a blue sphere.

Negative:
A blue cube and a red sphere.
```

这里图像不改变，只检测模型是否正确利用属性—对象绑定信息。

主要指标：

```text
I2T 2AFC Accuracy
Matching Margin
95% Bootstrap CI
```

第一阶段不做 T2I controlled image pair，因为那需要重新生成只差一个结构因素的图像，属于第二阶段。

如果：

```text
Relation / Binding-style
```

稳定弱于：

```text
Object / Attribute
```

则说明现有统一表示在结构关系利用上存在明显不均衡。

---

# 13. 第一阶段 E2：Single-Factor Representation Response Diagnosis

E2 不直接证明真正的 Structural Delta。

它只回答：

> **原始预训练表示对已有单因素变化是否具有可重复的 response pattern？**

核心使用 PUG: Animals，PUG: ImageNet 作为可选辅助。

从现有 metadata 中配对：

```text
Background Change
Texture Change
Size Change
Camera Change
```

所有 pair 均来自公开数据中已有图片，不重新渲染。

计算：

```text
Δ_h = h_after - h_before
Δ_z = z_after - z_before
```

对于离散因素，主分析单位定义为：

```text
transition_key
=
(factor, source_value, target_value)
```

重点比较：

```text
Same Transition + Different Context
vs
Different Factor + Different Context
vs
Random
```

而不是假设“所有 texture change”天然共享一个固定方向。

同时加入 nuisance 参考，判断观察到的 response 是否超出普通输入扰动水平。

第一阶段只能得到：

> **representation response 是否具有可重复规律。**

真正的 Relation / Binding Structural Delta 必须等第二阶段 Controlled Edit。

---

# 14. 第一阶段 E3：Composition Difficulty Diagnosis

E3 使用：

```text
CLEVR-CoGenT val_A
CLEVR-CoGenT val_B
```

比较：

```text
Condition A
vs
Condition B
```

下的属性—形状图文匹配难度。

必须明确：

> 冻结的通用预训练 baseline 并没有在本研究中“train on A”，因此第一阶段不能把 A / B 的差异直接解释成“seen → unseen composition drop”。

第一阶段只允许下结论：

> **不同颜色—形状组合条件存在系统性难度不对称。**

同时使用 sphere 作为对照，因为 sphere 在 A / B 中都可取全部颜色。

严格的：

```text
primitive seen
+
composition unseen
```

迁移验证放到后续有明确训练划分的阶段。

---

# 15. 第一阶段的判定标准

第一阶段最支持继续研究的结果是：

```text
1. E1：
Relation / Binding-style
稳定弱于 Object / Attribute

2. E1：
PUG: SPAR 与 CLEVR
出现相似趋势

3. E2：
Same Transition + Different Context
比 Different Factor + Different Context 更一致，
且 response 超过 nuisance 水平

4. E3：
CoGenT 的 cube / cylinder 组合条件
出现稳定难度不对称，
而 sphere 对照明显更弱或没有同样趋势
```

这些结果共同说明：

> 现有 baseline 的结构利用存在缺陷，组合条件会影响匹配难度，并且原始表示对某些已有变化保留了可重复响应。

它们只能支持：

```text
“值得进入第二阶段”
```

不能直接宣称：

```text
“Structural Delta 已经被证明”
或
“unseen composition transfer 已经被验证”
```

---

# 16. 第二阶段：Controlled Edit Mechanism Validation

只有第一阶段结果支持后，才进入第二阶段。

第二阶段才真正构造：

```text
Before
→
After
```

的严格受控编辑。

可使用：

```text
EditCLEVR
```

验证：

```text
Color
Shape
Size
Material
```

并使用 CLEVR generator 自建：

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

严格控制其他因素尽可能不变。

此时才真正定义：

```text
Δ_v = h_v(after) - h_v(before)
Δ_t = h_t(after) - h_t(before)
```

并研究：

```text
Matched Edit
Same Edit / Same Transition
Different Edit
Random
```

---

# 17. 第二阶段：可识别性、跨模态一致性与迁移

第二阶段可以加入轻量分析工具：

```text
Linear Edit Probe
Linear Cross-modal Projection
```

这些工具只用于回答机制问题，不是最终 GraphDelta。

重点验证：

```text
1. Edit / Transition 是否可以从 Δ 中识别？

2. 视觉 Δ 和文本 Δ 对同一 edit 是否比 mismatch 更一致？

3. 相同 edit 是否能迁移到 held-out context？

4. primitive 已出现但完整组合未出现时，
   Structural Delta 是否仍可识别？
```

这时才能严格构造：

```text
Unseen Attribute-Object
Unseen Relation
Unseen Binding
```

等 controlled split。

如果这些结果成立，说明：

> **Structural Delta 不只是一个直觉，而是具有可识别、跨模态和可迁移的结构。**

---

# 18. 第三阶段：进入 GraphDelta 方法验证

第二阶段之后才训练完整 GraphDelta。

完整方法加入：

```text
Scene Graph Edit Encoding
Context-conditioned Structural Transformation
Cross-modal Delta Alignment
Edit / Transformation Discrimination
```

核心比较：

```text
Frozen / Fine-tuned VLM
Hard Negative
State-level Structure Alignment
Fixed Delta
Delta Alignment
Context-conditioned GraphDelta
```

最终真正验证：

> **显式学习结构变化规律，是否能够改善严格定义的 unseen composition transfer。**

# 19. 完整 GraphDelta 的训练目标

如果预实验支持继续，完整方法可以包含四类目标。

## 19.1 原始图文状态对齐

保留基础视觉—语言目标：

```text
Image ↔ Text
```

保证统一表示仍具有正常的跨模态匹配能力。

---

## 19.2 Structural Delta Prediction

根据：

```text
Context + Scene Graph Edit
```

预测表示变化：

```text
d = F(Context, Edit)
```

并约束：

```text
d ≈ Δ
```

---

## 19.3 Cross-modal Delta Alignment

约束：

```text
Visual Delta ≈ Text Delta
```

使图像和文本对于同一种结构变化形成共享变化表示。

---

## 19.4 Edit Discrimination

让：

```text
Same Edit
```

在不同 context 中仍具有共同结构，同时：

```text
Different Edit
```

能够区分。

可以采用：

```text
classification
contrastive objective
in-batch negatives
```

第一版优先使用简单方案。

---

# 20. 后续扩展：Edit Composition

这一部分不属于第一阶段核心方法，可以等基础结果成立后再加入。

例如：

```text
red small cube
↓ Color Change
blue small cube
↓ Size Change
blue large cube
```

研究：

```text
e1
然后 e2
```

和：

```text
e1 + e2
```

是否具有一致的最终表示结果。

不强制：

```text
Δ_e1+e2 = Δ_e1 + Δ_e2
```

而采用更一般的：

```text
T_e2(T_e1(z))
≈
T_(e2∘e1)(z)
```

该扩展主要验证模型是否进一步学到：

> **多种结构变化之间的组合规律。**

---

# 21. 数据准备

## 21.1 第一阶段：固定公开数据

第一阶段固定使用：

```text
PUG: SPAR
PUG: Animals
PUG: ImageNet（辅助）
CLEVR v1.0
CLEVR-CoGenT v1.0
```

第一阶段只允许：

```text
读取 metadata / scene annotation
生成 deterministic caption
生成文本 hard negative
根据 metadata 匹配已有图片 pair
evaluation-only nuisance perturbation
```

不允许：

```text
重新渲染 CLEVR
创建新的 Binding / Relation 图像
生成新的训练数据
```

CLEVR 主要用于静态结构诊断；PUG: Animals 主要用于单因素 representation response；CoGenT 主要用于组合难度不对称诊断。

## 21.2 完整研究的属性范围

Controlled Edit 阶段主要考虑：

Object：

```text
cube
sphere
cylinder
```

Color：

```text
red
blue
green
yellow
```

Size：

```text
small
large
```

Relation：

```text
left
right
front
behind
```

并重点加入：

```text
Binding Swap
```

## 21.3 第二阶段最小 Controlled Edit 版本

第二阶段不要一开始就同时生成所有 Edit。

建议先做：

```text
Color
Binding
```

原因：

- Color Change 是简单正控制；
- Binding Change 最接近 GraphDelta 的核心难点。

第一轮可先构造：

```text
1000–1500 Controlled Edit Pairs
```

用于机制验证。

如果结果成立，再扩展：

```text
Object
Color
Size
Relation
Binding
```

总规模：

```text
3000–5000 pairs
```

这些规模属于第二阶段，不属于第一阶段固定公开数据诊断。

# 22. 模型准备

第一阶段 baseline 首选：

```text
CLIP ViT-L/14
```

建议在关键结果明确后增加冻结对照：

```text
SigLIP2
其他可获得的预训练 VLM baseline
```

第一阶段所有 baseline：

```text
Encoder Frozen
Original Pretrained Weights
Embedding Cache
No LoRA
No Fine-tuning
No Projection Training
```

模型可以增加，但其角色始终是：

> **baseline / phenomenon replication**

而不是 GraphDelta 的核心模型或方法模块。

不同模型必须使用各自官方 preprocess 和 tokenizer。

建议保存：

```text
pre-normalization projection feature（模型支持时）
normalized embedding
```

第二阶段仍然优先冻结 backbone 做机制分析。

第三阶段 GraphDelta 方法验证时，再比较：

```text
frozen
projection-only
partial fine-tuning / LoRA
```

等设置。

# 23. 第一阶段不需要准备什么

第一阶段当前不需要：

```text
ConceptNet
Wikidata
External Knowledge Graph
GNN
Graph Transformer
Visual Genome Scene Graph Generator
Object Detector
Region Feature
LLM-generated Scene Graph
Controlled CLEVR Re-rendering
Linear Edit Probe
Cross-modal Projection Training
```

第一阶段只需要公开数据的：

```text
metadata
scene annotation
deterministic caption
frozen VLM embedding
```

Scene Graph Edit 本身属于第二阶段及之后的机制 / 方法监督，不应提前混入第一阶段。

# 24. 完整实验结构

完整研究的实验链条建议分成六组。

## E1：State-level Structural Diagnosis

阶段：第一阶段。

回答：

> 不同结构因素的图文区分能力是否不均衡？

指标：

```text
I2T 2AFC Accuracy
Matching Margin
Bootstrap CI
```

数据：

```text
PUG: SPAR
CLEVR
```

## E2：Single-Factor Representation Response

阶段：第一阶段。

回答：

> 原始预训练表示对已有单因素变化是否有可重复响应？

指标：

```text
Delta Norm
Same-transition Similarity
Different-factor Similarity
Nuisance Reference
```

数据：

```text
PUG: Animals
PUG: ImageNet（辅助）
```

## E3：Composition Difficulty Diagnosis

阶段：第一阶段。

回答：

> CoGenT A / B 两组属性—形状组合条件是否存在系统性难度不对称？

指标：

```text
2AFC Accuracy
Matching Margin
Sphere Control
Per-(color, shape) Analysis
```

注意：这一阶段不直接声称 unseen composition transfer。

## E4：Controlled Structural Delta Diagnosis

阶段：第二阶段。

回答：

> 严格 Controlled Edit 下是否存在可识别、跨模态、可迁移的 Structural Delta？

指标：

```text
Same / Different Edit Similarity
Edit Classification
Cross-modal Delta Consistency
Held-out Context Accuracy
Controlled Unseen Composition
```

## E5：GraphDelta 方法验证

阶段：第三阶段。

比较：

```text
Frozen / Fine-tuned VLM
Hard Negative
State-level Structure Alignment
Fixed Delta
Delta Alignment
Context-conditioned GraphDelta
```

核心问题：

> 增加变化级监督之后，是否真正改善严格定义的 unseen composition transfer？

## E6：真实数据与模型泛化

在机制和方法成立后，再扩展到：

```text
ARO
SugarCrepe
EqBen
Visual Genome Attribution / Relation
What'sUp
更多 VLM backbone
```

目标是验证结论不依赖单一合成数据或单一 backbone。

# 25. 最重要的 Baseline

必须至少包含以下几类。

## Baseline 1：冻结预训练 VLM

```text
CLIP ViT-L/14
SigLIP2
其他可获得的预训练 VLM（可选）
```

这些都是 baseline，不属于 GraphDelta 方法模块。

---

## Baseline 2：Fine-tuning

普通图文 fine-tuning。

验证：

> 收益是否只是因为额外训练。

---

## Baseline 3：Hard Negative

只加入结构反事实样本：

```text
red cube + blue sphere
vs
blue cube + red sphere
```

但不显式学习 Delta。

验证：

> 收益是否只是来自更难的负样本。

---

## Baseline 4：State-level Structure Alignment

利用 Scene Graph 结构监督，但只对齐：

```text
当前结构状态
```

不学习：

```text
结构变化
```

这是非常关键的 baseline。

因为它可以回答：

> **GraphDelta 的收益到底来自 Scene Graph，还是来自“结构变化建模”本身。**

---

## Baseline 5：Fixed Delta

假设：

```text
Δ = Δ_e
```

同一种 Edit 使用固定变化向量。

与：

```text
Δ = F(e, s)
```

进行比较。

这是 Context-conditioned Transformation 最关键的消融。

---

# 26. 核心消融

至少包含：

### Ablation 1

去掉 Scene Graph Edit。

验证 Edit supervision 是否必要。

### Ablation 2

只使用 Hard Negative。

验证是否只是 hard-negative effect。

### Ablation 3

只做 State-level Alignment。

验证：

```text
状态级结构学习
vs
变化级结构学习
```

### Ablation 4

```text
Fixed Delta
vs
Context-conditioned Delta
```

验证相同 Edit 是否应该依赖 context。

### Ablation 5

后期去掉 Edit Composition。

验证多步组合规律是否有效。

---

# 27. 后续代码开发顺序

代码开发必须跟阶段验证顺序一致。

## Step 1：第一阶段数据 Manifest

先整理已有公开数据：

```text
PUG: SPAR
PUG: Animals
PUG: ImageNet
CLEVR val
CoGenT val_A / val_B
```

生成统一 manifest。

此时不生成新图像。

## Step 2：冻结 Baseline Encoder

先实现统一 VLM 接口：

```text
encode_image
encode_text
```

首选：

```text
CLIP ViT-L/14
```

关键现象明确后再接：

```text
SigLIP2
```

所有模型只 cache feature，不训练。

## Step 3：运行第一阶段 E1

完成：

```text
PUG: SPAR zero-shot structural diagnosis
CLEVR hard-negative 2AFC
Binding-style static diagnosis
```

## Step 4：运行第一阶段 E2

完成：

```text
PUG: Animals matched pairs
Single-factor response Delta
Same-transition vs Different-factor
Nuisance reference
```

## Step 5：运行第一阶段 E3

完成：

```text
CoGenT A / B difficulty diagnosis
Sphere control
Per-combination analysis
```

然后先汇总第一阶段结果，决定是否继续。

## Step 6：第二阶段 Controlled Edit 数据

只有第一阶段支持后，才生成：

```text
Atomic Controlled Edit
Relation Change
Binding Swap
```

并允许加入 EditCLEVR。

## Step 7：第二阶段 Structural Delta

完成：

```text
Δ_v / Δ_t
Same / Different Edit
Linear Edit Probe
Cross-modal Delta
Held-out Context
Controlled Unseen Composition
```

## Step 8：完整 GraphDelta

机制成立后，再实现：

```text
Edit Encoder
Context-conditioned Transformation
Delta Prediction
Cross-modal Delta Alignment
Edit Discrimination
```

这时才开始 GraphDelta 的训练和方法比较。

# 28. 后续工作的阶段划分

## 第一阶段：问题验证

目标：

> 证明研究问题真的存在。

阶段划分以《GraphDelta_完整预实验方案_统一修订版》为准：

```text
固定公开数据（PUG: SPAR / Animals、CLEVR val、CoGenT val）
一个或多个冻结预训练 VLM baseline（CLIP 必做，SigLIP2 建议复现）
2AFC + Matching Margin
单因素 representation response + nuisance 基线
CoGenT 难度不对称诊断
不重新渲染、不自建 Edit Pair
```

CLEVR Edit Pair（重新渲染的 before / after）属于第二阶段。

---

## 第二阶段：机制验证

目标：

> 证明表示中存在一定的 Structural Delta 信息。

完成：

```text
Delta Analysis
Same vs Different Edit
Linear Edit Classification
Held-out Context Transfer
```

---

## 第三阶段：方法验证

目标：

> 用 GraphDelta 主动塑造 Structural Transformation Geometry。

完成：

```text
Edit Encoding
Context-conditioned Delta
Cross-modal Alignment
Unseen Composition
```

---

## 第四阶段：扩展验证

目标：

> 证明方法不仅对 CLEVR 有效。

扩展：

```text
真实图像 benchmark
更多 Edit
多步 Edit Composition
不同 VLM backbone
```

---

# 29. 每个阶段都应该回答一个明确问题

整个工作不要变成“不断往模型里加模块”。

每一步都必须回答一个问题。

### Stage 1

> 现有模型真的对 Relation / Binding 等结构变化不敏感吗？

### Stage 2

> 在 Controlled Edit 下，embedding 中到底有没有可识别、跨模态、可迁移的 Structural Delta？

### Stage 3

> 显式学习这种变化规律，是否比普通 fine-tuning、hard negative 和静态结构对齐更有效？

### Stage 4

> 学到的规律是否真正可以迁移到训练中没有出现的组合？

只要这四个问题形成闭环，论文逻辑就是完整的。

---

# 30. 最终创新点

后续论文建议只强调三个核心创新。

## 创新一：从“结构状态”转向“结构变化”

现有工作主要研究：

```text
当前结构是什么？
```

GraphDelta 进一步研究：

```text
结构从 A 变成 B 时，
统一表示应该怎样变化？
```

研究对象从：

```text
Graph State
```

扩展为：

```text
Graph Structural Transformation
```

---

## 创新二：上下文相关的结构变化表示

不把：

```text
red → blue
```

简单视为固定全局方向。

而是建模：

```text
Δ = F(Edit, Context)
```

使同一种 Edit：

```text
具有共同规律
```

但：

```text
允许在不同语义场景中有不同具体响应
```

---

## 创新三：用结构变化规律实现未见组合迁移

GraphDelta 最终不是只追求 seen data 上更高的 matching score。

真正验证的是：

> primitive 已经见过、完整组合未见过时，模型能否把学到的结构变化规律迁移过去。

包括：

```text
Unseen Attribute-Object
Unseen Relation
Unseen Binding
```

---

# 31. 与 BindRepair 的关系

原 BindRepair 关注：

> 图像和文本中可能已经存在绑定信息，但跨模态匹配没有正确使用这些信息。

GraphDelta 将问题进一步推进：

> 不只研究已有 binding 信息如何用于 matching，而是研究表示空间本身如何响应属性、关系和绑定结构变化，并学习这种变化规律。

可以理解为：

```text
BindRepair
研究：
binding 信息有没有被正确利用？

        ↓

GraphDelta
研究：
binding / relation 变化以后，
representation 应该怎样系统变化？
```

因此 GraphDelta 更偏向：

> **Representation Learning**

而不是单纯：

> **Matching Repair**

---

# 32. 与 MCDisp-Align 的关系

MCDisp-Align 研究：

> 一张图像对应多个合理描述时，如何保留语义范围和描述多样性。

GraphDelta 研究：

> 当图像内部的对象、属性、关系和绑定发生变化时，统一表示如何产生正确并可迁移的结构响应。

二者都属于：

> **复杂视觉语义条件下的统一视觉—语言表示学习。**

可以形成连续研究主线：

```text
MCDisp-Align
多描述条件下：
如何表示语义范围和多样性？

        ↓

BindRepair
细粒度语义已经存在时：
为什么绑定信息没有被正确用于匹配？

        ↓

GraphDelta
语义结构发生变化时：
统一表示应该怎样变化，
并如何把这种规律迁移到新组合？
```

---

# 33. 当前最重要的工作

目前不要直接实现完整 GraphDelta。

第一阶段的执行细则以《GraphDelta_完整预实验方案_统一修订版》为准。当前最重要的是完成该方案的 Step 1–Step 7：

```text
1. 整理 PUG: SPAR / Animals / CLEVR val / CoGenT val。

2. 以 CLIP ViT-L/14 为首选冻结 baseline，
   关键结果明确后可加入 SigLIP2 等冻结 baseline 做复现，
   完成 Feature Extraction。

3. PUG: SPAR 结构诊断（复现官方协议，
   与已发表数字对齐）。

4. CLEVR 文本 hard-negative 结构诊断
   （含负例有效性检查）。

5. PUG: Animals 单因素 representation response 分析
   （按 transition_key 比较、含 nuisance 基线与聚类 bootstrap）。

6. CoGenT 难度不对称诊断（含 sphere 对照）。

7. 汇总结果，决定是否进入第二阶段。
```

严格的 Controlled Edit Pair、Relation / Binding before-after、Linear Edit Probe、Held-out Context / Unseen Composition Split 都属于第二阶段，在第一阶段验证问题成立后再做。

如果这四步结果足够明确，后续方法设计就会非常自然。

如果这四步不成立，也能够尽早调整研究方向，而不是在完整模型训练之后才发现核心假设有问题。

---

# 34. 整个工作最简逻辑

最后，把整个 GraphDelta 压缩成一条逻辑链：

```text
现有视觉—语言模型
已经能够识别对象 / 属性 / 关系
        ↓
但“识别这些概念”
不等于
“正确表示它们之间的结构”
        ↓
关系 / Binding 变化以后，
全局 embedding 可能仍然非常相似
        ↓
因此不能只研究
“当前场景是什么”
        ↓
进一步研究
“当前场景是怎么变的”
        ↓
使用 Scene Graph Edit
明确表示结构变化
        ↓
提取并学习
Structural Delta
        ↓
让视觉变化和文本变化
表达同一种结构规律
        ↓
学习
Context-conditioned Transformation
        ↓
不记忆具体 tuple，
而学习可以迁移的变化规律
        ↓
应用到
Unseen Composition
```

---

# 35. 一句话概括

> **GraphDelta 研究视觉—语言统一表示如何响应语义结构变化：利用 Scene Graph Edit 明确对象、属性、关系和绑定发生了什么变化，并学习上下文相关的跨模态表示变化规律，使模型不仅能够表示“当前场景是什么”，还能够理解“当前场景是怎么变化的”，最终将这种结构变化规律迁移到训练中没有出现过的新组合。**

---

# 36. 后续工作的核心原则

后续所有方法、代码和实验都应遵循以下原则：

1. **先证明问题，再设计复杂模型。**
2. **优先研究变化规律，而不是继续堆叠静态结构模块。**
3. **Scene Graph 主要作为结构变化监督，不作为推理依赖。**
4. **第一阶段固定公开数据和评估协议；模型可以增加，但新增模型只能作为冻结 baseline。**
5. **每个新增模块都必须对应一个明确研究问题。**
6. **严格的 unseen composition split 是第二阶段机制验证和第三阶段方法验证的关键，不强加给第一阶段冻结 baseline 诊断。**
7. **最终评价重点不是“训练集上提升多少”，而是“变化规律能否迁移”。**

后续代码生成、实验实现和论文撰写均以本框架为准。
