# E3 独立审计笔记（2026-10-28）

本笔记仅核对本地现有全量结果与实现，没有重新运行模型。主表来自 `../{model}/seed_{0,1,2}/tables/`；不使用 `_smoke` 结果。

## 1. 原报告需要纠正的事实

原报告 §2.9 的“CoGenT 规定 A 条件 cube/cylinder 只配 red、blue，B 条件配其余六色”不正确。`../../configs/e3_cogent.yaml` 中：

| 条件 | cube | cylinder | sphere |
|---|---|---|---|
| A | gray、blue、brown、yellow | red、green、purple、cyan | 全部八色 |
| B | red、green、purple、cyan | gray、blue、brown、yellow | 全部八色 |

因此每一种颜色在 A/B 中对应的非球形正例形状发生互换。按颜色的 A−B 不能解释成控制形状后的纯颜色效应。

ViT-L/14 主 gap 的 seed 0/1/2 分别为 −0.651、−0.926、+0.733pp，三个 CI 均含零，点估计符号不一致。“主要结论稳定”可以成立，但不宜写成所有点估计方向一致。

## 2. 新发现：形状方向的不对称解释了大量颜色单元差异

从 `e3_items_cogent.jsonl` 按 condition、shape、negative_type 聚合得到以下换形题准确率（seed 0）：

| 模型 | A cube | A cylinder | B cube | B cylinder |
|---|---:|---:|---:|---:|
| CLIP B/32 | 89.33% (754) | 58.33% (734) | 82.69% (698) | 54.10% (752) |
| CLIP L/14 | 87.93% (754) | 33.20% (734) | 88.35% (698) | 33.40% (752) |
| SigLIP2 | 59.28% (754) | 77.27% (734) | 57.25% (698) | 69.66% (752) |

括号为 item 数，每个 item 的准确率先对七模板平均，模板不是七倍独立样本。

这个结构跨 seed 稳定。L/14 的 cube 换形题 A/B 准确率跨三 seed 为 87.93–89.49%，cylinder 为 31.70–33.40%。B/32 也明显偏向 cube；SigLIP2 的方向相反，cylinder 高于 cube。

所以 L/14 的 red 单元 A−B 为负、yellow 为正，与 red 从 A cylinder 变成 B cube、yellow 从 A cube 变成 B cylinder 一致。它是强烈的正例形状不对称与颜色交互的综合现象，不能只解释为某些颜色的组合能力特别差。现有数据也不足以把它完全归因于文本先验：场景共现、负例候选集合和渲染可见性都可能参与。

在 L/14 内还存在形状相同后的颜色差异：A cylinder 换形题 red 16.15%、green 53.37%；A cube 中 brown 77.72%、yellow 97.96%。故“形状不对称”也没有解释全部异质性。

## 3. 整体均值稳定与否，须结合球形对照

题型等权 cube/cylinder A−B：

| 模型 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| CLIP B/32 | +3.965pp [1.898,5.968] | +5.678pp [3.554,7.839] | +4.346pp [2.364,6.521] |
| CLIP L/14 | −0.651pp [−2.646,1.418] | −0.926pp [−3.218,1.274] | +0.733pp [−1.377,2.847] |
| SigLIP2 | +3.580pp [1.586,5.705] | +6.813pp [4.598,8.930] | +5.212pp [3.272,7.132] |

B/32 与 SigLIP2 的 A>B 在三 seed 都成立，但其幅度不应被写死为约 4pp；SigLIP2 seed 1 为 6.8pp。

sphere 对照：B/32 为 −0.200/−0.895/−1.010pp（CI 均含零）；L/14 为 +0.260/−0.591/+0.140pp（均含零）；SigLIP2 为 +2.331/−0.041/+0.153pp，仅 seed 0 CI [0.452,4.364] 不含零。因此 SigLIP2 seed 0 的整体非球形 gap 不能全归因于组合约束，存在对照组的 A/B 难度变化。可把“非球形 gap − 球形 gap”的差中差作为下一步补充统计，但需要按图片联合 bootstrap 给出 CI；不能直接相减两个独立 CI。

球形也并非完全控制：其换形候选因场景中是否存在目标颜色的 cube/cylinder 而变化，而这又受 A/B 调色板约束。所以 sphere 控制的是正例形状和颜色覆盖，而不是全部场景与负例分布。

## 4. 能复核的例题与不能复核的信息

L/14 seed 0：

- `CLEVR_valA_003948.png` red cylinder：换色 item_acc=5/7、margin=+0.0010524；换形 item_acc=0/7、margin=−0.0516321。
- `CLEVR_valA_011076.png` yellow cube：换形 7/7，margin=+0.0803726。
- `CLEVR_valB_005512.png` red cube：换形 7/7，margin=+0.0752920。
- `CLEVR_valB_005890.png` yellow cylinder：换形 0/7，margin=−0.0585204。

有意思的是同一 `valA_003948` 换形题在三个模型均 0/7，margin 分别为 B/32 −0.0282425、L/14 −0.0516321、SigLIP2 −0.0239208；但是 red cylinder 组整体在三个模型差异很大（57.14%、16.15%、72.40%）。因此单个共同失败样例不能代表总体模型排序。

本地没有原始 CoGenT scenes、图像或特征缓存。现有 E3 JSONL 不含 content_pos/content_neg、negative_color、negative_shape、目标 object_index，因此仅凭 JSONL 无法独立重建“具体负例是 red cube”的断言，也无法检查全部视觉可见性、负例难度及文本先验。现有样例评分可独立确认，具体负例词值只能标为原报告依据外部标注复原。

## 5. 指标和推论边界

- `e3_cogent.csv` pooled accuracy 是 item 加权平均；`e3_difficulty_gap.csv` 主指标按换色/换形两题型等权，二者不能混写。
- `e3_unit_gaps.csv` 的 pooled_equal_type 行中 gap 是题型等权差，而 acc_A/acc_B 字段仍是 pooled item 均值；直接用那两个字段相减一般不会等于 gap。例如 L/14 red pooled 差为 −54.116pp，等权主 gap 为 −55.214pp。
- 每个 (condition,shape_group) 抽 3000 items，总共每模型每seed 12000 items；不是每种颜色形状各 3000。
- 同 seed 的题目生成与抽样不依赖模型，三模型 item 结构一致；不同 seed 变化的是对象/负例/题目抽样。冻结模型没有训练随机性，三个 seed 不是三个独立训练模型。
- 单元 CI 是逐单元探索性结果，脚本明确没有做多重比较校正；只能说明较强描述信号，不能把所有 CI 不含零的单元合称严格联合检验通过。
- A/B 不是匹配的同一场景受控编辑，不能把 A−B 解释成单一组合因素的因果效应。模型也没有 train-on-A，所以不能称为 CoGenT seen/unseen 泛化准确率。

## 6. 对 GraphDelta 的具体含义

E3 强力支持“必须分层诊断和设计成对编辑”；它没有测视觉变化或跨模态 Delta，更没有证明 GraphDelta 有效。下一步可从同一场景构造 red cylinder↔red cube、yellow cylinder↔yellow cube 的对称编辑，平衡两个方向、同等干扰，保存完整负例和 object_index，并与文本 Delta 配对。若静态打分的形状先验很强，需加入文本偏置对照以及双向编辑，防止视觉 Delta 改进其实只是打分先验变化。
