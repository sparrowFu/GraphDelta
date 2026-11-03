# STAGE1_SUMMARY — 预实验阶段总结与阶段判定

- 日期：2026-11-03；数据基准：git `2ffa587`（E1 平衡采样重跑 10-28 16:57–16:59；E2 重跑 10-28~10-30；E3、seed_stability、五图重跑 10-30 09:27–09:35）
- 详细依据：`outputs/analysis/预实验复查与再分析_平衡采样后_2026-11-03.md`（含 HTML 版）；审计 `outputs/analysis/audit_e1_sampling_2026-11-03.{md,json}`（检查 A–G 全过）
- 判定口径：方案 §66（CI 是否含 0/0.5，5pp 实践参考）与 §68–§72（情况 A/B/C/D）

## 判定结论

**建议进入第二阶段，路线以小规模受控编辑（Controlled Edit）验证为主。**

总体属于**情况 A 的不完整版本**：E1 条件完全满足；E2、E3 条件部分满足（各满足一半左右）。没有触发情况 C（CoGenT 完全无差距）或 D（全部很好）的排查分支。

## 三个核心问题的回答（方案 §3）

### 问题一：冻结 baseline 对结构因素的区分是否不均衡 —— **成立，且比采样修复前更干净**

平衡采样（按 pos_value 分层等额 + position 视角翻转）剥离词偏好后的读数：

- 真辨别（9/9 run CI 高于 0.5）：scene 0.749/0.887/0.990；color 0.683–0.705；object 0.575–0.639（cube 侧 0.88–0.92、cylinder 侧 0.14–0.38，值严重不均）；material 三模型显著高于 0.5，但 vitb32 偏好为主（分侧 metal 0.136 / rubber 0.918）。
- **可证明的零辨别**：size 分侧 ≈1.000/≈0.000（恒选 small）、总体恰 0.500；clevr/relation 恒选方位词（vitb32 选 left、vitl14 选 right）；pug_spar/position 总体 0.489–0.503。三者旧读数的"略高于随机"全部是词偏好伪影。
- 免疫因素干净失败：binding 0.495–0.515（9/9 含 0.5）——全局零样本打分未能利用绑定信息；vitb32 attribute 0.399（3/3 低于 0.5，系统性反向）。
- 颜色−绑定差距：三模型 3-seed 均值 17.8–20.7pp（逐 run 17.2–21.1pp），9/9 CI 高于 0。SPAR 官方协议 hardneg（0.477–0.584）与自建协议互相印证。

### 问题二：单因素 Δ 是否有可分析规律 —— **部分成立（environment 完整成立，其余弱或依赖数据集）**

- environment：transition similarity gap（z）0.669/0.641/0.726，CI 全部远大于 0；Δ 范数超出 nuisance（raw +4.82/+7.86/+4.54）——"具体转换跨 context 可复现"成立。
- 但规律严格依赖**同一具体转换**（同转换跨 context 0.643 vs 不同转换 0.006）；texture 数据集依赖（Animals 0.117–0.263 / ImageNet 0.222–0.354）；size/camera 弱；SigLIP2 的 nuisance Δ 自身有方向性（余弦 ≈0.41）。
- 不支持"统一背景轴"表述，支持"逐转换可复现"。

### 问题三：CoGenT A/B 搭配难度是否系统不对称 —— **2/3 模型成立**

- vitb32：+3.97/+5.68/+4.35pp（3/3 CI>0），sphere 对照 3/3 含 0 ✓
- siglip2：+3.58/+6.81/+5.21pp（3/3 CI>0），sphere 对照 seed0 +2.33 [0.45,4.36] CI>0（例外，audit_e3 已标记）
- vitl14：−0.65/−0.93/+0.73pp（含 0、符号不稳），无整体差距
- 附带发现：换形题形状不对称（vitb32/vitl14 cube≈88% vs cylinder≈32–59%，SigLIP2 反向）；E3 只支持"条件相关、模型相关的难度异质性"，不支持"未见组合泛化下降"表述。§71 排查清单（负例太易/颜色拉平/sphere 上限）已在 audit_e3 过一遍。

## 情况 A 四条件对照（§68）

| 条件 | 判定 | 依据 |
|---|---|---|
| Relation/Binding 明显弱 | ✓ 完全 | binding 零辨别（免疫、9/9 含 0.5）；relation/position/size 可证明零辨别 |
| SPAR 与 CLEVR 趋势一致 | ✓ 完全 | 两套数据、两套协议（自建+官方）一致给出 position/relation ≈0.5 |
| E2 transition 规律超出 nuisance | 部分 | environment 成立（CI>0、超 nuisance）；texture/size/camera 不完整 |
| CoGenT cube/cylinder 不对称且 sphere 无 | 部分 | 2/3 模型成立；vitl14 不成立；siglip2 有一个 sphere 例外种子 |

## 保留意见

1. 种子只覆盖抽样/负例/自助法稳定性，不是独立训练；CI 不构成对模型总体的保证。
2. 逐单元/逐值比较未做多重比较校正；本总结只引用 9/9 一致的判定。
3. 记录缺口：`e3_items_cogent.jsonl` 缺 negative_color/negative_shape；E2 配对清单未落盘（影响可复核性，不影响当前结论）。
4. 方案文档无"采样平衡"条款（只在 README Step 3 与代码 docstring），建议增补使规范层级一致。
5. vitb32 attribute 显著低于随机（0.399）的机制未解释，建议第二阶段一并分析。

## 第二阶段入口建议

- 优先做小规模受控编辑验证（情况 B 路线）：对 relation/binding 的失败做"视觉 Δ 是否不足"的直接检验（第一阶段未测，见深化分析 §9 证据分级）。
- 受控编辑设计需同时覆盖 environment（E2 已有强规律的对照因素）与 relation/binding（E1 失败因素），以区分"表示不编码"与"打分不利用"。
- 相关材料：`outputs/analysis/预实验结论深化分析_证据强度与机制边界_2026-10-28.md` §9–10（其 E1 数字已被 2026-11-03 复查报告取代，E2/E3 与证据分级仍有效）。
