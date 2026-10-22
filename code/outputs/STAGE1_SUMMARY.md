# GraphDelta 第一阶段总结（收尾版）

- 日期：2026-10-22；模型：CLIP ViT-L/14（冻结，唯一 backbone）。
- 数据与协议：见 `code/README.md`。规范结果目录 `outputs/tables` = 预注册 seed 0
  （配置 `seeds: [0,1,2]`）；seed 1/2 快照在 `outputs/tables_seed{1,2}`。
- 本文档 = 四个实验结果 + 三项收尾分析（E3 逐单元、多种子稳定性、§67 五图）
  + 第二阶段建议。所有数字直接来自 CSV，未四舍五入处见原表。

## 1. 四个实验的主结果

| 实验 | 关键数字（95% CI，图片聚类 bootstrap） | 结论 |
|---|---|---|
| E1 自建 2AFC（CLEVR val） | color 0.676 [0.658,0.694] > object 0.561 > material 0.557 > size 0.530 > relation 0.516 [0.495,0.536] ≈ binding 0.508 [0.488,0.526] | binding 与 relation 与随机无差别；color−binding 差 0.168 [0.141,0.195] |
| E1 自建 2AFC（PUG:SPAR） | scene 0.897 [0.885,0.908]；attribute 0.513（含 0.5）；relation 0.466 [0.448,0.483]；position 0.398 [0.382,0.413] | position / relation 显著低于随机 |
| E1 SPAR 官方协议 | hard-negative 二选一 0.48–0.56（全部 ≈ 随机）；two_objects_attribute 检索 top1 0.087–0.116 | 官方口径下双对象属性绑定同样失败 |
| E2 PUG:Animals Δ | environment：‖Δ‖=12.86（nuisance 5.18，比值 2.48），Norm Gap 7.67 [7.61,7.73]，Transition Gap 0.642 [0.637,0.647]；camera：2.10 [2.05,2.15] / 0.016 [0.013,0.020]；texture：−0.022 [−0.072,0.024] / 0.111 [0.107,0.115]；size：−0.359 [−0.409,−0.310] / 0.050 [0.047,0.054] | environment/camera 的 Δ 超出 nuisance 且跨 context 可重复；texture/size 范数不超过 nuisance、只有相似性判据为正 |
| E3 CoGenT | pooled 等权 A−B：cube/cylinder −0.007 [−0.026,0.014]；sphere +0.003 [−0.017,0.023] | 聚合口径无难度不对称；但见 §3 逐单元改写 |

## 2. 对照方案 §68–72 的判定

| §69 条件 | 判定 |
|---|---|
| 1. Relation/Binding 明显弱于 Object/Attribute | 成立（两个数据集一致） |
| 2. SPAR 与 CLEVR 趋势一致 | 成立 |
| 3. Δ 响应超出 nuisance 且跨 context 可重复 | 部分成立（environment/camera 成立；texture/size 仅相似性判据成立） |
| 4. CoGenT 难度不对称 | 聚合口径不成立（§3 表明该口径本身有局限） |

综合：主体落在情况 A（适合进入第二阶段）；E2 的 texture/size 部分对应情况 B
（第二阶段用更严格的 Controlled Edit 检验）；E3 对应情况 C，且 §3 的逐单元分析
把"无不对称"改写为"单元级效应被聚合抵消"。情况 C 按方案只削弱"组合难度不对称"
这一条证据，不阻塞进入第二阶段。

## 3. E3 逐单元分析（新增，`scripts/analyze_e3_units.py`）

方法：`e3_items_cogent.jsonl`（12000 条 item 级结果）→ 按 (shape_group, color)
16 个主单元算 A−B 题型等权 gap + 图片聚类 bootstrap CI（与主指标同函数同口径）；
总 gap 复算与 `e3_difficulty_gap.csv` 一致到 1e-16。16 个单元各 95% CI，
属多重比较，只做描述。

结果（`e3_unit_gaps.csv`）：

- 6 个单元 CI>0（A 更容易），6 个 CI<0（B 更容易），4 个不显著；
  gap 范围 [−0.552, +0.431]，std 0.244。
- 方向与官方调色板完全对应：正 gap 的 {gray, blue, brown, yellow} 恰为
  条件 A 的 cube 调色板（B 的 cylinder 调色板）；负 gap 的 {red, purple, cyan}
  （green 不显著）恰为条件 B 的 cube 调色板。
- 机制：cube/cylinder 的每个 (color, shape) 组合按设计只出现在一个条件里；
  按颜色聚合的单元 gap 实际比较的是"A 条件该颜色在 cube 上"vs"B 条件该
  颜色在 cylinder 上"。8 色按调色板 4/4 对半分成方向相反的两组，等权平均
  后 pooled gap ≈ 0 是算术结果，不等于"各组合难度相同"。
- sphere 对照（两条件全色）单元效应小得多（|gap| ≤ 0.13）。
- 稳定性：16/16 单元 gap 点估计在三种子下同号（`seed_stability.csv`）。

E3 结论改写：**"CoGenT A/B 聚合难度无不对称"应表述为"单元级组合难度差异
很大（最大 ±0.55）、方向沿调色板分裂、被 4/4 对半等权平均抵消"**。该发现
支持"组合难度存在且不可被聚合指标观察"，与 E1 的结构缺陷证据相互独立；
仍不构成 seen/unseen 结论（方案 §51，留给第二阶段）。

## 4. 多种子稳定性（`scripts/summarize_seed_stability.py`）

seeds 0/1/2 全量重跑四个实验（规范目录已恢复为 seed 0）。
`seed_stability.csv`：115 项判定，109 项稳定，6 项翻转：

| 项 | 三种子值 | 翻转内容 |
|---|---|---|
| clevr/relation acc | 0.516/0.505/0.521 | CI 含 0.5 → 不含（above） |
| clevr/size acc | 0.530/0.517/0.516 | CI 不含（above）→ 含 0.5 |
| pug_spar/relation acc | 0.466/0.485/0.471 | CI 低于 0.5 → 含 0.5 |
| clevr size−material gap | −0.026/−0.040/−0.050 | ns → neg |
| pug_spar attribute−relation gap | 0.047/0.027/0.020 | pos → ns |
| e2 texture norm gap | −0.022/−0.041/−0.093 | ns → neg |

6 项全部是点估计贴着 CI 边界的临界项（围绕 0.5 或 0 摆动），点估计本身
跨种子变化很小。所有结构性结论不变：binding 含 0.5、position 低于随机、
scene 0.9、environment Δ 远超 nuisance、E3 聚合 gap 含 0、E3 单元方向
16/16 同号、SPAR 官方 hardneg ≈ 随机。写作时对 relation/size/attribute
这三个贴近 0.5 的格子只写"约随机水平"，不写显著方向。

## 5. §67 五图（`outputs/figures/`，`scripts/make_figures.py`）

| 文件 | 读法 |
|---|---|
| fig1_2afc_accuracy.png | 两数据集 factor 降序 + CI + 0.5 随机线；结构类因素贴线、scene 远离 |
| fig2_matching_margin.png | 同口径 margin；margin 与 accuracy 排序一致 |
| fig3_delta_magnitude.png | 四因素 ‖Δ‖（raw / normalized 两面板）+ nuisance 虚线；environment/camera 高于虚线，texture/size 贴线 |
| fig4_delta_similarity_ladder.png | 每因素六层相似性阶梯（same-transition 带 CI）；environment 的 same-transition 0.644 显著高于 random 0.002 |
| fig5_cogent_ab.png | (a) A vs B 各题型 accuracy + pooled gap CI；(b) 16 个逐颜色单元 gap + CI，蓝=A 更容易 / 红=B 更容易 / 灰=ns |

合并 PDF：`GraphDelta_stage1_figures.pdf`。配色经 CVD / 对比度验证器检查通过。

## 6. 下一步（已具备条件）

1. **进入第二阶段**：BindRepair pilot（代码已完成并通过冒烟测试，阶段 0a 组合
   审计已跑完）。E1 结果直接支撑其前提（binding 2AFC ≈ 随机、官方 hardneg ≈
   随机）；E2 的 environment 主导（2.48×nuisance）被其 (矩形×绑定×环境) 设计覆盖。
2. **阻塞项**：PUG:SPARE 数据（CSV + 四图片目录）与 CLIP ViT-B/32 权重尚不在
   本机。到位后按 README 阶段表执行 0b → 1 → 2 → 3 → 4，每阶段有退出点。
3. **不做**：SigLIP2 对照（本地无权重，结论限定 CLIP）；第三阶段方法验证
   （等第二阶段结论）。

## 附：本轮新增 / 变更文件

- 代码：`scripts/analyze_e3_units.py`、`scripts/summarize_seed_stability.py`、
  `scripts/make_figures.py`（新增）；`src/experiments/e3_cogent_diagnosis.py`
  （仅新增 item 级 jsonl 落盘，重跑 seed 0 后四个既有 CSV 与改前逐字节一致）。
- 表格：`e3_items_cogent.jsonl`、`e3_unit_gaps.csv`、`seed_stability.csv`；
  `tables_seed{0,1,2}` 三份种子快照。
- 图：`outputs/figures/` 五张 PNG + 合并 PDF。
- 日志：`outputs/logs/`（追加式，含三种子全部运行记录）。
