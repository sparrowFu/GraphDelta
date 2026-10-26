# GraphDelta 审查问题核实与修复报告（2026-10-26）

对照《GraphDelta_代码审查报告_2026-10-26》逐条核实当前代码（基线 `422b1c9`），
属实的问题已修复并用测试验证。本报告为该审查报告的核实结论与修复记录。

**状态更新：本文件记录 A1–A7 原触发路径的实现修复，并保留当时“14 个实现/文档文件、39 项检查、2 个合成数据集集成流程”的验证记录。这些记录不能等同于所有逻辑问题关闭或五数据集、三模型的真实端到端验收。后续独立复核已确认原 NaN-CI、目录身份、new/legacy 拒绝及比较上限等修复有效，同时发现 ImageNet 实例覆盖和 SPAR 逐环境汇总等问题。最新状态以[独立复核修订版](D:/code/GraphDelta/GraphDelta_修复复核报告_2026-10-26.md)为准。**

第 3 节稀疏比较风险按样本量披露处理，不改变联合均匀抽样定义；该处理不保证实际样本量足以支持结论。

## 1. 逐条核实与修复

### A1 [P1] PUG_ImageNet 未接入 —— 入口已接通，实例约束未完成验收

核实：`extract_embeddings.py` 的 `--dataset` 无 `pug_imagenet`；`e2_change_diagnosis.py`
的 metadata、特征抽取、nuisance、dataset 标签全部写死 `pug_animals`；`run_e2.py`
无数据集参数；`pug_imagenet.py` 的 `META_COLUMNS` 缺 3 个旋转字段。

修复（按方案 §14 的因素选择：background/size/texture/camera）：

- `src/data/pug_imagenet.py`：`META_COLUMNS` 补 `character_rotation_yaw/roll/pitch`；
  统一语义列 environment←world_name、texture←character_texture、size←character_scale
  （原始数值）、camera←camera_yaw；固定 context 包含 character_label、camera_pitch、
  camera_roll、scene_light 和全部旋转字段，约束这些字段不变；但未保留并固定
  character_name，不能据此保证同一具体物体下仅目标因素变化，见最新复核 B1。
- `src/experiments/e2_change_diagnosis.py`：改为 `run(dataset=...)` 分发（DATASETS
  注册表），metadata / 抽取 / nuisance / caption / 输出标签全部按数据集选择；
  ImageNet 结果写 `e2_delta_stats_pug_imagenet.csv`（Animals 沿用原文件名，
  下游 summarize/make_figures 不受影响）。
- `scripts/run_e2.py` 加 `--dataset {pug_animals,pug_imagenet}`；
  `scripts/extract_embeddings.py` 加 `pug_imagenet` 抽取分支。
- `src/data/pair_matcher.py`：`build_factor_pairs` 加 `meta_cols` 参数（默认
  Animals 五列，行为不变）；canonical 顺序支持"未声明则按观测值排序"的回退。
- `configs/e2_change.yaml`：`factors` 与 `canonical_orders` 按数据集隔离
  （Animals 声明标签顺序；ImageNet 不声明，按观测值排序——不能沿用
  small/medium/large 标签顺序匹配 ImageNet 的原始数值 0.7/1.0/1.3）。
- `src/data/captions.py` 加 `pug_imagenet_caption`，当前只描述 texture + environment。
  原理由“character_label 是编号，无法构成自然语言”不成立：官方字段为
  African_crocodile 等字符串，可转为空格分词的类别名，见最新复核来源。

当时验证：合成 parquet + 假 encoder 跑通 ImageNet E2 流程，覆盖类别/姿态/
灯光/其余相机角固定、数值因素 source<target、CSV 标签和 n_cmp_* 列。
这不能证明同类多实例的严格单因素配对；最新真实元数据连续样本已发现实例覆盖压缩。

### A2 [P1] 旧缓存被错误认证 —— 属实，已修复

核实：`cache.py` 的 `_check_protocol` 在无 `protocol.json` 时直接写入当前协议，
不检查目录里是否已有 NPZ；`extract_embeddings.py` 先构造缓存（触发协议校验）
再处理 `--force`，协议不匹配时删除分支不可达。

修复：

- `cache.py`：只有**空目录**（无任何 NPZ）才首次登记协议；无协议但已有 NPZ
  的目录直接报错，提示用 `--force` 重建或手动移入历史目录。
- `extract_embeddings.py`：`--force` 移到缓存构造**之前**执行，作为整套重建
  入口——协议匹配时删本数据集图像缓存 + 文本缓存（这是当前保守实现，
  不是同协议图像重抽必须清除全部文本的逻辑要求）；协议不匹配或无协议时删整个模型缓存目录（目录内
  新旧特征无法区分，部分保留会重新引入本问题）。

验证：旧 NPZ 无协议被拒绝、空目录登记、不匹配报错、force 三种情形（匹配/
不匹配/无协议）行为全部符合预期；TextCache 命中/补抽行为不变。

### A3 [P2] 裁剪面积语义 —— 属实，已修复（按用户选择：面积比例）

核实：`delta.py` 注释声明"面积比例"，实现两边各乘 s，实际保留面积 s²，
`[0.85,1.0]` 只剩 72.25%–100%。

修复：边长缩放系数改为 `sqrt(s)`，向上取整（保证实际面积不低于采样的 s，
1000×1000、s=0.85 时得 922×922 = 85.01% 面积）。`[0.85,1.0]` 现在表示保留
85%–100% 面积，与注释和配置意图一致。配置注释同步更新。

验证：s=0.85 固定值与 200 次随机采样，面积全部落在 [0.85, 1.0]。

**影响：视觉 nuisance 扰动变轻，E2 的 nuisance 范数/相似度与 Factor/Nuisance
Norm Gap 需按新语义重算；原图像特征可复用，无需重抽。**（本仓库输出已在
上一轮清理中删除，无历史结果需要作废。）

### A4 [P2] 缺失 CI 判为"不显著且稳定" —— 属实，已修复

核实：`ci_vs_ref`/`ci_sign` 不检查有限性，NaN 落入 `contains`/`ns`；E3 单元
gap 的点估计 NaN 时 `{np.sign(v)}` 集合仍可能长度为 1。

修复：两个分类函数对非有限 CI 返回 `invalid`；新增 `stable_classes()`——
分类一致**且**无 invalid 才判 stable；E3 单元 gap 加点估计有限性检查；
SPAR 官方描述行在含缺失值时标注"含缺失值"。

验证：`ci_vs_ref(NaN)`/`ci_sign(NaN)` 返回 invalid；全 invalid、invalid+ns
混合均不判稳定；正常一致分类仍判稳定。

### A5 [P2] 显式目录绕过身份校验 —— 属实，已修复

核实：`verify_run_meta` 在无 `run_meta.json` 时直接放行；`make_figures.py`
的 `--tables-dir` 可指向任意无元数据目录，标题仍取请求的 `--model/--seed`。

修复：`io.py` 新增 `verify_tables_identity()`——有 `run_meta.json` 则核对；
无则只接受已知旧布局目录（`outputs/tables`、`outputs/tables_seed{1,2}`，
属主固定为 clip_vitl14 对应种子）且请求身份必须与属主一致；其余无元数据
目录一律拒绝。`make_figures.py` 改用该校验。

验证：run_meta 匹配/不匹配、旧布局属主匹配/种子不符、未知目录、目录外
路径共 5 种情形行为符合预期。

### A6 [P2] 跨协议种子稳定性 —— 属实，已修复

核实：`summarize_seed_stability.py` 检测到新旧布局混用后只打警告，仍计算
并输出正式 CSV，且不记录各行数据来源。

修复：混用布局时默认 `SystemExit`（信息说明口径不可比、需补齐同版本三种子）；
新增 `--allow-mixed-legacy` 供显式要求临时对照；CSV 固定新增
`layout_seed0/1/2` 列记录各种子来源（new/legacy）。

验证：代码路径检查（该脚本的全量运行需要三个种子的真实结果目录，本机
不可用，见第 4 节边界）。

### A7 [P2] 比较次数超预算 —— 属实，已修复

核实：`same_transition_dc`/`same_factor_dt_sc` 按"每组下限 32 × 分组数"拼接，
无全局上限；实测（384 行、上限 100）得 192/576 条。

修复：`delta_stats.py` 拼接后超过 `max_comparisons` 时无放回抽至上限；
所有分组一律受限。实测同一场景各组均 ≤ 上限。

### 第 3 节 diff_factor_sc 样本量风险 —— 属实，部分处理

核实：same-context 匹配靠拒绝采样（3×上限次尝试），完整数据中很稀疏；
结果 CSV 只输出均值，无各组实际比较数；日志只记录 diff_factor_dc 的来源计数。

已修复的部分：

- 拒绝采样尝试次数提高到 20×上限（仍是"联合均匀抽样 + 筛选"，声明语义不变）；
- `e2_delta_stats.csv` 新增 6 个 `n_cmp_*` 列记录各组实际比较数；
- 任一组不足预算时日志告警；`diff_factor_sc` 的来源计数也写入日志。

**未修改的部分及原因：报告建议的"直接索引可匹配的 shared context"会改变
抽样分布（从条件筛选后的联合均匀变为按索引结构的其他分布），属于协议层
面的决定。当前代码声明的口径（对其他 delta 联合均匀抽样）与实现一致，
核心 Transition Gap 使用 different-context 对照、不受影响。若要改分布，
需要先明确目标分布和权重定义，再由你确认后单独修改。**

## 2. 与报告第 1 节（模型支持）、第 4 节（已验证项）相关的处理

- 三模型工厂、SigLIP2 FixRes/小写化、encoder 冻结：报告已确认符合方案，
  本轮核实未见问题，**未修改**。
- 报告第 4 节列出的已验证项（CLEVR 只读 val、SPAR blank 过滤、2AFC 判据、
  聚类 bootstrap 等）：本轮在阅读相关代码时复核，与描述一致，**未修改**。

## 3. 修改文件清单（14 个）

| 文件 | 对应问题 |
|---|---|
| src/features/delta.py | A3 |
| src/features/cache.py | A2 |
| scripts/extract_embeddings.py | A2、A1 |
| src/data/pug_imagenet.py | A1 |
| src/data/pair_matcher.py | A1 |
| src/data/captions.py | A1 |
| src/metrics/delta_stats.py | A7、§3、A1 |
| src/experiments/e2_change_diagnosis.py | A1、§3 |
| scripts/run_e2.py | A1 |
| configs/e2_change.yaml | A1、A3 |
| scripts/summarize_seed_stability.py | A4、A6 |
| src/utils/io.py | A5 |
| scripts/make_figures.py | A5 |
| code/README.md | 文档同步 |

## 4. 验证方式与边界

已做：

- 全部 Python 文件 `py_compile` 通过（41 个）。
- 39 项单元检查：A2 协议与 force 重建、A3 面积保证、A4 分类与稳定判定、
  A5 目录身份、A7 上限、A1 配对器（Animals 行为不变 + ImageNet 列固定）。
- E2 端到端集成测试：合成 parquet + 假 encoder，`pug_animals` 与
  `pug_imagenet` 两个数据集的 `run(smoke=True)` 全流程跑通（配对、抽取、
  nuisance、bootstrap、CSV 写出、n_cmp 列、告警、协议保护）。
- 集成测试过程中发现并修复了两个实现问题：`canonical_orders` 中
  `environment: alphabetical` 字符串值的回退处理；canonical 顺序未按数据集
  隔离导致 ImageNet 的 size 因素配对为空。

边界（与审查报告第 5 节一致）：

- 当时未加载真实权重；合成数据 + 假 encoder 的测试只证明**被覆盖用例的流程可运行**，
  不证明全部配对语义、输入边界或真实 checkpoint 下的数值结果正确。
- `summarize_seed_stability.py` 的完整运行、`make_figures.py` 的实际绘图
  依赖真实结果目录和 matplotlib，本机不可用，仅做代码级验证。
- PUG:ImageNet 真实数据的取值集合（scale/texture/灯光等）未见过，按观测值
  排序的回退逻辑需在首次真实运行时确认（日志会输出各因素配对数）。

## 5. 后续动作建议（按审查报告第 5 节顺序，结合本轮修改更新）

1. 服务器上核查 `data/cache/` 来源；来源不明或确为旧 SigLIP2 特征时，应先隔离
   整个受影响模型缓存，再从空目录逐数据集重建。当前 `--force` 对匹配协议只删除
   指定数据集图像和文本缓存，不等于重建其他数据集；损坏 JSON 的恢复边界见最新复核 B7。
2. 真实数据端到端：三模型 × 五数据集各自任务各跑一次 `--smoke`，确认
   ImageNet 链路（首次运行观察各因素取值与配对日志）。
3. E2 按新 nuisance 语义（面积比例 85%–100%）重算；原图像特征可复用。
4. 补齐同版本三种子结果后再运行 `summarize_seed_stability.py`（混用布局
   现在默认会拒绝）；需要历史对照时用 `--allow-mixed-legacy`。
