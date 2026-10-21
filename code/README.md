# GraphDelta 第一阶段：冻结 baseline 诊断流水线

对应《GraphDelta_完整预实验方案_统一修订版》。**纯评估代码：无 optimizer / loss / backward /
LoRA / fine-tuning / projection training**（方案 §55 / §83 / §84）。

## 环境

```bash
conda run -n CudaVersion128Fuxp python <script>
# python 3.10 / torch 2.7.0+cu128 / torchvision 0.22 / transformers 5.x / pandas / pyarrow
```

## 运行顺序（方案 §82）

```bash
cd GraphDelta/code

# 0)（可选）生成 PUG 元数据 manifest
conda run -n CudaVersion128Fuxp python scripts/build_manifests.py --dataset all

# 1) 特征抽取（每个数据集一次；不运行则 run_eX 会自动抽取；缓存按 id 增量补齐）
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset clevr_val
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset pug_spar
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset cogent_valA
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset cogent_valB
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset pug_animals

# 2) 四个实验
conda run -n CudaVersion128Fuxp python scripts/run_e1.py                 # SPAR + CLEVR 自建 2AFC
conda run -n CudaVersion128Fuxp python scripts/run_e1_spar_official.py   # SPAR 官方协议（P2-11）
conda run -n CudaVersion128Fuxp python scripts/run_e3.py                 # CoGenT 难度不对称 + sphere 对照
conda run -n CudaVersion128Fuxp python scripts/run_e2.py                 # PUG: Animals 单因素响应（最重，最后跑）

# 检验多次采样稳定性（方案 seeds 0/1/2）：
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --seed 1   # e2/e3 同理，覆盖 outputs 前自行备份
```

快速自检（小样本，输出带 `_smoke` 后缀）：

```bash
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e1_spar_official.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e3.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e2.py --smoke
```

## 输出（outputs/tables/）

| 文件 | 内容 |
|---|---|
| e1_results.csv | 按 dataset × factor：2AFC acc + CI、margin、模板方差（§63）；含 `protocol=2afc_self_built` 标记 |
| e1_factor_gaps.csv | E1 内 dataset 各 factor 两两 Accuracy Gap + bootstrap CI（共享图片按图片并集同步重采样） |
| e1_clevr_filter_stats.json / e1_spar_filter_stats.json | 各过滤原因的候选计数（§20a） |
| e1_items_clevr/spar.jsonl | 每条 2AFC 样本（正/负内容、图片、factor），可复查 |
| e1_spar_official.csv | SPAR 官方协议：scene / one_object / one_object_position / two_objects / two_objects_relation / two_objects_attribute × {全候选检索 top1/top5, hard-negative} × {无背景, caption 带背景}（P2-11）。检索候选集只含真实行的正例 caption，role-swap 负例只用于 hard-negative 二选一（二轮问题 7）。查询图片按 `spar_official.max_per_task=2000` 种子随机抽样（与自建 E1 的 n 对齐；候选池仍取完整子集，bg/no-bg 复用同一组查询图片；设 null 恢复官方全量；scene 保留官方 10 行 quirk） |
| e2_delta_stats.csv | 按因素：Δ 范数（raw/z）、before/after 余弦、nuisance 范数与相似度、分层相似度（raw/z）、Transition Gap + CI、Factor/Nuisance Norm Gap + CI（§64；无后缀列 = normalized 空间，`_raw` = 归一化前空间，一轮 P2-8）。diff_factor 基准从其他因素联合均匀抽样，来源计数写入日志（二轮问题 1）；相似度 CI 为双端点 context 聚类 bootstrap（二轮问题 3）；Norm Gap 的因素端按 context、nuisance 端按图像聚类（二轮问题 4） |
| e2_transition_stats.csv | 逐 transition：(factor, source, target) 的 n_contexts、相似度（raw/z）、Δ 范数（raw/z）；n_contexts<2 时 mean_sim 为 NaN 并在日志报告数量 |
| e3_cogent.csv | 按 condition × shape_group × negative_type（§65）；CI 按图片聚类 bootstrap |
| e3_difficulty_gap.csv | A−B：题型等权 accuracy（主指标，图片聚类 bootstrap）+ 汇总口径（敏感性对照，CI 同样按图片聚类）；cube/cylinder 主指标 + sphere 对照（一轮 P2-9、二轮问题 5） |
| e3_cogent_by_combo.csv | 逐 (condition, shape_group, color, shape, negative_type)（含 pooled 行）：n、accuracy、mean_margin（P2-10） |
| e3_cogent_by_color.csv | 逐 (condition, shape_group, color)：n、accuracy、mean_margin（P2-10） |

## 关键实现约定（与方案对应）

1. **Split（16a）**：CLEVR 只用 val；CoGenT 只用 valA/valB；PUG: Animals 全量 215,040；
   SPAR 全量 43,560。
2. **PUG 行号**：按分片文件名字典序累计的全局 row_id；sample_id = `pug_{data}_{row_id:06d}`。
3. **CLEVR 关系**：直接用场景 JSON 的官方 `relationships` 字段（left/right/front/behind），
   不从坐标自行推导。语义（官方生成器）：`j ∈ relationships[rel][i]` 表示 **objs[j] 在
   objs[i] 的 rel 方向**，正例主语取 objs[j]（审查报告 P1-1；已用官方 `directions`
   三维向量逐条验证 21,164/21,164 为真）。
4. **负例有效性（20a）**：正例存在、负例对整图为假、（建议）正例唯一；无效样本直接丢弃，
   各过滤原因计数写入 `e1_*_filter_stats.json`。
5. **SPAR 自建 2AFC**：`character2_name=="blank"` 为单动物场景，position/attribute/relation
   先过滤；**双对象同种动物场景整体过滤**（正负文本可能相同或同时为真，P1-2）；
   position 负例 = 方位词取反，水平用 "to the left/right of"、垂直用 "below/above"
   （P2-11，不产生 "to the below of"）；relation 负例 = 主客体互换；
   attribute 负例 = 纹理互换（要求 tex1≠tex2）。
6. **模板（31）**：phrase 7 个 + sentence 7 个（configs/templates.yaml），报告跨模板平均与
   模板间方差；平局 s+==s- 记错。
7. **E2 方向（38）**：size 0.7<1.0<1.3；camera 按数值序；texture/environment 按固定列表 /
   字母序。顺序只用于统一符号。每个 (context, transition) 只保留一个方向。
8. **E2 配对（59）**：每个 context 按"全局已选次数越少的 transition 权重越高"无放回抽样
   （P2-7）；全量 environment 因素覆盖 2016/2016 transitions、每 transition ≥3 contexts。
9. **E2 分层（40/41/66）**：same_transition（不同 context 天然成立）、same_factor_dt（分同/异
   context，dc 组同时要求 transition 不同）、diff_factor（d2 从其他因素**联合均匀抽样**，
   same/different context 按两因素**共同固定字段**是否全部相等判定，一轮 P2-6、二轮问题 1）、
   random。比较记录两个端点的 context，CI 用双端点聚类 bootstrap（比较权重 = 两端点被抽中
   次数之积，与端点登记顺序无关，二轮问题 3）；全部聚类 bootstrap 保证 n_boot 次**有效**
   重采样，不足时 CI 置 NaN 而非静默返回低精度区间（三轮问题 1）。聚类单位 = 完整 metadata
   组合（`context_key`，§66）。比较采样直接抽不相等的索引对。
10. **nuisance（39a）**：同图两个确定性增强视图（crop 0.85–1.0 + JPEG 90/75，seed 由 row_id
    派生）；文本为同内容不同模板。输出 raw/z 两空间的范数与 nuisance delta 两两相似度；
    Factor/Nuisance Norm Gap 附 bootstrap CI（P2-8）。只在内存中产生，不落盘图像。
11. **E3（46–51）**：Difficulty Gap 只在 cube/cylinder 上计算，sphere 为对照；A/B 各自按
    image_id 聚类 bootstrap、跨 negative_type 等权平均（P2-9）；结论只写"难度不对称"，
    不写 seen/unseen。
12. **特征缓存（62）**：`data/cache/<model>/<dataset>_images.npz`（fp16，raw+normalized，
    raw = projection 后归一化前）；文本按**实际编码文本**的 md5 去重缓存（补句号与否
    进入键，二轮问题 8）。**缓存按 id 增量校验**：已有样本跳过、缺失样本补抽后合并写回
    （一轮 P2-12）。SPAR 官方协议特征单独使用 Resize(224,224)+ImageNet mean/std 预处理，
    缓存 key 为 `pug_spar_official`；add_background 的 world 展开按全量 metadata。
13. **SPAR 官方协议（9，P2-11）**：复现 `facebookresearch/PUG` 的
    `PUG_SPAR/run_eval_vlms_on_spar.py` CLIP 路径（对照过 facebookresearch/PUG
    仓库原脚本，历轮审查已核对逐项一致；审查材料已清理）：
    官方 caption 模板（无句号）、全候选检索 top1/top5 + hard-negative 二选一；
    **检索候选集只含真实行的正例 caption**（role-swap 负例仅用于 hard-negative，
    二轮问题 7）；add_background=True 时每个正例组合按**全部 world** 展开候选、
    图片的正标签与 hard-negative 仍用图片自身 world（三轮问题 2）；官方脚本的
    quirk 保留（`character1_pos` 只有 left/below，"bottom/top" 子集为空）。
    hard-negative 平局判错（与自建 2AFC 一致）；官方原脚本按随机换序 + argmax
    处理，严格大于口径与其等价（平局除外）。自建 2AFC 与官方协议结果分开报告，
    同名指标含义不同。查询图片按 `spar_official.max_per_task`（默认 2000，
    seeds[0]）随机抽样，与自建 E1 的 `target_per_cell` 对齐；检索候选池始终由
    完整任务子集生成，不随抽样缩减，bg 与 no-bg 两轮复用同一组查询图片（五轮
    P2-1）；此时数值不再与官方发布结果直接可比，设 null 恢复官方全量；scene
    任务的 10 行 quirk 不抽样。

## 许可

PUG 系列为 CC-BY-NC（禁止用于训练生成式模型）；本流水线只做评估。开源或写论文时注明。

## 已知边界

- `e2_delta_stats.csv` 的相似度列为组内均值（点估计），same_transition 与 gap 附聚类
  bootstrap CI；其余组只报点估计（比较用）。
- E1 的 CI 按 image_id 聚类 bootstrap；`per_scene_cap=1` 生效后每个 cell 内一张图最多
  一条样本（P2-5）。E3 每图最多 2 对象 × 2 题型，CI 按图片聚类。
- SPAR 官方协议 smoke 模式只取前 3000 行，部分任务样本少或为空（如 blank-blank 场景
  识别）；全量运行才有可与论文对齐的数字。
- SigLIP2 为第二轮对照：本地无权重，`siglip_encoder.py` 为占位。
- §67 的结果图未生成（汇总图，非核心算法）；如需可从 CSV 直接绘制。
