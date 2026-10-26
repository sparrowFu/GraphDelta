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
# PUG: ImageNet（辅助，方案 §14 / 审查 A1；跑 E2-ImageNet 前需要）
# conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset pug_imagenet

# 2) 四个实验（默认 CLIP ViT-L/14；其他 baseline 加 `--model`：clip_vitb32 / siglip2_so400m）
conda run -n CudaVersion128Fuxp python scripts/run_e1.py                 # SPAR + CLEVR 自建 2AFC
conda run -n CudaVersion128Fuxp python scripts/run_e1_spar_official.py   # SPAR 官方协议（P2-11）
conda run -n CudaVersion128Fuxp python scripts/run_e3.py                 # CoGenT 难度不对称 + sphere 对照
conda run -n CudaVersion128Fuxp python scripts/run_e2.py                 # PUG: Animals 单因素响应（最重，最后跑）
# PUG: ImageNet 辅助验证（方案 §14 / 审查 A1）：--dataset pug_imagenet
# 配对固定 character_name（实例身份，复核 B1）等全部非目标因素；
# 某因素 0 对配对默认停止，局部诊断加 --allow-empty-factors
# conda run -n CudaVersion128Fuxp python scripts/run_e2.py --dataset pug_imagenet
# 例：SigLIP2 / CLIP B/32 对照（特征缓存与结果目录均按模型隔离，互不影响）
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --model siglip2_so400m
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --model clip_vitb32

# 检验多次采样稳定性（方案 seeds 0/1/2）：
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --seed 1   # e2/e3 同理；结果自动写入 seed_1 目录
# 收尾脚本（2026-10-22 已按此跑完 seeds 0/1/2，当时为旧布局）：
conda run -n CudaVersion128Fuxp python scripts/analyze_e3_units.py        # E3 逐单元 gap（读 e3_items_cogent.jsonl）
conda run -n CudaVersion128Fuxp python scripts/summarize_seed_stability.py # 三种子判定稳定性汇总（--model 可选）
conda run -n CudaVersion128Fuxp python scripts/make_figures.py            # §67 五图（--model/--seed 标注标题）
```

结果目录约定（R2 隔离）：每个运行写入 `outputs/<model>/seed_<k>/tables/`，
切换 `--model` / `--seed` 不再互相覆盖；目录内 `run_meta.json` 记录 model/seed、
git commit（含是否有未提交改动）与 encoder `cache_protocol`（复核 R1 的来源记录），
`summarize_seed_stability.py` / `make_figures.py` / `analyze_e3_units.py` 读取时
校验身份。旧布局 `outputs/tables`（seed 0）与 `outputs/tables_seed{1,2}` 是
2026-10-22 **clip_vitl14** 三种子运行的历史快照，只读保留；读取端仅对
clip_vitl14（及 seed 0 绘图）允许回退旧布局，其他模型缺少新结果时直接报错，
不会把历史 CLIP-L/14 数据错当其他模型的结果（R2/R7）。阶段总结见
`outputs/STAGE1_SUMMARY.md`。

特征缓存约定（审查 §3.8 / A2）：`data/cache/<model>/` 内 `protocol.json` 记录该
encoder 的预处理语义指纹（`cache_protocol`，随 encoder 类声明）。encoder 的
图像/文本预处理变更时指纹更新，旧缓存立即报错失效（提示删除重抽），不会按
ID/文本键静默复用旧特征；无 `protocol.json` 但已有 NPZ 的历史目录会被拒绝
登记（新指纹不能证明旧特征语义），重建入口为 `extract_embeddings.py --force`
（协议不匹配时删整个模型缓存目录；协议匹配时删该数据集图像缓存 + 文本缓存）；
siglip2_so400m 当前指纹为 `siglip2-fixres-lowercase-v1`（R1 FixRes + R4 小写化），
旧缓存已删除。

快速自检（小样本，输出带 `_smoke` 后缀）：

```bash
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e1_spar_official.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e3.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e2.py --smoke
```

## 输出（outputs/<model>/seed_<k>/tables/；旧结果在 outputs/tables*）

| 文件 | 内容 |
|---|---|
| e1_results.csv | 按 dataset × factor：2AFC acc + CI、margin、模板方差（§63）；含 `protocol=2afc_self_built` 标记 |
| e1_factor_gaps.csv | E1 内 dataset 各 factor 两两 Accuracy Gap + bootstrap CI（共享图片按图片并集同步重采样） |
| e1_clevr_filter_stats.json / e1_spar_filter_stats.json | 各过滤原因的候选计数（§20a） |
| e1_items_clevr/spar.jsonl | 每条 2AFC 样本（正/负内容、图片、factor），可复查 |
| e1_spar_official.csv | SPAR 官方协议：scene / one_object / one_object_position / two_objects / two_objects_relation / two_objects_attribute × {全候选检索 top1/top5, hard-negative} × {无背景, caption 带背景}（P2-11）。检索候选集只含真实行的正例 caption，role-swap 负例只用于 hard-negative 二选一（二轮问题 7）。查询图片按 `spar_official.max_per_task=2000` 种子随机抽样（与自建 E1 的 n 对齐；候选池仍取完整子集，bg/no-bg 复用同一组查询图片；设 null 恢复官方全量；scene 保留官方 10 行 quirk） |
| e2_delta_stats.csv | 按因素：Δ 范数（raw/z）、before/after 余弦、nuisance 范数与相似度、分层相似度（raw/z）、Transition Gap + CI、Factor/Nuisance Norm Gap + CI（§64；无后缀列 = normalized 空间，`_raw` = 归一化前空间，一轮 P2-8）。diff_factor 基准从其他因素联合均匀抽样，来源计数写入日志（二轮问题 1）；相似度 CI 为双端点 context 聚类 bootstrap（二轮问题 3）；Norm Gap 的因素端按 context、nuisance 端按图像聚类（二轮问题 4）。各组实际比较数见 `n_cmp_*` 列，不足预算时日志告警（审查 A7/§3）。ImageNet 跑 E2 时输出为 `e2_delta_stats_pug_imagenet.csv`（A1） |
| e2_transition_stats.csv | 逐 transition：(factor, source, target) 的 n_contexts、相似度（raw/z）、Δ 范数（raw/z）；n_contexts<2 时 mean_sim 为 NaN 并在日志报告数量 |
| e3_cogent.csv | 按 condition × shape_group × negative_type（§65）；CI 按图片聚类 bootstrap |
| e3_difficulty_gap.csv | A−B：题型等权 accuracy（主指标，图片聚类 bootstrap）+ 汇总口径（敏感性对照，CI 同样按图片聚类）；cube/cylinder 主指标 + sphere 对照（一轮 P2-9、二轮问题 5） |
| e3_cogent_by_combo.csv | 逐 (condition, shape_group, color, shape, negative_type)（含 pooled 行）：n、accuracy、mean_margin（P2-10） |
| e3_cogent_by_color.csv | 逐 (condition, shape_group, color)：n、accuracy、mean_margin（P2-10） |
| e3_items_cogent.jsonl | E3 每条 item 级结果（image_id、condition、shape_group、negative_type、color、shape、item_acc、item_margin），供逐单元分析复查 |
| e3_unit_gaps.csv | 逐 (shape_group, color) 主单元 A−B 题型等权 gap + 图片聚类 bootstrap CI（`pooled_equal_type` 行）+ 逐题型明细行；单元级效应方向沿 CoGenT 调色板分裂，详见 `analyze_e3_units.py` 与 STAGE1_SUMMARY §3。写出前先复算总 gap 与主表对照，不一致默认停止（复核 B6；`--no-strict` 探索模式），检查结果写 `e3_unit_consistency.json` |
| seed_stability.csv | seeds 0/1/2 三种子关键判定（E1 CI-vs-0.5、gap 符号、SPAR 官方最大差、E2 CI 符号、E3 总/单元 gap）稳定性汇总。SPAR 描述行 stable 为 NA 不计稳定数；三种子均未定义的指标（hardneg 的 top5）跳过；`per_world` 输出时 world 计入 key（复核 B3/B8）；各种子来源布局记录在 `layout_seed*` 列 |
| figures/ | §67 五图：fig1 2AFC accuracy、fig2 margin、fig3 Δ 范数 vs nuisance、fig4 相似性阶梯、fig5 CoGenT A/B + 逐单元 gap（PNG 300dpi + 合并 PDF） |

## 关键实现约定（与方案对应）

1. **Split（16a）**：CLEVR 只用 val；CoGenT 只用 valA/valB；PUG: Animals 全量 215,040；
   SPAR 全量 43,560。
1a. **多 baseline（§84-2/§84-22）**：`--model` 选择冻结 baseline（`clip_vitl14` /
   `clip_vitb32` / `siglip2_so400m`），按 `configs/<model>.yaml` 的 `model_type`
   分发到各自 encoder；特征缓存按模型分目录（`data/cache/<model>/`），结果目录
   按模型 + 种子隔离（R2），不同模型不跨模型计算 cosine。
   SigLIP2 so400m-patch14-384 是 **FixRes checkpoint**（R1）：`SiglipModel` +
   `SiglipImageProcessor` 直接加载原始目录（固定 384×384 resize），无需转换；
   曾用 `convert_siglip2_checkpoint.py` 转成 NaFlex 布局加载的做法会改变预处理
   语义（动态分辨率 + 保持纵横比，同图特征实测 cosine 0.994 非等价），已废弃，
   脚本仅留档。文本统一**显式小写**后编码（R4：tokenizer_config 的
   do_lower_case 对 GemmaTokenizer 不生效，SigLIP2 官方要求小写），
   `padding="max_length"` 到 64（池化取序列末位置，官方要求）；SPAR 官方
   协议对其使用自身 384 预处理（§84-22），故 SigLIP2 官方协议数字不复现任何
   官方发布值，只作 backbone 对照。
2. **PUG 行号**：按分片文件名字典序累计的全局 row_id；sample_id = `pug_{data}_{row_id:06d}`。
3. **CLEVR 关系**：直接用场景 JSON 的官方 `relationships` 字段（left/right/front/behind），
   不从坐标自行推导。语义（官方生成器）：`j ∈ relationships[rel][i]` 表示 **objs[j] 在
   objs[i] 的 rel 方向**，正例主语取 objs[j]（审查报告 P1-1；已用官方 `directions`
   三维向量逐条验证 21,164/21,164 为真）。
4. **负例有效性（20a）**：正例存在、负例对整图为假、（建议）正例唯一；无效样本直接丢弃，
   各过滤原因计数写入 `e1_*_filter_stats.json`。
5. **SPAR 自建 2AFC**：blank 可能在 `character_name` 或 `character2_name` 任意一侧
   （各 1,320 行；R3），双对象任务（position/attribute/relation）要求**两侧均非
   blank**；双侧均 blank 的纯背景行（40 行）不进入动物描述任务；scene 单动物行
   描述实际存在的一侧。**双对象同种动物场景整体过滤**（正负文本可能相同或同时
   为真，P1-2）；position 负例 = 方位词取反，水平用 "to the left/right of"、垂直用
   "below/above"（P2-11，不产生 "to the below of"）；relation 负例 = 主客体互换；
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
    one_object 候选池含 "blank"（官方 `character_name.unique()` 不去 blank，
    blank 为干扰候选，R6）；**检索候选集只含真实行的正例 caption**（role-swap
    负例仅用于 hard-negative，
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
- SigLIP2 已按 FixRes 路径重实现（`siglip_encoder.py`，R1/R4 修复后需重抽该模型
  全部图像与文本特征并重跑其已有结果；旧的 NaFlex 路径产物已废弃）；
  尚未跑全量对照（需重抽约 35 万张图特征，与 BindRepair 并行安排）。
- §67 五图已生成（`scripts/make_figures.py` → `outputs/figures/`）。
