# GraphDelta 第一阶段：冻结 baseline 诊断流水线

对应《GraphDelta_完整预实验方案_统一修订版》。**纯评估代码：无 optimizer / loss / backward /
LoRA / fine-tuning / projection training**（方案 §55 / §83 / §84）。

## 环境

```bash
conda run -n CudaVersion128Fuxp python <script>
# python 3.10 / torch 2.7.0+cu128 / transformers 5.x / pandas / pyarrow / matplotlib
```

## 运行顺序（方案 §82）

```bash
cd GraphDelta/code

# 0)（可选）生成 PUG 元数据 manifest
conda run -n CudaVersion128Fuxp python scripts/build_manifests.py --dataset all

# 1) 特征抽取（每个数据集一次；不运行则 run_eX 会自动抽取）
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset clevr_val
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset pug_spar
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset cogent_valA
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset cogent_valB
conda run -n CudaVersion128Fuxp python scripts/extract_embeddings.py --dataset pug_animals

# 2) 三个实验
conda run -n CudaVersion128Fuxp python scripts/run_e1.py   # SPAR + CLEVR 结构诊断
conda run -n CudaVersion128Fuxp python scripts/run_e3.py   # CoGenT 难度不对称 + sphere 对照
conda run -n CudaVersion128Fuxp python scripts/run_e2.py   # PUG: Animals 单因素响应（最重，最后跑）
```

快速自检（小样本，输出带 `_smoke` 后缀）：

```bash
conda run -n CudaVersion128Fuxp python scripts/run_e1.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e3.py --smoke
conda run -n CudaVersion128Fuxp python scripts/run_e2.py --smoke
```

## 输出（outputs/tables/）

| 文件 | 内容 |
|---|---|
| e1_results.csv | 按 dataset × factor：2AFC acc + CI、margin、模板方差（§63） |
| e1_items_clevr/spar.jsonl | 每条 2AFC 样本（正/负内容、图片、factor），可复查 |
| e2_delta_stats.csv | 按因素：Δ 范数、nuisance 基线、分层相似度、Transition Gap + CI（§64） |
| e2_transition_stats.csv | 逐 transition：(factor, source, target) 的 n_contexts、相似度、Δ 范数 |
| e3_cogent.csv | 按 condition × shape_group × negative_type（§65） |
| e3_difficulty_gap.csv | A−B 难度差（cube/cylinder 主指标 + sphere 对照） |
| e3_cogent_by_combo.csv / by_color.csv | 逐 (color, shape) / 逐颜色 |

## 关键实现约定（与方案对应）

1. **Split（16a）**：CLEVR 只用 val；CoGenT 只用 valA/valB；PUG: Animals 全量 215,040；
   SPAR 全量 43,560。
2. **PUG 行号**：按分片文件名字典序累计的全局 row_id；sample_id = `pug_{data}_{row_id:06d}`。
3. **CLEVR 关系**：直接用场景 JSON 的官方 `relationships` 字段（left/right/front/behind），
   不从坐标自行推导。
4. **负例有效性（20a）**：正例存在、负例对整图为假、（建议）正例唯一；无效样本直接丢弃，
   数量写入日志。
5. **SPAR**：`character2_name=="blank"` 为单动物场景，position/attribute/relation 先过滤；
   position 负例 = 方位词取反（left↔right / below↔above）；relation 负例 = 主客体互换；
   attribute 负例 = 纹理互换（要求 tex1≠tex2）。
6. **模板（31）**：phrase 7 个 + sentence 4 个（configs/templates.yaml），报告跨模板平均与
   模板间方差；平局 s+==s- 记错。
7. **E2 方向（38）**：size 0.7<1.0<1.3；camera 按数值序；texture/environment 按固定列表 /
   字母序。顺序只用于统一符号。每个 (context, transition) 只保留一个方向。
8. **E2 分层（40/41）**：same_transition（不同 context 天然成立）、same_factor_dt（分同/异
   context）、diff_factor（按 animal|environment 是否相同分桶）、random。聚类 bootstrap 的
   cluster = animal|environment（environment 因素自身变化时记为 animal|*）。
9. **nuisance（39a）**：同图两个确定性增强视图（crop 0.85–1.0 + JPEG 90/75，seed 由 row_id
   派生）；文本为同内容不同模板。只在内存中产生，不落盘图像。
10. **E3（46–51）**：Difficulty Gap 只在 cube/cylinder 上计算，sphere 为对照；结论只写
    "难度不对称"，不写 seen/unseen。
11. **特征缓存（62）**：`data/cache/<model>/<dataset>_images.npz`（fp16，raw+normalized，
    raw = projection 后归一化前）；文本按 md5 去重缓存在同目录。

## 许可

PUG 系列为 CC-BY-NC（禁止用于训练生成式模型）；本流水线只做评估。开源或写论文时注明。

## 已知边界

- `e2_delta_stats.csv` 的相似度列为组内均值（点估计），same_transition 与 gap 附聚类
  bootstrap CI；其余组只报点估计（比较用）。
- E1 的 bootstrap CI 按 item 重采样；同一图片可能出现在不同 factor 的样本里，但每个
  cell 内部一张图最多一条样本（per_scene_cap=1）。
- SigLIP2 为第二轮对照：本地无权重，`siglip_encoder.py` 为占位。
