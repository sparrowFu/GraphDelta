# GraphDelta 预实验代码总览

本目录是预实验的全部代码。总体说明（每个环节做什么、为什么、怎么执行）见
[`../README.md`](../README.md)；各阶段操作手册见
[`stage1_frozen_baseline/README.md`](stage1_frozen_baseline/README.md) 与
[`stage2_controlled_edit/README.md`](stage2_controlled_edit/README.md)。本文件讲
代码本身的组织方式和公共机制的设计理由。

## 1. 目录结构与职责

```text
PreExperiments/
├── common/                     # 两阶段共用（import 名 src.*）
│   └── src/
│       ├── models/             # 冻结编码器工厂：resolve_models / build_encoder /
│       │                       # free_encoder_memory；CLIP 与 SigLIP2 实现
│       ├── features/           # ImageFeatureCache / TextCache（protocol 指纹、
│       │                       # 增量补齐、float16/float32 可选存储）；抽取与 Δ
│       ├── metrics/            # 聚类 bootstrap（cluster_mean_ci / cluster_gap_ci /
│       │                       # pair_cluster_*）、2AFC 打分、Δ 分组统计
│       ├── data/               # CLEVR / CoGenT / PUG 装载、caption 构造、模板
│       └── utils/              # io.py（ROOT 锚定、load_paths、run_meta 登记 /
│       │                       # 提交）、logging.py（stdout + outputs/logs）
├── stage1_frozen_baseline/     # 阶段一（E1–E3，已完成，结果冻结）
│   ├── scripts/                # run_e1* / run_e2 / run_e3 / extract_embeddings /
│   │                           # analyze_e3_units / summarize_seed_stability /
│   │                           # make_figures / build_manifests / audit_e1_sampling /
│   │                           # render_*_html / convert_siglip2_checkpoint
│   └── src/experiments/        # e1 / e2 / e3 实验模块 + common.py（2AFC 评测核）
└── stage2_controlled_edit/     # 阶段二（E4）
    ├── scripts/                # export_audit_items / build_edit_pairs /
    │                           # render_edit_pairs / run_e4 / summarize_e4
    ├── src/generation/         # 场景规格（BaseSceneSpec、放置约束同官方）与
    │                           # 编辑算子 + validate_edit（纯逻辑，无 Blender）
    ├── src/e4/                 # pairs（清单/划分）、render（调度/校验/账本）、
    │                           # captions、data（装载/候选池）、features（f32 缓存）、
    │                           # paired / delta / text_delta / transfer（A–D）、
    │                           # stats（置换/ridge）、run、summarize
    ├── render/                 # blend_render_driver.py（Blender 2.78 / py3.5 内
    │                           # 运行，规格走 JSON，不 import 本仓库代码）
    └── tests/                  # 单元测试（见 §4）
```

## 2. 公共机制与设计理由

### 2.1 路径锚定：代码可搬、数据不动

- 所有脚本与模块以 `__file__` 向上定位，不用当前目录；`common/src/utils/io.py`
  的 `ROOT` 锚定到 `code/`，因此 `configs/`、`data/`、`outputs/` 留在 `code/`
  下不随代码目录搬移——已有结果、缓存和报告里的相对链接全部保持有效。
- 数据 / 模型 / 渲染环境的绝对路径集中在 `configs/paths.yaml`；`load_paths()`
  把 `dirs` 段解析为绝对路径并建目录。换机器只改这一个文件。

### 2.2 import 约定：共用一套、阶段各一套

- 共用包目录名 `common/src/`，import 名保持 `src.*`（阶段一代码的内部导入
  迁移时零改动）；
- 阶段包用裸名：`experiments.*`（阶段一）、`generation.*` / `e4.*`（阶段二）；
- 脚本统一开头插入两条路径：

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # 阶段包
```

### 2.3 特征缓存：一次抽取、指纹失效

- 图像特征按 dataset 存 NPZ、文本按内容 md5 去重，`protocol.json` 记录
  encoder 预处理语义指纹——语义变更后旧缓存直接报错要求重建，防止静默串用；
- 阶段二使用独立 float32 命名空间（`data/cache/f32/<model>/`，dtype 参数控制
  存储），与阶段一 float16 缓存互不影响；E4 的缓存键额外带编辑对清单指纹
  （范式同 CoGenT 的 `condition_cache_key`），清单重建自动换键。

### 2.4 run_meta：失败运行不能冒充结果

- 每次实验经 `output_tables_dir` 在 `outputs/<model>/seed_<k>/tables/` 登记
  `running`，全部产物成功写出后才由 `commit_experiment` 置 `success` 并记录
  产物清单、git commit 与脏标记、缓存协议指纹；
- smoke 运行用独立 experiment 键与 `_smoke` 文件名，不更新正式结果；
  下游汇总脚本按 `success` 状态守门。

### 2.5 渲染账本：唯一状态记账、确定性失败不重试

- 渲染单位是唯一场景状态（before/after/换色/平移各一态，逆向共享），账本
  `ledger.jsonl` 逐态记录秒数、重试、字节数与校验问题；吞吐含校验与重试，
  重试不重复计费；
- 官方脚本无随机种子 → 正式数据由带种子规格驱动；可见性检查失败由驱动以
  退出码 3 报告（规格确定、重试结果相同），登记后剔除该状态。

## 3. 阶段状态

| 阶段 | 状态 | 正式运行前还须完成（冻结时点见工作方案开篇待办表） |
|---|---|---|
| 阶段一 | 已完成，结果冻结 | — |
| 阶段二 | 代码完成、小批端到端验证通过 | 题目人工审计；转换清单与配额冻结；E4 协议复核；首批人工核验流程；多进程吞吐实测 |

## 4. 测试与自检

```bash
# 单元测试（不依赖 Blender / 模型 / 数据集）
conda run -n CudaVersion128Fuxp python -m pytest PreExperiments/stage2_controlled_edit/tests/
```

覆盖三块：场景规格放置约束与编辑算子往返（test_generation.py，29 用例）；
清单构建确定性 / 划分完整性 / 序列化后编辑校验 / 账本去重（test_edit_pairs.py）；
E4 统计口径——平局记错、C ≡ Δv·Δt 数值等式、符号翻转置换小样本精确性、
ridge 探针、候选检索（test_e4_stats.py）。

端到端冒烟（分钟级，见阶段二 README §6）：小批清单 → `--limit` 渲染 →
`run_e4 --smoke` → `summarize_e4`。2026-11-04 已按此链路验证：21 态渲染
全部通过规格复核，E4-A/B/C/D 与汇总产物落盘，代数自检
`max|C − Δv·Δt| ≈ 2.3e-08`。


怎么具体解决提出的问题，给出各个模块的思路。
研究问题需要讲清楚这个问题会带来什么后果。
哪些不足导致了这个问题，然后针对不足提出改进方案，方法和发现的问题要有对应。
讲别人文章的时候需要把这篇文章前后逻辑讲清楚，这篇文章怎么讲？motivation、创新点等等。把别人文章的讲的清楚一点。

