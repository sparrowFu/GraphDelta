# E1 平衡采样审计（2026-11-04）

- git commit：`24dea3c790b478b2802aa6d2f6eb8dec61a4de5d`；数据源 45 个文件，SHA256 见同名 JSON。
- 检查 A–G 通过判定：全部通过；H 为描述性统计。
- 消费文件与产物均为正式（非 smoke）表。

## G. run_meta 实验状态

| run | e1_state | e1_spar_official | e2_pug_animals | e2_pug_imagenet | e3_cogent |
|---|---|---|---|---|---|
| clip_vitb32/seed_0 | success | success | success | success | success |
| clip_vitb32/seed_1 | success | success | success | success | success |
| clip_vitb32/seed_2 | success | success | success | success | success |
| clip_vitl14/seed_0 | success | success | success | success | success |
| clip_vitl14/seed_1 | success | success | success | success | success |
| clip_vitl14/seed_2 | success | success | success | success | success |
| siglip2_so400m/seed_0 | success | success | success | success | success |
| siglip2_so400m/seed_1 | success | success | success | success | success |
| siglip2_so400m/seed_2 | success | success | success | success | success |

## A. pos_value 配额平衡（count: seed0/seed1/seed2）

| dataset/factor | clip_vitb32 | clip_vitl14 | siglip2_so400m |
|---|---|---|---|
| clevr/object | 667/667/666 | 667/667/666 | 667/667/666 |
| clevr/color | 250/250/250/250/250/250/250/250 | 250/250/250/250/250/250/250/250 | 250/250/250/250/250/250/250/250 |
| clevr/size | 1000/1000 | 1000/1000 | 1000/1000 |
| clevr/material | 1000/1000 | 1000/1000 | 1000/1000 |
| clevr/relation | 500/500/500/500 | 500/500/500/500 | 500/500/500/500 |
| clevr/binding | 回退（无 pos_value） | 回退（无 pos_value） | 回退（无 pos_value） |
| pug_spar/scene | 回退（无 pos_value） | 回退（无 pos_value） | 回退（无 pos_value） |
| pug_spar/position | 500/500/500/500 | 500/500/500/500 | 500/500/500/500 |
| pug_spar/attribute | 1000/1000 | 1000/1000 | 1000/1000 |
| pug_spar/relation | 1000/1000 | 1000/1000 | 1000/1000 |

## D. e1_results.csv 重算一致性（90 个 run×factor 的最大绝对偏差）

- n_samples: 0.000e+00（容差 1e-09）
- accuracy: 2.220e-16（容差 1e-09）
- mean_margin: 0.000e+00（容差 1e-09）
- median_margin: 1.863e-09（容差 1e-07）
- margin_std: 1.683e-09（容差 1e-07）

## F. “恒选某词”策略期望准确率上界（max_share，argmax 词）

| dataset/factor | clip_vitb32 | clip_vitl14 | siglip2_so400m |
|---|---|---|---|
| clevr/object | 0.334（cube）；0.334（cylinder） | 0.334（cube）；0.334（cylinder） | 0.334（cube）；0.334（cylinder） |
| clevr/color | 0.125（blue）；0.125（cyan）；0.125（gray） | 0.125（blue）；0.125（cyan）；0.125（gray） | 0.125（blue）；0.125（cyan）；0.125（gray） |
| clevr/size | 0.500（large）；0.500（small） | 0.500（large）；0.500（small） | 0.500（large）；0.500（small） |
| clevr/material | 0.500（metal）；0.500（rubber） | 0.500（metal）；0.500（rubber） | 0.500（metal）；0.500（rubber） |
| clevr/relation | 0.250（in）；0.250（right） | 0.250（in）；0.250（right） | 0.250（in）；0.250（right） |
| clevr/binding | immune | immune | immune |
| pug_spar/scene | 0.037（zebra）；0.038（orca）；0.042（tarantula） | 0.037（zebra）；0.038（orca）；0.042（tarantula） | 0.037（zebra）；0.038（orca）；0.042（tarantula） |
| pug_spar/position | 0.250（left）；0.250（right） | 0.250（left）；0.250（right） | 0.250（left）；0.250（right） |
| pug_spar/attribute | immune | immune | immune |
| pug_spar/relation | immune | immune | immune |

## B. SPAR position 视角翻转比例（right+above 占比，z 检验 |z|<=3）

| model | seed0 | seed1 | seed2 |
|---|---|---|---|
| clip_vitb32 | 0.500 (z=+0.0) | 0.500 (z=+0.0) | 0.500 (z=+0.0) |
| clip_vitl14 | 0.500 (z=+0.0) | 0.500 (z=+0.0) | 0.500 (z=+0.0) |
| siglip2_so400m | 0.500 (z=+0.0) | 0.500 (z=+0.0) | 0.500 (z=+0.0) |

## H. 跨 seed 图片重叠（同因素三种子两两交集最大占比，描述性）

| dataset/factor | clip_vitb32 | clip_vitl14 | siglip2_so400m |
|---|---|---|---|
| clevr/object | 0.139 | 0.139 | 0.139 |
| clevr/color | 0.135 | 0.135 | 0.135 |
| clevr/size | 0.146 | 0.146 | 0.146 |
| clevr/material | 0.147 | 0.147 | 0.147 |
| clevr/relation | 0.143 | 0.143 | 0.143 |
| clevr/binding | 0.151 | 0.151 | 0.151 |
| pug_spar/scene | 0.051 | 0.051 | 0.051 |
| pug_spar/position | 0.067 | 0.067 | 0.067 |
| pug_spar/attribute | 0.107 | 0.107 | 0.107 |
| pug_spar/relation | 0.073 | 0.073 | 0.073 |

## E. e1_acc_by_pos_value.csv 一致性

- 全部 90 个 run×factor 中，分组 n 之和与总体 n、加权准确率与总体准确率一致性：全部通过（容差 1e-9）。
