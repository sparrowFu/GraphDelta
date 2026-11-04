"""阶段二 E4：受控结构编辑诊断（工作方案第 5–7 节）。

子模块：
- pairs     编辑对清单构建 / 划分 / 落盘（Step 2）
- render    规格驱动渲染的调度、校验与账本（Step 3，配合 render/ 驱动）
- captions  before/after 文本内容构造（第 3、5 节）
- data      清单 + 渲染产物装载、候选池构造
- features  float32 特征缓存接入（复用 common/src/features）
- paired    E4-A 配对结构判断
- delta     E4-B 视觉变化诊断 + 线性探针
- text_delta E4-C 转换级跨模态诊断
- transfer  E4-D 跨场景方向迁移
- stats     置换检验与聚类 bootstrap 封装
- run       编排（Step 4 入口，脚本 run_e4.py 调用）
"""
