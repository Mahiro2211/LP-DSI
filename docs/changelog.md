# 改动说明：相对 RT-DETRv4 上游，除 CROMA 教师之外的改动记录

> 日期：2026-08-29
> 范围说明：本仓库相对 RT-DETRv4 上游的改动分两部分——
> ①**此前已完成的**（不在本文档展开）：引入 CROMA SAR 基础模型作为冻结蒸馏教师（`engine/rtv4/croma_teacher.py`、`croma/` vendor 代码、`*_croma.yml` 配置），以及 NWD / SA-FDR / P2 等模块；
> ②**本次改动**（本文档）：把创新点 A 从 FD-DSI 频域解耦蒸馏简化为 **LP-DSI 低频对齐蒸馏**，并删除配套的 SSM 投影器、qroi / GAP / SCD 等外围模块，统一消融配置到 CROMA 基准。
> 设计文档见 `docs/innovations.md`（已同步重写）。

---

## 1. 核心改动：FD-DSI → LP-DSI（低频对齐蒸馏）

### 1.1 之前的设计（已删除）

创新点 A 原为四层机制堆叠：

1. 单级 2D Haar 小波分解（`haar_dwt2d`，29 行，含奇数尺寸 replicate padding 及其修过的 bug）；
2. 频段解耦对齐损失（LL 强对齐 + LH/HL/HH 以 `w_hf=0.25` 软化）；
3. 关系蒸馏（token 自相似 Gram 矩阵对齐，`w_rel` + top-K 稀疏化 `rel_topk`）；
4. 任务感知前景加权（GT 框栅格化权重 `fg_weight`）。

此外 criterion 还有 7 个蒸馏相关构造超参，配套 qroi / GAP / SCD / SSM 投影器等模块，蒸馏相关代码约 **900 行、约 20 个对外超参**。作为会议论文的单个创新点过重，且实验未带来收益。

### 1.2 现在的设计

损失只有一行——对学生投影特征 $S$ 与教师特征 $T$（已对齐网格）做 2×2 平均池化后计算逐位置余弦损失：

$$
\mathcal{L}_{\text{LP}} = \mathrm{CosineLoss}\big(\mathrm{AvgPool}_{2\times2}(S),\ \mathrm{AvgPool}_{2\times2}(T)\big)
$$

- 实现仅约 5 行（`engine/rtv4/distill_modules.py::low_pass_alignment_loss`），**零新增超参**；
- `distill_mode` 从 3 种（`cosine / fd / fd_rel`）精简为 2 种（`cosine / lp`），criterion 蒸馏构造参数从 7 个减到 1 个；
- 论文叙事不变："斑噪/模态差异集中在高频，跨模态只迁移低频语义"，公式从 4 子带组合缩为 1 行。

### 1.3 为什么可以放心删掉小波（数学依据）

1. **严格等价**：Haar 低频子带 $\mathrm{LL} = 2\cdot\mathrm{AvgPool}_{2\times2}(x)$，而余弦相似度对正常数缩放不变 → "只对齐 LL 子带"与"avg-pool 后做余弦对齐"**逐位等价**。小波框架不提供任何额外能力。
2. **Parseval 论证**：Haar 是正交变换，若损失为 L2 且 `w_hf=1`，各子带损失之和恒等于原始特征上的 L2——原设计的实际贡献只剩"高频降权"一个标量操作，包装与收益不成比例。
3. **机制层面**：蒸馏作用在 stride-32 的 token 网格（约 19×21），一个 2×2 Haar 块对应 64 像素尺度；而斑点噪声是**像素级**乘性噪声，token 级"高频子带"早已不是斑噪。这既解释了原设计无收益，也是简化后叙事更稳的原因。

### 1.4 消融设计的变化

| | 之前 | 现在 |
|---|---|---|
| 核心消融 | fd / fd_rel / full 等多组，含 `w_hf`、`w_rel`、`topk`、`fg` 旋钮 | **三组单变量闭环**：nodistill / cosine（全频段）/ lp（仅低频） |
| 新增超参 | `w_hf=0.25`、`w_rel=1.0`、`rel_topk=128`、`fg_weight=3` 等 | **0 个** |
| 频域故事 | Haar 四子带公式组 | 一行公式 + LL≡AvgPool 等价性论证 |

GAM 梯度自适应权重**机制与代码保留不动**（继承自 RT-DETRv4 基线），论文中降级为基线框架的一句话描述，不再作为本工作贡献点展开。

---

## 2. 删除的外围模块

| 模块 | 原位置 | 删除原因 |
|---|---|---|
| SS4D / SSM 蒸馏投影器 | `distill_projector.py`（SS4DBlock 85 行 + SSMProjector）、`hybrid_encoder.py` 的 ssm 分支与 2 个构造参数 | 最大单点复杂度（6 组参数、FFT 卷积、fp32 孤岛），无收益证据；投影器回归 Linear / MLP 两种 |
| Query 级 RoI 蒸馏（qroi） | `rtv4_criterion.loss_query_roi`、`rtv4.py` 的 query 投影、`dfine_decoder.py` 的 `query_embeddings` 训练期导出 | 属被砍掉的创新 E（TAD）的一部分，整块移除后 decoder 恢复原版输出契约 |
| GAP 空间锚定 | `gap_anchor_loss`、`croma_teacher.py` 的 `return_gap`、criterion 的 `loss_gap_anchor` 与 InfoNCE 温度超参 | 创新F，整块删除；教师恒返回特征图 |
| SCD 散斑一致性 | `apply_speckle_noise`、`det_engine.py`/`det_solver.py` 的 speckle 数据流与配置块 | 创新 D，整块删除；训练循环回到"师生看同一张图"的简单数据流 |
| 关系蒸馏 + 前景加权 | `relation_distillation_loss`、`foreground_weight_map`、criterion 的 4 个相关超参 | 随 fd_rel / TAD 一起移除，保证 lp 的"零超参"故事成立 |
| 过期副本 | `engine/rtv4/.ipynb_checkpoints/rtv4_criterion-checkpoint.py` | 纯 cruft |

删除后 `engine/` 内无任何残留引用（已全局 grep 验证）。

## 3. 配置系统改动

- **`teacher_model: null` 支持**（`engine/core/yaml_config.py`）：显式置空可跳过教师构建。用途：无蒸馏消融配置可以直接继承蒸馏基准配置，只覆写蒸馏相关开关，不再需要手工拼装一份"长得像"的独立配置。
- **三个核心消融配置统一到 CROMA 基准**（`configs/rtv4/ablation/`）：
  - `rtv4_s_hrsid_nodistill.yml` —— 继承 `rtv4_hgnetv2_s_hrsid_croma.yml`，`teacher_model: null` + 去蒸馏项 + GAM 关闭；
  - `rtv4_s_hrsid_cosine.yml` —— CROMA 基准的逐项展开版（独立 output_dir，显式 `distill_mode: cosine`），作为论文基线的单文件可读记录；
  - `rtv4_s_hrsid_lp.yml` —— 继承 CROMA 基准，仅 `distill_mode: lp`。
  三组同数据/同调度/同 seed，单变量递进。
- **删除 14 个失效消融配置**（fd / fdrel / full / ssmproj / croma_gap / croma_scd / croma_tad / croma_ssmproj 及其 SSDD 镜像）——它们引用已删除的模式与模块，不删会直接跑飞。保留 nwd / p2 / safdr 系列。
- **主配置未动**（`rtv4_hgnetv2_s_{hrsid,ssdd}_croma*.yml`）：现有复现结果不受影响；待消融验证 lp ≥ cosine 后再切 `distill_mode: lp`。
- **`tools/train_sar_ablation.sh` 重写**：核心验证链 nodistill → cosine → lp（保留 safdr / nwd / p2 阶段与自动续训逻辑）。

## 4. 测试与文档

- `tools/smoke_test_model.py`：纯函数检查改为 cosine / low-pass 等价性验证（含"avg-pool 与 Haar LL 缩放不变"断言）、奇数网格（19×19、21×17）覆盖；模型级用例更新为存留配置清单；
- `tools/smoke_test_train_step.py`：移除 speckle / GAP 分支描述，保留 AMP 边界与 GAM 探针断言；
- `docs/innovations.md`：创新 A 重写为 LP-DSI（含等价性论证），删除 D/E/F 章节，GAM 降级为基线描述，消融表与代码索引同步。

## 5. 验证结果（均通过）

| 验证项 | 结果 |
|---|---|
| 模块导入（全部改动文件） | 通过 |
| 纯函数冒烟（cosine / lp / 奇数网格 / 缩放不变性） | 通过 |
| 模型级冒烟（nodistill / cosine / lp / CROMA 主配置：前向+反向+推理） | 通过 |
| 训练循环级冒烟（lp @ 608 奇数网格 + AMP：`loss_distill≈2.98`，GAM 探针 2 次） | 通过 |
| nodistill：无蒸馏项（33 个损失项），GAM 探针 0 次，教师未加载 | 通过 |
| cosine 展开版 vs CROMA 基准配置合并等价性 | 仅 output_dir 与显式 distill_mode 不同 |

## 6. 净效果

- 蒸馏相关代码约 **900 行 → 约 300 行**（`distill_modules.py` 216→50 行，`distill_projector.py` 166→27 行，criterion 删去 3 个损失函数与 5 个超参，decoder / rtv4 / solver 删去死代码路径）；
- 对外蒸馏超参约 **20 个 → 1 个**（`distill_mode`）；
- 消融配置 **19 个 → 8 个**，核心验证只需 3 组；
- 推理图与 checkpoint 格式不受影响（所有删除均为训练期路径）。

## 7. 文件改动清单

| 文件 | 改动 |
|---|---|
| `engine/rtv4/distill_modules.py` | 重写：仅保留 `cosine_alignment_loss` + `low_pass_alignment_loss` |
| `engine/rtv4/rtv4_criterion.py` | 删 fd/fd_rel 分支、qroi/gap 损失、5 个构造超参 |
| `engine/rtv4/rtv4.py` | 删 gap 传参与 query 投影块 |
| `engine/rtv4/dfine_decoder.py` | 删 `query_embeddings` 训练期导出 |
| `engine/rtv4/hybrid_encoder.py` | 删 ssm 投影器分支与 2 个构造参数 |
| `engine/rtv4/distill_projector.py` | 删 SS4DBlock / SSMProjector，仅留 MLPProjector |
| `engine/rtv4/croma_teacher.py` | 删 `return_gap` 与 GAP-FFN 分支 |
| `engine/solver/det_engine.py` | 删 speckle 数据流与 teacher tuple 解包 |
| `engine/solver/det_solver.py` | 删 speckle 配置读取与传参（GAM 块保留不动） |
| `engine/core/yaml_config.py` | 新增 `teacher_model: null` 支持 |
| `configs/rtv4/ablation/` | 删 14 个失效配置；新增 nodistill / cosine / lp |
| `tools/train_sar_ablation.sh` | 重写为 3 组核心验证 + safdr/nwd/p2 |
| `tools/smoke_test_model.py` / `smoke_test_train_step.py` | 同步 lp 模式与存留配置 |
| `docs/innovations.md` | 创新 A 重写为 LP-DSI，删 D/E/F 章节 |
| `docs/changelog.md` | 本文档（新增） |

---

# 2026-09-03：按最终论文配置（rtv4_s_{hrsid,ship,ssdd}_lp.yml）裁剪代码库

> 范围：以三个 lp 配置解析后的完整设置为准，删除所有不可达/未启用的代码路径
> 与此前论文遗留的实验代码。数值等价性已验证（固定种子前向+反向，
> 42 个损失项逐项 bit-identical，梯度总和一致）。

## 1. 修复（配置链在目录扁平化时已断裂）

- 恢复缺失的 `configs/dfine/dfine_hgnetv2_s_{hrsid,ship_v0,ssdd}.yml` 与 `configs/base/dfine_hgnetv2.yml`（自 zip 备份）；
- 恢复缺失的 `croma/use_croma.py`（同上）；
- 修正 include 相对路径：lp 配置与 croma 配置扁平化到 `configs/` 后 `../` 前缀失效；
- **修复 `engine/rtv4/dfine_decoder.py` 的隐蔽 bug**：dn 去噪组 teacher corners 误为
  `out_corners[-1]`（300 长，与 198 长的 dn pred 不匹配，训练即崩溃；此前被过期的
  `__pycache__` 掩盖）。恢复为 D-FINE 上游的 `dn_out_corners[-1]` / `dn_out_logits[-1]`。

## 2. 删除的教师与模型族（其他论文/上游变体）

| 删除项 | 原因 |
|---|---|
| `dinov3/`（整个 vendored 仓库，173 个 py 文件） | DINOv3 教师是 RT-DETRv4 基线的教师，lp 配置全部使用 CROMA |
| `engine/rtv4/dinov3_teacher.py`、`dinov2_teacher.py` | 同上；教师工厂只保留 `CROMATeacherModel`（`teacher_model: null` 支持保留） |
| `engine/rtv4/rtdetrv2_decoder.py` | RT-DETRv2 decoder，无配置引用 |
| `engine/backbone/{presnet,csp_resnet,csp_darknet,timm_model,torchvision_model,test_resnet}.py` | lp 配置只用 HGNetv2 |
| `engine/data/dataset/voc_detection.py`、`voc_eval.py` | VOC 数据集未使用 |
| `engine/solver/clas_solver.py`、`clas_engine.py` | 分类任务未使用，TASKS 仅剩 detection |

## 3. 删除的未启用模块路径（lp 配置未开启，属前序探索）

| 删除项 | 位置 |
|---|---|
| NWD 损失与匹配代价 | `box_ops.nwd_cxcywh`、`matcher.cost_nwd`、`criterion.loss_nwd`（losses 列表无 `nwd`） |
| SA-FDR 逐查询曲率 | `dfine_utils.{weighting_function_batch,translate_gt_batch,bbox2distance_batch}`、decoder 的 `adaptive_reg_scale` 分支、criterion 的 2-D reg_scale 分支 |
| 顺带删除未引用的 `box_ops.masks_to_boxes` | 无任何调用方 |
| obj365→COCO 类权重映射 | `engine/solver/_solver.py`（`obj365_ids` 表 + `map_class_weights` + 尾部 ~340 行注释类名表；单类 SAR 配置不会从 obj365 checkpoint 调参，尺寸不匹配的头参数改为跳过并提示） |

上述路径在 lp 配置下本就不触发（数值等价验证成立）；备份在 `RT-DETRv4-minimal.zip`。

## 4. tools 清理

- 删除：`convert_dinov3_weights.py`（DINOv3 专用）、`train_ship_compare.sh`（DEIM/D-FINE 对比，前序工作）、`reference/safe_training.sh`（obj365/coco 上游启动器）、`dataset/{remap,resize}_obj365.py`、`dataset/{augment_ssdd_rotation,verify_ssdd_rot}.py`（旋转增广实验，最终配置用标准 train.json）；
- 重写：`train_sar_ablation.sh` → 三个 lp 配置的启动器（hrsids/ship/ssdd）；
- 更新：`smoke_test_model.py`（CROMA-only、当前配置清单）、`smoke_test_train_step.py`、两个数据集冒烟测试、`export_onnx.py` / `get_info.py` 默认配置路径；
- 删除配置：`configs/dataset/custom_detection.yml`（无引用）。

## 5. 文档

- `README.md` 重写：面向本论文（LP-DSI + CROMA、三个数据集配置、继承链、冒烟测试），移除 COCO 性能表 / DINOv3 教师准备 / deim·dfine·rtv2 复现说明；
- `docs/innovations.md` 重写：只保留 LP-DSI、CROMA 教师，GAM/EMA-search 降级为继承机制，删除 SA-FDR / NWD / P2 章节。

## 6. 验证（均通过）

| 验证项 | 结果 |
|---|---|
| 三个 lp 配置完整加载（include 链全解析） | 通过（teacher=CROMA、distill_mode=lp、losses 无 nwd） |
| 固定种子前向+损失+反向，42 项逐项对比 | 与清理前 bit-identical（total / grad_sum 一致） |
| 全局 grep 无残留引用（dinov3/nwd/adaptive_reg_scale/VOC/Clas 等） | 通过 |
