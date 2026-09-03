# 面向 SAR 船舶检测的低频对齐蒸馏（LP-DSI + CROMA 教师）

> 基线：RT-DETRv4-S（HGNetv2-B0 + HybridEncoder + DFINETransformer，DINOv3 ViT-B/16 教师蒸馏）
> 数据集：HRSID（主实验）/ ship_dataset_v0、SSDD（泛化），单类（ship），输入 640×640
> 最终配置：`configs/rtv4_s_{hrsid,ship,ssdd}_lp.yml`（创新点全部由这三个配置承载）
> 推理额外开销为零：LP-DSI 仅训练期存在，推理图与 checkpoint 不变。

---

## 0. 基线回顾（创新的出发点）

### 0.1 RT-DETRv4 原始蒸馏（DSI）

学生侧在混合编码器的 AIFI 输出 $F_5 \in \mathbb{R}^{B\times 256\times H/32\times W/32}$ 上接线性投影器 $\mathcal{P}: \mathbb{R}^{256}\to\mathbb{R}^{768}$；教师 DINOv3 ViT-B/16（冻结）输出 patch token 图 $T \in \mathbb{R}^{B\times 768\times H_t\times W_t}$，双线性插值到学生网格后，逐 patch 做余弦对齐：

$$
\mathcal{L}_{\text{DSI}} = \frac{1}{HW}\sum_{i=1}^{HW}\left(1 - \frac{\langle \hat{s}_i, \hat{t}_i\rangle}{\lVert \hat{s}_i\rVert \, \lVert \hat{t}_i\rVert}\right),\qquad \hat{s}_i = \frac{s_i}{\lVert s_i\rVert_2},\ \hat{t}_i = \frac{t_i}{\lVert t_i\rVert_2}
$$

其中 $s_i = \mathcal{P}(F_5)_i$，$t_i = T_i$ 为第 $i$ 个位置的通道向量。

---

## 1. 核心创新 A：LP-DSI 低频对齐蒸馏

> 实现：`engine/rtv4/distill_modules.py` → `low_pass_alignment_loss`、`rtv4_criterion.loss_distillation`（`distill_mode: lp`）

### 1.1 动机

- 教师与学生存在**模态差距**（DINOv3 为自然光学图像预训练；即便换用 CROMA，教师预训练数据与检测数据的纹理实现也不同）；
- SAR 的**斑点噪声（speckle）**是像素级乘性噪声，其能量在特征图上体现为高频、局部化的响应；跨模态真正可迁移的是**低频语义结构**（哪里有什么），而非高频纹理实现（表面细节长什么样）；
- 原始 DSI 对全部频段无差别对齐 → 强迫学生在"教师也不可信的高频响应"上模仿教师，引入有害梯度。

**假设**：只对齐低频（平滑）分量，蒸馏信号的信噪比更高。

### 1.2 方法：一行公式，零新超参

对学生投影特征 $S=\mathcal{P}(F_5)$ 与教师特征 $T$（已插值对齐到同一网格），低通算子 $P$ 取 2×2 平均池化，损失即 0.1 节的余弦损失作用在平滑后的特征图上：

$$
\mathcal{L}_{\text{LP}} = \frac{1}{H'W'}\sum_{i=1}^{H'W'}\left(1 - \cos\big\langle P(S)_i,\ P(T)_i \big\rangle\right),\qquad P = \mathrm{AvgPool}_{2\times2}
$$

**与小波分解的等价性**（审稿人层面的论证）：单级 2D Haar 变换的低频子带 $\mathrm{LL}(i,j) = \tfrac12\sum_{a,b} x(2i{+}a, 2j{+}b) = 2\cdot \mathrm{AvgPool}_{2\times2}(x)(i,j)$，而余弦相似度对正常数缩放不变，因此"只对齐 LL 子带"与"avg-pool 后做余弦对齐"**严格等价**。平均池化即 Haar 低通的最简形式，无需引入小波框架、子带拆分或高频权重等任何额外机制与超参。奇数网格由池化自然 floor-crop，无需 padding。

**与原始 DSI 的关系**：`cosine` 模式（全频段对齐）即 $P=\mathrm{Id}$ 的特例；消融只需三组：无蒸馏 / cosine / lp，无需任何旋钮。

### 1.3 实现与开销

`low_pass_alignment_loss` 为两次 `F.avg_pool2d` 加一次逐位置余弦（~5 行代码），无参数、仅训练期存在，推理图与 checkpoint 完全不变。训练目标中直接替换原蒸馏项 $\mathcal{L}_{\text{DSI}} \to \mathcal{L}_{\text{LP}}$。

---

## 2. 核心创新 B：CROMA SAR 教师替换

> 实现：`engine/rtv4/croma_teacher.py`（`CROMATeacherModel`）、vendor 代码 `croma/`（官方 `use_croma.py` 原样拷贝）
> 配置：`configs/rtv4_hgnetv2_s_{hrsid,ship,ssdd}_croma.yml`

### 2.1 动机

LP-DSI（创新 A）从**损失端**缓解教师-学生域差距（不对齐高频）；CROMA 替换则从**教师端**直接缩小差距：DINOv3 在自然光学图像（LVD-1689M）上预训练，而 CROMA（Contrastive Radar-Optical Masked Autoencoders，NeurIPS 2023，arXiv 2311.00566）在 Sentinel-1/2 大规模配对数据上预训练，其 SAR 编码器对斑点噪声与 SAR 成像机理有原生建模能力。两者正交：教师换为 CROMA 后，LP-DSI / GAM 等损失端机制照常叠加。

### 2.2 教师结构与网格对齐

CROMA-base 的 SAR 编码器为 ViT-B 级（768 维、6 层、16 头），`patch_size=8`，输入 2 通道（VV/VH 后向散射）。教师包装的关键设计：

- **有效步幅 32 对齐**：输入 AvgPool 4×（640→160）后送入 CROMA，$4\times 8 = 32$ 与 DINOv3 方案（AvgPool 2× + patch 16）同构——教师 patch 网格（640 下 20×20）与学生 F5 网格**逐位置 1:1**，任意 /32 整除的多尺度输入（608~672）均自动保持对齐；160×160 同时接近 CROMA 的 120×120 预训练分辨率；
- **动态 ALiBi**：CROMA 位置编码是无参数 2D ALiBi 相对偏置，按固定 patch 数在初始化时预计算；多尺度训练中网格变化时按 (网格, 设备) 缓存重算（`_ensure_attn_bias`），实现支持矩形网格（方形网格下与官方 `get_2dalibi` 数值一致）；
- **通道映射**：HRSID/SSDD 为单极化灰度（磁盘上复制为 3 通道 JPG），取通道均值后广播到 VV/VH 两个槽位；
- **归一化匹配**：CROMA 预训练将 SAR 逐通道裁剪到 8bit 值域再 /255（即编码器消费 [0,1] 输入），与数据管线输出的 8-bit [0,1] 图像天然匹配，无需额外归一化。

### 2.3 输出契约与框架兼容

`CROMATeacherModel.forward` 返回 detached 的 $[B, 768, H/32, W/32]$ 特征图，与基线 DINOv3 教师完全同契约：

- 学生侧 `HybridEncoder.distill_teacher_dim: 768` 与投影器（Linear / MLP）不变；
- 蒸馏损失（cosine / lp）与 GAM 自适应权重机制不变；
- 教师冻结、不进 checkpoint、不包 DDP、在 autocast 之外以 fp32 运行——均继承自现有蒸馏框架；
- `teacher_model.type` 配置分发（`engine/core/yaml_config.py`）。

### 2.4 权重

```bash
wget https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt -O pretrain/CROMA_base.pt
# HF 不可达时用镜像：https://hf-mirror.com/antofuller/CROMA/resolve/main/CROMA_base.pt
```

---

## 3. 继承自基线框架的配套机制（非本文贡献点）

- **GAM 梯度自适应蒸馏权重**（RT-DETRv4 基线）：逐 epoch 统计 encoder-transformer 梯度占比，围绕目标 $\rho\pm\delta=11\%\pm1\%$ 自动调节 $\lambda$（`distill_adaptive_params`，`engine/solver/det_solver.py`）。
- **两阶段训练与 EMA-search 回滚**：`stop_epoch` 后重启 EMA；HRSID/SSDD 的 lp 配置启用了带 patience/cooldown/decay 下限/次数上限的门控（`collate_fn.ema_search_*`），ship 配置沿用默认。
- **Dense O2O 数据增广、FlatCosine 调度、MAL 匹配感知分类损失**：均继承自 RT-DETRv4-S 基线（`configs/base/rtv4.yml`）。

---

## 4. 总训练目标

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mal}} + 5\,\mathcal{L}_{\text{L1}} + 2\,\mathcal{L}_{\text{GIoU}} + 0.15\,\mathcal{L}_{\text{fgl}} + 1.5\,\mathcal{L}_{\text{ddf}} + \lambda(t)\,\mathcal{L}_{\text{LP}}
$$

各项：MAL（DEIM 匹配感知分类损失）、L1+GIoU 框损失、FGL（分布 focal 损失）、DDF（层间自蒸馏 KL）、**LP-DSI（本文，替换原余弦蒸馏）**。$\lambda(t)$ 由继承自 v4 的 GAM 逐 epoch 调节。辅助监督（逐解码层 aux、编码器 one2many、第一层传统头 pre、CDN 去噪组）结构与基线完全一致。

---

## 5. 最终配置与继承链

| 配置 | 数据集 | 说明 |
|---|---|---|
| `configs/rtv4_s_hrsid_lp.yml` | HRSID | 主实验（`distill_mode: lp` + EMA-search 门控） |
| `configs/rtv4_s_ship_lp.yml` | ship_dataset_v0 | 泛化 |
| `configs/rtv4_s_ssdd_lp.yml` | SSDD | 泛化（`distill_mode: lp` + EMA-search 门控） |
| `configs/rtv4_hgnetv2_s_{*}_croma.yml` | — | 三个 lp 配置的父配置（CROMA 教师、GAM、优化器与 epoch 计划） |
| `configs/dfine/dfine_hgnetv2_s_{*}.yml` | — | HGNetv2-B0 模型形状与 D-FINE 骨架默认值 |

LP-DSI 引入的**新超参数为零**；$\lambda_0$ 由 GAM 目标 $\rho\pm\delta=11\%\pm1\%$ 自动调节。

## 6. 代码位置索引

| 模块 | 文件 |
|---|---|
| 余弦对齐 / 低频对齐损失 | `engine/rtv4/distill_modules.py` |
| 蒸馏模式分发（cosine / lp） | `engine/rtv4/rtv4_criterion.py` → `loss_distillation` |
| 蒸馏投影器（Linear / MLP，训练期专用） | `engine/rtv4/hybrid_encoder.py`、`engine/rtv4/distill_projector.py` |
| CROMA 教师（动态 ALiBi / 通道映射 / 步幅 32 对齐） | `engine/rtv4/croma_teacher.py` |
| CROMA 官方推理代码（vendor，未修改） | `croma/use_croma.py` |
| 教师前向与注入、GAM 梯度探针 | `engine/solver/det_engine.py` → `train_one_epoch` |
| GAM 逐 epoch 权重调节、EMA-search 回滚 | `engine/solver/det_solver.py` → `fit` |
| 教师工厂分发（CROMA，支持 `teacher_model: null`） | `engine/core/yaml_config.py` → `teacher_model` |
| 权重函数 / GT 离散化 / 距离转换（D-FINE 原版） | `engine/rtv4/dfine_utils.py` |
| 模型级冒烟测试 | `tools/smoke_test_model.py` |
| 训练循环级冒烟测试（AMP / GAM 探针） | `tools/smoke_test_train_step.py` |
| 训练启动脚本 | `tools/train_sar_ablation.sh` |
