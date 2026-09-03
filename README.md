<h2 align="center">LP-DSI: Low-Pass Feature Distillation from a SAR Foundation Model for Ship Detection</h2>

---

Official implementation of the conference paper: **low-pass feature alignment
distillation (LP-DSI) on a CROMA SAR teacher** for SAR ship detection, built
on the RT-DETRv4-S detector (HGNetv2-B0 + HybridEncoder + D-FINE decoder).

> The paper's three experiment configs are
> [`configs/rtv4_s_hrsid_lp.yml`](./configs/rtv4_s_hrsid_lp.yml) (main, HRSID),
> [`configs/rtv4_s_ship_lp.yml`](./configs/rtv4_s_ship_lp.yml) (ship_dataset_v0)
> and [`configs/rtv4_s_ssdd_lp.yml`](./configs/rtv4_s_ssdd_lp.yml) (SSDD
> generalization). Everything else in `configs/` is their inheritance chain.

## Method

1. **CROMA teacher** (teacher side): the frozen distillation teacher is
   [CROMA](https://arxiv.org/abs/2311.00566) (NeurIPS 2023), a foundation model
   pretrained on Sentinel-1/2 SAR–optical pairs, replacing the natural-image
   DINOv3 teacher of the RT-DETRv4 baseline. The SAR encoder (ViT-B, patch 8)
   sees the input avg-pooled 4x, so the teacher grid stays 1:1 with the
   student's stride-32 F5 feature; ALiBi position biases are recomputed
   dynamically for multi-scale training (`engine/rtv4/croma_teacher.py`).
2. **LP-DSI** (loss side): only the low-frequency band of the student-projected
   and teacher feature maps is aligned — a 2x2 average pooling (exactly the
   Haar LL sub-band) followed by the per-position cosine loss. Speckle- and
   texture-dominated high-frequency responses are ignored. Zero extra
   hyperparameters (`distill_mode: lp`,
   `engine/rtv4/distill_modules.py::low_pass_alignment_loss`).
3. **GAM adaptive distill weight** (inherited from the RT-DETRv4 baseline):
   the `loss_distill` weight is retuned every epoch so the encoder-transformer
   gradient share stays at rho +/- delta percent, plus stage-2 EMA-search
   rollback gating (`engine/solver/det_solver.py`).

Design details and the math behind the LL == AvgPool equivalence: see
[docs/innovations.md](./docs/innovations.md).

---

## 1. Getting Started

### 1.1 Environment setup

```shell
conda create -n rtv4 python=3.11.9
conda activate rtv4
pip install -r requirements.txt
```

Main dependencies: `torch`, `torchvision`, `faster-coco-eval`, `PyYAML`,
`tensorboard`, `scipy`, `calflops`, `einops` (`wandb` optional for W&B logging).

### 1.2 Download pretrained weights

Two sets of pretrained weights are needed. Put them under `pretrain/` in the
repository root:

```
pretrain/
├── hgnetv2/
│   └── PPHGNetV2_B0_stage1.pth      # student backbone (ImageNet stage-1)
└── CROMA_base.pt                    # frozen distillation teacher
```

**(a) HGNetv2-B0 student backbone** — released by the D-FINE authors
(~13 MB):

```shell
mkdir -p pretrain/hgnetv2
wget https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth \
     -O pretrain/hgnetv2/PPHGNetV2_B0_stage1.pth
```

If the file is missing, training also tries to download it automatically
(rank 0 downloads, other ranks wait). With a restricted network, download it
manually with the command above — training **exits** if the backbone weights
cannot be loaded.

**(b) CROMA-base distillation teacher** — from the official
[antofuller/CROMA](https://github.com/antofuller/croma) release (~650 MB,
loads directly, no conversion needed):

```shell
# HuggingFace
wget https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt -O pretrain/CROMA_base.pt

# ... or the hf-mirror.com mirror if HuggingFace is unreachable
wget https://hf-mirror.com/antofuller/CROMA/resolve/main/CROMA_base.pt -O pretrain/CROMA_base.pt
```

> CROMA-large can be used by setting `croma_size: "large"` together with
> `HybridEncoder.distill_teacher_dim: 1024` in the config (the paper uses base).

The exact paths are configured in the yml files and can be changed there:

| Weights | Config key | Default path |
|---|---|---|
| HGNetv2 backbone | `HGNetv2.local_model_dir` ([base/dfine_hgnetv2.yml](./configs/base/dfine_hgnetv2.yml)) | `./pretrain/hgnetv2/` |
| CROMA teacher | `teacher_model.croma_weights_path` ([rtv4_hgnetv2_s_*_croma.yml](./configs/rtv4_hgnetv2_s_hrsid_croma.yml)) | `pretrain/CROMA_base.pt` |

### 1.3 Dataset preparation

All three datasets are single-class (`ship`, `num_classes: 1`) COCO-format
datasets. Download them and edit the paths in the corresponding
`configs/dataset/*.yml`:

| Dataset | Config | Notes |
|---|---|---|
| [HRSID](https://github.com/chaozhong2010/HRSID) | [hrsid_detection.yml](./configs/dataset/hrsid_detection.yml) | remap `category_id` 1 -> 0 first (step below) |
| [SSDD](https://github.com/TianwenZhang0825/Official-SSDD) | [ssdd_detection.yml](./configs/dataset/ssdd_detection.yml) | uses `category_id: 0` as-is |
| ship_dataset_v0 | [ship_v0_detection.yml](./configs/dataset/ship_v0_detection.yml) | built with `tools/dataset/convert_ship_dataset_v0_to_coco.py` |

Example — edit `img_folder` / `ann_file` in the chosen yml:

```yaml
train_dataloader:
  dataset:
    img_folder: /path/to/HRSID_JPG/JPEGImages
    ann_file: /path/to/HRSID_JPG/annotations/train2017_contiguous.json
```

HRSID prerequisite — the original annotations use `category_id: 1`, but with
`remap_mscoco_category: False` the id is used directly as the training label,
so remap it once (originals are left untouched):

```shell
python tools/dataset/remap_coco_category_ids.py --ann_file /path/to/HRSID_JPG/annotations/train2017.json
python tools/dataset/remap_coco_category_ids.py --ann_file /path/to/HRSID_JPG/annotations/test2017.json
```

This produces the `*_contiguous.json` files the config points to. SSDD and
ship_dataset_v0 already use 0-based ids and need no conversion.

---

## 2. How to Run

### 2.1 Training

**Via the launcher** (recommended; auto-resumes from `last.pth` if a run was
interrupted):

```shell
bash tools/train_sar_ablation.sh hrsid   # one dataset: hrsid | ship | ssdd
bash tools/train_sar_ablation.sh all     # all three, sequentially
```

**Directly with torchrun** (single GPU):

```shell
torchrun --master_port=7777 --nproc_per_node=1 train.py \
    -c configs/rtv4_s_hrsid_lp.yml --use-amp --seed=0
```

Multi-GPU (e.g. 4 GPUs; `total_batch_size: 32` in
[configs/base/dataloader.yml](./configs/base/dataloader.yml) is split across
ranks automatically):

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py \
    -c configs/rtv4_s_hrsid_lp.yml --use-amp --seed=0
```

Replace `hrsid` with `ship` or `ssdd` for the other two datasets. Outputs
(weights + `log.txt` + eval stats) land in the config's `output_dir`, e.g.
`./outputs/rtv4_s_hrsid_lp/`:

| File | Content |
|---|---|
| `last.pth` | rolling checkpoint for `--resume` |
| `best_stg1.pth` / `best_stg2.pth` | best weights before / after the EMA restart (ranked by AP50) |
| `log.txt` | per-epoch train/test metrics (JSON lines) |
| `eval/` | COCOeval result dumps |

Useful flags:

| Flag | Effect |
|---|---|
| `--use-amp` | mixed-precision training |
| `--seed 0` | fix the seed for reproducibility |
| `--test-only -r model.pth` | evaluation only |
| `-r outputs/rtv4_s_hrsid_lp/last.pth` | resume an interrupted run |
| `-t model.pth` | fine-tune from a checkpoint (heads are re-initialized) |
| `--wandb` | log metrics to Weights & Biases (soft dependency) |
| `--no-ckpt` | skip all `.pth` writes (metrics still logged) |
| `-u key=value` | override any config entry, e.g. `-u total_batch_size=16` |

### 2.2 Evaluation only

```shell
torchrun --master_port=7777 --nproc_per_node=1 train.py \
    -c configs/rtv4_s_hrsid_lp.yml --test-only -r outputs/rtv4_s_hrsid_lp/best_stg2.pth
```

For deployment, extract the EMA weights from a checkpoint:

```shell
python tools/reference/convert_weight.py outputs/rtv4_s_hrsid_lp
```

### 2.3 Smoke tests (recommended before the first real run)

```shell
# pure-function checks: cosine / low-pass losses, odd token grids (no downloads)
python tools/smoke_test_model.py --ops

# data pipeline: loads each config, pulls a batch, checks the labels
python tools/dataset/smoke_test_hrsid_ssdd.py     # requires HRSID + SSDD on disk
python tools/dataset/smoke_test_ship_v0.py        # requires ship_dataset_v0

# model level: teacher + forward + loss + backward + inference
# (requires pretrain/CROMA_base.pt)
python tools/smoke_test_model.py configs/rtv4_s_hrsid_lp.yml

# training-loop level: AMP boundaries + GAM gradient probe
python tools/smoke_test_train_step.py configs/rtv4_s_hrsid_lp.yml
```

### 2.4 Inference & deployment

```shell
# PyTorch inference / visualization
python tools/inference/torch_inf.py -c configs/rtv4_s_hrsid_lp.yml -r model.pth \
    --input image.jpg --device cuda:0

# export ONNX (uses the EMA-extracted weights)
pip install onnx onnxsim
python tools/deployment/export_onnx.py --check -c configs/rtv4_s_hrsid_lp.yml -r model.pth

# ONNX / TensorRT inference on images or videos
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
python tools/inference/trt_inf.py --trt model.engine --input image.jpg

# FLOPs / MACs / params (calflops is already in requirements.txt)
python tools/benchmark/get_info.py -c configs/rtv4_s_hrsid_lp.yml
```

---

## 3. Config inheritance chain

```
rtv4_s_{hrsid,ship,ssdd}_lp.yml          # distill_mode: lp  (+ EMA-search gating, hrsid/ssdd)
└── rtv4_hgnetv2_s_{hrsid,ship,ssdd}_croma.yml   # CROMA teacher, GAM, optimizer, epoch plan
    ├── dfine/dfine_hgnetv2_s_{hrsid,ship_v0,ssdd}.yml   # HGNetv2-B0 model shape
    │   ├── dataset/{hrsid,ship_v0,ssdd}_detection.yml   # dataset paths
    │   ├── runtime.yml / base/{dataloader,optimizer}.yml
    │   └── base/dfine_hgnetv2.yml                      # model skeleton + criterion defaults
    └── base/rtv4.yml                                   # dense o2o aug, flatcosine, mal losses
```

## 4. Citation

If you find this work helpful, please consider citing:

```bibtex
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Zijun Liao and Yian Zhao and Xin Shan and Yu Yan and Chang Liu and Lei Lu and Xiangyang Ji and Jie Chen},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
```

## 5. Acknowledgement

Our work is built upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR),
[D-FINE](https://github.com/Peterande/D-FINE), [DEIM](https://github.com/Intellindust-AI-Lab/DEIM),
[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) and the teacher model
[CROMA](https://github.com/antofuller/croma). Thanks to these remarkable works!
