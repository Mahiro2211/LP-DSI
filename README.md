<h2 align="center">LP-DSI: Low-Pass Feature Distillation from a SAR Foundation Model for Ship Detection</h2>

---

Official implementation of the conference paper: **low-pass feature alignment
distillation (LP-DSI) on a CROMA SAR teacher** for SAR ship detection


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
python train.py -c ./configs/rtv4_s_hrsid_lp.yml --use-amp --seed=3401
python train.py -c ./configs/rtv4_s_ssdd_lp.yml --use-amp --seed=3401
python train.py -c ./configs/rtv4_s_ship_lp.yml --use-amp --seed=3401
```

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


### 2.2 Inference & deployment

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



## 3. Acknowledgement

Our work is built upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR),
[D-FINE](https://github.com/Peterande/D-FINE), [DEIM](https://github.com/Intellindust-AI-Lab/DEIM),
[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) and the teacher model
[CROMA](https://github.com/antofuller/croma). Thanks to these remarkable works!
