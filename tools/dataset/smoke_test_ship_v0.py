"""Smoke test for the ship_dataset_v0 configs: load config, build val dataloader,
pull one batch and verify labels are within [0, num_classes) and images are 3-channel."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from engine.core import YAMLConfig

for cfg_file in [
    'configs/dfine/dfine_hgnetv2_s_ship_v0.yml',
    'configs/rtv4_s_ship_lp.yml',
]:
    print(f'===== {cfg_file} =====')
    cfg = YAMLConfig(cfg_file)

    num_classes = cfg.yaml_cfg['num_classes']
    print(f'num_classes = {num_classes}')
    assert num_classes == 1

    val_dl = cfg.val_dataloader
    batch = next(iter(val_dl))
    imgs, targets = batch
    labels = torch.cat([t['labels'] for t in targets])
    print(f'val batch: {len(imgs)} images, {len(labels)} boxes, '
          f'label values = {labels.unique().tolist()}, '
          f'img shape = {tuple(imgs[0].shape)}')
    assert labels.numel() > 0, 'no boxes in batch'
    assert labels.min() >= 0 and labels.max() < num_classes, 'label out of range!'
    assert imgs[0].shape[0] == 3, f'expected 3-channel images, got {tuple(imgs[0].shape)}'

    train_ds = cfg.train_dataloader.dataset
    print(f'train dataset: {len(train_ds)} images, img_folder={train_ds.img_folder}')
    img, tgt = train_ds[0]
    print(f'train sample[0]: labels = {tgt["labels"].unique().tolist()}, boxes = {tuple(tgt["boxes"].shape)}')
    assert tgt['labels'].min() >= 0 and tgt['labels'].max() < num_classes

print('\nAll smoke tests passed.')
