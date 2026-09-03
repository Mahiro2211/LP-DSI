"""
Convert the ship_dataset_v0 dataset (flat YOLO-format: <stem>.jpg + <stem>.txt
with normalized `class_id cx cy w h` lines, single class 0 = ship) into COCO
json annotations with a train/val split.

Split modes:
- stratified (default): shuffle within each source prefix (Gao_ship_hh/hv/vh/vv,
  Sen_ship_hh/hv/vh/vv, newship, ship) so both splits keep the same
  sensor/polarization mix.
- random: shuffle all images and split once (paper protocol for
  SAR-Ship-Dataset: random 8:2 train/val after deduplication).

In both modes, byte-identical duplicate images (same md5) are treated as one
unit and always assigned to the same split, so duplicates never leak across
train/val. Images are NOT copied or modified (unless --convert-grayscale is
passed): `img_folder` keeps pointing at the flat dataset dir, and the split
membership is defined by the `file_name` entries of each json.

Category ids are 0-based contiguous (category_id 0), required by RT-DETR
configs that use `remap_mscoco_category: False`.

Usage:
    python tools/dataset/convert_ship_dataset_v0_to_coco.py \
        [--data-dir /root/autodl-tmp/Dataset/ship_dataset_v0] \
        [--split-mode random] [--val-ratio 0.2] [--seed 0]
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict

from PIL import Image

# source prefix -> stratification group, e.g. Gao_ship_hh_0201... -> Gao_ship_hh
PREFIX_RE = re.compile(r'^(Gao_ship_[a-z]{2}|Sen_ship_[a-z]{2}|newship|ship)')


def parse_prefix(stem):
    m = PREFIX_RE.match(stem)
    return m.group(1) if m else 'other'


def parse_yolo_labels(txt_path, img_w, img_h):
    """YOLO normalized cx,cy,w,h -> COCO absolute x,y,w,h (top-left corner)."""
    boxes = []
    with open(txt_path) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f'  [warn] {txt_path}:{line_no}: expected 5 fields, got {len(parts)}, skipped')
                continue
            class_id = int(parts[0])
            if class_id != 0:
                print(f'  [warn] {txt_path}:{line_no}: unexpected class id {class_id}, skipped')
                continue
            cx, cy, w, h = (float(v) for v in parts[1:])
            x = (cx - w / 2) * img_w
            y = (cy - h / 2) * img_h
            w *= img_w
            h *= img_h
            # clamp to image bounds
            x, y = max(x, 0.0), max(y, 0.0)
            w, h = min(w, img_w - x), min(h, img_h - y)
            if w <= 0 or h <= 0:
                print(f'  [warn] {txt_path}:{line_no}: box outside image after clamp, dropped')
                continue
            boxes.append((x, y, w, h))
    return boxes


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Convert ship_dataset_v0 (YOLO) to COCO with stratified train/val split.')
    parser.add_argument('--data-dir', type=str, default='/root/autodl-tmp/Dataset/ship_dataset_v0',
                        help='Dataset root containing flat *.jpg / *.txt pairs.')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation fraction (default: 0.1).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for the split (default: 0).')
    parser.add_argument('--split-mode', choices=['stratified', 'random'], default='stratified',
                        help="stratified: split within each source prefix (Gao/Sen/newship/ship); "
                             'random: shuffle all images and split once, matching the paper protocol '
                             '(SAR-Ship-Dataset: random 8:2 train/val). '
                             'Both modes keep byte-identical duplicates in a single split.')
    parser.add_argument('--convert-grayscale', action='store_true',
                        help='Convert non-RGB (grayscale) images to RGB in place (JPEG quality 95). '
                             'Required for the training pipeline: pil_to_tensor keeps the PIL channel count.')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(out_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.txt'))
    if not txt_files:
        raise RuntimeError(f'No .txt label files found in {data_dir}')

    groups = defaultdict(list)  # prefix -> [(stem, img_w, img_h, boxes), ...]
    skipped = 0
    non_rgb = 0
    for txt_name in txt_files:
        stem, _ = os.path.splitext(txt_name)
        img_path = os.path.join(data_dir, stem + '.jpg')
        if not os.path.exists(img_path):
            print(f'[warn] missing image for label {txt_name}, skipped')
            skipped += 1
            continue
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            if img.mode != 'RGB':
                if args.convert_grayscale:
                    rgb = img.convert('RGB')
                    rgb.save(img_path, quality=95)
                    rgb.close()
                else:
                    non_rgb += 1
                    print(f'  [warn] {stem}.jpg: PIL mode {img.mode} (not RGB) '
                          f'{"- converted" if args.convert_grayscale else "- use --convert-grayscale to fix"}')
        boxes = parse_yolo_labels(os.path.join(data_dir, txt_name), img_w, img_h)
        groups[parse_prefix(stem)].append((stem, img_w, img_h, boxes, file_md5(img_path)))

    rng = random.Random(args.seed)
    splits = {'train': [], 'val': []}

    def split_units(units, quota):
        """Shuffle duplicate-groups (images with identical bytes stay together)
        and fill val up to `quota` images, so duplicates never straddle splits."""
        rng.shuffle(units)
        val, train, n_val = [], [], 0
        for unit in units:
            if n_val < quota:
                val.extend(unit)
                n_val += len(unit)
            else:
                train.extend(unit)
        return train, val

    if args.split_mode == 'random':
        units = defaultdict(list)
        for prefix in sorted(groups):
            for entry in groups[prefix]:
                units[entry[4]].append(entry)
        n_total = sum(len(v) for v in groups.values())
        splits['train'], splits['val'] = split_units(list(units.values()), round(n_total * args.val_ratio))
    else:
        for prefix in sorted(groups):
            units = defaultdict(list)
            for entry in groups[prefix]:
                units[entry[4]].append(entry)
            tr, va = split_units(list(units.values()), round(len(groups[prefix]) * args.val_ratio))
            splits['train'].extend(tr)
            splits['val'].extend(va)

    dup_groups = defaultdict(int)
    for entries in groups.values():
        for e in entries:
            dup_groups[e[4]] += 1
    n_dup = sum(1 for c in dup_groups.values() if c > 1)
    if n_dup:
        print(f'[info] {n_dup} byte-identical image groups; each group is assigned to a single split')

    ann_id = 0
    stats = defaultdict(lambda: {'images': 0, 'boxes': 0})
    for split, entries in splits.items():
        images, annotations = [], []
        for img_id, (stem, img_w, img_h, boxes, _digest) in enumerate(entries):
            images.append({
                'id': img_id,
                'file_name': stem + '.jpg',
                'width': img_w,
                'height': img_h,
            })
            for (x, y, w, h) in boxes:
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': 0,
                    'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    'area': round(w * h, 2),
                    'iscrowd': 0,
                })
                ann_id += 1

            prefix = parse_prefix(stem)
            stats[f'{prefix}/{split}']['images'] += 1
            stats[f'{prefix}/{split}']['boxes'] += len(boxes)

        coco = {
            'images': images,
            'annotations': annotations,
            'categories': [{'id': 0, 'name': 'ship', 'supercategory': 'ship'}],
        }
        out_path = os.path.join(out_dir, f'{split}.json')
        with open(out_path, 'w') as f:
            json.dump(coco, f)
        print(f'{split}: {len(images)} images, {len(annotations)} boxes -> {out_path}')

    print(f'\nSplit stats (images/boxes), mode={args.split_mode}, val_ratio={args.val_ratio}, seed={args.seed}:')
    for prefix in sorted(groups):
        tr, va = stats[f'{prefix}/train'], stats[f'{prefix}/val']
        print(f'  {prefix:12s} train: {tr["images"]:6d} img / {tr["boxes"]:6d} boxes   '
              f'val: {va["images"]:5d} img / {va["boxes"]:6d} boxes')
    n_tr_img = sum(stats[f'{p}/train']['images'] for p in groups)
    n_va_img = sum(stats[f'{p}/val']['images'] for p in groups)
    n_tr_box = sum(stats[f'{p}/train']['boxes'] for p in groups)
    n_va_box = sum(stats[f'{p}/val']['boxes'] for p in groups)
    print(f'  {"TOTAL":12s} train: {n_tr_img:6d} img / {n_tr_box:6d} boxes   '
          f'val: {n_va_img:5d} img / {n_va_box:6d} boxes')
    if skipped:
        print(f'[warn] {skipped} label files skipped (missing image)')
    if non_rgb:
        print(f'[warn] {non_rgb} images are not RGB mode - rerun with --convert-grayscale to fix in place')


if __name__ == '__main__':
    main()
