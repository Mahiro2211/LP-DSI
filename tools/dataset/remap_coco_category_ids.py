"""
Remap COCO category ids to 0-based contiguous ids (required by RT-DETR training
configs, which use category_id directly as the class label when
`remap_mscoco_category: False`).

Writes the result to a new file; the original annotation file is left untouched.

Usage:
    python tools/dataset/remap_coco_category_ids.py --ann_file <coco.json> [--suffix contiguous]
"""

import json
import argparse
import os


def remap_category_ids(data):
    old_ids = sorted(cat['id'] for cat in data['categories'])
    old2new = {old: new for new, old in enumerate(old_ids)}

    data['categories'] = [
        {**cat, 'id': old2new[cat['id']]} for cat in data['categories']
    ]
    for ann in data['annotations']:
        ann['category_id'] = old2new[ann['category_id']]

    return data, old2new


def main():
    parser = argparse.ArgumentParser(description='Remap COCO category ids to 0-based contiguous ids.')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to the COCO annotation json file.')
    parser.add_argument('--suffix', type=str, default='contiguous', help='Suffix of the output file (default: contiguous).')
    args = parser.parse_args()

    with open(args.ann_file) as f:
        data = json.load(f)

    data, old2new = remap_category_ids(data)

    stem, ext = os.path.splitext(args.ann_file)
    output_file = f'{stem}_{args.suffix}{ext}'
    with open(output_file, 'w') as f:
        json.dump(data, f)

    print(f'Category id mapping: {old2new}')
    print(f'Number of images: {len(data["images"])}, annotations: {len(data["annotations"])}')
    print(f'Saved to {output_file}')


if __name__ == '__main__':
    main()
