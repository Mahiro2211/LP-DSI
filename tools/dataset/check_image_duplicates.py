"""
Report-only duplicate check for a flat image dataset (optionally with YOLO
sidecar labels): md5 over file bytes.

- Exact duplicates: images with identical md5 (same content, possibly under
  different names).
- Informational: distinct images whose label txt content is byte-identical
  (common and benign for single-class 256x256 chips; reported, not removed).

Usage:
    python tools/dataset/check_image_duplicates.py /root/autodl-tmp/Dataset/ship_dataset_v0
"""

import argparse
import hashlib
import os
from collections import defaultdict

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Report exact duplicate images (md5) and identical label contents.')
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset files.')
    args = parser.parse_args()

    names = sorted(os.listdir(args.data_dir))
    img_names = [n for n in names if n.lower().endswith(IMG_EXTS)]
    txt_names = [n for n in names if n.endswith('.txt')]
    print(f'Scanning {len(img_names)} images, {len(txt_names)} label files in {args.data_dir}')

    img_md5 = {}  # filename -> md5
    md5_groups = defaultdict(list)  # md5 -> [filenames]
    for name in img_names:
        digest = file_md5(os.path.join(args.data_dir, name))
        img_md5[name] = digest
        md5_groups[digest].append(name)

    dup_groups = {d: files for d, files in md5_groups.items() if len(files) > 1}
    n_extra = sum(len(files) - 1 for files in dup_groups.values())
    if dup_groups:
        print(f'\n[!] EXACT DUPLICATE IMAGES: {len(dup_groups)} groups, {n_extra} redundant copies, e.g.:')
        for digest, files in list(sorted(dup_groups.items()))[:10]:
            print(f'    {digest}: {files}')
        if len(dup_groups) > 10:
            print(f'    ... and {len(dup_groups) - 10} more groups')
    else:
        print('\nOK: no exact duplicate images (all md5 unique).')

    if txt_names:
        label_groups = defaultdict(list)  # label content md5 -> [stems]
        for name in txt_names:
            digest = file_md5(os.path.join(args.data_dir, name))
            label_groups[digest].append(name[:-4])

        # only count groups mixing at least two byte-distinct images
        mixed = []
        for stems in label_groups.values():
            if len(stems) > 1 and len({img_md5[s + '.jpg'] for s in stems}) > 1:
                mixed.append(stems)
        n_imgs = sum(len(stems) for stems in mixed)
        print(f'Identical label content across distinct images: {len(mixed)} groups / {n_imgs} images '
              f'(informational only - benign for single-class chips), e.g.:')
        for stems in sorted(mixed, key=len, reverse=True)[:5]:
            print(f'    {len(stems)} images share one label file: {stems[:6]}{"..." if len(stems) > 6 else ""}')


if __name__ == '__main__':
    main()
