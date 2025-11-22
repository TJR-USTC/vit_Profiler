"""Split a folder-per-class dataset into train/val/test with given ratios.

Example structure expected for source_dir:
  dataset/
    Angry/
      1.jpg
      2.png
    Happy/
      a.jpg

This script will create the following under dst_dir (default: src_dir_split):
  train/<class>/*
  val/<class>/*
  test/<class>/*

Usage example:
  python -m utils.split_dataset --src dataset/RiceLeafDisease --dst dataset_split --seed 42 --copy

Supports copy (default), move (--move) or create symlinks (--symlink).
Supports dry-run to preview file counts without making changes.
"""
from __future__ import annotations

import os
import argparse
import shutil
import random
from typing import List, Tuple


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def is_image_file(fn: str) -> bool:
    return os.path.splitext(fn)[1].lower() in IMAGE_EXTS


def list_class_files(src_dir: str) -> dict:
    """Return mapping class_name -> list of full file paths."""
    classes = {}
    for entry in os.scandir(src_dir):
        if entry.is_dir():
            cls = entry.name
            paths = []
            for root, _, files in os.walk(entry.path):
                for f in files:
                    if is_image_file(f):
                        paths.append(os.path.join(root, f))
            classes[cls] = sorted(paths)
    return classes


def split_list(items: List[str], ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
    """Split items into three lists according to ratios (train,val,test).

    Uses integer counts: train = int(n*ratios[0]); val = int(n*ratios[1]); test = remainder.
    """
    n = len(items)
    r0, r1, r2 = ratios
    n_train = int(n * r0)
    n_val = int(n * r1)
    n_test = n - n_train - n_val
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def transfer_file(src: str, dst: str, mode: str = 'copy') -> None:
    ensure_dir(os.path.dirname(dst))
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'move':
        shutil.move(src, dst)
    elif mode == 'symlink':
        # create relative symlink if possible
        try:
            if os.path.exists(dst):
                os.remove(dst)
            rel = os.path.relpath(src, os.path.dirname(dst))
            os.symlink(rel, dst)
        except OSError:
            # fallback to absolute symlink
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def split_dataset(src_dir: str, dst_dir: str, ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                  seed: int = 42, mode: str = 'copy', dry_run: bool = False) -> None:
    """Main function to split dataset.

    mode: 'copy'|'move'|'symlink'
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    classes = list_class_files(src_dir)
    random_state = random.Random(seed)

    summary = {}
    for cls, files in classes.items():
        files_copy = list(files)
        random_state.shuffle(files_copy)
        train_files, val_files, test_files = split_list(files_copy, ratios)
        summary[cls] = (len(files), len(train_files), len(val_files), len(test_files))

        for subset_name, subset_files in (('train', train_files), ('val', val_files), ('test', test_files)):
            for src_path in subset_files:
                rel = os.path.relpath(src_path, src_dir)
                # keep filename, place under dst_dir/subset/cls/
                dst_path = os.path.join(dst_dir, subset_name, cls, os.path.basename(src_path))
                if dry_run:
                    # only print
                    print(f"[DRY] {subset_name}: {rel} -> {dst_path}")
                else:
                    transfer_file(src_path, dst_path, mode=mode)

    # print summary
    print("\nSplit summary:")
    print("class,total,train,val,test")
    for cls, (total, ntrain, nval, ntest) in sorted(summary.items()):
        print(f"{cls},{total},{ntrain},{nval},{ntest}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Split dataset into train/val/test by class')
    p.add_argument('--src', required=True, help='source dataset dir (folder-per-class)')
    p.add_argument('--dst', default=None, help='destination root dir (if not set, uses <src>_split)')
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--test-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--copy', action='store_true', help='copy files (default)')
    group.add_argument('--move', action='store_true', help='move files')
    group.add_argument('--symlink', action='store_true', help='create symlinks instead of copying')
    p.add_argument('--dry-run', action='store_true', help='print actions without performing them')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    src = args.src
    dst = args.dst or (src.rstrip(os.sep) + '_split')
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if args.move:
        mode = 'move'
    elif args.symlink:
        mode = 'symlink'
    else:
        mode = 'copy'

    split_dataset(src, dst, ratios=ratios, seed=args.seed, mode=mode, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
