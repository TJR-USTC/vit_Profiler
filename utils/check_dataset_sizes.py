"""Check image sizes under a dataset directory and report per-class and overall stats.

Usage examples:
    # print summary to stdout
    python -m utils.check_dataset_sizes --dir dataset

    # write a CSV report
    python -m utils.check_dataset_sizes --dir dataset --out-csv report.csv

The script walks `--dir` recursively. The 'class' for each image is taken as the
first path component under the root (i.e. dataset/<class>/image.jpg). Files that
cannot be opened are listed with an error status.
"""
from __future__ import annotations

import os
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover - runtime will show import error
    Image = None


def iter_image_files(root: str, exts: Optional[List[str]] = None):
    """Yield (full_path, rel_path) for files under root matching extensions."""
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                yield full, rel


def get_image_size(path: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (width, height, error_msg). If success, error_msg is None."""
    if Image is None:
        return None, None, "Pillow not installed"
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(w), int(h), None
    except Exception as e:
        return None, None, str(e)


def check_dataset_image_sizes(root: str) -> Tuple[Dict[str, List[Tuple[int, int]]], List[Dict[str, str]]]:
    """Walk root, read image sizes and return per-class sizes and per-file records.

    Returns:
      - class_map: dict class_name -> list of (w,h)
      - records: list of dicts with keys: path, rel_path, class, width, height, status, error
    """
    class_map: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    records: List[Dict[str, str]] = []

    for full, rel in iter_image_files(root):
        parts = rel.split(os.sep)
        cls = parts[0] if len(parts) > 1 else "__root__"
        w, h, err = get_image_size(full)
        if err is None:
            class_map[cls].append((w, h))
            records.append({"path": full, "rel_path": rel, "class": cls, "width": str(w), "height": str(h), "status": "ok", "error": ""})
        else:
            records.append({"path": full, "rel_path": rel, "class": cls, "width": "", "height": "", "status": "error", "error": err})

    return class_map, records


def summarize(class_map: Dict[str, List[Tuple[int, int]]]) -> Dict[str, Dict[str, float]]:
    """Return summary stats per class and overall (min/max/mean for width and height).

    Output format: {class: {"count": n, "w_min":.., "w_max":.., "w_mean":.., "h_min":.., ...}}
    """
    out: Dict[str, Dict[str, float]] = {}
    overall_ws: List[int] = []
    overall_hs: List[int] = []
    total = 0
    for cls, sizes in class_map.items():
        if not sizes:
            out[cls] = {"count": 0}
            continue
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        total += len(sizes)
        overall_ws.extend(ws)
        overall_hs.extend(hs)
        out[cls] = {
            "count": len(sizes),
            "w_min": min(ws),
            "w_max": max(ws),
            "w_mean": sum(ws) / len(ws),
            "h_min": min(hs),
            "h_max": max(hs),
            "h_mean": sum(hs) / len(hs),
        }

    if overall_ws:
        out["__overall__"] = {
            "count": len(overall_ws),
            "w_min": min(overall_ws),
            "w_max": max(overall_ws),
            "w_mean": sum(overall_ws) / len(overall_ws),
            "h_min": min(overall_hs),
            "h_max": max(overall_hs),
            "h_mean": sum(overall_hs) / len(overall_hs),
        }
    else:
        out["__overall__"] = {"count": 0}

    return out


def write_csv(records: List[Dict[str, str]], out_csv: str) -> None:
    fieldnames = ["path", "rel_path", "class", "width", "height", "status", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Check image sizes under a dataset directory")
    p.add_argument("--dir", required=True, help="dataset root directory")
    p.add_argument("--out-csv", default=None, help="optional CSV report path")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    root = args.dir
    class_map, records = check_dataset_image_sizes(root)
    stats = summarize(class_map)

    # print per-class stats
    for cls, s in sorted(stats.items()):
        if cls == "__overall__":
            print("== Overall ==")
        else:
            print(f"Class: {cls}")
        for k, v in s.items():
            print(f"  {k}: {v}")
        print("")

    if args.out_csv:
        write_csv(records, args.out_csv)
        print(f"Wrote CSV report to {args.out_csv}")


if __name__ == "__main__":
    main()
