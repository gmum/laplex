"""Build ImageNet10 as a tree of symlinks into the full ImageNet on /shared.

The team ran the projection experiments with ``--root_dataset`` pointing at a full
ImageNet (``--root_dataset /media/dane/datasets/ImageNet``); the loader
``CountSketch_LogisticRegression/imagenet10_dataset.py`` keeps only the ten
synsets of ``IMAGENET10_SYNSETS`` and expects ``root/{train,val}/<synset>/``.
The cluster copy has per-synset train folders but a flat val folder, so val is
rebuilt from ``LOC_val_solution.csv`` (50 images per synset).

    python3 make_imagenet10.py                 # -> nn_training/data/imagenet10
    python3 make_imagenet10.py --out DIR --force
    python3 make_imagenet10.py --all-classes   # all 1000 synsets -> nn_training/data/imagenet
                                               # (classification_heads/: DATA_ROOT, VAL_ROOT=.../val)

Standard library only (runs on the login node). The links point into /shared, so the
container must bind /shared/sets/datasets (sif.sh does).
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOADER = HERE / "CountSketch_LogisticRegression" / "imagenet10_dataset.py"
DEFAULT_SRC = Path("/shared/sets/datasets/imagenet/CLS-LOC")
DEFAULT_VAL_CSV = Path("/shared/sets/datasets/ImageNet/LOC_val_solution.csv")
DEFAULT_OUT = HERE / "data" / "imagenet10"
DEFAULT_OUT_ALL = HERE / "data" / "imagenet"


def loader_synsets() -> list[str]:
    """Read IMAGENET10_SYNSETS from the loader without importing torch."""
    tree = ast.parse(LOADER.read_text())
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == "IMAGENET10_SYNSETS" for t in targets):
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"IMAGENET10_SYNSETS not found in {LOADER}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="ImageNet root with train/<synset>/ and flat val/")
    p.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV, help="LOC_val_solution.csv")
    p.add_argument("--out", type=Path, help=f"default {DEFAULT_OUT}, with --all-classes {DEFAULT_OUT_ALL}")
    p.add_argument("--force", action="store_true", help="remove an existing --out first")
    p.add_argument("--all-classes", action="store_true", help="every synset of --src/train instead of ImageNet10")
    args = p.parse_args()
    if args.out is None:
        args.out = DEFAULT_OUT_ALL if args.all_classes else DEFAULT_OUT

    if args.all_classes:
        synsets = sorted(d.name for d in (args.src / "train").iterdir() if d.is_dir())
    else:
        synsets = loader_synsets()
    if args.out.exists():
        if not args.force:
            print(f"{args.out} exists; use --force to rebuild", file=sys.stderr)
            return 1
        shutil.rmtree(args.out)

    for syn in synsets:
        src = args.src / "train" / syn
        if not src.is_dir():
            raise FileNotFoundError(src)
        dst = args.out / "train" / syn
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst, target_is_directory=True)

    wanted = set(synsets)
    counts = {s: 0 for s in synsets}
    with open(args.val_csv, newline="") as f:
        for row in csv.DictReader(f):
            syn = row["PredictionString"].split()[0]
            if syn not in wanted:
                continue
            src = args.src / "val" / f"{row['ImageId']}.JPEG"
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = args.out / "val" / syn / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, dst)
            counts[syn] += 1

    if args.all_classes:
        print(f"{len(synsets)} synsets, val {sum(counts.values())} images")
    else:
        for syn in synsets:
            n_train = sum(1 for _ in (args.out / "train" / syn).iterdir())
            print(f"{syn}: train {n_train:5d}  val {counts[syn]:3d}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
