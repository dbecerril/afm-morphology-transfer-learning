#!/usr/bin/env python3
"""Create train/val/test splits for the AFM patches HDF5.

Usage examples:
  python scripts/make_splits.py --h5 afm_patches_256_2ch.h5 --out splits --mode random --ratios 0.8 0.1 0.1 --seed 0
  python scripts/make_splits.py --h5 afm_patches_256_2ch.h5 --out splits --mode by-base-id --ratios 0.8 0.1 0.1 --seed 0

Outputs (in --out dir):
  train.npy, val.npy, test.npy : arrays of integer indices
  metadata.json : records mode, seed, ratios, counts
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import h5py


def parse_args():
    p = argparse.ArgumentParser(description="Make dataset splits (random or by-base-id)")
    p.add_argument("--h5", required=True, help="HDF5 file with patches")
    p.add_argument("--out", default="splits", help="Output directory for split .npy files")
    p.add_argument("--mode", choices=("random", "by-base-id"), default="random")
    p.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), help="Train/val/test ratios (must sum to 1)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def random_split(n, ratios, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    t = int(ratios[0] * n)
    v = int((ratios[0] + ratios[1]) * n)
    train = perm[:t]
    val = perm[t:v]
    test = perm[v:]
    return train, val, test


def by_base_id_split(h5path, ratios, seed=0):
    with h5py.File(h5path, "r") as f:
        if "patches/base_id" not in f:
            raise RuntimeError("HDF5 file does not contain patches/base_id needed for by-base-id split")
        base_ids = f["patches/base_id"][:].astype(str)
    groups = defaultdict(list)
    for i, b in enumerate(base_ids):
        groups[b].append(i)
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    total = len(base_ids)
    target_train = int(ratios[0] * total)
    target_val = int(ratios[1] * total)

    train_inds = []
    val_inds = []
    test_inds = []
    acc = 0
    for k in keys:
        g = groups[k]
        if len(train_inds) < target_train:
            train_inds.extend(g)
        elif len(val_inds) < target_val:
            val_inds.extend(g)
        else:
            test_inds.extend(g)
    # convert to numpy arrays
    return np.array(train_inds, dtype=np.int64), np.array(val_inds, dtype=np.int64), np.array(test_inds, dtype=np.int64)


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    if args.mode == "random":
        with h5py.File(args.h5, "r") as f:
            n = int(f["patches/proc"].shape[0])
        train, val, test = random_split(n, ratios, args.seed)
    else:
        train, val, test = by_base_id_split(args.h5, ratios, args.seed)

    np.save(outdir / "train.npy", train)
    np.save(outdir / "val.npy", val)
    np.save(outdir / "test.npy", test)

    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "ratios": ratios,
        "counts": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote splits to {outdir} - counts: {meta['counts']}")


if __name__ == "__main__":
    main()
