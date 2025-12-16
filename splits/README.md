Splits directory

This folder contains `.npy` index files produced by `scripts/make_splits.py`.

Files:
- `train.npy`, `val.npy`, `test.npy`: integer index arrays for each split.
- `metadata.json`: records the split `mode`, `seed`, `ratios`, and counts.

How to regenerate:
```bash
python scripts/make_splits.py --h5 afm_patches_256_2ch.h5 --out splits --mode by-base-id --ratios 0.8 0.1 0.1 --seed 0
```

Usage with training script:
```bash
python scripts/train_autoencoder.py --h5 afm_patches_256_2ch.h5 --split-dir splits --split-name train --val-split val
```
