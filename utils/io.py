import json
from pathlib import Path
import numpy as np
import h5py

from compute_stats import compute_channel_mean_std
from make_splits import by_base_id_split, random_split

AUTO_SPLIT_RATIOS = (0.8, 0.1, 0.1)
def resolve_h5_path(h5_path: str) -> Path:
    candidate = Path(h5_path)
    if candidate.exists():
        return candidate

    repo_candidate = REPO_ROOT / h5_path
    if repo_candidate.exists():
        return repo_candidate

    datasets_candidate = REPO_ROOT / "datasets" / h5_path
    if datasets_candidate.exists():
        return datasets_candidate

    raise FileNotFoundError(
        f"HDF5 file '{h5_path}' not found. Pass --h5 with the correct dataset path (absolute or relative to repo)."
    )



def ensure_split_files(args, ratios=AUTO_SPLIT_RATIOS):
    split_dir = Path(args.split_dir)
    train_path = split_dir / f"{args.split_name}.npy"
    val_path = split_dir / f"{args.val_split}.npy" if args.val_split else None
    needs_train = not train_path.exists()
    needs_val = val_path is not None and not val_path.exists()
    if not (needs_train or needs_val):
        return {}

    split_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_inds, val_inds, test_inds = by_base_id_split(args.h5, ratios, args.seed)
        split_mode = "by-base-id"
    except Exception as exc:
        print(f"Auto split (by-base-id) unavailable ({exc}); falling back to random split.")
        with h5py.File(args.h5, "r") as f:
            total = int(f["patches/proc"].shape[0])
        train_inds, val_inds, test_inds = random_split(total, ratios, args.seed)
        split_mode = "random"

    def save_unique(paths, arr, label):
        seen = set()
        for p in paths:
            if p is None:
                continue
            path_obj = Path(p)
            if path_obj in seen:
                continue
            np.save(path_obj, arr)
            seen.add(path_obj)
            print(f"Wrote {label} split ({len(arr)} samples) -> {path_obj}")

    save_unique([split_dir / "train.npy", train_path], train_inds, "train")
    val_targets = [split_dir / "val.npy"]
    if val_path is not None:
        val_targets.append(val_path)
    save_unique(val_targets, val_inds, "val")
    save_unique([split_dir / "test.npy"], test_inds, "test")

    meta = {
        "mode": split_mode,
        "seed": args.seed,
        "ratios": ratios,
        "counts": {"train": int(len(train_inds)), "val": int(len(val_inds)), "test": int(len(test_inds))},
        "generated_by": "scripts/train_autoencoder.py",
    }
    with open(split_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"train": train_inds, "val": val_inds, "test": test_inds}


def ensure_channel_norm_file(args, preferred_train_indices=None):
    stats_path = Path(args.stats)
    if stats_path.exists():
        return

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    train_indices = preferred_train_indices
    if train_indices is None:
        split_dir = Path(args.split_dir)
        train_file = split_dir / f"{args.split_name}.npy"
        if train_file.exists():
            train_indices = np.load(train_file)

    if train_indices is not None:
        print(f"Stats file {stats_path} missing. Computing channel stats from {len(train_indices)} train samples.")
    else:
        print(f"Stats file {stats_path} missing. Computing channel stats from entire dataset.")

    mean, std = compute_channel_mean_std(args.h5, indices=train_indices)
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote channel norm stats -> {stats_path}")
