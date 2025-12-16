from pathlib import Path
import json
import numpy as np
import h5py


def compute_channel_mean_std(
    h5_path: str,
    dataset: str = "patches/proc",
    indices=None,
    batch: int = 256,
):
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset]  # (N, C, H, W)
        N, C, _, _ = ds.shape

        if indices is None:
            indices = np.arange(N)
        else:
            indices = np.asarray(indices, dtype=np.int64)
            # h5py fancy indexing expects monotonically increasing indices
            if indices.ndim != 1:
                indices = indices.reshape(-1)
            indices = np.sort(indices)

        count = np.zeros((C,), dtype=np.int64)
        mean = np.zeros((C,), dtype=np.float64)
        m2 = np.zeros((C,), dtype=np.float64)

        for i in range(0, len(indices), batch):
            idx = indices[i:i+batch]
            x = ds[idx]  # (B,C,H,W)
            x = x.reshape(x.shape[0], C, -1)

            for c in range(C):
                vals = x[:, c, :].ravel()
                for v in vals:
                    count[c] += 1
                    delta = v - mean[c]
                    mean[c] += delta / count[c]
                    m2[c] += delta * (v - mean[c])

        var = m2 / (count - 1)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


if __name__ == "__main__":
    h5 = "./datasets/afm_patches_256.h5"

    # example: random 80% train split
    with h5py.File(h5, "r") as f:
        N = f["patches/proc"].shape[0]

    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    n_train = int(0.8 * N)
    train_idx = idx[:n_train]

    mean, std = compute_channel_mean_std(h5, indices=train_idx)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    Path("stats").mkdir(exist_ok=True)
    with open("stats/channel_norm.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Saved stats:", stats)
