""" 
Process gwyddion files in a given folder
does background substraction, row alignment, clips outliers and saves to .h5 file
example run:
python preprocess/process_gwy.py --input_dir data/data_gwy/ 
--channel_title Topography 
--stride 128 --out_h5 afm_patches_256.h5
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple

import h5py
import numpy as np
import gwyfile


# ---------- Loading ----------

class AFMMeta(dict):
    """Simple metadata container (json-serializable)."""
    pass


def find_channel_index_by_title(obj, desired_title: str) -> int:
    desired = desired_title.strip().lower()
    for k, v in obj.items():
        if k.endswith("/title") and str(v).strip().lower() == desired:
            parts = k.split("/")
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    available = [str(v) for k, v in obj.items() if k.endswith("/title")]
    raise ValueError(f"Channel '{desired_title}' not found. Available: {available}")


def load_gwy_height(path: str | Path, channel_title: str = "Height") -> Tuple[np.ndarray, AFMMeta]:
    """Load a .gwy file and return (height_array, metadata) for the chosen channel."""
    path = Path(path)
    obj = gwyfile.load(str(path))
    chan = find_channel_index_by_title(obj, channel_title)
    df = obj[f"/{chan}/data"]

    z = np.array(df.data, dtype=np.float32)
    yres, xres = z.shape

    xreal_m = float(df.xreal)
    yreal_m = float(df.yreal)

    meta = AFMMeta(
        source_file=str(path),
        channel_title=channel_title,
        xres=int(xres),
        yres=int(yres),
        xreal_m=xreal_m,
        yreal_m=yreal_m,
        dx_m_per_px=xreal_m / xres,
        dy_m_per_px=yreal_m / yres,
    )
    return z, meta


# ---------- Preprocessing ----------

def crop_border(z: np.ndarray, frac: float = 0.0) -> np.ndarray:
    """Crop a fraction of the border on all sides (e.g. frac=0.05 crops 5%)."""
    if frac <= 0:
        return z
    if frac >= 0.5:
        raise ValueError("crop_border frac must be < 0.5")

    z = np.asarray(z)
    h, w = z.shape
    cy = int(round(h * frac))
    cx = int(round(w * frac))

    cy = min(cy, (h - 1) // 2)
    cx = min(cx, (w - 1) // 2)

    return z[cy : h - cy, cx : w - cx]


def subtract_best_fit_plane(z: np.ndarray) -> np.ndarray:
    """Fit plane z = a*x + b*y + c (least squares) and subtract it."""
    z = np.asarray(z, dtype=np.float32)
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(h * w, dtype=np.float32)]
    coeff, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    plane = (coeff[0] * xx + coeff[1] * yy + coeff[2]).astype(np.float32)
    return z - plane


def align_rows_by_median(z: np.ndarray) -> np.ndarray:
    """Line-by-line correction: subtract each row's median."""
    z = np.asarray(z, dtype=np.float32)
    return z - np.median(z, axis=1, keepdims=True)


def robust_clip(z: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """
    Clip outliers to ±sigma * (robust std) around the median.
    Uses MAD-based robust sigma (less sensitive to spikes than mean/std).
    """
    if sigma is None or sigma <= 0:
        return z

    z = np.asarray(z, dtype=np.float32)
    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-12
    robust_std = 1.4826 * mad

    lo = med - sigma * robust_std
    hi = med + sigma * robust_std
    return np.clip(z, lo, hi)


def preprocess_raw_and_proc(
    z_raw: np.ndarray,
    *,
    crop_border_frac: float = 0.0,
    do_plane: bool = True,
    do_row_median: bool = True,
    clip_sigma: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (raw_cropped, processed), keeping them aligned for patching.
    Crops first so edges don't bias plane/line corrections.
    """
    raw_cropped = crop_border(z_raw.astype(np.float32, copy=False), frac=crop_border_frac)

    proc = raw_cropped
    if do_plane:
        proc = subtract_best_fit_plane(proc)
    if do_row_median:
        proc = align_rows_by_median(proc)
    if clip_sigma is not None:
        proc = robust_clip(proc, sigma=float(clip_sigma))

    return raw_cropped, proc


# ---------- Patch extraction ----------

def iter_patch_coords(h: int, w: int, patch: int, stride: int) -> Iterator[Tuple[int, int]]:
    """Yield top-left (y, x) for a grid of patches."""
    if h < patch or w < patch:
        return
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            yield y, x


# ---------- HDF5 writer ----------

def append_to_resizable(ds, arr: np.ndarray):
    """Append along axis=0 to an HDF5 dataset with maxshape=(None, ...)."""
    n_old = ds.shape[0]
    n_new = n_old + arr.shape[0]
    ds.resize((n_new, *ds.shape[1:]))
    ds[n_old:n_new] = arr


def build_h5_from_folder(
    input_dir: Path,
    out_h5: Path,
    channel_title: str = "Height",
    patch: int = 256,
    stride: int = 256,
    crop_border_frac: float = 0.05,
    clip_sigma: float | None = 8.0,
    max_files: int | None = None,
):
    """
    Loads all .gwy files in input_dir, preprocesses (crop -> plane remove -> row median -> clip),
    extracts patches, and writes one HDF5 file.
    """
    gwy_files = sorted(input_dir.rglob("*.gwy"))
    if max_files is not None:
        gwy_files = gwy_files[:max_files]

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5, "w") as f:
        # Resizable datasets for patches
        d_raw = f.create_dataset(
            "patches/raw",
            shape=(0, patch, patch),
            maxshape=(None, patch, patch),
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(64, patch, patch),
        )
        d_proc = f.create_dataset(
            "patches/proc",
            shape=(0, patch, patch),
            maxshape=(None, patch, patch),
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(64, patch, patch),
        )

        # For traceability: which file + where did each patch come from?
        d_file = f.create_dataset(
            "patches/source_file",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=4,
            chunks=(1024,),
        )
        d_yx = f.create_dataset(
            "patches/top_left_yx",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype="int32",
            compression="gzip",
            compression_opts=4,
            chunks=(1024, 2),
        )

        # Store per-scan metadata as a JSON-lines string dataset
        meta_ds = f.create_dataset(
            "scans/meta_jsonl",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=4,
            chunks=(256,),
        )

        # Global params for reproducibility
        f.attrs["channel_title"] = channel_title
        f.attrs["patch"] = patch
        f.attrs["stride"] = stride
        f.attrs["preprocess"] = json.dumps(
            {
                "crop_border_frac": crop_border_frac,
                "plane": True,
                "row_median": True,
                "clip_sigma": clip_sigma,
            }
        )

        total_patches = 0

        for idx, fp in enumerate(gwy_files, start=1):
            try:
                z_raw, meta = load_gwy_height(fp, channel_title=channel_title)
            except Exception as e:
                print(f"[{idx}/{len(gwy_files)}] SKIP {fp.name}: load error: {e}")
                continue

            # Preprocess (keeps raw/proc aligned via shared cropping)
            z_raw_c, z_proc = preprocess_raw_and_proc(
                z_raw,
                crop_border_frac=crop_border_frac,
                do_plane=True,
                do_row_median=True,
                clip_sigma=clip_sigma,
            )

            # Extract patches on the same grid for both
            h, w = z_raw_c.shape
            coords = list(iter_patch_coords(h, w, patch, stride))
            if not coords:
                print(f"[{idx}/{len(gwy_files)}] SKIP {fp.name}: too small after crop ({h}x{w})")
                continue

            raw_patches = np.empty((len(coords), patch, patch), dtype=np.float32)
            proc_patches = np.empty((len(coords), patch, patch), dtype=np.float32)
            yx = np.empty((len(coords), 2), dtype=np.int32)

            for i, (y, x) in enumerate(coords):
                raw_patches[i] = z_raw_c[y:y+patch, x:x+patch]
                proc_patches[i] = z_proc[y:y+patch, x:x+patch]
                yx[i] = (y, x)

            # Append patches
            append_to_resizable(d_raw, raw_patches)
            append_to_resizable(d_proc, proc_patches)

            # Append provenance arrays
            n = len(coords)

            d_file.resize((d_file.shape[0] + n,))
            d_file[-n:] = [str(fp)] * n

            append_to_resizable(d_yx, yx)

            # Append scan metadata row (include preprocessing used)
            meta = dict(meta)
            meta.update(
                {
                    "crop_border_frac": crop_border_frac,
                    "clip_sigma": clip_sigma,
                }
            )
            meta_line = json.dumps(meta, ensure_ascii=False)
            meta_ds.resize((meta_ds.shape[0] + 1,))
            meta_ds[-1] = meta_line

            total_patches += n
            print(f"[{idx}/{len(gwy_files)}] OK {fp.name}: {n} patches (total {total_patches})")

        print(f"Done. Wrote {total_patches} patches to {out_h5}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--out_h5", type=str, required=True)
    p.add_argument("--channel_title", type=str, default="Height")
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)  # set e.g. 128 for overlap
    p.add_argument("--crop_border_frac", type=float, default=0.05)
    p.add_argument("--clip_sigma", type=float, default=4.0)
    p.add_argument("--max_files", type=int, default=None)
    args = p.parse_args()

    build_h5_from_folder(
        input_dir=Path(args.input_dir),
        out_h5=Path(args.out_h5),
        channel_title=args.channel_title,
        patch=args.patch,
        stride=args.stride,
        crop_border_frac=args.crop_border_frac,
        clip_sigma=args.clip_sigma,
        max_files=args.max_files,
    )
