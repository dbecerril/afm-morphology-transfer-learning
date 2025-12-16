""" 
Process gwyddion files in a given folder
does background substraction, row alignment, clips outliers and saves to .h5 file
This version works when you want to merge topogrpahy and phase/friction
python preprocess/process_gwy_multichannel.py --input_dir data/data_gwy/ --topo_channel_title Topography --aux_channel_title Topography --stride 128 --out_h5 afm_patches_256.h5
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import gwyfile


# ---------- Loading ----------

class AFMMeta(dict):
    """Simple metadata container (json-serializable)."""
    pass


def _list_channel_titles(obj) -> List[Tuple[str, str]]:
    titles: List[Tuple[str, str]] = []
    for k, v in obj.items():
        if k.endswith("/title"):
            titles.append((k, str(v)))
    return titles


def _find_first_channel_index(obj) -> int:
    """
    Many .gwy exports contain exactly one channel. This finds the smallest index
    that has a '/<idx>/data' entry.
    """
    idxs = []
    for k in obj.keys():
        m = re.match(r"^/(\d+)/data$", str(k))
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        raise ValueError("No '/<idx>/data' fields found in .gwy container.")
    return min(idxs)


def find_channel_index_by_title(obj, desired_title: str) -> int:
    """Find channel index whose title matches desired_title (case-insensitive)."""
    desired = desired_title.strip().lower()
    for k, v in obj.items():
        if k.endswith("/title") and str(v).strip().lower() == desired:
            parts = k.split("/")
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    available = [str(v) for k, v in obj.items() if k.endswith("/title")]
    raise ValueError(f"Channel '{desired_title}' not found. Available: {available}")


def load_gwy_channel(
    path: str | Path,
    *,
    channel_title: Optional[str] = None,
) -> Tuple[np.ndarray, AFMMeta]:
    """
    Load a .gwy file and return (array, metadata).

    If channel_title is None, loads the first channel found.
    """
    path = Path(path)
    obj = gwyfile.load(str(path))

    if channel_title is None:
        chan = _find_first_channel_index(obj)
        title = str(obj.get(f"/{chan}/title", ""))
    else:
        chan = find_channel_index_by_title(obj, channel_title)
        title = channel_title

    df = obj[f"/{chan}/data"]
    z = np.array(df.data, dtype=np.float32)
    yres, xres = z.shape

    xreal_m = float(df.xreal)
    yreal_m = float(df.yreal)

    meta = AFMMeta(
        source_file=str(path),
        channel_title=title,
        xres=int(xres),
        yres=int(yres),
        xreal_m=xreal_m,
        yreal_m=yreal_m,
        dx_m_per_px=xreal_m / xres,
        dy_m_per_px=yreal_m / yres,
        available_titles=_list_channel_titles(obj),
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


def preprocess_pair_raw_and_proc(
    topo_raw: np.ndarray,
    aux_raw: np.ndarray,
    *,
    crop_border_frac: float = 0.05,
    clip_sigma: float | None = 8.0,
    do_plane: bool = True,
    do_row_median: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess two channels with identical cropping so they stay aligned.
    Returns:
      raw_stacked: (2, Hc, Wc)
      proc_stacked: (2, Hc, Wc)
    """
    if topo_raw.shape != aux_raw.shape:
        raise ValueError(f"Topo and aux shapes differ: {topo_raw.shape} vs {aux_raw.shape}")

    topo_c = crop_border(topo_raw.astype(np.float32, copy=False), frac=crop_border_frac)
    aux_c = crop_border(aux_raw.astype(np.float32, copy=False), frac=crop_border_frac)

    topo_p = topo_c
    aux_p = aux_c

    # Process each channel independently (often best: phase/friction shouldn't share topo plane)
    if do_plane:
        topo_p = subtract_best_fit_plane(topo_p)
        aux_p = subtract_best_fit_plane(aux_p)
    if do_row_median:
        topo_p = align_rows_by_median(topo_p)
        aux_p = align_rows_by_median(aux_p)
    if clip_sigma is not None:
        topo_p = robust_clip(topo_p, sigma=float(clip_sigma))
        aux_p = robust_clip(aux_p, sigma=float(clip_sigma))

    raw = np.stack([topo_c, aux_c], axis=0)
    proc = np.stack([topo_p, aux_p], axis=0)
    return raw, proc


# ---------- Pairing files ----------

AUX_SUFFIXES = [
    "PHASE SHIFT",
    "PHASE_SHIFT",
    "PHASE",
    "FRICTION",
    "DAC1",
]

TOPO_SUFFIXES = [
    "T",
    "TOPO",
    "TOPOGRAPHY",
    "HEIGHT",
]


def _normalize_stem(stem: str) -> str:
    # normalize repeated spaces/underscores for matching
    s = stem.replace("__", "_")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def classify_file(path: Path) -> Tuple[str, str]:
    """
    Returns (base_id, kind) where kind in {'topo', 'aux:<type>', 'unknown'}.

    Handles names like:
      perovskite_s101024_23_10_T.gwy
      perovskite_s101024_23_10_FRICTION.gwy
      samples070425_2_PHASE SHIFT.gwy
    """
    stem = _normalize_stem(path.stem)

    # Try underscore-separated suffix first
    parts_us = stem.split("_")
    last_us = parts_us[-1].upper() if parts_us else ""

    # Try space-separated suffix too (e.g., "PHASE SHIFT")
    last_space = stem.split(" ")[-1].upper() if stem else ""

    # Detect topo suffixes
    if last_us in TOPO_SUFFIXES:
        base = "_".join(parts_us[:-1])
        return base, "topo"

    # Detect aux suffixes (including PHASE SHIFT)
    upper = stem.upper()
    for suf in AUX_SUFFIXES:
        suf_u = suf.upper()
        if upper.endswith("_" + suf_u) or upper.endswith(" " + suf_u) or upper == suf_u:
            # strip suffix with either separator
            base = re.sub(r"([_ ]+)" + re.escape(suf_u) + r"$", "", upper)
            # recover original-case-ish base from original stem by slicing length
            # safer: strip from original stem with regex ignoring case
            base2 = re.sub(r"([_ ]+)" + re.escape(suf) + r"$", "", stem, flags=re.IGNORECASE)
            return base2, f"aux:{suf_u}"

    return stem, "unknown"


def group_pairs(gwy_files: List[Path]) -> List[dict]:
    """
    Groups files into topo+aux pairs. If multiple aux types exist for a topo base,
    returns multiple pairs (one per aux file).
    """
    groups: Dict[str, Dict[str, List[Path]]] = {}
    for fp in gwy_files:
        base, kind = classify_file(fp)
        base = _normalize_stem(base)
        groups.setdefault(base, {}).setdefault(kind, []).append(fp)

    pairs = []
    for base, d in groups.items():
        topo_files = d.get("topo", [])
        aux_kinds = [k for k in d.keys() if k.startswith("aux:")]

        if not topo_files or not aux_kinds:
            continue

        # If multiple topo files for same base, take first (you can refine if needed)
        topo_fp = sorted(topo_files)[0]

        for ak in sorted(aux_kinds):
            aux_fp = sorted(d[ak])[0]
            aux_type = ak.split(":", 1)[1]
            pairs.append(
                {
                    "base_id": base,
                    "topo_file": topo_fp,
                    "aux_file": aux_fp,
                    "aux_type": aux_type,
                }
            )

    return pairs


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
    *,
    patch: int = 256,
    stride: int = 256,
    crop_border_frac: float = 0.05,
    clip_sigma: float | None = 8.0,
    topo_channel_title: Optional[str] = None,
    aux_channel_title: Optional[str] = None,
    max_pairs: int | None = None,
):
    """
    Finds topo+aux .gwy pairs in input_dir (based on filename suffix),
    preprocesses (crop -> plane -> row-median -> clip) per channel,
    extracts aligned patches, and writes one HDF5 file.

    Output datasets:
      patches/raw   : (N, 2, patch, patch)  [channel0=topo, channel1=aux]
      patches/proc  : (N, 2, patch, patch)
      patches/source_topo_file, patches/source_aux_file, patches/base_id, patches/aux_type, patches/top_left_yx
      scans/meta_jsonl : one line per pair
    """
    gwy_files = sorted(input_dir.rglob("*.gwy"))
    pairs = group_pairs(gwy_files)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    if not pairs:
        raise SystemExit(
            "No topo+aux pairs found. Make sure files end with _T and one of "
            + ", ".join([f"_{s}" for s in AUX_SUFFIXES])
        )

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5, "w") as f:
        # Resizable datasets for patches (2 channels)
        d_raw = f.create_dataset(
            "patches/raw",
            shape=(0, 2, patch, patch),
            maxshape=(None, 2, patch, patch),
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(32, 2, patch, patch),
        )
        d_proc = f.create_dataset(
            "patches/proc",
            shape=(0, 2, patch, patch),
            maxshape=(None, 2, patch, patch),
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(32, 2, patch, patch),
        )

        # Provenance
        d_topo_file = f.create_dataset(
            "patches/source_topo_file",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=4,
            chunks=(1024,),
        )
        d_aux_file = f.create_dataset(
            "patches/source_aux_file",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=4,
            chunks=(1024,),
        )
        d_base = f.create_dataset(
            "patches/base_id",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=4,
            chunks=(1024,),
        )
        d_aux_type = f.create_dataset(
            "patches/aux_type",
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

        # Pair-level metadata
        meta_ds = f.create_dataset(
            "scans/meta_jsonl",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="gzip",
            compression_opts=4,
            chunks=(256,),
        )

        # Global attrs
        f.attrs["patch"] = patch
        f.attrs["stride"] = stride
        f.attrs["channel_order"] = json.dumps(["topography", "aux"])
        f.attrs["aux_suffixes"] = json.dumps(AUX_SUFFIXES)
        f.attrs["topo_suffixes"] = json.dumps(TOPO_SUFFIXES)
        f.attrs["preprocess"] = json.dumps(
            {
                "crop_border_frac": crop_border_frac,
                "plane": True,
                "row_median": True,
                "clip_sigma": clip_sigma,
            }
        )

        total_patches = 0

        for i, pair in enumerate(pairs, start=1):
            topo_fp = pair["topo_file"]
            aux_fp = pair["aux_file"]
            base_id = pair["base_id"]
            aux_type = pair["aux_type"]

            try:
                topo_raw, topo_meta = load_gwy_channel(topo_fp, channel_title=topo_channel_title)
                aux_raw, aux_meta = load_gwy_channel(aux_fp, channel_title=aux_channel_title)
            except Exception as e:
                print(f"[{i}/{len(pairs)}] SKIP {base_id}: load error: {e}")
                continue

            try:
                raw2, proc2 = preprocess_pair_raw_and_proc(
                    topo_raw,
                    aux_raw,
                    crop_border_frac=crop_border_frac,
                    clip_sigma=clip_sigma,
                    do_plane=True,
                    do_row_median=True,
                )
            except Exception as e:
                print(f"[{i}/{len(pairs)}] SKIP {base_id}: preprocess error: {e}")
                continue

            # raw2/proc2 are (2, H, W)
            _, h, w = raw2.shape
            coords = list(iter_patch_coords(h, w, patch, stride))
            if not coords:
                print(f"[{i}/{len(pairs)}] SKIP {base_id}: too small after crop ({h}x{w})")
                continue

            raw_patches = np.empty((len(coords), 2, patch, patch), dtype=np.float32)
            proc_patches = np.empty((len(coords), 2, patch, patch), dtype=np.float32)
            yx = np.empty((len(coords), 2), dtype=np.int32)

            for j, (y, x) in enumerate(coords):
                raw_patches[j, 0] = raw2[0, y:y+patch, x:x+patch]
                raw_patches[j, 1] = raw2[1, y:y+patch, x:x+patch]
                proc_patches[j, 0] = proc2[0, y:y+patch, x:x+patch]
                proc_patches[j, 1] = proc2[1, y:y+patch, x:x+patch]
                yx[j] = (y, x)

            append_to_resizable(d_raw, raw_patches)
            append_to_resizable(d_proc, proc_patches)
            append_to_resizable(d_yx, yx)

            n = len(coords)
            # string datasets
            d_topo_file.resize((d_topo_file.shape[0] + n,))
            d_topo_file[-n:] = [str(topo_fp)] * n

            d_aux_file.resize((d_aux_file.shape[0] + n,))
            d_aux_file[-n:] = [str(aux_fp)] * n

            d_base.resize((d_base.shape[0] + n,))
            d_base[-n:] = [base_id] * n

            d_aux_type.resize((d_aux_type.shape[0] + n,))
            d_aux_type[-n:] = [aux_type] * n

            # pair meta
            meta_line = json.dumps(
                {
                    "base_id": base_id,
                    "topo_file": str(topo_fp),
                    "aux_file": str(aux_fp),
                    "aux_type": aux_type,
                    "topo_meta": topo_meta,
                    "aux_meta": aux_meta,
                    "crop_border_frac": crop_border_frac,
                    "clip_sigma": clip_sigma,
                    "patch": patch,
                    "stride": stride,
                },
                ensure_ascii=False,
            )
            meta_ds.resize((meta_ds.shape[0] + 1,))
            meta_ds[-1] = meta_line

            total_patches += n
            print(f"[{i}/{len(pairs)}] OK {base_id} ({aux_type}): {n} patches (total {total_patches})")

        print(f"Done. Wrote {total_patches} paired patches to {out_h5}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--out_h5", type=str, required=True)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--crop_border_frac", type=float, default=0.05)
    p.add_argument("--clip_sigma", type=float, default=8.0)
    # Optional: if your .gwy contains multiple channels and you want a specific one
    p.add_argument("--topo_channel_title", type=str, default=None)
    p.add_argument("--aux_channel_title", type=str, default=None)
    p.add_argument("--max_pairs", type=int, default=None)
    args = p.parse_args()

    build_h5_from_folder(
        input_dir=Path(args.input_dir),
        out_h5=Path(args.out_h5),
        patch=args.patch,
        stride=args.stride,
        crop_border_frac=args.crop_border_frac,
        clip_sigma=args.clip_sigma,
        topo_channel_title=args.topo_channel_title,
        aux_channel_title=args.aux_channel_title,
        max_pairs=args.max_pairs,
    )