"""AFMPatchesH5Dataset

Lightweight PyTorch Dataset for loading multichannel AFM image patches
stored in an HDF5 file. This module provides `AFMPatchesH5Dataset` which
reads patch arrays and optional metadata and returns them as PyTorch
tensors (optionally normalized) and an optional metadata dict.

Inputs (HDF5 file structure expected):
- `patches/proc`: ndarray, shape (N, C, H, W), dtype float32 — main image
    patches (required by default, dataset path configurable via `x_dataset`).
- `patches/aux_type` (optional): (N,) strings like "FRICTION", "T" — used
    for filtering by `aux_types`.
- `patches/base_id` (optional): (N,) strings identifying the source/sample.
- `patches/top_left_yx` (optional): (N, 2) ints containing top-left coordinates
    of each patch in (y, x) order.

Constructor arguments:
- `h5_path` (str | Path): path to the HDF5 file.
- `x_dataset` (str): dataset path for patches (default: "patches/proc").
- `norm` (ChannelNorm | None): per-channel mean/std for normalization.
- `aux_types` (Sequence[str] | None): list of aux types to include (case-ins).
- `return_meta` (bool): if True, `__getitem__` returns `(tensor, meta_dict)`.

Outputs (`__getitem__`):
- If `return_meta=False`: returns a `torch.FloatTensor` of shape (C, H, W),
    dtype float32. If `norm` is provided the tensor is normalized ((x-mean)/std).
- If `return_meta=True`: returns `(tensor, meta_dict)` where `meta_dict`
    contains keys such as `idx`, and when present `aux_type`, `base_id`, and
    `top_left_yx`.

Notes:
- HDF5 file handles are opened lazily per worker to support PyTorch
    `DataLoader` with multiple workers.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


@dataclass(frozen=True)
class ChannelNorm:
    mean: torch.Tensor  # shape (C,)
    std: torch.Tensor   # shape (C,)


class AFMPatchesH5Dataset(Dataset):
    """
    Loads multichannel AFM patches from an HDF5 file.

    Expected dataset shape:
      patches/proc: (N, C, H, W) float32
      patches/aux_type: (N,) string  (optional but recommended)
      patches/base_id: (N,) string    (optional)
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        x_dataset: str = "patches/proc",
        norm: Optional[ChannelNorm] = None,
        aux_types: Optional[Sequence[str]] = None,   # e.g. ["PHASE", "FRICTION"]
        indices: Optional[Sequence[int]] = None,     # explicit indices to use (subset)
        return_meta: bool = False,
    ):
        self.h5_path = str(h5_path)
        self.x_dataset = x_dataset
        self.norm = norm
        self.aux_types = [a.upper() for a in aux_types] if aux_types is not None else None
        self.return_meta = return_meta

        # Build index list once (fast).
        # If `indices` is provided, use it (useful for train/val/test splits).
        with h5py.File(self.h5_path, "r") as f:
            x = f[self.x_dataset]
            self._shape = tuple(x.shape)  # (N,C,H,W)

            if indices is not None:
                self.indices = np.array(indices, dtype=np.int64)
            else:
                if self.aux_types is None or "patches/aux_type" not in f:
                    self.indices = np.arange(self._shape[0], dtype=np.int64)
                else:
                    aux = f["patches/aux_type"][:].astype(str)
                    mask = np.isin(np.char.upper(aux), np.array(self.aux_types))
                    self.indices = np.nonzero(mask)[0].astype(np.int64)

            # basic validation: clamp indices to valid range
            if np.any(self.indices < 0) or np.any(self.indices >= self._shape[0]):
                raise ValueError("Some provided indices are out of range for the HDF5 dataset")

        # Lazy file handle (opened per-worker)
        self._h5 = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, i: int):
        self._ensure_open()
        idx = int(self.indices[i])

        x = self._h5[self.x_dataset][idx]  # (C,H,W) np.float32
        x = torch.from_numpy(x)            # float32 tensor

        if self.norm is not None:
            # broadcast (C,) -> (C,1,1)
            mean = self.norm.mean.view(-1, 1, 1)
            std = self.norm.std.clamp_min(1e-6).view(-1, 1, 1)
            x = (x - mean) / std

        if not self.return_meta:
            return x

        meta: Dict[str, Any] = {"idx": idx}
        if "patches/aux_type" in self._h5:
            meta["aux_type"] = self._h5["patches/aux_type"][idx].astype(str)
        if "patches/base_id" in self._h5:
            meta["base_id"] = self._h5["patches/base_id"][idx].astype(str)
        if "patches/top_left_yx" in self._h5:
            meta["top_left_yx"] = tuple(self._h5["patches/top_left_yx"][idx].tolist())
        return x, meta

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
