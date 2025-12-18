import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.afm_h5_dataset import AFMPatchesH5Dataset, ChannelNorm
from utils.reproducibility import seed_worker

def load_channel_norm(stats_path: str) -> ChannelNorm:
    with open(stats_path) as f:
        stats = json.load(f)
    return ChannelNorm(
        mean=torch.tensor(stats["mean"], dtype=torch.float32),
        std=torch.tensor(stats["std"], dtype=torch.float32),
    )

def build_dataloaders(args, norm: ChannelNorm,use_aux: bool = True):
    split_dir = Path(args.split_dir)

    # train indices
    indices_path = split_dir / f"{args.split_name}.npy"
    train_inds = np.load(indices_path) if indices_path.exists() else None
    aux_types = args.aux_types if use_aux else []
    train_ds = AFMPatchesH5Dataset(
        args.h5,
        norm=norm,
        aux_types=aux_types,
        indices=train_inds
    )
    pin_memory = torch.cuda.is_available()
    persistent_workers = args.num_workers > 0
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=g,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=False,
    )

    # val
    val_loader = None
    val_ds = None
    if args.val_split:
        val_path = split_dir / f"{args.val_split}.npy"
        if val_path.exists():
            val_inds = np.load(val_path)
            val_ds = AFMPatchesH5Dataset(args.h5, norm=norm, aux_types=args.aux_types, indices=val_inds)
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=max(1, args.num_workers // 2),
                pin_memory=pin_memory,
                persistent_workers=(args.num_workers // 2) > 0,
                worker_init_fn=seed_worker if (args.num_workers // 2) > 0 else None,
                generator=g,
                prefetch_factor=2 if (args.num_workers // 2) > 0 else None,
                drop_last=False,
            )

    return train_loader, val_loader, train_ds, val_ds
