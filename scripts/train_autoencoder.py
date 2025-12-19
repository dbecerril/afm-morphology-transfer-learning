"""Train AFM U-Net autoencoder

This script trains the `AFMUNetAutoencoder` on AFM patch datasets stored in
HDF5. It supports loading precomputed index splits (`.npy`) produced by
`scripts/make_splits.py`, TensorBoard and optional Weights & Biases logging,
checkpointing, and resuming from checkpoints.

Quick usage:

1) Create splits (recommended `by-base-id` to avoid leakage):
     python scripts/make_splits.py --h5 afm_patches_256_2ch.h5 --out splits \
             --mode by-base-id --ratios 0.8 0.1 0.1 --seed 0

2) Train using the splits with TensorBoard:
     python scripts/train_autoencoder.py --h5 afm_patches_256_2ch.h5 \
             --split-dir splits --split-name train --val-split val --epochs 20 \
             --batch-size 32 --log-tb --log-dir logs --run-name myrun

3) Resume training from a checkpoint:
     python scripts/train_autoencoder.py --resume checkpoints/ae_epoch005.pth

Main CLI options (see `--help`):
- `--h5`, `--stats`: paths to HDF5 and channel-norm JSON
- `--split-dir`, `--split-name`, `--val-split`: use `.npy` files in `split-dir`
- `--epochs`, `--batch-size`, `--lr`, `--num-workers`, `--seed`
- `--checkpoint-dir`, `--resume`
- `--log-tb`, `--log-dir`, `--log-interval`
- `--log-wandb`, `--wandb-project`, `--run-name`

Notes:
- Missing channel stats or split files are generated automatically (80/10/10
    ratios, by-base-id when available) before training starts.
- TensorBoard and W&B are optional; the script continues if those packages
    are not installed.
How to run:
    python scripts/train_autoencoder.py \
  --h5 /content/afm_patches.h5 \
  --in-channels 2 \
  --out-channels 1 \
  --target-channel 0 \
  --log-tb

"""
#!/usr/bin/env python3
"""
Train AFM U-Net autoencoder (Colab-friendly).

Improvements vs your version:
- Full reproducibility seeding (numpy/torch/random + worker seeds)
- AMP mixed precision (faster on Colab GPUs)
- Sample-weighted epoch/val loss (handles last small batch correctly)
- Optional LR scheduler (ReduceLROnPlateau on val loss)
- Persistent workers / prefetch tuning for speed
- Saves run config + stats copy into checkpoint dir
- Optional reconstruction snapshots (input / recon / residual) every N epochs
"""

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import sys

import h5py

# Ensure repo root is on PYTHONPATH (works on Colab + local)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from datasets.afm_h5_dataset import AFMPatchesH5Dataset, ChannelNorm
from model.autoencoder_model import AFMUNetAutoencoder


from utils.reproducibility import seed_everything
from utils.io import resolve_h5_path, ensure_split_files, ensure_channel_norm_file
from utils.data import load_channel_norm, build_dataloaders
from utils.compute_stats import sobel_grad_mag_1ch
from utils.logging import setup_tensorboard,setup_wandb,log_train_step,log_epoch,close_loggers

def parse_args():
    p = argparse.ArgumentParser(description="Train AFM U-Net autoencoder")
    p.add_argument("--h5", default="afm_patches_256.h5", help="HDF5 file with patches")
    p.add_argument("--stats", default="stats/channel_norm.json", help="Channel norm JSON")
    p.add_argument("--split-dir", default="splits", help="Directory containing split .npy files")
    p.add_argument("--split-name", default="train", help="Which split to use (train/val/test)")
    p.add_argument("--val-split", default="val", help="Validation split name (optional)")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    p.add_argument("--log-tb", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--log-dir", default="logs", help="Directory for TensorBoard logs")
    p.add_argument("--log-interval", type=int, default=100, help="Batches between log messages")

    p.add_argument("--log-wandb", action="store_true", help="Enable Weights & Biases logging if available")
    p.add_argument("--wandb-project", default="afm-autoencoder", help="W&B project name")
    p.add_argument("--run-name", default=None, help="Run name for logs/wandb")

    # training extras
    p.add_argument("--use-amp", action="store_true", help="Use mixed precision AMP (recommended on Colab GPU)")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Max grad norm (0 disables)")
    p.add_argument("--scheduler", action="store_true", help="Enable ReduceLROnPlateau scheduler (uses val if present)")
    p.add_argument("--patience", type=int, default=5, help="Scheduler patience")
    p.add_argument("--min-lr", type=float, default=1e-6, help="Scheduler min lr")


    # snapshotting
    p.add_argument("--save-snapshots", action="store_true", help="Save recon snapshots every N epochs")
    p.add_argument("--snapshot-every", type=int, default=5, help="Epoch frequency for snapshots")
    p.add_argument("--snapshot-max", type=int, default=8, help="Max samples to snapshot")
    p.add_argument("--snapshot-dir", default="snapshots", help="Directory (inside checkpoint-dir) for snapshots")

    # dataset channels (keep your default; adjust if you change dataset)
    p.add_argument("--aux-types", nargs="+", default=["PHASE", "FRICTION"], help="Aux channels to use")

    #FLI options
    p.add_argument("--use-aux-conditioning", action="store_true",
                help="Use aux as late FiLM conditioning (topo-only encoder input).")
    p.add_argument("--aux-dropout", type=float, default=0.3,
                help="Probability of dropping aux during training.")
    p.add_argument("--grad-loss", type=float, default=0.0,
                help="Weight for gradient (Sobel) loss on topo reconstruction.")

    return p.parse_args()

@torch.no_grad()
def save_recon_snapshot(model, device, batch, out_dir: Path, epoch: int, max_n: int = 8):
    """
    Saves a simple numpy snapshot: input, recon, residual for topography (channel 0).
    This avoids extra deps; you can visualize later or add matplotlib if you like.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x = batch[:max_n].to(device, non_blocking=True)
    topo = x[:, :1]
    aux  = x[:, 1:] if x.size(1) > 1 else None
    y = model(topo, aux) if (aux is not None and model.aux_channels > 0) else model(topo)


    # Move to CPU
    x_np = x.detach().cpu().float().numpy()
    y_np = y.detach().cpu().float().numpy()
    r_np = np.abs(x_np[:, 0:1] - y_np)  # residual on topo only

    # Save as npz (easy to load/plot later)
    np.savez_compressed(
        out_dir / f"epoch_{epoch:03d}.npz",
        x=x_np,
        y=y_np,
        residual=r_np,
    )


def main():
    args = parse_args()
    args.h5 = str(resolve_h5_path(args.h5, REPO_ROOT))
    seed_everything(args.seed)

    split_cache = ensure_split_files(args)
    ensure_channel_norm_file(args, preferred_train_indices=split_cache.get("train"))
    
    norm = load_channel_norm(args.stats)
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(args, norm,use_aux=args.use_aux_conditioning)
    # ---- Infer actual channel counts from data (robust) ----
    x0 = next(iter(train_loader))            # CPU batch is fine
    n_ch = x0.size(1)
    aux_ch = max(0, n_ch - 1)
    print(f"[Sanity] Detected channels from dataset: total={n_ch}, aux={aux_ch}")

    # run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ae_{timestamp}"

    # checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # store run config and stats copy for reproducibility
    (ckpt_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    try:
        shutil.copy(args.stats, ckpt_dir / "channel_norm.json")
    except Exception:
        pass

    tb_writer = setup_tensorboard(args.log_dir, run_name) if args.log_tb else None
    wandb_run = setup_wandb(
        args.log_wandb,
        args.wandb_project,
        run_name,
        vars(args),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AFMUNetAutoencoder(
        in_channels=1,                       # topo-only encoder input
        out_channels=1,       # should be 1 for topo recon
        aux_channels=(aux_ch if args.use_aux_conditioning else 0),
        aux_dropout=args.aux_dropout,
    ).to(device)

    #if args.out_channels == 1 and args.target_channel >= args.in_channels:
    #    raise ValueError(f"--target-channel must be < --in-channels (got {args.target_channel} vs {args.in_channels})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss(reduction="mean")

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == "cuda"))

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.patience,
            min_lr=args.min_lr,
            verbose=True,
        )

    # resume support
    start_epoch = 1
    best_val = float("inf")
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ck = torch.load(resume_path, map_location=device)
            if isinstance(ck, dict) and "model_state_dict" in ck:
                model.load_state_dict(ck["model_state_dict"])
                if "optimizer_state_dict" in ck:
                    optimizer.load_state_dict(ck["optimizer_state_dict"])
                if "scaler_state_dict" in ck and scaler is not None:
                    scaler.load_state_dict(ck["scaler_state_dict"])
                start_epoch = int(ck.get("epoch", 0)) + 1
                best_val = float(ck.get("best_val", best_val))
                global_step = int(ck.get("global_step", 0))
            else:
                # Allow loading raw state_dict
                model.load_state_dict(ck)
            print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
        else:
            print(f"Resume checkpoint {resume_path} not found, starting from scratch")

    print(f"Training on device: {device}, dataset size: {len(train_ds)} patches")
    print(f"Model in_channels=1 (topo-only); aux_conditioning={args.use_aux_conditioning}; aux_types={args.aux_types}")

    # quick sanity check on the first batch shape
    first_batch_checked = False
    margin = 16  # crop margin for loss computation
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()

            train_loss_sum = 0.0
            train_n = 0

            for batch_idx, x in enumerate(train_loader, start=1):
                if not first_batch_checked:
                    print(f"[Sanity] First batch shape: {tuple(x.shape)} dtype={x.dtype}")
                    first_batch_checked = True

                x = x.to(device, non_blocking=True)
                bs = x.size(0)

                optimizer.zero_grad(set_to_none=True)
                topo = x[:, :1]
                aux = x[:, 1:] if x.size(1) > 1 else None

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    if args.use_aux_conditioning:
                        out = model(topo, aux)
                    else:
                        out = model(topo)
                    out_c  = out[:, :, margin:-margin, margin:-margin]
                    topo_c = topo[:, :, margin:-margin, margin:-margin]

                    loss_l1 = F.l1_loss(out_c, topo_c)

                    if args.grad_loss > 0:
                        go = sobel_grad_mag_1ch(out_c)
                        gt = sobel_grad_mag_1ch(topo_c)
                        loss_g = F.l1_loss(go, gt)
                        loss = loss_l1 + args.grad_loss * loss_g
                    else:
                        loss = loss_l1
                        
                scaler.scale(loss).backward()

                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                # sample-weighted accumulation
                train_loss_sum += loss.detach().item() * bs
                train_n += bs
                global_step += 1

                if batch_idx % args.log_interval == 0:
                    log_train_step(
                        tb_writer,
                        wandb_run,
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                        global_step,
                    )

            epoch_train_loss = train_loss_sum / max(1, train_n)
            log_epoch(tb_writer, wandb_run, "train", epoch_train_loss, epoch)
            print(f"Epoch {epoch} completed. Train loss: {epoch_train_loss:.6f}")

            # validation
            val_loss = None
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_n = 0

                with torch.no_grad():
                    for xb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        bs = xb.size(0)
                        topob = xb[:, :1]
                        auxb  = xb[:, 1:] if xb.size(1) > 1 else None
                        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                            outb = model(topob, auxb) if args.use_aux_conditioning else model(topob)
                        out_c  = out[:, :, margin:-margin, margin:-margin]
                    topo_c = topo[:, :, margin:-margin, margin:-margin]

                    loss_l1 = F.l1_loss(out_c, topo_c)

                    if args.grad_loss > 0:
                        go = sobel_grad_mag_1ch(out_c)
                        gt = sobel_grad_mag_1ch(topo_c)
                        loss_g = F.l1_loss(go, gt)
                        lb = loss_l1 + args.grad_loss * loss_g
                    else:
                        lb = loss_l1 #criterion(outb, topob)
                        val_loss_sum += lb.item() * bs
                        val_n += bs

                val_loss = val_loss_sum / max(1, val_n)
                print(f"Epoch {epoch} validation loss: {val_loss:.6f}")
                log_epoch(tb_writer, wandb_run, "val", val_loss, epoch)


            # scheduler step
            if scheduler is not None:
                # if no val, fall back to train loss
                scheduler_metric = val_loss if val_loss is not None else epoch_train_loss
                scheduler.step(scheduler_metric)

            # save checkpoint (every epoch)
            ckpt_path = ckpt_dir / f"ae_epoch{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val": best_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "train_loss": epoch_train_loss,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

            # save best checkpoint
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_path = ckpt_dir / "ae_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val": best_val,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                    },
                    best_path,
                )
                print(f"New best validation loss: {best_val:.6f}, saved {best_path}")

            # snapshots (uses first batch from train_loader for consistency)
            if args.save_snapshots and (epoch % args.snapshot_every == 0):
                # grab one batch quickly (no need to restart loader; just reuse the last x if available)
                snap_dir = ckpt_dir / args.snapshot_dir
                try:
                    save_recon_snapshot(model, device, x.detach(), snap_dir, epoch, max_n=args.snapshot_max)
                    print(f"Saved snapshot npz to: {snap_dir / f'epoch_{epoch:03d}.npz'}")
                except Exception as e:
                    print(f"Snapshot save failed: {e}")
        

    except KeyboardInterrupt:
        print("Training interrupted by user, saving last checkpoint...")
        ckpt_path = ckpt_dir / "ae_interrupt.pth"
        torch.save(
            {
                "epoch": epoch if "epoch" in locals() else 0,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            },
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")
    finally:
        close_loggers(tb_writer, wandb_run)


if __name__ == "__main__":
    main()
