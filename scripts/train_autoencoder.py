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
- If `--split-dir/{split_name}.npy` exists it will be used; otherwise the
    whole dataset is used. Validation is run only if `--val-split` file exists.
- TensorBoard and W&B are optional; the script continues if those packages
    are not installed.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from datasets.afm_h5_dataset import AFMPatchesH5Dataset, ChannelNorm
from torch.utils.data import DataLoader

from model.autoencoder_model import AFMUNetAutoencoder


def parse_args():
    p = argparse.ArgumentParser(description="Train AFM U-Net autoencoder")
    p.add_argument("--h5", default="afm_patches_256_2ch.h5", help="HDF5 file with patches")
    p.add_argument("--stats", default="stats/channel_norm.json", help="Channel norm JSON")
    p.add_argument("--split-dir", default="splits", help="Directory containing split .npy files")
    p.add_argument("--split-name", default="train", help="Which split to use (train/val/test)")
    p.add_argument("--val-split", default="val", help="Validation split name (optional)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--log-tb", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--log-dir", default="logs", help="Directory for TensorBoard logs")
    p.add_argument("--log-interval", type=int, default=100, help="Batches between log messages")
    p.add_argument("--log-wandb", action="store_true", help="Enable Weights & Biases logging if available")
    p.add_argument("--wandb-project", default="afm-autoencoder", help="W&B project name")
    p.add_argument("--run-name", default=None, help="Run name for logs/wandb")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    with open(args.stats) as f:
        stats = json.load(f)

    norm = ChannelNorm(mean=torch.tensor(stats["mean"]), std=torch.tensor(stats["std"]))

    # Load indices for split if provided
    split_dir = Path(args.split_dir)
    indices_path = split_dir / f"{args.split_name}.npy"
    if indices_path.exists():
        indices = np.load(indices_path)
        print(f"Using indices from {indices_path} (count={len(indices)})")
    else:
        indices = None
        print(f"No split file {indices_path}, using entire dataset")

    train_ds = AFMPatchesH5Dataset(args.h5, norm=norm, aux_types=["PHASE", "FRICTION"], indices=indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # optional validation loader
    val_loader = None
    val_path = split_dir / f"{args.val_split}.npy"
    if val_path.exists():
        val_inds = np.load(val_path)
        val_ds = AFMPatchesH5Dataset(args.h5, norm=norm, aux_types=["PHASE", "FRICTION"], indices=val_inds)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)
        print(f"Validation split loaded from {val_path} (count={len(val_inds)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFMUNetAutoencoder(in_channels=2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # logging setup
    tb_writer = None
    if args.log_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = Path(args.log_dir)
            run_name = args.run_name or f"run_{tb_log_dir.name}"
            tb_writer = SummaryWriter(log_dir=str(tb_log_dir / run_name))
            print(f"TensorBoard logging enabled at {tb_log_dir / run_name}")
        except Exception as e:
            tb_writer = None
            print(f"TensorBoard not available: {e}")

    wandb_run = None
    if args.log_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
            print(f"Weights & Biases logging enabled (project={args.wandb_project})")
        except Exception as e:
            wandb_run = None
            print(f"wandb not available or failed to init: {e}")

    # resume support
    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ck = torch.load(resume_path, map_location=device)
            model.load_state_dict(ck.get("model_state_dict", ck))
            if "optimizer_state_dict" in ck:
                optimizer.load_state_dict(ck["optimizer_state_dict"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
        else:
            print(f"Resume checkpoint {resume_path} not found, starting from scratch")

    print(f"Training on device: {device}, dataset size: {len(train_ds)} patches")

    global_step = 0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_idx, x in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                global_step += 1

                if batch_idx % args.log_interval == 0:
                    avg = running_loss / batch_idx
                    msg = f"Epoch {epoch} | Batch {batch_idx} | Avg loss: {avg:.6f}"
                    print(msg)
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/batch_loss", loss.item(), global_step)
                    if wandb_run is not None:
                        wandb.log({"train/batch_loss": loss.item(), "global_step": global_step})

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch} completed. Loss: {epoch_loss:.6f}")
            if tb_writer is not None:
                tb_writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
            if wandb_run is not None:
                wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})

            # run validation if available
            val_loss = None
            if val_loader is not None:
                model.eval()
                val_loss_acc = 0.0
                with torch.no_grad():
                    for xb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        outb = model(xb)
                        val_loss_acc += criterion(outb, xb).item()
                val_loss = val_loss_acc / len(val_loader)
                print(f"Epoch {epoch} validation loss: {val_loss:.6f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("val/epoch_loss", val_loss, epoch)
                if wandb_run is not None:
                    wandb.log({"val/epoch_loss": val_loss, "epoch": epoch})

            # save checkpoint
            ckpt_path = ckpt_dir / f"ae_epoch{epoch:03d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # save best checkpoint (by validation loss)
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_path = ckpt_dir / "ae_best.pth"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}, best_path)
                print(f"New best validation loss: {best_val:.6f}, saved {best_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user, saving last checkpoint...")
        ckpt_path = ckpt_dir / "ae_interrupt.pth"
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)
        print(f"Saved: {ckpt_path}")
    finally:
        if tb_writer is not None:
            tb_writer.close()
        if wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
