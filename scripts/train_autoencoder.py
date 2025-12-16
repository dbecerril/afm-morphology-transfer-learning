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

from datasets.afm_h5_dataset import AFMPatchesH5Dataset, ChannelNorm
from model.autoencoder_model import AFMUNetAutoencoder

from compute_stats import compute_channel_mean_std
from make_splits import by_base_id_split, random_split


AUTO_SPLIT_RATIOS = (0.8, 0.1, 0.1)




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
    p.add_argument("--in-channels", type=int, default=2, help="Input channels expected by model")

    return p.parse_args()


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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility settings (slower but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Make DataLoader workers deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def save_recon_snapshot(model, device, batch, out_dir: Path, epoch: int, max_n: int = 8):
    """
    Saves a simple numpy snapshot: input, recon, residual for first channel (and second if present).
    This avoids extra deps; you can visualize later or add matplotlib if you like.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x = batch[:max_n].to(device, non_blocking=True)
    y = model(x)

    # Move to CPU
    x_np = x.detach().cpu().float().numpy()
    y_np = y.detach().cpu().float().numpy()
    r_np = np.abs(x_np - y_np)

    # Save as npz (easy to load/plot later)
    np.savez_compressed(
        out_dir / f"epoch_{epoch:03d}.npz",
        x=x_np,
        y=y_np,
        residual=r_np,
    )


def main():
    args = parse_args()
    args.h5 = str(resolve_h5_path(args.h5))
    seed_everything(args.seed)

    split_cache = ensure_split_files(args)
    ensure_channel_norm_file(args, preferred_train_indices=split_cache.get("train"))

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

    # load stats
    with open(args.stats) as f:
        stats = json.load(f)
    norm = ChannelNorm(
        mean=torch.tensor(stats["mean"], dtype=torch.float32),
        std=torch.tensor(stats["std"], dtype=torch.float32),
    )

    # splits
    split_dir = Path(args.split_dir)
    indices_path = split_dir / f"{args.split_name}.npy"
    if indices_path.exists():
        indices = np.load(indices_path)
        print(f"Using indices from {indices_path} (count={len(indices)})")
    else:
        indices = None
        print(f"No split file {indices_path}, using entire dataset")

    train_ds = AFMPatchesH5Dataset(args.h5, norm=norm, aux_types=args.aux_types, indices=indices)

    # DataLoader perf knobs (Colab)
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

    # optional validation
    val_loader = None
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
        print(f"Validation split loaded from {val_path} (count={len(val_inds)})")
    else:
        print(f"No validation split found at {val_path} (training only)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFMUNetAutoencoder(in_channels=args.in_channels).to(device)

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

    # logging setup
    tb_writer = None
    if args.log_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = Path(args.log_dir) / run_name
            tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"TensorBoard logging enabled at {tb_log_dir}")
        except Exception as e:
            tb_writer = None
            print(f"TensorBoard not available: {e}")

    wandb_run = None
    if args.log_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            print(f"Weights & Biases logging enabled (project={args.wandb_project})")
        except Exception as e:
            wandb_run = None
            print(f"wandb not available or failed to init: {e}")

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
    print(f"Model expects in_channels={args.in_channels}; dataset aux_types={args.aux_types}")

    # quick sanity check on the first batch shape
    first_batch_checked = False

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

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model(x)
                    loss = criterion(out, x)

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
                    avg = train_loss_sum / max(1, train_n)
                    lr_now = optimizer.param_groups[0]["lr"]
                    msg = f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Train avg loss: {avg:.6f} | LR: {lr_now:.2e}"
                    print(msg)
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/batch_loss", loss.item(), global_step)
                        tb_writer.add_scalar("train/lr", lr_now, global_step)
                    if wandb_run is not None:
                        wandb_run.log({"train/batch_loss": loss.item(), "train/lr": lr_now, "global_step": global_step})

            epoch_train_loss = train_loss_sum / max(1, train_n)
            print(f"Epoch {epoch} completed. Train loss: {epoch_train_loss:.6f}")

            if tb_writer is not None:
                tb_writer.add_scalar("train/epoch_loss", epoch_train_loss, epoch)
            if wandb_run is not None:
                wandb_run.log({"train/epoch_loss": epoch_train_loss, "epoch": epoch})

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
                        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                            outb = model(xb)
                            lb = criterion(outb, xb)
                        val_loss_sum += lb.item() * bs
                        val_n += bs

                val_loss = val_loss_sum / max(1, val_n)
                print(f"Epoch {epoch} validation loss: {val_loss:.6f}")

                if tb_writer is not None:
                    tb_writer.add_scalar("val/epoch_loss", val_loss, epoch)
                if wandb_run is not None:
                    wandb_run.log({"val/epoch_loss": val_loss, "epoch": epoch})

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
