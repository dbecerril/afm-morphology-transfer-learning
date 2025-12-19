# Copilot / AI Agent Instructions — afm-morphology-transfer-learning

This project trains and evaluates an AFM U-Net autoencoder on patch HDF5 datasets. The goal of this file is to give an AI coding agent the minimal, high-value context to be productive quickly: where data comes from, how training and evaluation are run, important file formats and conventions, and representative code examples.

## Big picture (what matters)
- Core components:
  - `datasets/afm_h5_dataset.py` — the PyTorch Dataset expected input: HDF5 with `patches/proc` (N,C,H,W) and optional `patches/aux_type`, `patches/base_id`, `patches/top_left_yx`.
  - `model/autoencoder_model.py` — `AFMUNetAutoencoder` (U-Net style) with optional late FiLM aux conditioning. `embed()` returns a per-patch embedding (global avg pool of bottleneck).
  - `scripts/train_autoencoder.py` — main training CLI: split creation, checkpointing, TensorBoard/W&B logging, mixed-precision support, and optional aux conditioning.
  - `utils/` — helpers for IO, stats, dataloaders, reproducibility, logging.
  - `preprocess/` — scripts to convert raw AFM files into the .h5 used by the dataset (e.g. `afm_to_gwy.sh`, `process_gwy_multichannel.py`).

## Developer workflows & important commands
- Create dataset splits (recommended `by-base-id` to avoid leakage):
  - python scripts/make_splits.py --h5 <h5file> --out splits --mode by-base-id --ratios 0.8 0.1 0.1 --seed 0
- Train (defaults in `scripts/train_autoencoder.py`):
  - python scripts/train_autoencoder.py --h5 afm_patches_256.h5 --split-dir splits --split-name train --val-split val --epochs 20 --batch-size 32 --log-tb --log-dir logs --run-name myrun
- Resume training from a checkpoint:
  - python scripts/train_autoencoder.py --resume checkpoints/ae_epoch005.pth
- Preprocess raw AFM files (from README):
  1. Place AFM files under `data/data_afm/`
  2. Run `preprocess/afm_to_gwy.sh`
  3. Run `preprocess/process_gwy_multichannel.py` to produce the HDF5
- Quick inference / latent extraction (example):
  - Load model and checkpoint: see `notebooks/latent_aux_correlation.ipynb` — load `AFMUNetAutoencoder`, load `ck = torch.load("checkpoints/ae_best.pth")`, `state = ck['model_state_dict']`, `model.load_state_dict(state)`.
  - Use `model.embed(topo_tensor)` to get per-patch embeddings (B, D).

## File / format specifics agents should know
- Expected HDF5 layout:
  - `patches/proc` : (N, C, H, W) float32 primary patches
  - Optional: `patches/aux_type` (N,) string, `patches/base_id` (N,) string, `patches/top_left_yx` (N,2)
  - `AFMPatchesH5Dataset` will use `patches/norm` if present (x_dataset='auto').
- Channel normalization: stored in JSON at `stats/channel_norm.json` (structure: {"mean":[...], "std":[...]}) — training auto-generates it if missing using `ensure_channel_norm_file()`.
- Checkpoint content: saved as dicts with keys like `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `train_loss`, `val_loss`. The `train` script also writes `run_config.json` and copies `channel_norm.json` to the checkpoint dir.
- Default aux channels: `--aux-types` default is `["PHASE", "FRICTION"]` and this repository often uses these. Aux conditioning is optional (`--use-aux-conditioning` flag) and when enabled the model expects aux tensors passed to `forward(topo, aux)`.

## Patterns & conventions specific to this codebase
- Topography-first: models are built to use topo as the encoded input; aux channels are used only for late conditioning (FiLM) and are intentionally not injected into the encoder. Use `use_aux_conditioning` when you want the aux to condition the decoder.
- Split generation: prefer `by-base-id` (keeps patches from same base together) to avoid leakage; `ensure_split_files()` will try that first.
- DataLoader settings tuned for reproducibility and speed: `pin_memory` when CUDA available, `persistent_workers` when `num_workers>0`, `seed_worker` used for deterministic shuffling.
- Minimal external tooling: TensorBoard and W&B are optional. Training proceeds if these packages are not installed.

## Quick code examples (copy-paste friendly)
- Train with val logging and TensorBoard:
  - python scripts/train_autoencoder.py --h5 afm_patches_256.h5 --split-dir splits --split-name train --val-split val --epochs 50 --batch-size 32 --log-tb --log-dir logs
- Load and run embed on CPU/GPU:
  - model = AFMUNetAutoencoder(in_channels=1, out_channels=1, aux_channels=0).to(device)
  - ck = torch.load('checkpoints/ae_best.pth', map_location=device)
  - state = ck.get('model_state_dict', ck)
  - model.load_state_dict(state)
  - model.eval(); emb = model.embed(topo_tensor.to(device))

## Where to look for common tasks
- Add a new metric/logging hook: `utils/logging.py` and `scripts/train_autoencoder.py` (calls to `log_epoch` / `log_train_step`)
- Change dataset filtering or normalization: `datasets/afm_h5_dataset.py` and `utils/data.py`
- Modify model architecture or conditioning: `model/autoencoder_model.py` (see `FiLM` and `embed()`)
- Reproduce/inspect training artifacts: `checkpoints/<run>/run_config.json` and `checkpoints/<run>/channel_norm.json` are authoritative per-run records.

## Small gotchas / developer tips
- HDF5 file path resolution: `utils/io.resolve_h5_path()` will search project root and `datasets/` automatically; pass relative names used in the repo to be robust.
- `AFMPatchesH5Dataset` opens HDF5 lazily per-worker; avoid storing open handles across process forks.
- When aux conditioning is enabled in the model (aux_channels>0) you must pass `aux` to `forward()` or it will raise a `ValueError`.

---
Please review: are there additional workflows, helper scripts, or project behaviours you want emphasized (e.g., CI/test commands, preferred evaluation scripts, or common debugging steps)? I'll iterate the file accordingly.