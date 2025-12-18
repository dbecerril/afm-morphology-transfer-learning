from pathlib import Path

def setup_tensorboard(log_dir: str, run_name: str):
    """
    Creates and returns a TensorBoard SummaryWriter, or None if unavailable.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir = Path(log_dir) / run_name
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logging enabled at {tb_log_dir}")
        return writer
    except Exception as e:
        print(f"TensorBoard not available: {e}")
        return None


def setup_wandb(enable: bool, project: str, run_name: str, config: dict):
    """
    Initializes a Weights & Biases run if enabled.
    Returns the wandb run object or None.
    """
    if not enable:
        return None

    try:
        import wandb
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
        )
        print(f"Weights & Biases logging enabled (project={project})")
        return run
    except Exception as e:
        print(f"wandb not available or failed to init: {e}")
        return None


def log_train_step(tb_writer, wandb_run, loss, lr, global_step):
    if tb_writer is not None:
        tb_writer.add_scalar("train/batch_loss", loss, global_step)
        tb_writer.add_scalar("train/lr", lr, global_step)

    if wandb_run is not None:
        wandb_run.log({
            "train/batch_loss": loss,
            "train/lr": lr,
            "global_step": global_step,
        })


def log_epoch(tb_writer, wandb_run, split: str, loss: float, epoch: int):
    """
    split: 'train' or 'val'
    """
    if tb_writer is not None:
        tb_writer.add_scalar(f"{split}/epoch_loss", loss, epoch)

    if wandb_run is not None:
        wandb_run.log({
            f"{split}/epoch_loss": loss,
            "epoch": epoch,
        })


def close_loggers(tb_writer, wandb_run):
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
