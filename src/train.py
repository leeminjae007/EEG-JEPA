import os
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from src.datasets.tuheeg_loader import make_tuheeg
from src.models.hibrainmj import HiBrainMJ, HiBrainMJConfig

try:
    import yaml
except ImportError:
    yaml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else default


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _save(path, model, optimizer, epoch):
    model = _unwrap(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "opt": optimizer.state_dict(),
        },
        path,
    )


def _load(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    _unwrap(model).load_state_dict(state, strict=False)
    if "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])
    return int(checkpoint.get("epoch", 0))


def _distributed_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _set_lr(optimizer, step, total_steps, warmup_steps, base_lr, final_lr):
    if warmup_steps > 0 and step < warmup_steps:
        lr = base_lr * float(step + 1) / float(warmup_steps)
    else:
        denom = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _mean_across_ranks(value, device):
    x = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x)
        x /= dist.get_world_size()
    return float(x.cpu())


def _init_wandb(args, log_cfg, rank):
    wandb_cfg = _get(log_cfg, "wandb", {})
    if not (rank == 0 and isinstance(wandb_cfg, dict) and bool(_get(wandb_cfg, "enabled", False))):
        return None
    if wandb is None:
        print("wandb is not installed; run `pip install wandb` to enable W&B logging")
        return None
    run = wandb.init(
        project=_get(wandb_cfg, "project", "hibrainmj"),
        entity=_get(wandb_cfg, "entity", None),
        name=_get(wandb_cfg, "name", None),
        group=_get(wandb_cfg, "group", None),
        tags=_get(wandb_cfg, "tags", None),
        dir=_get(log_cfg, "folder", None),
        config=args,
        resume="allow",
    )
    return run


def main(args, resume_preempt=False):
    model_cfg = _get(args, "model", {})
    data_cfg = _get(args, "data", {})
    opt_cfg = _get(args, "optimization", {})
    log_cfg = _get(args, "logging", {})
    meta_cfg = _get(args, "meta", {})

    world_size, rank = _distributed_info()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HiBrainMJ(HiBrainMJConfig.from_dict(args)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_get(opt_cfg, "lr", 5e-4)),
        weight_decay=float(_get(opt_cfg, "weight_decay", 0.04)),
    )
    scaler = torch.amp.GradScaler("cuda") if bool(_get(meta_cfg, "use_bfloat16", False)) and device.type == "cuda" else None

    patch_len = int(_get(model_cfg, "patch_len", 16))
    num_patches = _get(data_cfg, "num_patches")
    window_size = _get(data_cfg, "window_size", None)
    if window_size is None and num_patches is not None:
        window_size = int(num_patches) * patch_len

    _, loader, sampler = make_tuheeg(
        batch_size=int(_get(data_cfg, "batch_size", 64)),
        split="train",
        root_dir=_get(data_cfg, "root_path"),
        window_size=window_size,
        patch_size=patch_len,
        test_ratio=float(_get(data_cfg, "test_ratio", 0.2)),
        collator=None,
        pin_mem=bool(_get(data_cfg, "pin_mem", True)),
        num_workers=int(_get(data_cfg, "num_workers", 4)),
        persistent_workers=bool(_get(data_cfg, "persistent_workers", False)),
        max_samples=_get(data_cfg, "max_samples", None),
        num_subjects=_get(data_cfg, "num_subjects", None),
        subset_subjects_path=_get(data_cfg, "subset_subjects_path", None),
        packed_data_path=_get(data_cfg, "packed_data_path", None),
        packed_keys_path=_get(data_cfg, "packed_keys_path", None),
        prefetch_factor=_get(data_cfg, "prefetch_factor", None),
        world_size=world_size,
        rank=rank,
    )

    folder = _get(log_cfg, "folder", None)
    writer = SummaryWriter(os.path.join(folder, "tb")) if (
        rank == 0 and SummaryWriter is not None and folder and bool(_get(log_cfg, "tensorboard", True))
    ) else None
    if rank == 0 and folder:
        os.makedirs(folder, exist_ok=True)
        if yaml is not None:
            with open(os.path.join(folder, "params-hibrainmj.yaml"), "w") as f:
                yaml.safe_dump(args, f)
    wandb_run = _init_wandb(args, log_cfg, rank)

    start_epoch = 0
    if bool(_get(meta_cfg, "load_checkpoint", False)) or resume_preempt:
        ckpt = _get(meta_cfg, "read_checkpoint", None)
        ckpt = ckpt or (os.path.join(folder, "hibrainmj-latest.pth.tar") if folder else None)
        if ckpt:
            start_epoch = _load(ckpt, model, optimizer, device)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[0] if device.type == "cuda" else None)

    epochs = int(_get(opt_cfg, "epochs", 30))
    momentum = float(_get(_get(args, "hibrainmj", {}), "ema_momentum", _get(opt_cfg, "ema_momentum", 0.996)))
    total_steps = max(len(loader) * max(epochs - start_epoch, 1), 1)
    warmup_steps = int(float(_get(opt_cfg, "warmup", 0)) * len(loader))
    base_lr = float(_get(opt_cfg, "lr", 5e-4))
    final_lr = float(_get(opt_cfg, "final_lr", 1e-6))
    step = start_epoch * len(loader)
    for epoch in range(start_epoch, epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        losses = []
        iterator = tqdm(loader, mininterval=10) if rank == 0 else loader
        for batch in iterator:
            eeg = batch["eeg"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            lr = _set_lr(optimizer, step, total_steps, warmup_steps, base_lr, final_lr)
            if scaler is None:
                out = model(eeg)
                out["loss"].backward()
                optimizer.step()
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(eeg)
                scaler.scale(out["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
            _unwrap(model).update_target_encoder(momentum)
            loss_value = _mean_across_ranks(out["loss"].detach(), device)
            losses.append(loss_value)
            if writer:
                writer.add_scalar("loss/train", loss_value, step)
                writer.add_scalar("lr", lr, step)
            if wandb_run:
                wandb.log({"loss/train": loss_value, "lr": lr, "epoch": epoch + 1}, step=step)
            step += 1

        mean_loss = sum(losses) / max(len(losses), 1)
        if rank == 0:
            print(f"epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}")
        if wandb_run:
            wandb.log({"loss/epoch": mean_loss, "epoch": epoch + 1}, step=step)
        if rank == 0 and folder:
            _save(os.path.join(folder, "hibrainmj-latest.pth.tar"), model, optimizer, epoch + 1)

    if writer:
        writer.close()
    if wandb_run:
        wandb.finish()
