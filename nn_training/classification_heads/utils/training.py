import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .misc import AverageMeter, accuracy


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """AMP + DDP trainer. Mixed precision is always enabled."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device,
        config,
        is_main_process: bool = True,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.is_main_process = is_main_process
        self.world_size = world_size

        self.best_val_acc = 0.0
        self.scaler = torch.amp.GradScaler("cuda")

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        it = tqdm(self.train_loader) if self.is_main_process else self.train_loader
        for images, targets in it:
            images = images.to(self.device, non_blocking=True,
                               memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if getattr(self.config, "GRAD_CLIP", None) is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc = accuracy(outputs, targets)[0]
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        it = tqdm(self.val_loader) if self.is_main_process else self.val_loader
        for images, targets in it:
            images = images.to(self.device, non_blocking=True,
                               memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            acc = accuracy(outputs, targets)[0]
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))

        return loss_meter.avg, acc_meter.avg

    def train(self) -> None:
        for epoch in range(self.config.EPOCHS):
            if self.is_main_process:
                print(f"\nEpoch [{epoch + 1}/{self.config.EPOCHS}]")
            t0 = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            if self.is_main_process:
                epoch_time = time.time() - t0
                lrs = [g["lr"] for g in self.optimizer.param_groups]

                print(f"Train Loss: {train_loss:.4f}  Acc@1: {train_acc:.2f}%")
                print(f"Val   Loss: {val_loss:.4f}  Acc@1: {val_acc:.2f}%")
                print(f"Time: {epoch_time:.1f}s  LRs: {lrs}")

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint("best_model.pth")
                    print("Saved best model.")

    def save_checkpoint(self, path: str) -> None:
        m = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(m.state_dict(), path)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    config,
    optimizer_type: str = "laplex",
) -> torch.optim.Optimizer:
    """Build AdamW with parameter groups appropriate for the model type.

    optimizer_type:
        "laplex"  — two groups: exponential params (a, b, log_tau) vs rest
        "lora"    — two groups: LoRA matrices (A, B) vs rest
        "simple"  — single group
    """
    betas = getattr(config, "BETAS", (0.9, 0.999))
    eps = getattr(config, "EPS", 1e-8)

    if optimizer_type == "laplex":
        exp_params, base_params = [], []
        for m in model.modules():
            if hasattr(m, "a") and hasattr(m, "b") and hasattr(m, "log_tau"):
                exp_params += [m.a, m.b, m.log_tau]
                base_params.append(m.c)
                if getattr(m, "bias", None) is not None:
                    base_params.append(m.bias)
        exp_ids = {id(p) for p in exp_params}
        base_ids = {id(p) for p in base_params}
        for p in model.parameters():
            if id(p) not in exp_ids and id(p) not in base_ids:
                base_params.append(p)
        return optim.AdamW(
            [
                {"params": base_params, "lr": config.LR_BASE,
                 "weight_decay": getattr(config, "WD_BASE", config.WEIGHT_DECAY)},
                {"params": exp_params,  "lr": config.LR_EXP,
                 "weight_decay": getattr(config, "WD_EXP",  config.WEIGHT_DECAY)},
            ],
            betas=betas, eps=eps,
        )

    elif optimizer_type == "lora":
        lora_params, base_params = [], []
        for m in model.modules():
            if hasattr(m, "A") and hasattr(m, "B"):
                lora_params += [m.A, m.B]
                if getattr(m, "bias", None) is not None:
                    base_params.append(m.bias)
        lora_ids = {id(p) for p in lora_params}
        base_ids = {id(p) for p in base_params}
        for p in model.parameters():
            if id(p) not in lora_ids and id(p) not in base_ids:
                base_params.append(p)
        return optim.AdamW(
            [
                {"params": base_params, "lr": config.LR_BASE,
                 "weight_decay": getattr(config, "WD_BASE", config.WEIGHT_DECAY)},
                {"params": lora_params, "lr": config.LR_EXP,
                 "weight_decay": getattr(config, "WD_EXP",  0.0)},
            ],
            betas=betas, eps=eps,
        )

    elif optimizer_type == "simple":
        return optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=getattr(config, "LR", config.LR_BASE),
            weight_decay=config.WEIGHT_DECAY,
            betas=betas, eps=eps,
        )

    raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config,
    scheduler_type: str = "warmup_cosine",
):
    """Build a learning-rate scheduler.

    scheduler_type:
        "warmup_cosine"  — linear warmup then cosine annealing
        "cosine"         — cosine annealing from epoch 0
    """
    if scheduler_type == "warmup_cosine":
        warmup = getattr(config, "WARMUP_EPOCHS", 0)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch) / float(max(1, warmup))
            progress = (epoch - warmup) / float(max(1, config.EPOCHS - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS,
            eta_min=getattr(config, "ETA_MIN", 1e-6),
        )

    raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}")
