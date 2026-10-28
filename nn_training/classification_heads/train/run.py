"""Shared training pipeline used by all four train scripts."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import LaplexLinear, LoRALinear
from train.setup import build_model, _BACKBONES
from utils import (
    Trainer,
    build_dataloaders,
    build_optimizer,
    build_scheduler,
    count_parameters,
    fix_seed,
    get_torch_generator,
    get_worker_init_fn,
)


def run(config, head_cls: type, optimizer_type: str) -> None:
    """Full training pipeline.

    Args:
        config:         plain-module config (e.g. ``import configs.laplex_frozen as config``)
        head_cls:       ``LaplexLinear`` or ``LoRALinear``
        optimizer_type: ``"laplex"``, ``"lora"``, or ``"simple"``
    """
    # ------------------------------------------------------------------
    # DDP setup
    # ------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    fix_seed(
        seed=config.SEED + local_rank,
        cudnn_benchmark=config.CUDNN_BENCHMARK,
        cudnn_deterministic=config.CUDNN_DETERMINISTIC,
        cublas_workspace_config=config.CUBLAS_WORKSPACE_CONFIG,
    )

    # ------------------------------------------------------------------
    # Data  (env vars DATA_ROOT / VAL_ROOT override config file values)
    # ------------------------------------------------------------------
    if os.environ.get("DATA_ROOT"):
        config.DATA_ROOT = os.environ["DATA_ROOT"]
    if os.environ.get("VAL_ROOT"):
        config.VAL_ROOT = os.environ["VAL_ROOT"]

    _, weights = _BACKBONES[config.MODEL]
    transform = weights.transforms()

    generator = get_torch_generator(config.SEED + local_rank)
    worker_init_fn = get_worker_init_fn(config.SEED + local_rank)

    train_loader, val_loader = build_dataloaders(
        config, transform, transform,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(config, head_cls)
    model = model.to(memory_format=torch.channels_last).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )
        actual_model = model.module
    else:
        actual_model = model

    if is_main:
        total, trainable = count_parameters(actual_model)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:    {total - trainable:,}")

    # torch.compile: skip under DDP (can be unstable) and CUDA laplex backend
    # (load_inline extensions don't always survive compilation).
    if torch.__version__ >= "2" and world_size == 1:
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = build_optimizer(actual_model, config, optimizer_type)
    scheduler = build_scheduler(optimizer, config, "warmup_cosine")
    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0)
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        is_main_process=is_main,
        world_size=world_size,
    ).train()
