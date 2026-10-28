import os
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torchvision import datasets


def build_dataloaders(
    config,
    train_transform,
    val_transform,
    *,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Expects:
      config.DATA_ROOT — path whose train/ subdirectory is an ImageFolder tree.
      config.VAL_ROOT  — path to a pre-organized val set (ImageFolder layout,
                         i.e. one subfolder per class).
    """
    train_dataset = datasets.ImageFolder(
        os.path.join(config.DATA_ROOT, "train"),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        os.path.expanduser(config.VAL_ROOT),
        transform=val_transform,
        is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png")),
    )

    base_kw = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    if worker_init_fn is not None:
        base_kw["worker_init_fn"] = worker_init_fn
    if generator is not None:
        base_kw["generator"] = generator

    def _make(dataset, shuffle: bool) -> DataLoader:
        try:
            return DataLoader(dataset=dataset, shuffle=shuffle, **base_kw)
        except TypeError:
            kw = {k: v for k, v in base_kw.items()
                  if k not in ("generator", "persistent_workers")}
            return DataLoader(dataset=dataset, shuffle=shuffle, **kw)

    return _make(train_dataset, shuffle=True), _make(val_dataset, shuffle=False)
