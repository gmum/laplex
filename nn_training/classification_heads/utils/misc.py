import functools
import os
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    """Running mean tracker."""

    __slots__ = ("val", "sum", "count", "avg")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = self.sum = self.count = self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)) -> list[Tensor]:
    """Top-k accuracy, returned as percentages."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch_size)
            for k in topk
        ]


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_seed(
    seed: int = 42,
    cudnn_benchmark: bool = False,
    cudnn_deterministic: bool = True,
    cublas_workspace_config: str | None = ":4096:8",
) -> None:
    """Set global seeds and deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cublas_workspace_config is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _seed_worker(seed: int, worker_id: int) -> None:
    w = seed + worker_id
    np.random.seed(w)
    random.seed(w)
    torch.manual_seed(w)


def get_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Return a DataLoader worker_init_fn that seeds each worker differently."""
    # A partial of a module-level function, not a closure: Python >= 3.14 starts
    # DataLoader workers with forkserver, which pickles worker_init_fn.
    return functools.partial(_seed_worker, seed)


def get_torch_generator(seed: int) -> torch.Generator:
    """Return a seeded CPU Generator for deterministic DataLoader sampling."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g
