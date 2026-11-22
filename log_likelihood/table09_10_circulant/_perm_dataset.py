"""Fixed-permutation coordinate wrapper for the permutation control.

One permutation pi is drawn once per (n, seed) and cached to disk, so every model
in the run matrix sees the *same* scrambling.  Raster order = identity (no wrap).
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))


def perm_path(n: int, seed: int) -> str:
    return os.path.join(_HERE, f"perm_n{n}_seed{seed}.pt")


def load_or_build_perm(n: int, seed: int = 0) -> torch.Tensor:
    """Return a cached permutation of range(n), building + saving it if absent."""
    path = perm_path(n, seed)
    if os.path.exists(path):
        perm = torch.load(path)
        assert perm.numel() == n, f"cached perm has {perm.numel()} != {n}"
        return perm
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    torch.save(perm, path)
    return perm


def inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


class PermutedFlatDataset(Dataset):
    """Wrap a dataset returning flat (n,) tensors; apply x -> x[perm]."""

    def __init__(self, base: Dataset, perm: torch.Tensor):
        self.base = base
        self.perm = perm

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x[self.perm], y
