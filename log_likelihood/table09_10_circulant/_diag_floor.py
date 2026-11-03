"""Diagonal Gaussian floor: N(m, diag(var)), closed form (no training).

This is the covariance BCCB collapses *toward* under coordinate permutation
(a stationary model applied to scrambled indices can only keep per-coordinate
variance).  Reporting it makes the collapse quantitative rather than merely visual.
It is permutation-invariant up to relabeling, so raster == permuted.
"""
from __future__ import annotations

import math

import torch


def diag_gaussian_bits_per_dim(train_loader, test_loader, n, device="cpu", eps=1e-6):
    """Fit per-coordinate mean/var on train, return (mean test LL nats, bits/dim)."""
    # Welford in fp64
    mean = torch.zeros(n, dtype=torch.float64, device=device)
    M2 = torch.zeros(n, dtype=torch.float64, device=device)
    count = 0
    for x, _ in train_loader:
        x = x.view(x.shape[0], -1).to(device=device, dtype=torch.float64)
        for row in x:
            count += 1
            delta = row - mean
            mean += delta / count
            M2 += delta * (row - mean)
    var = M2 / max(count - 1, 1) + eps

    logdet = torch.log(var).sum()
    const = n * math.log(2.0 * math.pi)

    total, ntest = 0.0, 0
    for x, _ in test_loader:
        x = x.view(x.shape[0], -1).to(device=device, dtype=torch.float64)
        diff = x - mean
        quad = (diff * diff / var).sum(dim=1)
        ll = -0.5 * (quad + logdet + const)
        total += ll.sum().item()
        ntest += x.shape[0]
    mean_ll = total / ntest
    bpd = -mean_ll / (n * math.log(2.0))
    return mean_ll, bpd
