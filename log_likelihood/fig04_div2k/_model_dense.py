"""Dense low-rank Woodbury baseline (prob-PCA-like).

Analogous to `GaussianKerLapWoodbury`, but the first, dimension-reducing matrix
is simply dense, **without** the Laplace-kernel structure.

Model:
    B   = W · L^T,  W ∈ ℝ^{n × k_dense}  (dense), L lower-triangular k_d×k_d
    Σ   = diag(s²) + B Bᵀ
    N(m, Σ)

Use:  Woodbury and Cholesky as in the kernel model,
only `B^T · diag(w²/s²) · B` is computed EXPLICITLY (no LAPLEX):
    G = Lᵀ · Wᵀ · diag(1/s²) · W · L
(the dense model has no extra `w` — W already subsumes it).

Number of effective parameters:
    2n + n·k_d + k_d(k_d+1)/2
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDenseLowRank:
    def __init__(self, m, W, A_lat, s, eps=1e-12):
        self.m = m
        self.W = W                   # (n, k_d)  dense
        self.A_lat = A_lat           # (k_d, k_d) lower-tri
        self.s = torch.clamp(s, min=eps)
        self.eps = eps

    def log_likelihood(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        n = self.m.numel()
        k = self.A_lat.shape[0]

        D_inv = 1.0 / (self.s ** 2)                    # (n,)
        B = self.W @ self.A_lat.T                      # (n, k_d)

        # G = B^T diag(1/s²) B   (k_d, k_d)
        G = (B.T * D_inv) @ B
        G = 0.5 * (G + G.T)                            # symetryzacja

        M = torch.eye(k, device=x.device, dtype=x.dtype) + G
        L_chol = torch.linalg.cholesky(M)

        diff = x - self.m
        Dinv_diff = diff * D_inv                       # (B, n)
        BtDinv = Dinv_diff @ B                         # (B, k_d)

        sol = torch.cholesky_solve(BtDinv.unsqueeze(-1), L_chol).squeeze(-1)
        quad = (diff * Dinv_diff).sum(dim=1) - (sol * BtDinv).sum(dim=1)

        logdet = (
            torch.sum(torch.log(self.s ** 2))
            + 2 * torch.sum(torch.log(torch.diagonal(L_chol)))
        )

        return -0.5 * (
            quad + logdet
            + n * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
        )

    def map_project(self, x):
        """MAP posterior mean: x_hat = m + B @ sol.T, sol = M^{-1} BtDinv."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        D_inv = 1.0 / (self.s ** 2)
        B = self.W @ self.A_lat.T                      # (n, k_d)
        G = (B.T * D_inv) @ B
        G = 0.5 * (G + G.T)
        k = self.A_lat.shape[0]
        M = torch.eye(k, device=x.device, dtype=x.dtype) + G
        L_chol = torch.linalg.cholesky(M)
        diff = x - self.m
        BtDinv = (diff * D_inv) @ B                    # (B, k_d)
        sol = torch.cholesky_solve(BtDinv.unsqueeze(-1), L_chol).squeeze(-1)  # (B, k_d)
        return self.m + sol @ B.T                      # (B, n)

    def sample(self, num_samples):
        n = self.m.numel()
        k = self.A_lat.shape[0]

        z = torch.randn(num_samples, k, device=self.m.device)
        eps = torch.randn(num_samples, n, device=self.m.device)

        tmp = z @ self.A_lat                            # (B, k_d)
        signal = tmp @ self.W.T                         # (B, n)

        return self.m + signal + eps * self.s


class GaussianDenseLowRankWoodbury(nn.Module):
    """Trainable dense low-rank model.

    Args
    ----
    n        : data dimension
    k_dense  : latent dimension (rank B = k_dense)
    """

    def __init__(self, n, k_dense, eps=1e-4, single_tau=True, mean_init=None):
        super().__init__()
        self.n = n
        self.k_dense = k_dense
        self.eps = eps
        # `single_tau` is ignored — the dense model has no τ.  The flag is accepted
        # only for a signature uniform with the ModelClass of the runner.
        self.single_tau = single_tau

        if mean_init is not None:
            self.register_buffer("m", mean_init.detach().to(torch.float32).clone())
        else:
            self.m = nn.Parameter(torch.zeros(n))
        self.s_raw = nn.Parameter(torch.zeros(n))

        # W: dense low-rank factor  (He-like init of order 0.01, as in
        # the kernel model, where the total effect of w @ K @ A was of similar size)
        self.W = nn.Parameter(torch.randn(n, k_dense) * 0.01)

        # A_lat: lower-triangular with a positive diagonal (as in the kernel model)
        self.A_off = nn.Parameter(torch.randn(k_dense, k_dense) * 0.01)
        self.A_diag_raw = nn.Parameter(torch.full((k_dense,), -4.6))

    def s(self):
        return F.softplus(self.s_raw) + self.eps

    def A_lat(self):
        L = torch.tril(self.A_off, diagonal=-1)
        diag = F.softplus(self.A_diag_raw) + self.eps
        return L + torch.diag(diag)

    def to_functional(self):
        return GaussianDenseLowRank(
            self.m, self.W, self.A_lat(), self.s(),
        )

    def log_likelihood(self, x):
        return self.to_functional().log_likelihood(x)

    def sample(self, num_samples):
        return self.to_functional().sample(num_samples)

    # Placeholders for uniformity — the runner uses tau_L/tau_R of the kernel models.
    def tau_L(self):
        return torch.tensor(float("nan"))

    def tau_R(self):
        return torch.tensor(float("nan"))
