"""Dense low-rank Woodbury baseline (prob-PCA-like).

Analog do `GaussianKerLapWoodbury`, ale pierwsza macierz redukująca wymiar
jest po prostu gęsta (dense), **bez** struktury jądra Laplace.

Model:
    B   = W · L^T,  W ∈ ℝ^{n × k_dense}  (dense), L dolnotrójkątna k_d×k_d
    Σ   = diag(s²) + B Bᵀ
    N(m, Σ)

Użycie:  Woodbury i Cholesky identyczne jak w modelu kernelowym,
tylko `B^T · diag(w²/s²) · B` liczymy JAWNIE (bez LAPLEX):
    G = Lᵀ · Wᵀ · diag(1/s²) · W · L
(w modelu dense pomijamy dodatkowe `w` — W je już subsumuje).

Liczba efektywnych parametrów:
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
    """Trenowalny dense low-rank.

    Args
    ----
    n        : wymiar danych
    k_dense  : wymiar ukryty (rank B = k_dense)
    """

    def __init__(self, n, k_dense, eps=1e-4, single_tau=True, mean_init=None):
        super().__init__()
        self.n = n
        self.k_dense = k_dense
        self.eps = eps
        # `single_tau` jest ignorowany — dense analog nie ma τ.  Akceptujemy
        # flagę tylko dla jednolitości sygnatury z ModelClass w runnerze.
        self.single_tau = single_tau

        if mean_init is not None:
            self.register_buffer("m", mean_init.detach().to(torch.float32).clone())
        else:
            self.m = nn.Parameter(torch.zeros(n))
        self.s_raw = nn.Parameter(torch.zeros(n))

        # W: dense low-rank factor  (He-ish init dopasowany do rzędu 0.01 jak w
        # kernel-modelu, gdzie sumaryczny wpływ w @ K @ A był podobnej wielkości)
        self.W = nn.Parameter(torch.randn(n, k_dense) * 0.01)

        # A_lat: lower-tri z dodatnią przekątną (analog kernelowego modelu)
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

    # Placeholdery dla uniformowości — runner używa tau_L/tau_R dla kernel-modeli.
    def tau_L(self):
        return torch.tensor(float("nan"))

    def tau_R(self):
        return torch.tensor(float("nan"))
