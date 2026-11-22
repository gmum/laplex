"""Gaussian density model with a complex Laplace kernel.

Kernel of the model:
    K_c(α, β, γ, δ)[l, i] = exp(-|α_l - β_i|/τ_sign + i·(γ_l - δ_i))
    τ_sign = τ_R if α_l > β_i, otherwise τ_L

Effective REAL projection matrix:
    A[l, i] = Re(K_c[l, i]) = exp(-|α_l - β_i|/τ_sign) · cos(γ_l - δ_i)

Density model (as before, with the same Woodbury structure):
    B = diag(w) · A · L^T,   L lower-triangular k×k
    Σ = diag(s²) + B Bᵀ

**Key identity**: G = Aᵀ diag(D_α) A splits into three real
Laplace Gram matrices through the cos-cos identity:
    cos(γ_l - δ_i) cos(γ_l - δ_{i'}) = ½[cos(δ_i - δ_{i'}) + cos(2γ_l - δ_i - δ_{i'})]
Hence:
    G[i,i'] = ½ [ cos(δ_i - δ_{i'}) · G_base[i,i']
                + cos(δ_i + δ_{i'}) · G_{cos2γ}[i,i']
                + sin(δ_i + δ_{i'}) · G_{sin2γ}[i,i'] ]
where G_{·} = K_real · diag(D_{·}) · K_realᵀ (a plain real Laplace Gram
via `laplex_gram`, weights: D, D·cos(2γ), D·sin(2γ)).

Left multiplication v @ A = Re(v @ K_c) — `laplex_complex`.
Right multiplication A @ X = Re(K_c @ X) — `laplex_complex_right`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

from laplex_complex import laplex_complex, laplex_complex_right

# Reuse Ker_gram (real, two-τ) from the base model.
from _run_mnist import Ker_gram, _as_tau_vec
# Scan-based single-tau Gram  (O(n² + n·k) vs O(n²·k) dense)
from _gram_scan import ker_D_kerT_scan


def banded_A_times_G(A_lat, G, band_width):
    """Efficient A @ G when A_lat is lower-triangular with bandwidth `band_width`.

    A @ G: (k, k) · (k, k) = (k, k),  koszt O((band_width+1)·k²) flops.
    Vectorized version (no Python loop):
      1. Build G_stack ∈ ℝ^{(b+1, k, k)} with shifts of G along the first axis.
      2. Build weights ∈ ℝ^{(b+1, k)} from the b+1 diagonals of A (zero-padded).
      3. out[i, j] = Σ_d weights[d, i] · G_stack[d, i, j].

    Three torch ops instead of ~3·(b+1) — much less launch overhead.
    """
    k = G.shape[0]
    b = band_width
    # Shifts of G: pad the top with b zeros, then slice for every d ∈ [0, b]
    # G_pad[b + i - d, j] = G[i-d, j] if i >= d, else 0
    G_pad = F.pad(G, (0, 0, b, 0))                              # (k+b, k)
    # Stack of shifts: for d=0,...,b, a copy of G_pad[b-d : b-d+k, :]
    # as_strided or unfold would also work — here a stacked list comprehension (1 backend op)
    G_stack = torch.stack([G_pad[b - d : b - d + k] for d in range(b + 1)], dim=0)  # (b+1, k, k)

    # Diagonals of A_lat packed into (b+1, k) with zero padding
    weights = torch.zeros(b + 1, k, device=G.device, dtype=G.dtype)
    weights[0] = torch.diagonal(A_lat, offset=0)
    for d in range(1, b + 1):
        weights[d, d:] = torch.diagonal(A_lat, offset=-d)

    # One elementwise product + sum over d
    return (weights.unsqueeze(-1) * G_stack).sum(dim=0)


def banded_A_G_AT(A_lat, G, band_width):
    """Compute A @ G @ A.T for banded lower-triangular A, O(2·b·k²) flops."""
    AG = banded_A_times_G(A_lat, G, band_width)                 # (k, k)
    # (AG) @ A.T = (A @ (AG)^T)^T
    return banded_A_times_G(A_lat, AG.T.contiguous(), band_width).T.contiguous()


# ------------------------------------------------------------------
# Single-row (L=1) wrappers for the complex kernel.
# ------------------------------------------------------------------

def CX_mult_Kc(X, alpha, beta, gamma, delta, tau_L, tau_R=None):
    """X @ K_c(α, β, γ, δ).  X: (B, n_α) real → (B, n_β) complex.

    Tau convention: τ_R if α_i > β_j (as in the real laplex).
    """
    if tau_R is None:
        tau_R = tau_L
    s_ = torch.ones(1, device=X.device, dtype=X.dtype)
    Y = laplex_complex(
        X.unsqueeze(1),                 # (B, 1, n_α)
        alpha.unsqueeze(0),
        beta.unsqueeze(0),
        gamma.unsqueeze(0),
        delta.unsqueeze(0),
        _as_tau_vec(tau_L),
        _as_tau_vec(tau_R),
        s_,
        normalize=False,
    )                                   # (B, 1, n_β) complex
    return Y.squeeze(1)


def Kc_mult_X(X, alpha, beta, gamma, delta, tau_L, tau_R=None):
    """K_c(α, β, γ, δ) @ X.  X: (n_β, B) complex-or-real → (n_α, B) complex.

    Output na α-side (n_α).
    """
    if tau_R is None:
        tau_R = tau_L
    s_ = torch.ones(1, device=X.device, dtype=X.real.dtype if X.is_complex() else X.dtype)
    X3 = X.T.unsqueeze(1)               # (B, 1, n_β)
    Y3 = laplex_complex_right(
        X3,
        alpha.unsqueeze(0),
        beta.unsqueeze(0),
        gamma.unsqueeze(0),
        delta.unsqueeze(0),
        _as_tau_vec(tau_L),
        _as_tau_vec(tau_R),
        s_,
        normalize=False,
    )                                   # (B, 1, n_α) complex
    return Y3.squeeze(1).T


# ------------------------------------------------------------------
# Funkcyjny model.
# ------------------------------------------------------------------

class GaussianKerLapComplex:
    """Functional model: the original parametrization of A (lower-triangular, positive diagonal),
    Gram via the scan (fast), the whole critical path in fp64 for stability,
    with no tricks factoring A out.

        M = I + A·G·Aᵀ,   Σ⁻¹ via Woodbury as in the original model.
    """

    def __init__(self, m, w, alpha, beta, gamma, delta, A_lat, s, tau_L, tau_R=None,
                 eps=1e-12):
        self.m = m
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.A_lat = A_lat
        self.s = torch.clamp(s, min=eps)
        self.tau_L = tau_L
        self.tau_R = tau_R if tau_R is not None else tau_L
        self.eps = eps

    def log_likelihood(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        n = self.m.numel()
        k = self.A_lat.shape[0]

        D_inv = 1.0 / (self.s ** 2)

        # ===== Gram via the fp64 scan (3-way split, O(n²+n·k) per call) ====
        # Pure-PyTorch fp64 torch.cumsum (CUDA-executed).  ~0.86ms/call vs
        # dense fp32 1.48ms/call at (n=500, k=3072).  Stable thanks to fp64.
        same_tau = (self.tau_L is self.tau_R) or bool(
            (self.tau_L == self.tau_R).all()
        )
        w_tilde_sq = (self.w / self.s) ** 2
        cos2g = torch.cos(2.0 * self.gamma)
        sin2g = torch.sin(2.0 * self.gamma)

        if same_tau:
            tau = self.tau_L
            G_base  = ker_D_kerT_scan(self.beta, self.alpha, w_tilde_sq, tau)
            G_cos2g = ker_D_kerT_scan(self.beta, self.alpha, w_tilde_sq * cos2g, tau)
            G_sin2g = ker_D_kerT_scan(self.beta, self.alpha, w_tilde_sq * sin2g, tau)
        else:
            G_base  = Ker_gram(self.beta, self.alpha, w_tilde_sq,
                               tau_L=self.tau_R, tau_R=self.tau_L)
            G_cos2g = Ker_gram(self.beta, self.alpha, w_tilde_sq * cos2g,
                               tau_L=self.tau_R, tau_R=self.tau_L)
            G_sin2g = Ker_gram(self.beta, self.alpha, w_tilde_sq * sin2g,
                               tau_L=self.tau_R, tau_R=self.tau_L)

        # Everything to fp64 for the sum (the scan returns fp32 after the cast) —
        # re-cast so that the weighted sum is surely in fp64.
        G_base64  = G_base.to(torch.float64)
        G_cos2g64 = G_cos2g.to(torch.float64)
        G_sin2g64 = G_sin2g.to(torch.float64)

        d64 = self.delta.to(torch.float64)
        cos_diff = torch.cos(d64.unsqueeze(0) - d64.unsqueeze(1))
        cos_sum  = torch.cos(d64.unsqueeze(0) + d64.unsqueeze(1))
        sin_sum  = torch.sin(d64.unsqueeze(0) + d64.unsqueeze(1))

        G = 0.5 * (cos_diff * G_base64 + cos_sum * G_cos2g64 + sin_sum * G_sin2g64)
        G = 0.5 * (G + G.T)

        # ===== M = I + A G Aᵀ, Cholesky — w fp64 =======================
        A64 = self.A_lat.to(torch.float64)
        G64 = G                                                            # already fp64
        C64 = A64 @ G64 @ A64.T
        C64 = 0.5 * (C64 + C64.T)
        I_k64 = torch.eye(k, device=x.device, dtype=torch.float64)
        M64 = I_k64 + C64
        # Jitter safety (w fp64 mikrojitter wystarczy).
        L_chol64 = None
        for js in (0.0, 1e-12, 1e-8, 1e-4, 1e-2):
            M_try = M64 if js == 0.0 else (M64 + js * I_k64)
            L_try, info = torch.linalg.cholesky_ex(M_try)
            if int(info) == 0:
                L_chol64 = L_try
                break
        if L_chol64 is None:
            raise RuntimeError("Cholesky(M) failed even in fp64 with jitter 1e-2")

        # ===== quadratic form (computed in fp64 for quad + logdet) =====
        diff = x - self.m
        Dinv_diff = diff * D_inv
        v = Dinv_diff * self.w                                            # (B, n) fp32

        tmp_cplx = CX_mult_Kc(v, self.alpha, self.beta,
                              self.gamma, self.delta,
                              self.tau_L, self.tau_R)                     # (B, k) complex
        tmp = tmp_cplx.real

        BtDinv = tmp @ self.A_lat.T                                       # (B, k) fp32
        BtDinv64 = BtDinv.to(torch.float64)
        sol64 = torch.cholesky_solve(BtDinv64.unsqueeze(-1), L_chol64).squeeze(-1)
        quad_kernel = (sol64 * BtDinv64).sum(dim=1).to(x.dtype)            # (B,)

        quad = (diff * Dinv_diff).sum(dim=1) - quad_kernel

        # ===== logdet w fp64 ==========================================
        logdet = (
            torch.sum(torch.log(self.s ** 2))
            + (2.0 * torch.sum(torch.log(torch.diagonal(L_chol64)))).to(x.dtype)
        )

        return -0.5 * (
            quad + logdet
            + n * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
        )

    def map_project(self, x):
        """MAP posterior mean: x_hat = m + F(sol), sol = M^{-1} F^T(diff/s^2).

        Uses the same M and sol as log_likelihood, without repeating the code.
        Fast matrix operations (CX_mult_Kc / Kc_mult_X).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        D_inv = 1.0 / (self.s ** 2)

        same_tau = (self.tau_L is self.tau_R) or bool((self.tau_L == self.tau_R).all())
        k = self.A_lat.shape[0]
        w_tilde_sq = (self.w / self.s) ** 2
        cos2g = torch.cos(2.0 * self.gamma)
        sin2g = torch.sin(2.0 * self.gamma)
        if same_tau:
            tau = self.tau_L
            G_base  = ker_D_kerT_scan(self.beta, self.alpha, w_tilde_sq, tau)
            G_cos2g = ker_D_kerT_scan(self.beta, self.alpha, w_tilde_sq * cos2g, tau)
            G_sin2g = ker_D_kerT_scan(self.beta, self.alpha, w_tilde_sq * sin2g, tau)
        else:
            G_base  = Ker_gram(self.beta, self.alpha, w_tilde_sq, tau_L=self.tau_R, tau_R=self.tau_L)
            G_cos2g = Ker_gram(self.beta, self.alpha, w_tilde_sq * cos2g, tau_L=self.tau_R, tau_R=self.tau_L)
            G_sin2g = Ker_gram(self.beta, self.alpha, w_tilde_sq * sin2g, tau_L=self.tau_R, tau_R=self.tau_L)

        G_base64  = G_base.to(torch.float64)
        G_cos2g64 = G_cos2g.to(torch.float64)
        G_sin2g64 = G_sin2g.to(torch.float64)
        d64 = self.delta.to(torch.float64)
        cos_diff = torch.cos(d64.unsqueeze(0) - d64.unsqueeze(1))
        cos_sum  = torch.cos(d64.unsqueeze(0) + d64.unsqueeze(1))
        sin_sum  = torch.sin(d64.unsqueeze(0) + d64.unsqueeze(1))
        G = 0.5 * (cos_diff * G_base64 + cos_sum * G_cos2g64 + sin_sum * G_sin2g64)
        G = 0.5 * (G + G.T)

        A64 = self.A_lat.to(torch.float64)
        C64 = A64 @ G @ A64.T
        C64 = 0.5 * (C64 + C64.T)
        I_k64 = torch.eye(k, device=x.device, dtype=torch.float64)
        M64 = I_k64 + C64
        L_chol64 = None
        for js in (0.0, 1e-12, 1e-8, 1e-4, 1e-2):
            M_try = M64 if js == 0.0 else (M64 + js * I_k64)
            L_try, info = torch.linalg.cholesky_ex(M_try)
            if int(info) == 0:
                L_chol64 = L_try
                break

        diff = x - self.m
        v = (diff * D_inv) * self.w
        tmp_cplx = CX_mult_Kc(v, self.alpha, self.beta, self.gamma, self.delta, self.tau_L, self.tau_R)
        BtDinv = tmp_cplx.real @ self.A_lat.T
        BtDinv64 = BtDinv.to(torch.float64)
        sol64 = torch.cholesky_solve(BtDinv64.unsqueeze(-1), L_chol64).squeeze(-1)

        # decode: F(sol) = Kc_mult_X(A_lat @ sol.T).T.real * w
        sol32 = sol64.float()
        tmp_dec = sol32 @ self.A_lat                                      # (B, k)
        Kc_tmp = Kc_mult_X(tmp_dec.T, self.alpha, self.beta,
                           self.gamma, self.delta,
                           self.tau_L, self.tau_R)                        # (n, B) complex
        return self.m + Kc_tmp.T.real * self.w                           # (B, n)

    def sample(self, num_samples):
        n = self.m.numel()
        k = self.A_lat.shape[0]

        z = torch.randn(num_samples, k, device=self.m.device)
        eps_ = torch.randn(num_samples, n, device=self.m.device)

        tmp = z @ self.A_lat                                              # (B, k)
        Kc_tmp = Kc_mult_X(tmp.T, self.alpha, self.beta,
                           self.gamma, self.delta,
                           self.tau_L, self.tau_R)                        # (n, B) complex
        signal = Kc_tmp.T.real * self.w                                   # (B, n)

        return self.m + signal + eps_ * self.s


# ------------------------------------------------------------------
# Trainable module.
# ------------------------------------------------------------------

class GaussianKerLapComplexWoodbury(nn.Module):
    """Trainable variant with the complex kernel.

    Extra parameters compared with the real variant:
      - γ : (n,)   phases on the α side
      - δ : (k,)   phases on the β side
    (the phase temperature `s` is fixed to 1 and not trained — γ, δ
    absorb the scale themselves.)
    """

    def __init__(self, n, k, eps=1e-4, single_tau=True, mean_init=None,
                 band_width=None):
        """band_width: number of non-zero subdiagonals of A_lat (not counting
        the main one). For band_width=5, A_lat has 5 subdiagonals + the main one = 6
        non-zero diagonals. None = the full lower triangle."""
        super().__init__()
        self.n = n
        self.k = k
        self.eps = eps
        self.single_tau = single_tau
        self.band_width = band_width

        if mean_init is not None:
            self.register_buffer("m", mean_init.detach().to(torch.float32).clone())
        else:
            self.m = nn.Parameter(torch.zeros(n))
        self.w = nn.Parameter(torch.randn(n) * 0.01)

        self.alpha = nn.Parameter(torch.linspace(-1, 1, n))
        self.beta  = nn.Parameter(torch.linspace(-1, 1, k))
        # γ, δ: start from a different linspace to have non-zero phases at once
        # (the gradient of γ, δ is 0 when their difference is zero).
        self.gamma = nn.Parameter(torch.linspace(-1, 1, n))
        self.delta = nn.Parameter(torch.linspace(-1, 1, k))

        # A parametrization: lower-triangular with a positive diagonal (original init).
        if band_width is not None:
            self.band_subdiags = nn.ParameterList([
                nn.Parameter(torch.randn(k - d) * 0.01)
                for d in range(1, band_width + 1)
            ])
        else:
            self.A_off = nn.Parameter(torch.randn(k, k) * 0.01)

        self.A_diag_raw = nn.Parameter(torch.full((k,), -4.6))        # softplus ≈ 0.01
        self.s_raw = nn.Parameter(torch.zeros(n))

        if single_tau:
            self.tau_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.tau_L_raw = nn.Parameter(torch.tensor(0.0))
            self.tau_R_raw = nn.Parameter(torch.tensor(0.0))

    def tau_L(self):
        if self.single_tau:
            return F.softplus(self.tau_raw) + self.eps
        return F.softplus(self.tau_L_raw) + self.eps

    def tau_R(self):
        if self.single_tau:
            return F.softplus(self.tau_raw) + self.eps
        return F.softplus(self.tau_R_raw) + self.eps

    def s(self):
        return F.softplus(self.s_raw) + self.eps

    def A_lat(self):
        """Zbuduj A (lower-tri, pos-diag)."""
        diag = F.softplus(self.A_diag_raw) + self.eps
        if self.band_width is not None:
            A = torch.diag(diag)
            for d, subdiag in enumerate(self.band_subdiags, start=1):
                A = A + torch.diag(subdiag, diagonal=-d)               # below diagonal
            return A
        else:
            L = torch.tril(self.A_off, diagonal=-1)
            return L + torch.diag(diag)

    def to_functional(self):
        return GaussianKerLapComplex(
            self.m, self.w,
            self.alpha, self.beta,
            self.gamma, self.delta,
            self.A_lat(),
            self.s(), self.tau_L(), self.tau_R(),
        )

    def log_likelihood(self, x):
        return self.to_functional().log_likelihood(x)

    def sample(self, num_samples):
        return self.to_functional().sample(num_samples)
