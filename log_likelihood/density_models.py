"""
density_models.py  (App. I.2 and App. L: the covariance models of Tabs. 9 and 14)
=================
A downstream task and stronger structured covariance baselines for the density
model of Sec. 4.2:

  * Downstream task 1 — MAP image restoration (denoising): corrupt held-out
    images with Gaussian noise and restore by the Gaussian posterior mean under
    each fitted covariance model; report PSNR (and SSIM if scikit-image present).
  * (Optional, not in the paper) OOD detection: each model's log-likelihood as the
    score; AUROC separating in-distribution (ImageNette val) from OOD.
  * Stronger structured baseline — a Kronecker / tensor-normal Gaussian
    (separable per-axis covariance), which at the same budget affords full-rank
    per-axis covariances and explicitly encodes the pixel grid. LAPLEX is given
    no such spatial bias.

Covariance models (Gaussian x ~ N(m, Sigma)):
  LAPLEX     : factor-analysis  Sigma = F F^T + diag(sigma^2),  F = diag(w) A L,
               A = LAPLEX(a,b) in R^{n x k}, L in R^{k x k} lower-triangular.
               Trained by maximizing the Woodbury log-likelihood (Adam).
  dense-LR   : same FA template with a free F in R^{n x r} (budget-matched r).
  PPCA       : probabilistic PCA (closed form: top-r empirical eigenvectors).
  Kronecker  : tensor-normal Sigma = Sigma_C (x) Sigma_H (x) Sigma_W, flip-flop MLE.

All FA log-likelihoods/MAP use the Woodbury identity and never form Sigma.
Kronecker LL/MAP use per-axis eigendecompositions (exact, cheap).

Run `python density_models.py` first (self-test): validates every model's LL (and MAP
where it exists) against a brute-force dense Gaussian on a tiny problem (must print
rel-errors ~1e-6 or better). Then table09_14_density.py runs the full experiment.

Comments in English (repo convention).
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# laplex.py and prefix_suffix.py of ../benchmarks/comparison_experiments (the released operators)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "benchmarks", "comparison_experiments"))
import laplex  # exact LAPLEX CUDA kernels

LOG2PI = math.log(2.0 * math.pi)


# =============================================================================
# LAPLEX factor-analysis matvecs (never materialize F or A)
# =============================================================================
def A_matvec(y, a, b):
    """A @ y for y in R^{k} or (k, B) -> R^{n} / (n, B).  A = K(a,b), n x k."""
    sq = y.dim() == 1
    Y = y.unsqueeze(1) if sq else y
    out = laplex.Laplex_X(Y, a, b, threshold=0)      # K(a,b) @ Y
    return out.squeeze(1) if sq else out


def AT_matvec(r, a, b):
    """A^T @ r for r in R^{n} or (B, n) -> R^{k} / (B, k)."""
    sq = r.dim() == 1
    R = r.unsqueeze(0) if sq else r
    out = laplex.X_Laplex(R, a, b, threshold=0)      # R @ K(a,b) = (A^T r^T)^T
    return out.squeeze(0) if sq else out


def AT_diag_A(a, b, g):
    """A^T diag(g) A in R^{k x k} exactly, g in R^n.  Uses K(a,b)^T = K(b,a)."""
    return laplex.Laplex_D_LaplexT(b, a, g, threshold=0)   # (k, k)


# =============================================================================
# Factor-analysis model:  Sigma = F F^T + diag(d2),  F depends on `kind`
# =============================================================================
class FAModel(nn.Module):
    """Trainable factor-analysis Gaussian with either a LAPLEX or a free-dense F."""

    def __init__(self, n, k, kind, mean_init, dtype=torch.float64):
        super().__init__()
        self.n, self.k, self.kind = n, k, kind
        self.mean = nn.Parameter(mean_init.to(dtype).clone())
        self.log_d = nn.Parameter(torch.zeros(n, dtype=dtype) - 1.0)  # diag std (log)
        if kind == "laplex":
            self.a = nn.Parameter(torch.randn(n, dtype=dtype))
            self.b = nn.Parameter(torch.linspace(-3, 3, k, dtype=dtype))
            self.w = nn.Parameter(torch.randn(n, dtype=dtype) / math.sqrt(n))
            self.L = nn.Parameter(torch.eye(k, dtype=dtype) * 0.1
                                  + 0.01 * torch.randn(k, k, dtype=dtype))
        elif kind == "dense":
            self.Fdense = nn.Parameter(torch.randn(n, k, dtype=dtype) / math.sqrt(n))
        else:
            raise ValueError(kind)

    # --- F and F^T applied to (B, k)/(B, n) batches (rows = examples) ---
    def _Ltri(self):
        return torch.tril(self.L)

    def F_apply(self, Z):  # Z: (B, k) -> (B, n)
        if self.kind == "dense":
            return Z @ self.Fdense.T
        Lt = self._Ltri()
        t = Z @ Lt.T                       # (B, k)  = (L z)
        At = A_matvec(t.T, self.a, self.b)  # (n, B)
        return (self.w.unsqueeze(1) * At).T  # (B, n)

    def FT_apply(self, R):  # R: (B, n) -> (B, k)
        if self.kind == "dense":
            return R @ self.Fdense
        u = R * self.w                     # (B, n)
        Atu = AT_matvec(u, self.a, self.b)  # (B, k)
        return Atu @ self._Ltri()          # (B, k) = L^T (A^T u)

    def FT_Dinv_F(self, dinv):
        """F^T diag(dinv) F in R^{k x k}."""
        if self.kind == "dense":
            return (self.Fdense * dinv.unsqueeze(1)).T @ self.Fdense
        Lt = self._Ltri()
        G = AT_diag_A(self.a, self.b, self.w * self.w * dinv)  # A^T diag(w^2 dinv) A
        return Lt.T @ G @ Lt

    # --- Woodbury log-likelihood of a data batch X (B, n) ---
    def loglik(self, X):
        n, k = self.n, self.k
        d2 = torch.exp(2.0 * self.log_d)          # (n,)
        dinv = 1.0 / d2
        R = X - self.mean                          # (B, n)
        C = torch.eye(k, dtype=X.dtype, device=X.device) + self.FT_Dinv_F(dinv)
        Lc = torch.linalg.cholesky(C)              # capacitance Cholesky
        logdet = (2.0 * self.log_d).sum() + 2.0 * torch.log(torch.diagonal(Lc)).sum()
        # quad = r^T Sigma^{-1} r  (Woodbury)
        DiR = R * dinv                              # (B, n)
        FtDiR = self.FT_apply(DiR)                  # (B, k)
        sol = torch.cholesky_solve(FtDiR.T, Lc).T   # (B, k) = C^{-1} F^T Dinv r
        quad = (R * DiR).sum(1) - (FtDiR * sol).sum(1)
        return -0.5 * (n * LOG2PI + logdet + quad)  # (B,)

    # --- posterior-mean MAP denoising: y = x + N(0, s2 I) ---
    @torch.no_grad()
    def map_denoise(self, Y, s2):
        """x_hat = m + Sigma (Sigma + s2 I)^{-1} (y - m), via Woodbury on
        Sigma + s2 I = F F^T + diag(d2 + s2)."""
        d2 = torch.exp(2.0 * self.log_d) + s2
        dinv = 1.0 / d2
        R = Y - self.mean
        C = torch.eye(self.k, dtype=Y.dtype, device=Y.device) + self.FT_Dinv_F(dinv)
        Lc = torch.linalg.cholesky(C)
        DiR = R * dinv
        w1 = DiR - dinv * self.F_apply(torch.cholesky_solve(
            self.FT_apply(DiR).T, Lc).T)             # (Sigma+s2 I)^{-1} (y-m)
        # x_hat = m + Sigma w1 ,  Sigma w1 = F(F^T w1) + d2_orig * w1
        d2o = torch.exp(2.0 * self.log_d)
        Sw = self.F_apply(self.FT_apply(w1)) + d2o * w1
        return self.mean + Sw


# =============================================================================
# PPCA (closed form): top-r eigenvectors of the empirical covariance
# =============================================================================
class PPCAModel:
    def __init__(self, X, r):
        self.mean = X.mean(0)
        Xc = X - self.mean
        N = Xc.shape[0]
        # SVD trick: eigen of Xc^T Xc /(N-1) via the N x N Gram
        G = (Xc @ Xc.T) / (N - 1)
        evals, evecs = torch.linalg.eigh(G)          # ascending
        evals = evals.flip(0)[:r].clamp(min=1e-8)
        evecs = evecs.flip(1)[:, :r]
        V = Xc.T @ evecs                              # (n, r) unnormalised
        V = V / V.norm(dim=0, keepdim=True)
        self.V, self.lam = V, evals                   # columns, eigenvalues
        # residual (isotropic) noise = mean of the discarded variance
        total_var = (Xc * Xc).sum() / (N - 1)
        self.sigma2 = ((total_var - evals.sum()) / (X.shape[1] - r)).clamp(min=1e-6)

    def to(self, device):
        self.mean = self.mean.to(device); self.V = self.V.to(device)
        self.lam = self.lam.to(device); self.sigma2 = self.sigma2.to(device)
        return self

    def _woodbury(self, R, extra=0.0):
        s2 = self.sigma2 + extra
        lam = self.lam
        # Sigma = V diag(lam) V^T + s2 I  ->  Sigma^{-1} via Woodbury on V
        VtR = self.V.T @ R.T                          # (r, B)
        coef = (1.0 / s2) - (1.0 / s2) * (lam / (lam + s2))  # diag scaling per comp
        SinvR = R / s2 - ((self.V * ( (lam/(lam+s2))/s2 ).unsqueeze(0)) @ VtR).T
        return SinvR

    def loglik(self, X):
        R = X - self.mean
        n = X.shape[1]
        s2 = self.sigma2
        logdet = (n - self.lam.numel()) * torch.log(s2) + torch.log(self.lam + s2).sum()
        SinvR = self._woodbury(R)
        quad = (R * SinvR).sum(1)
        return -0.5 * (n * LOG2PI + logdet + quad)

    @torch.no_grad()
    def map_denoise(self, Y, s2n):
        R = Y - self.mean
        lam = self.lam
        s2 = self.sigma2
        # (Sigma + s2n I)^{-1}(y-m) then multiply by Sigma
        w1 = self._woodbury(R, extra=s2n)
        Vtw = self.V.T @ w1.T
        Sw = (self.V * lam.unsqueeze(0)) @ Vtw
        Sw = Sw.T + s2 * w1
        return self.mean + Sw


# =============================================================================
# Kronecker / tensor-normal:  Sigma = Sigma_1 (x) Sigma_2 (x) Sigma_3
# =============================================================================
class KroneckerModel:
    """Tensor-normal over an image tensor of shape dims=(d1,d2,d3) (C,H,W).
    Fitted by flip-flop MLE; LL and MAP via per-mode eigendecompositions."""

    def __init__(self, X, dims, iters=5, ridge=1e-4):
        self.dims = dims
        self.n = int(np.prod(dims))
        self.mean = X.mean(0)
        Xc = (X - self.mean).reshape(-1, *dims)       # (N, d1, d2, d3)
        N = Xc.shape[0]
        dev, dt = X.device, X.dtype
        S = [torch.eye(d, device=dev, dtype=dt) for d in dims]

        def mode_unfold(T, mode):
            # move `mode` axis (offset by 1 for batch) to front of feature axes
            perm = [0, mode + 1] + [i + 1 for i in range(len(dims)) if i != mode]
            return T.permute(*perm).reshape(N, dims[mode], -1)  # (N, d_mode, rest)

        for _ in range(iters):
            for mode in range(len(dims)):
                # whiten all other modes with their current Sigma^{-1}
                W = Xc
                for j in range(len(dims)):
                    if j == mode:
                        continue
                    Wj = torch.linalg.inv(S[j])
                    W = torch.tensordot(W, Wj, dims=([j + 1], [0]))
                    # tensordot moved contracted axis to the end; move it back
                    order = list(range(W.dim()))
                    order.insert(j + 1, order.pop(-1))
                    W = W.permute(*order).contiguous()
                U = mode_unfold(Xc, mode)              # (N, d, rest)
                Wu = mode_unfold(W, mode)              # (N, d, rest)
                rest = U.shape[2]
                Smode = torch.einsum("ndr,ner->de", U, Wu) / (N * rest)
                Smode = 0.5 * (Smode + Smode.T) + ridge * torch.eye(
                    dims[mode], device=dev, dtype=dt)
                S[mode] = Smode
        # normalize scale ambiguity: fix modes 1.. to unit mean-diagonal, keep total in S[0]
        for j in range(1, len(dims)):
            c = torch.diagonal(S[j]).mean()
            S[j] = S[j] / c
            S[0] = S[0] * c
        self.S = S
        self.eig = [torch.linalg.eigh(s) for s in S]  # (evals, evecs) per mode

    def to(self, device):
        self.mean = self.mean.to(device)
        self.S = [s.to(device) for s in self.S]
        self.eig = [(e.to(device), Q.to(device)) for (e, Q) in self.eig]
        return self

    def _to_eigen(self, R):
        # R: (B, n) -> tensor, apply Q_m^T along each mode
        B = R.shape[0]
        T = R.reshape(B, *self.dims)
        for m, (_, Q) in enumerate(self.eig):
            T = torch.tensordot(T, Q, dims=([m + 1], [0]))
            order = list(range(T.dim())); order.insert(m + 1, order.pop(-1))
            T = T.permute(*order).contiguous()
        return T  # (B, d1, d2, d3) in eigenbasis

    def _from_eigen(self, T):
        for m, (_, Q) in enumerate(self.eig):
            T = torch.tensordot(T, Q, dims=([m + 1], [1]))  # Q (not Q^T)
            order = list(range(T.dim())); order.insert(m + 1, order.pop(-1))
            T = T.permute(*order).contiguous()
        return T.reshape(T.shape[0], -1)

    def _lam(self):  # outer product of per-mode eigenvalues -> (d1,d2,d3)
        ev = [e[0] for e in self.eig]
        lam = ev[0]
        for e in ev[1:]:
            lam = lam.unsqueeze(-1) * e
        return lam

    def loglik(self, X):
        R = X - self.mean
        lam = self._lam().reshape(-1)                 # (n,)
        Te = self._to_eigen(R).reshape(R.shape[0], -1)
        quad = (Te * Te / lam.unsqueeze(0)).sum(1)
        logdet = torch.log(lam).sum()
        n = X.shape[1]
        return -0.5 * (n * LOG2PI + logdet + quad)

    @torch.no_grad()
    def map_denoise(self, Y, s2n):
        R = Y - self.mean
        lam = self._lam().reshape(-1)
        Te = self._to_eigen(R).reshape(R.shape[0], -1)
        gain = lam / (lam + s2n)                       # Wiener gain in eigenbasis
        Xe = Te * gain.unsqueeze(0)
        return self.mean + self._from_eigen(Xe.reshape(-1, *self.dims))


# =============================================================================
# FFT-circulant / stationary Gaussian: the canonical FIXED-geometry competitor
# =============================================================================
class CirculantModel:
    """Stationary (block-circulant) Gaussian: per-channel 2D-circulant covariance
    diagonalized by the 2D DFT.  Sigma = U diag(psd) U^H with U the unitary 2D
    DFT and psd the power spectrum (O(n) parameters, O(n log n) compute).  This
    is translation-invariant on a *prescribed* pixel ordering — the fixed-FFT
    foil to LAPLEX's *learned* geometry.  LL and MAP are exact and diagonal in
    the Fourier domain (MAP = Wiener filter)."""

    def __init__(self, X, dims, eps=1e-6, chunk=64):
        self.dims = dims
        C, H, W = dims
        self.mean = X.mean(0)
        psd = torch.zeros(C, H, W, dtype=X.dtype, device=X.device)
        for s in range(0, X.shape[0], chunk):           # chunked: DIV2K-size FFTs
            Xc = (X[s:s + chunk] - self.mean).reshape(-1, C, H, W)
            Fr = torch.fft.fft2(Xc, norm="ortho")       # (N,C,H,W) complex, unitary
            psd += (Fr.abs() ** 2).sum(0)
        self.psd = psd / X.shape[0] + eps               # (C,H,W) real eigenvalues

    def to(self, device):
        self.mean = self.mean.to(device); self.psd = self.psd.to(device); return self

    def loglik(self, X):
        C, H, W = self.dims
        R = (X - self.mean).reshape(-1, C, H, W)
        Fr = torch.fft.fft2(R, norm="ortho")
        quad = ((Fr.abs() ** 2) / self.psd).sum((1, 2, 3))
        logdet = torch.log(self.psd).sum()
        return -0.5 * (X.shape[1] * LOG2PI + logdet + quad)

    @torch.no_grad()
    def map_denoise(self, Y, s2):
        C, H, W = self.dims
        R = (Y - self.mean).reshape(-1, C, H, W)
        Fr = torch.fft.fft2(R, norm="ortho")
        xe = torch.fft.ifft2(Fr * (self.psd / (self.psd + s2)), norm="ortho").real
        return self.mean + xe.reshape(Y.shape[0], -1)


# =============================================================================
# Budget-matched circulant family: "Circulant (K)" of Tab. 9
# =============================================================================
class CirculantMixModel(nn.Module):
    """A single Gaussian whose precision factor is a product of K
    (per-pixel diagonal x per-channel 2D circulant) stages:

        z = L (x - m),   L = D_K C_K ... D_1 C_1,   Sigma = (L^T L)^{-1},

    C_j = U^H diag(lambda_j) U is a symmetric positive-definite circulant with a
    real, Hermitian-consistent spectrum stored on the rfft2 half-plane; D_j > 0
    is diagonal.  K is a covariance-capacity knob (not mixture components):
    each stage adds n + C*H*(W//2+1) parameters, which is how the baseline is
    budget-matched to LAPLEX.  The mean is the frozen empirical mean.  The LL is
    exact: log|det L| = sum_j (sum log D_j + sum over the full grid log lambda_j).
    Stage 1 starts at the closed-form stationary whitening (lambda = psd^{-1/2}),
    the others at identity, so training starts from the fitted FFT-circulant."""

    def __init__(self, mean, dims, K, psd_init=None, dtype=torch.float64):
        super().__init__()
        C, H, W = dims
        Wh = W // 2 + 1
        self.dims, self.K = dims, K
        self.register_buffer("mean", mean.to(dtype).clone())
        self.log_D = nn.Parameter(torch.zeros(K, C, H, W, dtype=dtype))
        log_lam = torch.zeros(K, C, H, Wh, dtype=dtype)
        if psd_init is not None:
            log_lam[0] = -0.5 * torch.log(psd_init[..., :Wh].to(dtype).cpu())
        self.log_lam = nn.Parameter(log_lam)
        # self-conjugate rfft columns (v = 0, and v = W/2 for even W) must be
        # symmetric under u -> -u and appear once on the full grid; others twice
        self.self_conj = [0] + ([W // 2] if W % 2 == 0 else [])
        mult = torch.full((Wh,), 2.0, dtype=dtype)
        mult[self.self_conj] = 1.0
        self.register_buffer("mult", mult)
        self.register_buffer("uflip", (-torch.arange(H)) % H)

    def _log_spec(self):
        c = self.self_conj
        ls = self.log_lam.clone()
        ls[..., c] = 0.5 * (self.log_lam[..., c] + self.log_lam[..., self.uflip, :][..., c])
        return ls                                        # (K, C, H, Wh)

    def whiten(self, R):  # R: (B, n) -> L R: (B, n)
        C, H, W = self.dims
        z = R.reshape(-1, C, H, W)
        lam = torch.exp(self._log_spec())
        D = torch.exp(self.log_D)
        for j in range(self.K):
            z = D[j] * torch.fft.irfft2(lam[j] * torch.fft.rfft2(z), s=(H, W))
        return z.reshape(R.shape[0], -1)

    def logdet_L(self):
        return self.log_D.sum() + (self._log_spec() * self.mult).sum()

    def loglik(self, X):
        z = self.whiten(X - self.mean)
        return -0.5 * (X.shape[1] * LOG2PI + (z * z).sum(1)) + self.logdet_L()


# =============================================================================
# Diagonal Gaussian: the independent-pixel floor
# =============================================================================
class DiagonalModel:
    """Sigma = diag(var), closed form (empirical per-pixel mean and variance)."""

    def __init__(self, X, eps=1e-6):
        self.mean = X.mean(0)
        self.var = X.var(0) + eps

    def to(self, device):
        self.mean = self.mean.to(device); self.var = self.var.to(device); return self

    def loglik(self, X):
        R = X - self.mean
        quad = (R * R / self.var).sum(1)
        return -0.5 * (X.shape[1] * LOG2PI + torch.log(self.var).sum() + quad)

    @torch.no_grad()
    def map_denoise(self, Y, s2):
        return self.mean + (Y - self.mean) * (self.var / (self.var + s2))


# =============================================================================
# Self-test: FA (laplex/dense), PPCA, Kronecker, Circulant vs dense Gaussian
# =============================================================================
def _dense_gaussian_ll(X, mean, Sigma):
    n = X.shape[1]
    R = (X - mean).double()
    Ls = torch.linalg.cholesky(Sigma.double())
    logdet = 2.0 * torch.log(torch.diagonal(Ls)).sum()
    sol = torch.cholesky_solve(R.T, Ls).T
    quad = (R * sol).sum(1)
    return -0.5 * (n * LOG2PI + logdet + quad)


def _dense_map(Y, mean, Sigma, s2):
    n = Y.shape[1]
    R = (Y - mean).double()
    A = Sigma.double() + s2 * torch.eye(n, dtype=torch.float64, device=Y.device)
    sol = torch.linalg.solve(A, R.T).T
    return mean + (Sigma.double() @ sol.T).T


def run_selftest(device):
    torch.manual_seed(0)
    dt = torch.float64
    print("SELF-TEST (rel-errors should be ~1e-6 or smaller)\n" + "-" * 50)

    # ---- FA models: build one, read its dense Sigma, compare LL + MAP ----
    n, k, B = 40, 6, 16
    X = torch.randn(B, n, dtype=dt, device=device)
    for kind in ("laplex", "dense"):
        fa = FAModel(n, k, kind, X.mean(0), dtype=dt).to(device)
        with torch.no_grad():
            # explicit F, Sigma for reference
            if kind == "dense":
                Fexp = fa.Fdense
            else:
                Kab = torch.exp(-(fa.a[:, None] - fa.b[None, :]).abs())
                Fexp = fa.w[:, None] * (Kab @ torch.tril(fa.L))
            Sigma = Fexp @ Fexp.T + torch.diag(torch.exp(2 * fa.log_d))
            ll = fa.loglik(X); ll_ref = _dense_gaussian_ll(X, fa.mean, Sigma)
            s2 = 0.3
            mp = fa.map_denoise(X, s2); mp_ref = _dense_map(X, fa.mean, Sigma, s2)
            # also check F/F^T adjointness
            z = torch.randn(3, k, dtype=dt, device=device)
            r = torch.randn(3, n, dtype=dt, device=device)
            adj = (fa.F_apply(z) * r).sum() - (z * fa.FT_apply(r)).sum()
        e_ll = (ll - ll_ref).abs().max().item() / ll_ref.abs().max().item()
        e_mp = (mp - mp_ref).abs().max().item() / mp_ref.abs().max().item()
        print(f"FA[{kind:6s}]  LL rel-err {e_ll:.2e} | MAP rel-err {e_mp:.2e} "
              f"| <Fz,r>-<z,F^Tr> {adj.abs().item():.2e}")

    # ---- PPCA ----
    Xp = torch.randn(200, n, dtype=dt, device=device)
    pp = PPCAModel(Xp, r=5)
    Sigma = pp.V @ torch.diag(pp.lam) @ pp.V.T + pp.sigma2 * torch.eye(n, dtype=dt, device=device)
    e_ll = (pp.loglik(X) - _dense_gaussian_ll(X, pp.mean, Sigma)).abs().max().item() \
        / _dense_gaussian_ll(X, pp.mean, Sigma).abs().max().item()
    e_mp = (pp.map_denoise(X, 0.3) - _dense_map(X, pp.mean, Sigma, 0.3)).abs().max().item() \
        / _dense_map(X, pp.mean, Sigma, 0.3).abs().max().item()
    print(f"PPCA         LL rel-err {e_ll:.2e} | MAP rel-err {e_mp:.2e}")

    # ---- Kronecker ----
    dims = (2, 4, 5); nk = 40
    Xk = torch.randn(300, nk, dtype=dt, device=device)
    kr = KroneckerModel(Xk, dims, iters=8)
    Sig = kr.S[0]
    for j in range(1, 3):
        Sig = torch.kron(Sig, kr.S[j])
    e_ll = (kr.loglik(X) - _dense_gaussian_ll(X, kr.mean, Sig)).abs().max().item() \
        / _dense_gaussian_ll(X, kr.mean, Sig).abs().max().item()
    e_mp = (kr.map_denoise(X, 0.3) - _dense_map(X, kr.mean, Sig, 0.3)).abs().max().item() \
        / _dense_map(X, kr.mean, Sig, 0.3).abs().max().item()
    print(f"Kronecker    LL rel-err {e_ll:.2e} | MAP rel-err {e_mp:.2e}")

    # ---- Circulant (single channel so the dense DFT reference is simple) ----
    dims_c = (1, 5, 8)                                   # n = 40, matches X
    Xc_data = torch.randn(300, n, dtype=dt, device=device)
    cm = CirculantModel(Xc_data, dims_c)

    def _dft1(N):
        jj = torch.arange(N, device=device)
        return (torch.exp(-2j * math.pi * torch.outer(jj, jj) / N)
                / math.sqrt(N)).to(torch.complex128)
    U = torch.kron(_dft1(5), _dft1(8))                  # unitary 2D DFT (40x40)
    Sigc = (U @ torch.diag(cm.psd.reshape(-1).to(torch.complex128))
            @ U.conj().T).real
    e_ll = (cm.loglik(X) - _dense_gaussian_ll(X, cm.mean, Sigc)).abs().max().item() \
        / _dense_gaussian_ll(X, cm.mean, Sigc).abs().max().item()
    e_mp = (cm.map_denoise(X, 0.3) - _dense_map(X, cm.mean, Sigc, 0.3)).abs().max().item() \
        / _dense_map(X, cm.mean, Sigc, 0.3).abs().max().item()
    print(f"Circulant    LL rel-err {e_ll:.2e} | MAP rel-err {e_mp:.2e}")

    # ---- Circulant mix: dense L from the whitening map, even and odd W ----
    for dims_m in ((2, 4, 5), (1, 5, 8)):                # n = 40, matches X
        cmx = CirculantMixModel(X.mean(0), dims_m, K=3).to(device)
        with torch.no_grad():
            cmx.log_D.normal_(0, 0.3); cmx.log_lam.normal_(0, 0.5)
            Lmat = cmx.whiten(torch.eye(n, dtype=dt, device=device)).T   # columns = L e_i
            Sigm = torch.linalg.inv(Lmat.T @ Lmat)
            e_ll = (cmx.loglik(X) - _dense_gaussian_ll(X, cmx.mean, Sigm)).abs().max().item() \
                / _dense_gaussian_ll(X, cmx.mean, Sigm).abs().max().item()
            # each circulant stage must be a symmetric operator (real spectrum)
            cmx.log_D.zero_()
            C1 = cmx.whiten(torch.eye(n, dtype=dt, device=device)).T
            e_sym = (C1 - C1.T).abs().max().item() / C1.abs().max().item()
        print(f"CircMix{dims_m} LL rel-err {e_ll:.2e} | circulant asym {e_sym:.2e}")

    # ---- Diagonal ----
    dg = DiagonalModel(Xc_data)
    Sigd = torch.diag(dg.var)
    e_ll = (dg.loglik(X) - _dense_gaussian_ll(X, dg.mean, Sigd)).abs().max().item() \
        / _dense_gaussian_ll(X, dg.mean, Sigd).abs().max().item()
    e_mp = (dg.map_denoise(X, 0.3) - _dense_map(X, dg.mean, Sigd, 0.3)).abs().max().item() \
        / _dense_map(X, dg.mean, Sigd, 0.3).abs().max().item()
    print(f"Diagonal     LL rel-err {e_ll:.2e} | MAP rel-err {e_mp:.2e}")
    print("-" * 50 + "\nIf all rel-errors are small, the math is correct.")


if __name__ == "__main__":
    # The full experiment is run by table09_14_density.py.
    run_selftest(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
