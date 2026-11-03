"""Stationary (BCCB / block-circulant-with-circulant-blocks) Gaussian baselines.

The strong fixed-geometry competitor for the LAPLEX density experiment: a
translation-invariant Gaussian on 3-channel images, diagonalized exactly by the
per-channel 2D DFT.  O(n log n) log-likelihood, no n x n matrix ever formed.

Two variants:
  * "scalar" : Sigma = C_color (x) C_spatial + sigma^2 I
               C_spatial diagonalized by 2D DFT, eigenvalues lambda_k >= 0;
               C_color = Lc Lc^T + eps I3 (shared 3x3 color shape across freq).
  * "full"   : arbitrary 3x3 Hermitian PSD block B_k per spatial frequency
               = the *most general* spatially-stationary Gaussian (the steelman).

Interface mirrors `_model_dense.GaussianDenseLowRankWoodbury`:
  .log_likelihood(x)  -> per-example LL in nats (full Gaussian, summed over dims)
  .sample(N)          -> (N, n) real samples
  .tau_L()/.tau_R()   -> NaN placeholders (runner uniformity)

All heavy linear algebra (FFT, per-freq 3x3 Cholesky / solve / logdet) runs in
fp64 / complex128, matching the paper's fp64 choice for Gram + capacitance.

Data layout: flat index = c*H*W + h*W + w  (channel-major, from `x.view(-1)` on a
(3,H,W) tensor), so Sigma = C_color (x) C_spatial with color as the outer factor.

Realness of Sigma is guaranteed by parameterizing on the rfft2 half-spectrum with
Hermitian multiplicities mult_k in {1,2}; imaginary parts are zeroed at the (<=4)
self-conjugate frequencies.  Verified against a dense reference in test_bccb.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Frequency-grid bookkeeping (rfft2 half-spectrum)                            #
# --------------------------------------------------------------------------- #
def mult_weights(H: int, W: int) -> torch.Tensor:
    """Hermitian multiplicity for each rfft2 coefficient (H, W//2+1).

    Full-spectrum sum  =  sum over half-spectrum weighted by these.
    Columns v=0 and v=W/2 (W even) are self-paired in v -> weight 1 (every entry
    counted once; conjugate partners u<->-u are both physically present).  All
    other columns -> weight 2 (their conjugate lives in the dropped half).
    """
    Wh = W // 2 + 1
    mult = torch.full((H, Wh), 2.0, dtype=torch.float64)
    mult[:, 0] = 1.0
    if W % 2 == 0:
        mult[:, W // 2] = 1.0
    return mult


def self_conj_mask(H: int, W: int) -> torch.Tensor:
    """Boolean (H, W//2+1): frequencies that are their own conjugate.

    These must carry a *real* block B_k for Sigma to be real; we zero the
    imaginary part of L_k there.  For H,W even there are exactly 4:
    (0,0), (0,W/2), (H/2,0), (H/2,W/2).
    """
    Wh = W // 2 + 1
    sc = torch.zeros(H, Wh, dtype=torch.bool)
    u_special = [0] + ([H // 2] if H % 2 == 0 else [])
    v_special = [0] + ([W // 2] if W % 2 == 0 else [])
    for u in u_special:
        for v in v_special:
            sc[u, v] = True
    return sc


# --------------------------------------------------------------------------- #
#  Model                                                                        #
# --------------------------------------------------------------------------- #
class GaussianBCCB(nn.Module):
    def __init__(
        self,
        H: int,
        W: int,
        model_type: str = "full",
        channels: int = 3,
        eps: float = 1e-4,
        mean_init: torch.Tensor | None = None,
        init_var: float = 0.05,
    ):
        super().__init__()
        assert model_type in ("scalar", "full")
        self.H, self.W, self.C = H, W, channels
        self.Wh = W // 2 + 1
        self.n = channels * H * W
        self.model_type = model_type
        self.eps = eps

        self.register_buffer("mult", mult_weights(H, W))            # (H,Wh) f64
        self.register_buffer("scmask", self_conj_mask(H, W))        # (H,Wh) bool
        self.register_buffer("I3", torch.eye(channels, dtype=torch.complex128))

        if mean_init is not None:
            self.register_buffer("m", mean_init.detach().to(torch.float64).clone())
        else:
            self.m = nn.Parameter(torch.zeros(self.n, dtype=torch.float64))

        # softplus^{-1}(init_var) so the model starts near a white Gaussian of the
        # right pixel-variance scale.
        inv_sp = math.log(math.expm1(init_var))

        if model_type == "scalar":
            # lambda_k = softplus(theta); start small so sigma^2 carries the variance.
            self.theta = nn.Parameter(torch.full((H, self.Wh), -4.0, dtype=torch.float64))
            self.Lc_off = nn.Parameter(torch.zeros(channels, channels, dtype=torch.float64))
            self.Lc_diag_raw = nn.Parameter(torch.full((channels,), inv_sp, dtype=torch.float64))
            self.sigma_raw = nn.Parameter(torch.tensor(inv_sp, dtype=torch.float64))
        else:  # full
            self.diag_raw = nn.Parameter(
                torch.full((H, self.Wh, channels), inv_sp, dtype=torch.float64)
            )
            n_off = channels * (channels - 1) // 2
            self.off_re = nn.Parameter(torch.zeros(H, self.Wh, n_off, dtype=torch.float64))
            self.off_im = nn.Parameter(torch.zeros(H, self.Wh, n_off, dtype=torch.float64))

    # -- block construction (H, Wh, 3, 3) complex128 Hermitian PD ------------ #
    def _symmetrize_special(self, B: torch.Tensor) -> torch.Tensor:
        """Enforce Sigma real: in self-paired columns v in {0, W/2}, the blocks at
        u and (-u) mod H must be conjugate (their conjugate partner lives in the
        same half-spectrum column).  Averaging B[u] with conj(B[-u]) enforces this
        and, at the 4 self-conjugate frequencies, forces a real block.  Non-special
        columns are unconstrained (their conjugate lives in the dropped half).
        """
        H, W = self.H, self.W
        flip = (-torch.arange(H, device=B.device)) % H
        B = B.clone()
        for v in [0] + ([W // 2] if W % 2 == 0 else []):
            col = B[:, v]                                   # (H,C,C)
            B[:, v] = 0.5 * (col + col[flip].conj())
        return B

    def _blocks(self) -> torch.Tensor:
        H, Wh, C = self.H, self.Wh, self.C
        if self.model_type == "scalar":
            lam = F.softplus(self.theta) + self.eps                 # (H,Wh)
            Lc = torch.tril(self.Lc_off, -1) + torch.diag(
                F.softplus(self.Lc_diag_raw) + self.eps
            )
            Ccol = Lc @ Lc.T + self.eps * torch.eye(C, dtype=torch.float64, device=Lc.device)
            sig2 = F.softplus(self.sigma_raw) + self.eps
            B = lam[..., None, None] * Ccol + sig2 * torch.eye(
                C, dtype=torch.float64, device=lam.device
            )
            return self._symmetrize_special(B.to(torch.complex128))
        else:  # full
            diag = F.softplus(self.diag_raw) + self.eps            # (H,Wh,3) real>0
            L = torch.zeros(H, Wh, C, C, dtype=torch.complex128, device=diag.device)
            idx = 0
            for i in range(C):
                L[..., i, i] = diag[..., i].to(torch.complex128)
                for j in range(i):
                    L[..., i, j] = self.off_re[..., idx] + 1j * self.off_im[..., idx]
                    idx += 1
            B = L @ L.conj().transpose(-1, -2) + self.eps * self.I3
            return self._symmetrize_special(B)

    # -- per-example log-likelihood (nats) ---------------------------------- #
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(torch.float64)
        Bsz = x.shape[0]
        H, W, Wh, C = self.H, self.W, self.Wh, self.C
        Nf = H * Wh

        diff = x - self.m
        img = diff.view(Bsz, C, H, W)
        xf = torch.fft.rfft2(img, norm="ortho")                   # (B,C,H,Wh) c128
        xt = xf.permute(0, 2, 3, 1).reshape(Bsz, Nf, C)           # (B,Nf,C)

        B_blocks = self._blocks().reshape(Nf, C, C)               # (Nf,C,C)
        Lb = torch.linalg.cholesky(B_blocks)                      # (Nf,C,C)

        rhs = xt.permute(1, 2, 0)                                 # (Nf,C,B)
        sol = torch.cholesky_solve(rhs, Lb)                       # (Nf,C,B)
        quad_f = (rhs.conj() * sol).sum(dim=1).real               # (Nf,B)
        mult = self.mult.reshape(Nf, 1)
        quad = (mult * quad_f).sum(dim=0)                         # (B,)

        logdet = self._logdet()

        const = self.n * math.log(2.0 * math.pi)
        return -0.5 * (quad + logdet + const)

    def _logdet(self) -> torch.Tensor:
        """log det Sigma = sum_half mult_k * logdet(B_k)  (real scalar)."""
        Nf = self.H * self.Wh
        Lb = torch.linalg.cholesky(self._blocks().reshape(Nf, self.C, self.C))
        logdiag = torch.log(torch.diagonal(Lb, dim1=-2, dim2=-1).real)   # (Nf,C)
        logdet_f = 2.0 * logdiag.sum(dim=-1)                             # (Nf,)
        return (self.mult.reshape(Nf) * logdet_f).sum()

    # -- coloring map  x = irfft( L_k * rfft(e) )  (used by sample + tests) -- #
    def _color(self, e: torch.Tensor) -> torch.Tensor:
        """e: (N, C, H, W) real -> (N, C, H, W) real with cov S S^T = Sigma."""
        N, C, H, W = e.shape
        Nf = H * self.Wh
        ef = torch.fft.rfft2(e.to(torch.float64), norm="ortho")   # (N,C,H,Wh)
        et = ef.permute(0, 2, 3, 1).reshape(N, Nf, C)             # (N,Nf,C)
        Lb = torch.linalg.cholesky(self._blocks().reshape(Nf, C, C))
        # s_k = L_k e_k
        s = torch.einsum("fij,nfj->nfi", Lb, et)                  # (N,Nf,C)
        s = s.reshape(N, H, self.Wh, C).permute(0, 3, 1, 2)       # (N,C,H,Wh)
        return torch.fft.irfft2(s, s=(H, W), norm="ortho")        # (N,C,H,W)

    def sample(self, num_samples: int) -> torch.Tensor:
        dev = self.m.device
        e = torch.randn(num_samples, self.C, self.H, self.W, dtype=torch.float64, device=dev)
        x = self._color(e).reshape(num_samples, self.n)
        return self.m + x

    # -- placeholders for runner uniformity --------------------------------- #
    def tau_L(self):
        return torch.tensor(float("nan"))

    def tau_R(self):
        return torch.tensor(float("nan"))


class GaussianBCCBMixture(nn.Module):
    """Mixture of K stationary BCCB Gaussians (each its own learned mean + blocks).

    p(x) = sum_k softmax(logits)_k * N(x; m_k, Sigma_k),  Sigma_k a BCCB covariance.

    Purpose: a *parameter-matched, strictly more expressive* stationary
    baseline than a single BCCB.  If even this cannot recover structure on permuted
    (no-grid) data, the point that stationary models are grid-dependent is airtight.
    """

    def __init__(self, H, W, K=3, model_type="full", channels=3, eps=1e-4,
                 mean_init=None, param_jitter=0.05):
        super().__init__()
        self.K = K
        # Same mean treatment as LAPLEX: ONE shared frozen Welford mean (a buffer, not
        # trainable), so the whole trainable budget goes into covariance.  The mixture
        # is over K BCCB covariances only -> trainable params = K * blocks.  DIV2K:
        # 3 x 4.73M ~= 14.2M ~= LAPLEX 13.6M.  Each component gets the same frozen mean.
        self.components = nn.ModuleList([
            GaussianBCCB(H, W, model_type=model_type, channels=channels,
                         eps=eps, mean_init=mean_init)
            for _ in range(K)
        ])
        # Means are shared+frozen, so break symmetry in the covariance parameters so
        # the components specialize instead of staying identical.
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for comp in self.components:
                for p in comp.parameters():          # trainable block params only (m is a buffer)
                    p.add_(param_jitter * torch.randn(p.shape, generator=g, dtype=p.dtype))
        self.logits = nn.Parameter(torch.zeros(K, dtype=torch.float64))

    def log_likelihood(self, x):
        comp_ll = torch.stack([c.log_likelihood(x) for c in self.components], dim=0)  # (K,B)
        logw = torch.log_softmax(self.logits, dim=0).unsqueeze(1)                     # (K,1)
        return torch.logsumexp(comp_ll + logw, dim=0)                                 # (B,)

    def sample(self, num_samples):
        w = torch.softmax(self.logits, dim=0)
        idx = torch.multinomial(w, num_samples, replacement=True)
        return torch.cat([self.components[i].sample(1) for i in idx.tolist()], dim=0)

    def tau_L(self):
        return torch.tensor(float("nan"))

    def tau_R(self):
        return torch.tensor(float("nan"))
