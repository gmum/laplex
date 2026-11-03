"""
laplex_complex.py
=================

Complex two-temperature Laplex kernel, soft-only.

Operator:

    K[l, k, j] = exp( -|a[l,k] - b[l,j]| / tau_sign
                      + i (c[l,k] - d[l,j]) / s[l] )

    tau_sign   = tau_right if a[l,k] > b[l,j] else tau_left

The phase factorises as exp(i c_k/s) * exp(-i d_l/s), so the operator
is implemented as:

    1. pre-modulate:  x_k^c = exp(i c_k/s) * x_k                    (complex)
    2. apply magnitude kernel (column-softmaxed):
                     y_l^c  = sum_k softmax_k(-|a-b|/tau_sign) x_k^c
    3. de-modulate:  y_l    = exp(-i d_l/s) * y_l^c

Column-normalisation is over the magnitude only (the only real, positive
factor). The sum sum_k |K| is always real positive, so no numerical
cancellation from opposite phases.

The hard (tau -> 0) branch is intentionally not included here. In the
hard limit a single k* dominates and the output reduces to
X[k*] * exp(i(c_{k*} - d_l)/s), i.e. a gather followed by a per-position
phase twist. Re/Im then gives two channels with matching magnitudes but
different phases -- expressively the same as a real hard kernel with 2L
rows. Interference across k, which is the whole reason to go complex,
only lives in the soft branch. Use `laplex.py` if you need hard.

On CUDA with float32/complex64, `_magnitude.magnitude_apply` dispatches
to a fused O(L n) DL-scan implementation that runs both temperatures in
a single kernel launch. Complex inputs are handled by splitting into
real and imaginary halves: the magnitude kernel K is real, so
K @ (Re + i Im) = K@Re + i K@Im. Elsewhere it falls back to the dense
softmax+einsum reference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from _magnitude import magnitude_apply as _magnitude_apply


# ----------------------------------------------------------------------------
# Functional API
# ----------------------------------------------------------------------------

def laplex_complex(
    X: torch.Tensor,          # (B, L, n_a) real or complex
    a: torch.Tensor,          # (L, n_a)    real
    b: torch.Tensor,          # (L, n_b)    real
    c: torch.Tensor,          # (L, n_a)    real  phase-in  anchors (a-side)
    d: torch.Tensor,          # (L, n_b)    real  phase-out anchors (b-side)
    tau_left:  torch.Tensor,  # (L,)        positive
    tau_right: torch.Tensor,  # (L,)        positive
    s:         torch.Tensor,  # (L,)        positive  (phase temperature)
    normalize: bool = True,
) -> torch.Tensor:
    """Soft complex two-tau Laplex, LEFT-mul.

    Y[b, l, j] = Σ_k Kbar[l, k, j] X[b, l, k]
    K[l,k,j]    = exp(-|a[l,k]-b[l,j]|/tau_sign + i (c[l,k] - d[l,j]) / s[l])
    tau_sign    = tau_R if a[l,k] > b[l,j] else tau_L.

    X: (B, L, n_a) real or complex; returns (B, L, n_b) COMPLEX.
    """
    cdtype    = torch.complex64 if X.dtype in (torch.float32, torch.complex64) else torch.complex128
    phase_in  = torch.exp( 1j * (c / s.unsqueeze(1))).to(cdtype)    # (L, n_a)
    phase_out = torch.exp(-1j * (d / s.unsqueeze(1))).to(cdtype)    # (L, n_b)
    Xc = X.to(cdtype) * phase_in.unsqueeze(0)                       # (B, L, n_a)
    Yc = _magnitude_apply(Xc, a, b, tau_left, tau_right, normalize) # (B, L, n_b)
    return Yc * phase_out.unsqueeze(0)


def laplex_complex_right(
    X: torch.Tensor,          # (B, L, n_b) real or complex  (b-side input)
    a: torch.Tensor,          # (L, n_a)    real             (a-side, output)
    b: torch.Tensor,          # (L, n_b)    real             (b-side, sum-over)
    c: torch.Tensor,          # (L, n_a)    real  phase anchors on a-side
    d: torch.Tensor,          # (L, n_b)    real  phase anchors on b-side
    tau_left:  torch.Tensor,  # (L,)        positive
    tau_right: torch.Tensor,  # (L,)        positive
    s:         torch.Tensor,  # (L,)        positive
    normalize: bool = True,
) -> torch.Tensor:
    """Soft complex two-tau Laplex, RIGHT-mul.

    Y[b, l, k] = Σ_j Kbar[l, k, j] X[b, l, j]
    (same K as in `laplex_complex`; X now on b-side, output on a-side).

    Returns (B, L, n_a) COMPLEX.

    Realizacja (analog `laplex_right`):
      - pre-demodulate X przez exp(-i d/s)  (b-side)
      - apply magnitude kernel z zamienionymi anchorami i τ-ami:
            _magnitude_apply(X', b, a, tau_R, tau_L, normalize)
      - post-modulate przez exp(i c/s)  (a-side)
    Normalizacja działa jak w real `laplex_right`: Z[l,j] = Σ_k |K[l,k,j]|
    (tylko magnituda, zawsze realna dodatnia) i skaluje X przed apply.
    """
    cdtype    = torch.complex64 if X.dtype in (torch.float32, torch.complex64) else torch.complex128
    n_a       = a.shape[-1]
    B, L, n_b = X.shape

    # Phase factors.
    phase_c = torch.exp( 1j * (c / s.unsqueeze(1))).to(cdtype)      # (L, n_a)  a-side
    phase_d = torch.exp(-1j * (d / s.unsqueeze(1))).to(cdtype)      # (L, n_b)  b-side

    if normalize:
        # Z[l, j] = Σ_k exp(-|a[l,k] - b[l,j]|/tau_sign) -- magnitudes only, REAL.
        ones_X = torch.ones((1, L, n_a), device=X.device, dtype=a.dtype)
        Z = _magnitude_apply(ones_X, a, b, tau_left, tau_right, False)  # (1, L, n_b)
        Xw = X.to(cdtype) / Z
    else:
        Xw = X.to(cdtype)

    # Pre-demodulate and apply transposed magnitude kernel.
    Xd = Xw * phase_d.unsqueeze(0)                                   # (B, L, n_b)
    Ym = _magnitude_apply(Xd, b, a, tau_right, tau_left, False)      # (B, L, n_a)

    # Post-modulate.
    return Ym * phase_c.unsqueeze(0)


# ----------------------------------------------------------------------------
# Module API
# ----------------------------------------------------------------------------

class LaplexComplex(nn.Module):
    """L independent complex two-tau Laplex rows -- pure matrix multiplication.

    Parameters per row i:
        a_i, b_i : (n,)  magnitude anchors
        c_i, d_i : (n,)  phase anchors
        tau_left_i, tau_right_i : (,) positive magnitude temperatures
                  (tied if single_tau=True; only log_tau_left is a Parameter)
        s_i      : (,)  positive phase temperature

    Flags:
        normalize   -- column-softmax over k (True) or raw exp weights (False)
        single_tau  -- tie tau_right = tau_left

    The output is the complex (B, L, n) tensor viewed as 2L real channels
    by stacking Re and Im along the row axis:  (B, 2L, n).

    No per-feature scaling is applied here (no "v"). For training, a
    row-mixer placed after this layer can take arbitrary linear
    combinations of the 2L real channels (see LaplexComplexBlock).
    """

    def __init__(
        self,
        n: int,
        l: int,
        tau_init: float = 0.1,
        s_init: float = 0.5,
        normalize: bool = True,
        single_tau: bool = False,
    ):
        super().__init__()
        self.n, self.l = n, l
        self.normalize  = normalize
        self.single_tau = single_tau

        def _init_anchors():
            return torch.stack(
                [torch.sort(torch.empty(n).uniform_(-1, 1))[0]
                 for _ in range(l)])

        self.a = nn.Parameter(_init_anchors())
        self.b = nn.Parameter(_init_anchors())
        self.c = nn.Parameter(torch.empty(l, n).uniform_(-1, 1))
        self.d = nn.Parameter(torch.empty(l, n).uniform_(-1, 1))

        self.log_tau_left = nn.Parameter(torch.full((l,), math.log(tau_init)))
        if single_tau:
            self.log_tau_right = None
        else:
            self.log_tau_right = nn.Parameter(
                torch.full((l,), math.log(tau_init)))
        self.log_s = nn.Parameter(torch.full((l,), math.log(s_init)))

    def _params(self):
        tau_l = self.log_tau_left.exp()
        tau_r = tau_l if self.single_tau else self.log_tau_right.exp()
        s     = self.log_s.exp()
        return tau_l, tau_r, s

    @property
    def out_channels(self) -> int:
        """Row-axis size of the output: 2L (Re and Im stacked)."""
        return 2 * self.l

    def forward(
        self,
        X: torch.Tensor,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """X : (B, L, n) real  ->  (B, 2L, n) real  [Re | Im]."""
        norm = self.normalize if normalize is None else normalize
        tau_l, tau_r, s = self._params()
        Yc = laplex_complex(
            X, self.a, self.b, self.c, self.d,
            tau_l, tau_r, s, normalize=norm)
        return torch.cat([Yc.real, Yc.imag], dim=1)


class LaplexComplexBlock(nn.Module):
    """Full complex layer:
        X (B, l_in, n) -> LaplexComplex -> (B, 2*l_in, n) -> mix -> (B, l_out, n).

    The mixer M has shape (l_out, 2*l_in) and acts along the row axis only,
    leaving the n axis untouched.
    """

    def __init__(
        self,
        n: int,
        l_in: int,
        l_out: int | None = None,
        tau_init: float = 0.1,
        s_init: float = 0.5,
        normalize: bool = True,
        single_tau: bool = False,
    ):
        super().__init__()
        l_out = l_in if l_out is None else l_out
        self.kernel = LaplexComplex(
            n, l_in,
            tau_init=tau_init, s_init=s_init,
            normalize=normalize, single_tau=single_tau,
        )
        self.M = nn.Parameter(torch.empty(l_out, self.kernel.out_channels))
        nn.init.orthogonal_(self.M)

    def forward(
        self,
        X: torch.Tensor,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        Y = self.kernel(X, normalize=normalize)
        return torch.einsum("ab,Bbn->Ban", self.M, Y)


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, n = 2, 4, 16
    x = torch.randn(B, L, n)
    for single_tau in (False, True):
        for normalize in (True, False):
            blk = LaplexComplexBlock(n, L, L,
                                     single_tau=single_tau, normalize=normalize)
            y = blk(x)
            assert y.shape == (B, L, n)
            y.sum().backward()
            k = blk.kernel
            assert k.a.grad is not None and k.c.grad is not None
            assert k.log_tau_left.grad is not None and k.log_s.grad is not None
            print(f"single_tau={single_tau} normalize={normalize}  "
                  f"out={tuple(y.shape)}  OK")
