"""
laplex.py
=========

Real two-temperature Laplex kernel: column-normalised (or raw) Laplace
kernel with two independent decay rates for the left/right sides of
each anchor.

Operator (L independent rows of size n each; batch dim B):

    K[l, k, j] = exp( -|a[l,k] - b[l,j]| / tau_sign )
    tau_sign   = tau_right  if a[l,k] > b[l,j]
                 tau_left   otherwise

Column-normalisation (default) divides by sum_k K over the k axis:

    Kbar[l, k, j] = K[l, k, j] / Z[l, j],   Z[l, j] = sum_k K[l, k, j]

Four functional entry points:

    laplex(X, a, b, tau_L, tau_R, normalize=True)
        Y[b,l,j] = sum_k Kbar[l,k,j] X[b,l,k]
        (left-multiplication: X on the a-side, output on the b-side)

    laplex_hard(X, a, b, tau_L, tau_R)
        Y[b,l,j] = X[b,l, k*],   k* = argmin_k |a[l,k]-b[l,j]|/tau_sign
        (the tau -> 0 limit of the normalised soft kernel)

    laplex_right(X, a, b, tau_L, tau_R, normalize=True)
        Y[b,l,k] = sum_j Kbar[l,k,j] X[b,l,j]
        (right-multiplication: X on the b-side, output on the a-side)

    laplex_gram(a, b, D, tau_L, tau_R, normalize=False)
        G[l,i,i'] = sum_j D[l,j] K[l,i,j] K[l,i',j]
        (Laplace Gram matrix, K * diag(D) * K^T)

On CUDA with float32, `_magnitude.magnitude_apply` dispatches to a fused
decoupled-lookback scan (O(L n)); elsewhere the dense O(L n^2) reference
is used. `laplex_right` with normalize=True costs two kernel launches:
first the denominator Z[l,j], then the transposed application to X/Z.
`laplex_gram` uses the dense reference by default.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from _magnitude import magnitude_apply as _magnitude_apply


# ----------------------------------------------------------------------------
# Functional API -- soft and hard forward
# ----------------------------------------------------------------------------

def laplex(
    X: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_left:  torch.Tensor,
    tau_right: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Soft two-tau Laplex, left-multiplication: Y[b,l,j] = sum_k Kbar[l,k,j] X[b,l,k].

    X : (B, L, n_a)  a-side input
    a : (L, n_a)     a-side (sum-over) anchors
    b : (L, n_b)     b-side (output) anchors
    returns  (B, L, n_b).
    """
    return _magnitude_apply(X, a, b, tau_left, tau_right, normalize)


def laplex_broadcast(
    X: torch.Tensor,          # (B, n_a)  -- single row vector, shared across L
    a: torch.Tensor,          # (L, n_a)
    b: torch.Tensor,          # (L, n_b)
    tau_left:  torch.Tensor,  # (L,)
    tau_right: torch.Tensor,  # (L,)
    normalize: bool = True,
) -> torch.Tensor:
    """Apply L independent Laplex kernels to the SAME input vector.

    Equivalent to  laplex(X.unsqueeze(1).expand(B, L, n_a), a, b, ...)
    but written explicitly so the expansion stays a stride-0 view along L
    (no L-fold copy of X). On the CUDA scan path X is still materialised
    internally via `.contiguous()`, but the materialisation is explicit
    and happens only inside _magnitude_apply; the call site hands in a
    view, not a clone. On the dense fallback the stride-0 view survives
    all the way into einsum.

    X : (B, n_a)
    returns (B, L, n_b).
    """
    B, n_a = X.shape
    L = a.shape[0]
    X_bcast = X.unsqueeze(1).expand(B, L, n_a)
    return _magnitude_apply(X_bcast, a, b, tau_left, tau_right, normalize)


def laplex_hard(
    X: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_left:  torch.Tensor,
    tau_right: torch.Tensor,
) -> torch.Tensor:
    """Hard two-tau Laplex (tau -> 0 limit of the normalised soft kernel).

    For each (l, j), gather X at k* = argmin_k |a[l,k]-b[l,j]|/tau_sign.
    Chosen from the two `searchsorted` neighbours using the weighted
    distances d/tau_L (left) and d/tau_R (right).

    X : (B, L, n_a), a : (L, n_a), b : (L, n_b)  ->  (B, L, n_b).
    """
    B, L, n_a = X.shape
    n_b = b.shape[-1]
    a_sorted, perm = torch.sort(a, dim=1)                         # (L, n_a)
    idx_r = torch.searchsorted(a_sorted, b).clamp(0, n_a - 1)     # (L, n_b)
    idx_l = (idx_r - 1).clamp(0, n_a - 1)
    a_r = torch.gather(a_sorted, 1, idx_r)
    a_l = torch.gather(a_sorted, 1, idx_l)
    d_r = (a_r - b).abs() / tau_right.unsqueeze(1)
    d_l = (b - a_l).abs() / tau_left .unsqueeze(1)
    idx_sorted = torch.where(d_r < d_l, idx_r, idx_l)
    idx = torch.gather(perm, 1, idx_sorted)                       # (L, n_b)
    gather_idx = idx.unsqueeze(0).expand(B, L, n_b)
    return torch.gather(X, 2, gather_idx)


# ----------------------------------------------------------------------------
# Functional API -- phase-modulated kernel
# ----------------------------------------------------------------------------

def laplex_phase(
    X: torch.Tensor,          # (B, L, n_a)
    a: torch.Tensor,          # (L, n_a)
    b: torch.Tensor,          # (L, n_b)
    c: torch.Tensor,          # (L, n_a)  phase offsets on a-side
    d: torch.Tensor,          # (L, n_b)  phase offsets on b-side
    tau_left:  torch.Tensor,  # (L,)
    tau_right: torch.Tensor,  # (L,)
    normalize: bool = True,
) -> torch.Tensor:
    """Phase-modulated two-tau Laplex:

        K[l, k, j] = exp(-|a[l,k] - b[l,j]| / tau_sign) * cos(c[l,k] - d[l,j])

    Implemented via the identity
        cos(c_k - d_j) = cos(c_k) cos(d_j) + sin(c_k) sin(d_j)
    so the output reduces to two real magnitude-kernel applies:

        Y_c = magnitude_apply(X * cos(c), a, b, tau_L, tau_R, normalize)
        Y_s = magnitude_apply(X * sin(c), a, b, tau_L, tau_R, normalize)
        Y   = cos(d) * Y_c + sin(d) * Y_s

    Equivalent to taking the real part of a complex kernel
        K_z[l,k,j] = exp(-|a-b|/tau_sign) * exp(i(c_k - d_j))
    applied to X (complex-scalar-multiplied).

    X : (B, L, n_a) -> (B, L, n_b).
    """
    cos_c = torch.cos(c).unsqueeze(0)     # (1, L, n_a)
    sin_c = torch.sin(c).unsqueeze(0)
    X_c = X * cos_c
    X_s = X * sin_c
    Y_c = _magnitude_apply(X_c, a, b, tau_left, tau_right, normalize)   # (B, L, n_b)
    Y_s = _magnitude_apply(X_s, a, b, tau_left, tau_right, normalize)
    cos_d = torch.cos(d).unsqueeze(0)     # (1, L, n_b)
    sin_d = torch.sin(d).unsqueeze(0)
    return cos_d * Y_c + sin_d * Y_s


def laplex_phase_broadcast(
    X: torch.Tensor,          # (B, n_a)
    a: torch.Tensor,          # (L, n_a)
    b: torch.Tensor,          # (L, n_b)
    c: torch.Tensor,          # (L, n_a)
    d: torch.Tensor,          # (L, n_b)
    tau_left:  torch.Tensor,
    tau_right: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Single-vector phase Laplex: X (B, n_a) broadcast to L rows -> (B, L, n_b)."""
    B, n_a = X.shape
    L = a.shape[0]
    X_bcast = X.unsqueeze(1).expand(B, L, n_a)
    return laplex_phase(X_bcast, a, b, c, d, tau_left, tau_right, normalize)


# ----------------------------------------------------------------------------
# Functional API -- right-multiplication
# ----------------------------------------------------------------------------

def laplex_right(
    X: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau_left:  torch.Tensor,
    tau_right: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Right-multiplication: Y[b,l,k] = sum_j Kbar[l,k,j] X[b,l,j].

    X : (B, L, n)  b-side input
    a : (L, n)     a-side (output) anchors
    b : (L, n)     b-side (sum-over) anchors
    returns  (B, L, n).

    normalize=False uses the raw (un-normalised) Laplace kernel:
        Y[b,l,k] = sum_j K[l,k,j] X[b,l,j]
    normalize=True uses the column-softmaxed kernel:
        Y[b,l,k] = sum_j (K[l,k,j]/Z[l,j]) X[b,l,j]
    Implemented via swapped-anchor magnitude_apply applied to X/Z:
        Z[l,j] = sum_k K[l,k,j]  (magnitude_apply on ones, un-normalised)
        Y      = magnitude_apply(X/Z, b, a, tau_right, tau_left, False)
    """
    B, L, n_b = X.shape
    n_a = a.shape[-1]
    if not normalize:
        return _magnitude_apply(X, b, a, tau_right, tau_left, False)

    ones_X = torch.ones((1, L, n_a), device=X.device, dtype=X.dtype)
    Z = _magnitude_apply(ones_X, a, b, tau_left, tau_right, False)  # (1, L, n_b)
    w = X / Z                                                         # (B, L, n_b)
    return _magnitude_apply(w, b, a, tau_right, tau_left, False)     # (B, L, n_a)


# ----------------------------------------------------------------------------
# Functional API -- Gramian  K diag(D) K^T
# ----------------------------------------------------------------------------

def laplex_gram(
    a: torch.Tensor,          # (L, n_a)
    b: torch.Tensor,          # (L, n_b)
    D: torch.Tensor,          # (L, n_b)
    tau_left:  torch.Tensor,  # (L,)
    tau_right: torch.Tensor,  # (L,)
    normalize: bool = False,
) -> torch.Tensor:
    """Laplex Gram matrix G[l,i,i'] = sum_j D[l,j] K[l,i,j] K[l,i',j].

    With the two-temperature kernel
        K[l,i,j] = exp(-|a[l,i]-b[l,j]|/tau_sign),
        tau_sign = tau_right if a[l,i] > b[l,j] else tau_left.

    normalize=False uses the raw kernel K (default, matches the density-
    model use case  Sigma = D + W K A A^T K^T W).
    normalize=True  uses the column-softmaxed Kbar; this is equivalent to
    un-normalised Gram with weights D / Z^2, where Z[l,j] = sum_k K[l,k,j].

    Reference implementation -- dense O(L (n_a + n_b) n_b).
    """
    if a.dim() == 1:    # allow single-row usage
        a = a.unsqueeze(0); b = b.unsqueeze(0); D = D.unsqueeze(0)
        tau_left = tau_left.unsqueeze(0); tau_right = tau_right.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    L, n_a = a.shape
    _,  n_b = b.shape

    diff = a.unsqueeze(2) - b.unsqueeze(1)                        # (L, n_a, n_b)
    tau_lr = torch.where(diff > 0,
                         tau_right.view(L, 1, 1),
                         tau_left .view(L, 1, 1))
    logits = -diff.abs() / tau_lr                                 # (L, n_a, n_b)

    if normalize:
        # Kbar = softmax over i (a-side).
        K = torch.softmax(logits, dim=1)
    else:
        K = torch.exp(logits)

    # G[l, i, i'] = sum_j D[l,j] K[l,i,j] K[l,i',j]
    KD = K * D.unsqueeze(1)                                        # (L, n_a, n_b)
    G  = torch.einsum("lij,lIj->liI", KD, K)                       # (L, n_a, n_a)

    if squeeze:
        G = G.squeeze(0)
    return G


# ----------------------------------------------------------------------------
# Module API
# ----------------------------------------------------------------------------

class Laplex(nn.Module):
    """L independent two-tau Laplex rows -- pure matrix multiplication.

    Parameters per row i:
        a_i, b_i                          : (n,) learnable anchors
        tau_left_i, tau_right_i           : (,) positive temperatures
                (tied if single_tau=True; only log_tau_left is a Parameter)

    Flags:
        normalize   -- column-softmax over k; False uses raw exp weights
                       (only affects the soft branch)
        single_tau  -- tie tau_right = tau_left (classic single-tau kernel)
        hard        -- module default: soft (False) or argmin-gather (True).
                       Override per-call via forward(..., hard=True/False).
        direction   -- "left" (default; X on the a-side, out on the b-side)
                       or "right" (X on the b-side, out on the a-side).
                       Override per-call via forward(..., direction=...).

    The module applies no per-feature scaling (no "v"): it is a pure
    application of the kernel matrix.
    """

    def __init__(
        self,
        n: int,
        l: int,
        n_out: int | None = None,
        tau_init: float = 0.1,
        normalize: bool = True,
        single_tau: bool = False,
        hard: bool = False,
        direction: str = "left",
    ):
        super().__init__()
        if hard and not normalize:
            raise ValueError(
                "hard=True requires normalize=True "
                "(un-normalised hard limit is degenerate).")
        if direction not in ("left", "right"):
            raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
        if hard and direction != "left":
            raise ValueError("hard=True is only defined for direction='left'.")
        n_out            = n if n_out is None else n_out
        self.n_in        = n
        self.n_out       = n_out
        self.n           = n          # kept for backward compatibility
        self.l           = l
        self.normalize   = normalize
        self.single_tau  = single_tau
        self.hard        = hard
        self.direction   = direction

        def _init_anchors(size):
            return torch.stack(
                [torch.sort(torch.empty(size).uniform_(-1, 1))[0]
                 for _ in range(l)])

        self.a = nn.Parameter(_init_anchors(n))      # (L, n_in)  -- a-side
        self.b = nn.Parameter(_init_anchors(n_out))  # (L, n_out) -- b-side

        self.log_tau_left = nn.Parameter(torch.full((l,), math.log(tau_init)))
        if single_tau:
            self.log_tau_right = None
        else:
            self.log_tau_right = nn.Parameter(
                torch.full((l,), math.log(tau_init)))

    def _taus(self):
        tau_l = self.log_tau_left.exp()
        tau_r = tau_l if self.single_tau else self.log_tau_right.exp()
        return tau_l, tau_r

    def forward(
        self,
        X: torch.Tensor,
        hard: bool | None = None,
        normalize: bool | None = None,
        direction: str | None = None,
    ) -> torch.Tensor:
        """X : (B, L, n) -> (B, L, n)."""
        use_hard  = self.hard      if hard      is None else hard
        norm      = self.normalize if normalize is None else normalize
        direction = self.direction if direction is None else direction
        if use_hard and not norm:
            raise ValueError(
                "hard requires normalize=True (degenerate otherwise).")
        if use_hard and direction != "left":
            raise ValueError("hard=True is only defined for direction='left'.")
        tau_l, tau_r = self._taus()
        if use_hard:
            return laplex_hard(X, self.a, self.b, tau_l, tau_r)
        if direction == "right":
            return laplex_right(X, self.a, self.b, tau_l, tau_r, normalize=norm)
        return laplex(X, self.a, self.b, tau_l, tau_r, normalize=norm)

    def gram(self, D: torch.Tensor, normalize: bool | None = None) -> torch.Tensor:
        """Gram G = K diag(D) K^T   for all L rows.

        D : (L, n) or (n,) broadcast to (L, n).
        Returns (L, n, n).
        """
        norm = self.normalize if normalize is None else normalize
        if D.dim() == 1:
            D = D.unsqueeze(0).expand(self.l, -1)
        tau_l, tau_r = self._taus()
        return laplex_gram(self.a, self.b, D, tau_l, tau_r, normalize=norm)


class LaplexPhase(nn.Module):
    """L independent phase-modulated Laplex rows.

        K[l, k, j] = exp(-|a[l,k] - b[l,j]| / tau_sign) * cos(c[l,k] - d[l,j])

    Parameters per row:
        a : (L, n_in)    magnitude anchors, a-side
        b : (L, n_out)   magnitude anchors, b-side
        c : (L, n_in)    phase offsets,     a-side
        d : (L, n_out)   phase offsets,     b-side
        log_tau_left, log_tau_right : (L,)
    """

    def __init__(
        self,
        n: int,
        l: int,
        n_out: int | None = None,
        tau_init: float = 0.1,
        normalize: bool = True,
        single_tau: bool = False,
        phase_init: float = 0.5,
    ):
        super().__init__()
        n_out = n if n_out is None else n_out
        self.n_in, self.n_out, self.l = n, n_out, l
        self.normalize = normalize
        self.single_tau = single_tau

        def _init_anchors(size):
            return torch.stack(
                [torch.sort(torch.empty(size).uniform_(-1, 1))[0]
                 for _ in range(l)])

        self.a = nn.Parameter(_init_anchors(n))
        self.b = nn.Parameter(_init_anchors(n_out))
        self.c = nn.Parameter(torch.empty(l, n    ).uniform_(-phase_init, phase_init))
        self.d = nn.Parameter(torch.empty(l, n_out).uniform_(-phase_init, phase_init))

        self.log_tau_left = nn.Parameter(torch.full((l,), math.log(tau_init)))
        if single_tau:
            self.log_tau_right = None
        else:
            self.log_tau_right = nn.Parameter(
                torch.full((l,), math.log(tau_init)))

    def _taus(self):
        tau_l = self.log_tau_left.exp()
        tau_r = tau_l if self.single_tau else self.log_tau_right.exp()
        return tau_l, tau_r

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X : (B, L, n_in) -> (B, L, n_out)."""
        tau_l, tau_r = self._taus()
        return laplex_phase(X, self.a, self.b, self.c, self.d,
                            tau_l, tau_r, self.normalize)


class LaplexBlock(nn.Module):
    """Full real layer:
        X (B, l_in, n_in) -> Laplex -> (B, l_in, n_out) -> mix rows ->
        (B, l_out, n_out).

    The row-mixer M has shape (l_out, l_in) and acts along the row axis
    only, leaving the n axis untouched. Pass `n_out` to use a rectangular
    kernel (n_in -> n_out features).
    """

    def __init__(
        self,
        n: int,
        l_in: int,
        l_out: int | None = None,
        n_out: int | None = None,
        tau_init: float = 0.1,
        normalize: bool = True,
        single_tau: bool = False,
        hard: bool = False,
        direction: str = "left",
    ):
        super().__init__()
        l_out = l_in if l_out is None else l_out
        self.kernel = Laplex(
            n, l_in,
            n_out=n_out,
            tau_init=tau_init,
            normalize=normalize,
            single_tau=single_tau,
            hard=hard,
            direction=direction,
        )
        self.M = nn.Parameter(torch.empty(l_out, l_in))
        nn.init.orthogonal_(self.M)

    def forward(
        self,
        X: torch.Tensor,
        hard: bool | None = None,
        normalize: bool | None = None,
        direction: str | None = None,
    ) -> torch.Tensor:
        Y = self.kernel(X, hard=hard, normalize=normalize, direction=direction)
        return torch.einsum("ab,Bbn->Ban", self.M, Y)


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, n = 2, 4, 16
    x = torch.randn(B, L, n)

    # Soft/hard x single/two tau — left direction (classic).
    for single_tau in (False, True):
        for hard in (False, True):
            blk = LaplexBlock(n, L, L, single_tau=single_tau, hard=hard)
            y = blk(x)
            assert y.shape == (B, L, n)
            y.sum().backward()
            if not hard:
                assert blk.kernel.a.grad is not None
                assert blk.kernel.log_tau_left.grad is not None
            print(f"soft/hard:  single_tau={single_tau} hard={hard}  "
                  f"out={tuple(y.shape)}  OK")

    # Right-multiplication, both normalize modes.
    for single_tau in (False, True):
        for normalize in (True, False):
            blk = LaplexBlock(n, L, L, single_tau=single_tau,
                              direction="right", normalize=normalize)
            y = blk(x)
            assert y.shape == (B, L, n)
            y.sum().backward()
            print(f"right-mul:  single_tau={single_tau} normalize={normalize}  "
                  f"out={tuple(y.shape)}  OK")

    # Gram -- compare two tau vs single tau (smoke only).
    k = Laplex(n, L, single_tau=False)
    D = torch.randn(L, n).abs()
    G = k.gram(D)
    assert G.shape == (L, n, n)
    # Symmetry check.
    assert torch.allclose(G, G.transpose(-1, -2), atol=1e-5)
    print(f"gram:       shape={tuple(G.shape)}  symmetric  OK")

    # Rejection cases.
    try:
        LaplexBlock(n, L, L, normalize=False, hard=True)
    except ValueError:
        print("normalize=False + hard=True correctly rejected.")
    try:
        LaplexBlock(n, L, L, hard=True, direction="right")
    except ValueError:
        print("hard=True + direction='right' correctly rejected.")
