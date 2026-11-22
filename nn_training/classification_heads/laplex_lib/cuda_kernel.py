"""
cuda.py  --  CUDA backend for the LAPLEX operations
====================================================

Optimised CUDA implementation of the three core operations on the real
Laplace kernel

    K(a, b)[i, j] = exp(-|a[i] - b[j]|),       a in R^n, b in R^k.

API matches the pure-PyTorch reference in `pure.py` so that the two
are drop-in replacements for each other.  The top-level ``laplex.py``
facade picks one at import time (CUDA when available, pure-PyTorch
otherwise) and re-exports the chosen backend's symbols, so callers
should normally do ``from laplex import X_Laplex, ...`` rather than
importing this module directly.

    X_Laplex(X, a, b)              = X @ K(a, b)              ->  (B, k) or (k,)
    Laplex_X(X, a, b)              = K(a, b) @ X              ->  (n, D) or (n,)
    Laplex_D_LaplexT(alpha, beta, D)
                                   = K(alpha, beta) @ diag(D) @ K(alpha, beta)^T
                                                              ->  (n, n)

Algorithm sketch
----------------
The Laplace kernel splits cleanly once one of the index sets is sorted:
for sorted `a` and any `b[j]` we have

    K[i, j] = exp(a[i] - b[j])  if a[i] <= b[j],
              exp(b[j] - a[i])  otherwise,

so the contribution of every `a[i]` to position `b[j]` depends only on
the *side* of `b[j]` relative to `a[i]`.  The product `X @ K` reduces to
two 1-D scans on the shorter of the two axes, plus one binary search.

Two branches:
  - branch A (n <= k):  sort `a`, scan along it, gather per `b[j]`     → O(n log n + k)
  - branch B (n  > k):  scatter `n` contributions into `k` buckets in
                         sorted `b`, then scan along `b`               → O(n + k log k)

The Gram matrix `K diag(D) K^T` admits an analogous trick: for sorted
`alpha` the kernel sum splits into three regimes (`beta_c` left of, between,
or right of every pair `(alpha_i, alpha_j)`), each of which is a separable
sum that becomes a prefix/suffix scan on `2*alpha` after a few segment
sums.  The resulting cost is `O(n^2 + k log n)` instead of the obvious
`O(n^2 k)`.

dtype dispatch
--------------
- fp64 anywhere in the inputs   -> whole call promoted to fp64
- otherwise                     -> fp32
- fp16 / bf16                   -> promoted to fp32 (CUDA scan is fp32-only at low precision)
The dispatch happens transparently inside `prefix_suffix.prefix_suffix_sep`.

This file has no external dependency beyond `prefix_suffix.py` and `torch`.

Module organisation
-------------------
This module hosts the SHARED-coords single-matrix CUDA API:

    X_Laplex,  Laplex_X,  Laplex_D_LaplexT,
    X_LaplexPhase,  LaplexPhase_X,
    LaplexPhase_D_LaplexPhaseT,  LaplexPhaseT_D_LaplexPhase

The per-row (\"grouped\") and L-kernel mixture variants used by
Sortformer-style attention live in a separate companion repository
(``laplex_grouped``).  Import them directly from there -- this module
does not re-export them.
"""
from __future__ import annotations

import os, sys
from typing import Optional

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_suffix import (
    prefix_suffix_sep, prefix_suffix, _GatherCombineFn,
)


__all__ = [
    # Public matvec / Gram API
    "X_Laplex", "Laplex_X", "Laplex_D_LaplexT",
    # Phased variants (single-temperature Laplace with cos modulation)
    "X_LaplexPhase", "LaplexPhase_X",
    "LaplexPhase_D_LaplexPhaseT", "LaplexPhaseT_D_LaplexPhase",
    # Dispatch constants
    "GPU_MEM_BUDGET", "THIN_DIM_LIMIT", "DENSE_SAFETY_CAP",
    "NAIVE_SCAN_THRESHOLD",
]


# ===========================================================================
# Tunable module-level constants for dispatch
# ===========================================================================
# The dispatcher chooses between two regimes:
#   - dense materialise + cuBLAS GEMM
#   - scan path (production or merged-events, depending on B)
# Empirically the dense path wins in TWO disjoint regions:
#   (1) "small symmetric"   --  n*k below a threshold (fits comfortably in
#                               L2, scan setup overhead dominates)
#   (2) "thin asymmetric"   --  min(n, k) is small (cuBLAS GEMM is super
#                               efficient for skewed shapes regardless of
#                               total n*k)
# The scan wins in between, when K is large and roughly square.
# Both rules are subject to a hard safety cap (`DENSE_SAFETY_CAP`) so a
# pathological  min(n, k)  shape doesn't try to materialise 10 GB of K.

# (1) Small-symmetric threshold: n*k <= GPU_MEM_BUDGET --> dense.
# Default 1 << 22 = 4M elements = 16 MB FP32; ~ quarter of L2 on a mid-range
# consumer GPU.  Bump up for datacentre cards with bigger L2; drop on small GPUs.
# Overridable per-call via `dense_budget=`.
GPU_MEM_BUDGET = 1 << 22

# (2) Thin-asymmetric shortcut: min(n, k) <= THIN_DIM_LIMIT --> dense even
# for nk well above GPU_MEM_BUDGET.  Empirically (RTX 5060):
#     min(n, k) = 384, nk = 19M   ->  dense wins 2x  over scan
#     min(n, k) = 384, nk = 38M   ->  dense wins 12x
#     min(n, k) = 64,  nk = 1.2M  ->  dense wins 29x
# 512 is comfortably above the 384 used in LAPLEX KDK and well below the
# 1000 dim at which scan starts to win.  Set None to disable the shortcut.
THIN_DIM_LIMIT = 512

# Hard safety cap on materialisation -- nk above this NEVER goes dense
# (would OOM or thrash memory).  Default 1 << 28 = 256M elem = 1 GB FP32;
# fits comfortably on an 8 GB consumer GPU alongside other tensors.
# Per-call override via `dense_safety_cap=`.
DENSE_SAFETY_CAP = 1 << 28


def _should_use_dense(n: int, k: int,
                      dense_budget: Optional[int] = None,
                      thin_dim_limit: Optional[int] = None,
                      dense_safety_cap: Optional[int] = None) -> bool:
    """Decision: materialise K(a, b) + cuBLAS GEMM, or take the scan path?

    See module docstring at top of this file for the two-regime motivation.
    Each parameter falls back to its module-level default when None.
    """
    budget = GPU_MEM_BUDGET    if dense_budget     is None else dense_budget
    thin   = THIN_DIM_LIMIT    if thin_dim_limit   is None else thin_dim_limit
    cap    = DENSE_SAFETY_CAP  if dense_safety_cap is None else dense_safety_cap
    nk = n * k
    if cap is not None and nk > cap:
        return False                             # too big to materialise
    if nk <= budget:
        return True                              # small-symmetric
    if thin is not None and min(n, k) <= thin:
        return True                              # thin-asymmetric
    return False

# When the batch axis is large enough to saturate the GPU on its own,
# the two-pass merged-events scan (see merged_events.py) wins over
# the production decoupled-lookback scan.  It avoids materialising the
# prefix/suffix (k, B) arrays entirely -- emits y directly during the
# sweep through the merged sorted event list -- so memory traffic is
# ~1.5x smaller.  Empirical crossover on an RTX 5060 is around B ~ 4096.
# The merged path is wrapped in a torch.autograd.Function and propagates
# gradients through (a, b, x), so it is used in BOTH training and
# inference.  Supports FP32 and FP64 (kernel templated).  The constant
# kept its historical 'NAIVE_SCAN' prefix for backward compatibility;
# set to None to disable the dispatch entirely.
NAIVE_SCAN_THRESHOLD = 4096


# ===========================================================================
# torch.compile compatibility note
# ===========================================================================
# All three execution paths (dense / merged-events / production-scan) are
# torch.compile-friendly:
#   * Dense path is pure PyTorch ops -- Inductor + Triton fuses the
#     element-wise sub/abs/exp/multiply into 2-3 generated kernels.
#   * Merged-events and production-scan paths each wrap a custom
#     torch.autograd.Function; Dynamo creates a graph break around them
#     but the surrounding code (LayerNorm, projections, MLPs) still
#     compiles, so the in-eager gains are not lost.
#
# Recommended user-side setup (place near the top of the training script):
#
#     import torch
#     torch.set_float32_matmul_precision('high')   # TF32 for GEMM
#     model = model.to(device)
#     model = torch.compile(model)                 # auto-fuses LAPLEX dense
#
# First training iteration pays ~30-60s for Inductor codegen (cached after).
# On Windows install `triton-windows`:  pip install triton-windows
# ===========================================================================


# ===========================================================================
# X_Laplex  =  X @ K(a, b)
# ===========================================================================
def X_Laplex(X: Tensor, a: Tensor, b: Tensor,
             threshold: int = 1024,
             a_sorted: bool = False, b_sorted: bool = False,
             dense_budget: Optional[int] = None,
             thin_dim_limit: Optional[int] = None,
             dense_safety_cap: Optional[int] = None) -> Tensor:
    """``X @ K(a, b)``,  K[i, j] = exp(-|a[i] - b[j]|).

    Parameters
    ----------
    X : (B, n) or (n,)        anchor-side input
    a : (n,)                  anchors on the X-side (sortable)
    b : (k,)                  anchors on the output side
    threshold : int           if `min(n, k) <= threshold` use a dense matmul
                              instead of the scan path (constant scan setup
                              dominates for very small problems)
    a_sorted, b_sorted        skip the corresponding internal sort if `a`
                              or `b` is already in non-decreasing order
    dense_budget : int        per-call override of module-level
                              `GPU_MEM_BUDGET` (small-symmetric threshold).
    thin_dim_limit : int      per-call override of `THIN_DIM_LIMIT`
                              (thin-asymmetric shortcut).
    dense_safety_cap : int    per-call override of `DENSE_SAFETY_CAP`
                              (hard cap on materialisation size).

    Returns
    -------
    (B, k) or (k,) tensor in the promoted floating dtype (any fp64 input
    promotes everything to fp64; otherwise fp32).

    Note
    ----
    No batch-axis chunking is performed here.  Very large B is handled by
    the merged-events fast path in `Laplex_X` (which dispatches to
    `merged_events.merged_K_at_x`); this internal scan only runs in
    the small-to-medium B regime where the production scan is optimal.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)
    B, n = X.shape
    k = b.numel()

    # Dense path: small-symmetric  OR  thin-asymmetric  (see _should_use_dense
    # for the two-regime explanation).  Guarded by a safety cap so the
    # thin-asymmetric shortcut never tries to materialise > 1 GB of K.
    if _should_use_dense(n, k, dense_budget, thin_dim_limit, dense_safety_cap):
        K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        y = X @ K.to(X.dtype)
        return y.squeeze(0) if squeeze else y

    if n <= k:
        y = _xA(X, a, b, a_sorted=a_sorted)
    else:
        y = _xAT(X, a, b, b_sorted=b_sorted)
    return y.squeeze(0) if squeeze else y


def _xA(X, a, b, a_sorted=False):
    """Branch A: scan along `a` (the shorter axis), gather per `b`."""
    B, n = X.shape
    k = b.numel()

    if a_sorted:
        a_s, X_s = a, X
    else:
        a_s, perm = torch.sort(a)
        X_s = X[:, perm]

    # Both prefix(a_s, X_s) and suffix(a_s, X_s) in a single fused scan.
    P, S = prefix_suffix_sep(a_s, X_s)

    # For each b[j]: bucket index c[j] = max i s.t. a_s[i] <= b[j],
    # or -1 if b[j] < a_s[0].
    c = torch.searchsorted(a_s, b, right=True) - 1            # (k,) in [-1, n-1]

    # Left contribution:  exp(a_s[c] - b) * P[:, c]    (only for c >= 0).
    c_l = c.clamp(min=0)
    gate_l = (c >= 0).to(X.dtype)
    w_l = torch.exp(a_s[c_l] - b) * gate_l                    # (k,)

    # Right contribution: exp(b - a_s[c+1]) * S[:, c+1] (only for c+1 < n).
    c_r = (c + 1).clamp(max=n - 1)
    gate_r = ((c + 1) < n).to(X.dtype)
    w_r = torch.exp(b - a_s[c_r]) * gate_r                    # (k,)

    return _GatherCombineFn.apply(P, S, w_l, w_r, c_l, c_r)


def _xAT(X, a, b, b_sorted=False):
    """Branch B: scatter `n` X-contributions into `k` buckets in sorted `b`,
    then scan along `b`.  Used when n > k."""
    B, n = X.shape
    k = b.numel()

    if b_sorted:
        b_s, inv_perm_b = b, None
    else:
        b_s, perm_b = torch.sort(b)
        inv_perm_b = torch.argsort(perm_b)

    # For each a[i], find its bucket in sorted b.
    idx = torch.searchsorted(b_s, a, right=True) - 1          # (n,)

    # Left scatter: u[c] += X[:, i] * exp(b_s[c] - a[i])
    i_u = idx.clamp(min=0)
    gate_u = (idx >= 0).to(X.dtype)
    vals_u = torch.exp(b_s[i_u] - a) * gate_u
    u = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_u.unsqueeze(0).expand(B, -1), X * vals_u.unsqueeze(0))

    # Right scatter: v[c+1] += X[:, i] * exp(a[i] - b_s[c+1])
    i_v = (idx + 1).clamp(max=k - 1)
    gate_v = ((idx + 1) < k).to(X.dtype)
    vals_v = torch.exp(a - b_s[i_v]) * gate_v
    v = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_v.unsqueeze(0).expand(B, -1), X * vals_v.unsqueeze(0))

    # Final answer is prefix(b_s, v) + suffix(b_s, u).
    result_sorted = prefix_suffix(b_s, v, u)
    if inv_perm_b is not None:
        return result_sorted[:, inv_perm_b]
    return result_sorted


# ===========================================================================
# Laplex_X  =  K(a, b) @ X    (transpose identity)
# ===========================================================================
def Laplex_X(X: Tensor, a: Tensor, b: Tensor,
             threshold: int = 1024,
             a_sorted: bool = False, b_sorted: bool = False,
             dense_budget: Optional[int] = None,
             thin_dim_limit: Optional[int] = None,
             dense_safety_cap: Optional[int] = None,
             merged_threshold: Optional[int] = None) -> Tensor:
    """``K(a, b) @ X``.

    X : (k, D) or (k,)     ->     (n, D) or (n,)

    `dense_budget`, `thin_dim_limit`, `dense_safety_cap` are forwarded to
    `X_Laplex` -- see its docstring.

    `merged_threshold`       when B >= merged_threshold AND the dense path
                             does NOT apply (per `_should_use_dense`),
                             dispatch to the two-pass merged-events scan in
                             `merged_events.py` (~1.5x faster than the
                             production scan in this regime, supports both
                             FP32 and FP64, and autograd).  Default `None`
                             uses module-level `NAIVE_SCAN_THRESHOLD`.  Set
                             to `None` to disable.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(1)

    k_in, B = X.shape
    n_a = a.numel(); n_b = b.numel()

    # Dense path takes priority over the merged-events path when applicable:
    #   - small-symmetric K (n_a * n_b <= GPU_MEM_BUDGET), or
    #   - thin-asymmetric K (min(n_a, n_b) <= THIN_DIM_LIMIT).
    # When dense applies, fall through to X_Laplex which materialises and GEMMs.
    use_dense = _should_use_dense(n_a, n_b,
                                  dense_budget, thin_dim_limit, dense_safety_cap)

    # ── Fast path for large B (autograd-aware): merged-events scan ───────
    # Supports FP16 (mixed precision: FP32 accumulator), FP32, FP64.
    thr = NAIVE_SCAN_THRESHOLD if merged_threshold is None else merged_threshold
    if (not use_dense
            and thr is not None and B >= thr
            and X.dtype in (torch.float16, torch.float32, torch.float64)
            and X.is_cuda
            and a.dtype == b.dtype == X.dtype):
        try:
            from merged_events import merged_K_at_x
            X_in = X if X.is_contiguous() else X.contiguous()
            out_m = merged_K_at_x(a, b, X_in)
            return out_m.squeeze(1) if squeeze else out_m
        except Exception:
            # Any failure (build issues, unsupported sizes) falls through
            # to the production scan.
            pass

    out = X_Laplex(X.T, b, a, threshold=threshold,
                   a_sorted=b_sorted, b_sorted=a_sorted,
                   dense_budget=dense_budget,
                   thin_dim_limit=thin_dim_limit,
                   dense_safety_cap=dense_safety_cap).T
    return out.squeeze(1) if squeeze else out


# ===========================================================================
# Laplex_D_LaplexT  =  K(alpha, beta) @ diag(D) @ K(alpha, beta)^T
# ===========================================================================
def Laplex_D_LaplexT(alpha: Tensor, beta: Tensor, D: Tensor,
                     threshold: int = 1024,
                     alpha_sorted: bool = False,
                     dense_budget: Optional[int] = None,
                     thin_dim_limit: Optional[int] = None,
                     dense_safety_cap: Optional[int] = None) -> Tensor:
    """``K(alpha, beta) @ diag(D) @ K(alpha, beta)^T``  ->  (n, n) symmetric.

    Cost ``O(n^2 + k log n)`` via three segment sums on `D` and one scan
    on `2 * alpha`, replacing the naive ``O(n^2 k)`` matmul.
    `dense_budget`, `thin_dim_limit`, `dense_safety_cap` -- see X_Laplex.

    Batched D
    ---------
    ``D`` may be either 1-D of shape ``(k,)`` (single problem; result is
    ``(n, n)``) or 2-D of shape ``(B, k)`` (B different right-hand sides
    sharing the SAME (alpha, beta); result is ``(B, n, n)``).  The shared-
    coords structure lets us amortise the sort, bucketize and prefix-
    suffix scan on ``2*alpha`` across the batch: only the three segment
    sums and the final (B, n, n) assembly carry the B factor, which is
    typically much cheaper than B independent ``Laplex_D_LaplexT`` calls.

    Both real and complex ``D`` are supported in both shapes; the result
    inherits ``D``'s dtype.
    """
    if D.dim() not in (1, 2):
        raise ValueError(f"D must be 1D (k,) or 2D (B, k); got {D.shape}")
    batched = (D.dim() == 2)

    n, k = alpha.numel(), beta.numel()
    if batched and D.size(1) != k:
        raise ValueError(
            f"D last dim {D.size(1)} != beta.numel() {k}")
    if not batched and D.size(0) != k:
        raise ValueError(
            f"D numel {D.size(0)} != beta.numel() {k}")

    out_dtype = D.dtype

    # ── Dense path ────────────────────────────────────────────────────
    if _should_use_dense(n, k, dense_budget, thin_dim_limit, dense_safety_cap):
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        K_c = K.to(out_dtype)
        if batched:
            # M[b, i, l] = Σ_c K[i,c] · D[b,c] · K[l,c]
            return torch.einsum('ic,bc,lc->bil', K_c, D, K_c)
        return (K_c * D.unsqueeze(0)) @ K_c.T

    # ── Scan path (with optional batched D) ───────────────────────────
    if alpha_sorted:
        alpha_s, inv_perm = alpha, None
    else:
        alpha_s, perm = torch.sort(alpha)
        inv_perm = torch.argsort(perm)

    device = alpha.device

    # Bucket each beta[c] into sorted alpha; idx[c] in [0, n].  Shared
    # across all B if D is batched -- (alpha, beta) are themselves shared.
    idx = torch.bucketize(beta, alpha_s, right=True)

    # alpha_ext: sentinels so that idx=0 and idx=n stay in range.
    alpha_ext = torch.cat([beta.min().unsqueeze(0),
                           alpha_s,
                           beta.max().unsqueeze(0)])

    # Per-beta exponent factors -- shape (k,), shared across batches.
    alpha_ip1 = alpha_ext[idx + 1]
    u_exp = torch.exp(2.0 * (beta - alpha_ip1)).to(out_dtype)
    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_exp = torch.exp(2.0 * (alpha_i - beta)).to(out_dtype)

    # Promote D to 2-D (B_eff, k) so the same code handles both shapes.
    D_2d = D.unsqueeze(0) if not batched else D
    B_eff = D_2d.size(0)

    u_vals = D_2d * u_exp.unsqueeze(0)            # (B_eff, k)
    v_vals = D_2d * v_exp.unsqueeze(0)            # (B_eff, k)
    idx_e  = idx.unsqueeze(0).expand(B_eff, -1)   # (B_eff, k)

    u = torch.zeros(B_eff, n + 1, device=device, dtype=out_dtype) \
              .scatter_add(-1, idx_e, u_vals)
    v = torch.zeros(B_eff, n + 1, device=device, dtype=out_dtype) \
              .scatter_add(-1, idx_e, v_vals)
    w = torch.zeros(B_eff, n + 1, device=device, dtype=out_dtype) \
              .scatter_add(-1, idx_e, D_2d)

    # U = prefix(2 alpha, u[:, :n]),  V = suffix(2 alpha, v[:, 1:n+1]).
    # The kernel takes (B, T) and broadcasts alpha across the batch dim.
    alpha2 = 2.0 * alpha_s
    U, V = prefix_suffix_sep(alpha2,
                             u[:, :n].clone(),
                             v[:, 1:n + 1].clone())          # (B_eff, n) each

    # Cumulative weight (per-batch).
    W = torch.cumsum(w[:, :n], dim=-1)                       # (B_eff, n)

    # Build (B_eff, n, n) lower triangle of the symmetric Gram matrix.
    # M[b, i, l] = diff[i,l] · ( U[b, l] + (W[b, i] - W[b, l]) + V[b, i] )
    ai = alpha_s.unsqueeze(1)
    aj = alpha_s.unsqueeze(0)
    diff = torch.exp(-torch.relu(ai - aj)).to(out_dtype)     # (n, n)

    M_raw = diff.unsqueeze(0) * (
        U.unsqueeze(-2)                                   # (B, 1, n)
        + (W.unsqueeze(-1) - W.unsqueeze(-2))             # (B, n, n)
        + V.unsqueeze(-1)                                 # (B, n, 1)
    )                                                     # (B, n, n)

    # Symmetrise per slice: forces exact M = M^T.
    M = torch.tril(M_raw) + torch.tril(M_raw, -1).transpose(-1, -2)

    if inv_perm is not None:
        # Apply inverse permutation to last two dims per batch.
        M = M.index_select(-2, inv_perm).index_select(-1, inv_perm)

    return M if batched else M.squeeze(0)


# ===========================================================================
# Phased Laplace kernel
#     A_phase[i, j] = exp(-|alpha_i - beta_j| / tau) * cos(gamma_i - delta_j)
# ===========================================================================
# Real-valued I/O implementation built on top of the real-Laplace primitives
# above:
#
#     A_phase = Re( exp(i*gamma) * K(alpha/tau, beta/tau) * exp(-i*delta) ),
#
# so X * A_phase factors as
#     Y_real = X_Laplex( X * cos(gamma), alpha/tau, beta/tau )
#     Y_imag = X_Laplex( X * sin(gamma), alpha/tau, beta/tau )
#     Y      = Y_real * cos(delta) + Y_imag * sin(delta).
# The two `X_Laplex` calls are concatenated along the batch axis and issued
# as a single 2B-row call -- one scan instead of two.  The Gram counterpart
# uses three real Gram scans (G0 + Re/Im split of a complex-weighted Gram).
#
# The pure-PyTorch backend in `pure.py` exposes the SAME phased API via a
# complex-Laplex path; the two are drop-in compatible.

def _phased_scale(t: Tensor, tau, ref: Tensor):
    """``t / tau`` with proper dtype/device for ``tau``;  ``t`` unchanged when
    ``tau`` is None."""
    if tau is None:
        return t
    tau_t = torch.as_tensor(tau, device=ref.device, dtype=ref.dtype)
    return t / tau_t


def X_LaplexPhase(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: int = 1024,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None) -> Tensor:
    """``X @ A_phase`` with
    ``A_phase[i,j] = exp(-|alpha_i - beta_j| / tau) * cos(gamma_i - delta_j)``.

    X : (B, n) or (n,)        real
    alpha, gamma : (n,)
    beta,  delta : (k,)
    tau   : optional positive scalar dividing both alpha and beta
    Returns (B, k) or (k,) with the same real dtype as `X`.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)
    B, n = X.shape

    alpha_s = _phased_scale(alpha, tau, alpha)
    beta_s  = _phased_scale(beta,  tau, alpha)

    cos_g = torch.cos(gamma);  sin_g = torch.sin(gamma)        # (n,)
    cos_d = torch.cos(delta);  sin_d = torch.sin(delta)        # (k,)

    # Stack [X*cos(gamma); X*sin(gamma)] (2B, n) -> one scan -> split.
    X_cs = torch.cat([X * cos_g.unsqueeze(0),
                      X * sin_g.unsqueeze(0)], dim=0)          # (2B, n)
    Y_cs = X_Laplex(X_cs, alpha_s, beta_s,
                    threshold=threshold,
                    a_sorted=alpha_sorted, b_sorted=beta_sorted)  # (2B, k)
    Y_real_part = Y_cs[:B]                                     # (B, k)
    Y_imag_part = Y_cs[B:]                                     # (B, k)

    # Re( (Re + i Im) * (cos(d) - i sin(d)) ) = Re*cos(d) + Im*sin(d).
    Y = Y_real_part * cos_d.unsqueeze(0) + Y_imag_part * sin_d.unsqueeze(0)
    return Y.squeeze(0) if squeeze else Y


def LaplexPhase_X(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: int = 1024,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None) -> Tensor:
    """``A_phase @ X``.

    X : (k, B) or (k,)     ->     (n, B) or (n,).
    Implemented via the transpose identity
        A_phase X = (X^T A_phase^T)^T,
    where ``A_phase^T`` is ``A_phase`` with (alpha, gamma) and (beta, delta)
    swapped.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(1)
    Y = X_LaplexPhase(X.T, beta, alpha, delta, gamma,
                      threshold=threshold,
                      alpha_sorted=beta_sorted, beta_sorted=alpha_sorted,
                      tau=tau).T
    return Y.squeeze(1) if squeeze else Y


def LaplexPhase_D_LaplexPhaseT(alpha: Tensor, beta: Tensor,
                               gamma: Tensor, delta: Tensor, D: Tensor,
                               threshold: int = 1024,
                               alpha_sorted: bool = False,
                               tau: Optional[float] = None) -> Tensor:
    """``A_phase @ diag(D) @ A_phase^T``  ->  shape (n, n).

    Three real Gram scans:
        G0      = K diag(D)               K^T
        Gc_real = K diag( D cos(2 delta)) K^T
        Gc_imag = K diag(-D sin(2 delta)) K^T
    Then
        M = 0.5 * (cos(gamma_i - gamma_j) * G0
                   + Re[ exp(i(gamma_i + gamma_j)) * (Gc_real + i Gc_imag) ]).
    """
    alpha_s = _phased_scale(alpha, tau, alpha)
    beta_s  = _phased_scale(beta,  tau, alpha)

    G0 = Laplex_D_LaplexT(alpha_s, beta_s, D,
                          threshold=threshold, alpha_sorted=alpha_sorted)

    cos_2d = torch.cos(2.0 * delta);  sin_2d = torch.sin(2.0 * delta)
    Gc_real = Laplex_D_LaplexT(alpha_s, beta_s, D * cos_2d,
                               threshold=threshold, alpha_sorted=alpha_sorted)
    Gc_imag = Laplex_D_LaplexT(alpha_s, beta_s, -D * sin_2d,
                               threshold=threshold, alpha_sorted=alpha_sorted)

    gi = gamma.unsqueeze(1)
    gj = gamma.unsqueeze(0)
    cos_diff = torch.cos(gi - gj)
    cos_sum  = torch.cos(gi + gj)
    sin_sum  = torch.sin(gi + gj)

    # Re[ exp(i(gi+gj)) * (Gc_real + i*Gc_imag) ] = cos_sum*Gc_real - sin_sum*Gc_imag
    re_T1 = cos_sum * Gc_real - sin_sum * Gc_imag
    return 0.5 * (cos_diff * G0 + re_T1)


def LaplexPhaseT_D_LaplexPhase(alpha: Tensor, beta: Tensor,
                               gamma: Tensor, delta: Tensor, D: Tensor,
                               threshold: int = 1024,
                               beta_sorted: bool = False,
                               tau: Optional[float] = None) -> Tensor:
    """``A_phase^T @ diag(D) @ A_phase``  ->  shape (k, k).
    Same construction with the two sides swapped.
    """
    return LaplexPhase_D_LaplexPhaseT(beta, alpha, delta, gamma, D,
                                      threshold=threshold,
                                      alpha_sorted=beta_sorted, tau=tau)


# ===========================================================================
# Self-test (run: python laplex.py)
# ===========================================================================
def _smoke_test():
    assert torch.cuda.is_available(), "CUDA required (this is the CUDA backend)."
    device = "cuda"
    torch.manual_seed(0)

    print("=" * 60)
    print("laplex.py: smoke test (vs dense reference)")
    print("=" * 60)

    passed = failed = 0

    def chk(name, got, ref, tol):
        nonlocal passed, failed
        ref64 = ref.detach().double()
        got64 = got.detach().double()
        err = (got64 - ref64).abs().max().item() / (ref64.abs().max().item() + 1e-30)
        ok = err < tol
        passed += ok; failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:48s} dtype={got.dtype} err={err:.2e}")

    def K_full(a, b):
        return torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())

    # ---- X_Laplex / Laplex_X (fp32) ----
    print("\n--- X_Laplex / Laplex_X (fp32) ---")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 300, 100), (2, 500, 50)]:
        a = torch.randn(n, device=device, dtype=torch.float32)
        b = torch.randn(k, device=device, dtype=torch.float32)
        X = torch.randn(B, n, device=device, dtype=torch.float32)
        Y = torch.randn(k, B, device=device, dtype=torch.float32)
        Kf = K_full(a, b)
        chk(f"X_Laplex B={B}, n={n}, k={k}",
            X_Laplex(X, a, b, threshold=0), X @ Kf, 1e-4)
        chk(f"Laplex_X B={B}, n={n}, k={k}",
            Laplex_X(Y, a, b, threshold=0), Kf @ Y, 1e-4)

    # ---- fp64 ----
    print("\n--- X_Laplex / Laplex_X (fp64) ---")
    for B, n, k in [(2, 80, 60), (4, 200, 100), (1, 150, 250)]:
        a = torch.randn(n, device=device, dtype=torch.float64)
        b = torch.randn(k, device=device, dtype=torch.float64)
        X = torch.randn(B, n, device=device, dtype=torch.float64)
        chk(f"X_Laplex fp64 B={B}, n={n}, k={k}",
            X_Laplex(X, a, b, threshold=0), X @ K_full(a, b), 1e-12)

    # ---- Laplex_D_LaplexT ----
    print("\n--- Laplex_D_LaplexT ---")
    for n, k in [(50, 100), (200, 300)]:
        alpha = torch.randn(n, device=device, dtype=torch.float32)
        beta  = torch.randn(k, device=device, dtype=torch.float32)
        D     = torch.randn(k, device=device, dtype=torch.float32)
        Kf = K_full(alpha, beta)
        chk(f"Gram fp32 n={n}, k={k}",
            Laplex_D_LaplexT(alpha, beta, D, threshold=0),
            (Kf * D.unsqueeze(0)) @ Kf.T, 1e-3)

    alpha = torch.randn(120, device=device, dtype=torch.float64)
    beta  = torch.randn(180, device=device, dtype=torch.float64)
    D     = torch.randn(180, device=device, dtype=torch.float64)
    Kf = K_full(alpha, beta)
    chk("Gram fp64 n=120, k=180",
        Laplex_D_LaplexT(alpha, beta, D, threshold=0),
        (Kf * D.unsqueeze(0)) @ Kf.T, 1e-12)

    # ---- Batched D: D shape (B, k) -> M shape (B, n, n) ----
    print("\n--- Laplex_D_LaplexT (batched D) ---")
    for n, k, B in [(50, 100, 4), (200, 300, 8)]:
        alpha = torch.randn(n, device=device, dtype=torch.float32)
        beta  = torch.randn(k, device=device, dtype=torch.float32)
        Db    = torch.randn(B, k, device=device, dtype=torch.float32)
        Kf = K_full(alpha, beta)
        M_ref = torch.einsum('ic,bc,lc->bil', Kf, Db, Kf)
        chk(f"Gram batched n={n}, k={k}, B={B}",
            Laplex_D_LaplexT(alpha, beta, Db, threshold=0), M_ref, 1e-3)

    # Scan path batched D (n large enough that we leave dense)
    alpha = torch.randn(800, device=device, dtype=torch.float32)
    beta  = torch.randn(1200, device=device, dtype=torch.float32)
    Db    = torch.randn(4, 1200, device=device, dtype=torch.float32)
    Kf = K_full(alpha, beta)
    M_ref = torch.einsum('ic,bc,lc->bil', Kf, Db, Kf)
    chk("Gram batched (scan path) n=800, k=1200, B=4",
        Laplex_D_LaplexT(alpha, beta, Db), M_ref, 5e-3)

    # ---- gradient sanity ----
    print("\n--- gradient sanity ---")
    a = torch.randn(40, device=device, requires_grad=True)
    b = torch.randn(60, device=device, requires_grad=True)
    X = torch.randn(2, 40, device=device, requires_grad=True)
    y = X_Laplex(X, a, b, threshold=0)
    y.sum().backward()
    ok = a.grad is not None and b.grad is not None and X.grad is not None
    passed += ok; failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  X_Laplex backward populates a.grad, b.grad, X.grad")

    print(f"\n{'='*60}\nResult: {passed} PASS, {failed} FAIL\n{'='*60}")
    return failed == 0


if __name__ == "__main__":
    _smoke_test()
