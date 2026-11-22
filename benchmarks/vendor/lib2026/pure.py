"""
pure.py  --  pure-PyTorch backend for the LAPLEX operations
============================================================

Pure-PyTorch reference implementation of the LAPLEX operations on the
real and phased Laplace kernels.  No custom CUDA kernels, no
``load_inline``, runs anywhere PyTorch runs (CPU, CUDA, MPS, ROCm, ...).

This is the drop-in fallback used by the auto-dispatch ``laplex.py``
facade when the CUDA backend (see ``cuda.py``) is unavailable.  The
public API is intentionally identical to ``cuda.py``:

Real Laplace kernel  K(a, b)[i, j] = exp(-|a_i - b_j|):

    X_Laplex(X, a, b)                  = X @ K(a, b)
    Laplex_X(X, a, b)                  = K(a, b) @ X
    Laplex_D_LaplexT(alpha, beta, D)   = K(alpha, beta) diag(D) K(alpha, beta)^T

Phased Laplace kernel  A_phase[i, j] = exp(-|alpha_i - beta_j| / tau)
                                       * cos(gamma_i - delta_j):

    X_LaplexPhase, LaplexPhase_X,
    LaplexPhase_D_LaplexPhaseT, LaplexPhaseT_D_LaplexPhase

Dispatch (mirrors the CUDA backend)
-----------------------------------
Same two-rule decision as ``cuda.py``: dense matmul wins for
small-symmetric (``n*k <= GPU_MEM_BUDGET``) or thin-asymmetric
(``min(n,k) <= THIN_DIM_LIMIT``) shapes, scan wins in between, hard
``DENSE_SAFETY_CAP`` prevents runaway allocations.  The constants are
shared by name so users can tune one set of knobs that affects both
backends.

For very large batch dimensions ``B`` the scan path materialises a
``(B, n)`` prefix and a ``(B, k)`` suffix tensor; we chunk along ``B``
in slabs of ``B_CHUNK`` to keep the working set bounded.  This mirrors
the merged-events kernel's internal ``B_chunk`` in the CUDA backend.

Stability trick
---------------
The exponential-weighted prefix scan
    y_i = sum_{j<=i} exp(alpha_j - alpha_i) * x_j
is rewritten as
    y_i = exp(-alpha_i + logcumsumexp_{j<=i}(alpha_j + log x_j)),
encoding the sign of `x` in the imaginary part of a complex logarithm
(`log|x| + i*pi*[x<0]`).  The scan then collapses to a single
``torch.logcumsumexp`` over a complex tensor -- parallel, differentiable,
numerically stable, and accepting genuine complex inputs for free.

Dtype support
-------------
Inputs may be float16 / bfloat16 / float32 / float64; complex64 /
complex128 are accepted by the underlying ``prefix`` / ``suffix`` and
by the phased operations.  The output dtype follows PyTorch's standard
promotion rules.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


__all__ = [
    # Public matvec / Gram API
    "X_Laplex", "Laplex_X", "Laplex_D_LaplexT",
    # Phased variants
    "X_LaplexPhase", "LaplexPhase_X",
    "LaplexPhase_D_LaplexPhaseT", "LaplexPhaseT_D_LaplexPhase",
    # Building blocks (useful for callers that want a stable scan)
    "prefix", "suffix", "prefix_suffix_sep",
    # Dispatch constants
    "GPU_MEM_BUDGET", "THIN_DIM_LIMIT", "DENSE_SAFETY_CAP", "B_CHUNK",
]


# =============================================================================
# Tunable module-level constants for dispatch
# =============================================================================
# Shared by name with cuda.py so a single set of knobs tunes both backends.

# Small-symmetric:  n*k <= GPU_MEM_BUDGET  ->  dense
GPU_MEM_BUDGET = 1 << 22

# Thin-asymmetric:  min(n, k) <= THIN_DIM_LIMIT  ->  dense
THIN_DIM_LIMIT = 512

# Hard cap on dense materialisation (1 GB FP32 by default)
DENSE_SAFETY_CAP = 1 << 28

# Batch chunk size for the scan path.  Above this we slice along the
# leading batch dimension to bound the (B, n) / (B, k) working set.
B_CHUNK = 4096


def _should_use_dense(n: int, k: int,
                      dense_budget: Optional[int] = None,
                      thin_dim_limit: Optional[int] = None,
                      dense_safety_cap: Optional[int] = None) -> bool:
    """Mirror of ``cuda._should_use_dense``: same two-rule + safety logic."""
    budget = GPU_MEM_BUDGET    if dense_budget     is None else dense_budget
    thin   = THIN_DIM_LIMIT    if thin_dim_limit   is None else thin_dim_limit
    cap    = DENSE_SAFETY_CAP  if dense_safety_cap is None else dense_safety_cap
    nk = n * k
    if cap is not None and nk > cap:
        return False
    if nk <= budget:
        return True
    if thin is not None and min(n, k) <= thin:
        return True
    return False


# =============================================================================
# Internal helpers
# =============================================================================
def _safe_complex_log(x: Tensor) -> Tensor:
    """``log x`` returned as a complex tensor (works for real or complex `x`).

    For real `x`, the sign is encoded in the imaginary part as 0 (positive)
    or pi (negative).  A small epsilon floor on ``|x|`` keeps the log finite
    when `x` is exactly zero.
    """
    if x.is_complex():
        eps = torch.finfo(x.real.dtype).eps
        mag = x.abs().clamp(min=eps)
        return torch.log(mag) + 1j * torch.angle(x)
    eps = torch.finfo(x.dtype).eps
    log_mag = torch.log(x.abs().clamp(min=eps))
    phase = torch.where(x < 0, torch.full_like(x, math.pi), torch.zeros_like(x))
    return torch.complex(log_mag, phase)


def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"unsupported real dtype {real_dtype}")


# =============================================================================
# Stable prefix / suffix scan -- the only primitive we need
# =============================================================================
def prefix(alpha: Tensor, x: Tensor) -> Tensor:
    """Stable left-cumulative weighted sum:

        y[..., i] = sum_{j <= i} exp(alpha[j] - alpha[i]) * x[..., j].

    One ``torch.logcumsumexp`` over a complex log; differentiable; works for
    real or complex `x`; output has the same shape and realness as `x`.
    """
    alpha_b = alpha.view(*([1] * (x.dim() - 1)), -1)
    log_x = _safe_complex_log(x)                            # complex
    log_s = torch.logcumsumexp(alpha_b + log_x, dim=-1)     # complex
    if x.is_complex():
        return torch.exp(-alpha_b + log_s)
    # Re(exp(-alpha + log_s)) = exp(-alpha + Re(log_s)) * cos(Im(log_s))
    return torch.exp(-alpha_b + log_s.real) * torch.cos(log_s.imag)


def suffix(alpha: Tensor, x: Tensor) -> Tensor:
    """Stable right-cumulative weighted sum:

        y[..., i] = sum_{j >= i} exp(alpha[i] - alpha[j]) * x[..., j].

    Reduces to ``prefix`` on flipped inputs.
    """
    return prefix(-alpha.flip(0), x.flip(-1)).flip(-1)


def prefix_suffix_sep(alpha: Tensor, u: Tensor, v: Optional[Tensor] = None):
    """Convenience: returns ``(prefix(alpha, u), suffix(alpha, v))``.
    If `v` is omitted, ``v = u``.
    """
    if v is None:
        v = u
    return prefix(alpha, u), suffix(alpha, v)


# =============================================================================
# Fast multiplication on the REAL Laplace kernel
# =============================================================================
def _X_Laplex_scan_branch(X: Tensor, a: Tensor, b: Tensor,
                          a_sorted: bool, b_sorted: bool,
                          real_dtype: torch.dtype) -> Tensor:
    """Scan path for a single (B, n) slab.  Caller handles B-chunking."""
    B, n = X.shape
    k = b.numel()

    # --- branch A: n <= k.  Scan along a (shorter axis), gather per b. -------
    if n <= k:
        if a_sorted:
            a_s, X_s = a, X
        else:
            a_s, perm = torch.sort(a)
            X_s = X[:, perm]

        P, S = prefix_suffix_sep(a_s, X_s)                  # (B, n) each

        c = torch.searchsorted(a_s, b, right=True) - 1      # (k,) in [-1, n-1]
        c_l = c.clamp(min=0)
        w_l = torch.exp(a_s[c_l] - b) * (c >= 0).to(real_dtype)
        c_r = (c + 1).clamp(max=n - 1)
        w_r = torch.exp(b - a_s[c_r]) * ((c + 1) < n).to(real_dtype)

        return w_l.unsqueeze(0) * P.index_select(1, c_l) \
             + w_r.unsqueeze(0) * S.index_select(1, c_r)

    # --- branch B: n > k.  Scatter from a into k buckets along b, scan k. ----
    if b_sorted:
        b_s, inv_perm_b = b, None
    else:
        b_s, perm_b = torch.sort(b)
        inv_perm_b = torch.argsort(perm_b)

    idx = torch.searchsorted(b_s, a, right=True) - 1        # (n,)

    i_u = idx.clamp(min=0)
    vals_u = torch.exp(b_s[i_u] - a) * (idx >= 0).to(real_dtype)
    u = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_u.unsqueeze(0).expand(B, -1), X * vals_u.unsqueeze(0))

    i_v = (idx + 1).clamp(max=k - 1)
    vals_v = torch.exp(a - b_s[i_v]) * ((idx + 1) < k).to(real_dtype)
    v = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_v.unsqueeze(0).expand(B, -1), X * vals_v.unsqueeze(0))

    y = prefix(b_s, v) + suffix(b_s, u)
    if inv_perm_b is not None:
        y = y[:, inv_perm_b]
    return y


def X_Laplex(X: Tensor, a: Tensor, b: Tensor,
             threshold: Optional[int] = None,
             a_sorted: bool = False, b_sorted: bool = False,
             dense_budget: Optional[int] = None,
             thin_dim_limit: Optional[int] = None,
             dense_safety_cap: Optional[int] = None,
             B_chunk: Optional[int] = None) -> Tensor:
    """``X @ K(a, b)`` where ``K[i, j] = exp(-|a[i] - b[j]|)``.

    X : (B, n) or (n,)        (real or complex)
    a : (n,)        b : (k,)
    Returns (B, k) or (k,) with the same dtype as `X`.

    Dispatch logic mirrors the CUDA backend: dense matmul for
    small-symmetric (``n*k <= GPU_MEM_BUDGET``) or thin-asymmetric
    (``min(n, k) <= THIN_DIM_LIMIT``) shapes, scan path otherwise.  The
    legacy ``threshold`` kwarg (from the original pure-PyTorch reference)
    is honoured as an alias for ``thin_dim_limit`` when set.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)
    B, n = X.shape
    k = b.numel()
    real_dtype = X.real.dtype if X.is_complex() else X.dtype

    # Legacy `threshold` kwarg: map to thin_dim_limit if not overridden.
    if threshold is not None and thin_dim_limit is None:
        thin_dim_limit = threshold

    # --- Dense path: small-symmetric OR thin-asymmetric (with safety cap) ---
    if _should_use_dense(n, k, dense_budget, thin_dim_limit, dense_safety_cap):
        K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        y = X @ K.to(X.dtype)
        return y.squeeze(0) if squeeze else y

    # --- Scan path with B-chunking for large batches ---
    chunk = B_CHUNK if B_chunk is None else B_chunk
    if chunk is None or B <= chunk:
        y = _X_Laplex_scan_branch(X, a, b, a_sorted, b_sorted, real_dtype)
    else:
        # Process B in slabs of `chunk` to bound the (chunk, n) / (chunk, k)
        # working set.  Concatenate at the end.
        out_chunks = []
        for s in range(0, B, chunk):
            X_chunk = X[s:s + chunk]
            out_chunks.append(_X_Laplex_scan_branch(
                X_chunk, a, b, a_sorted, b_sorted, real_dtype))
        y = torch.cat(out_chunks, dim=0)

    return y.squeeze(0) if squeeze else y


def Laplex_X(X: Tensor, a: Tensor, b: Tensor,
             threshold: Optional[int] = None,
             a_sorted: bool = False, b_sorted: bool = False,
             dense_budget: Optional[int] = None,
             thin_dim_limit: Optional[int] = None,
             dense_safety_cap: Optional[int] = None,
             B_chunk: Optional[int] = None) -> Tensor:
    """``K(a, b) @ X``.  Implemented via the transpose identity
    ``K(a,b) X = (X^T K(b,a))^T``.

    X : (k, D) or (k,)     ->     (n, D) or (n,).
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(1)
    out = X_Laplex(X.T, b, a,
                   threshold=threshold,
                   a_sorted=b_sorted, b_sorted=a_sorted,
                   dense_budget=dense_budget,
                   thin_dim_limit=thin_dim_limit,
                   dense_safety_cap=dense_safety_cap,
                   B_chunk=B_chunk).T
    return out.squeeze(1) if squeeze else out


def Laplex_D_LaplexT(alpha: Tensor, beta: Tensor, D: Tensor,
                     threshold: Optional[int] = None,
                     alpha_sorted: bool = False,
                     dense_budget: Optional[int] = None,
                     thin_dim_limit: Optional[int] = None,
                     dense_safety_cap: Optional[int] = None) -> Tensor:
    """``K(alpha, beta) @ diag(D) @ K(alpha, beta)^T``.

    Same dispatch as ``X_Laplex``.  Accepts real or complex ``D``;  the
    result inherits ``D``'s dtype.

    Batched D
    ---------
    ``D`` may be 1-D ``(k,)`` -> result ``(n, n)``, or 2-D ``(B, k)`` ->
    result ``(B, n, n)`` (B different right-hand sides sharing the SAME
    ``(alpha, beta)``).  The sort, bucketize and prefix/suffix scan on
    ``2*alpha`` are computed once and broadcast across the batch -- only
    the three segment sums and the final ``(B, n, n)`` assembly carry
    the B factor.
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

    device = alpha.device
    out_dtype = D.dtype

    if threshold is not None and thin_dim_limit is None:
        thin_dim_limit = threshold

    # ── Dense path ────────────────────────────────────────────────────
    if _should_use_dense(n, k, dense_budget, thin_dim_limit, dense_safety_cap):
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        K_c = K.to(out_dtype)
        if batched:
            return torch.einsum('ic,bc,lc->bil', K_c, D, K_c)
        return (K_c * D.unsqueeze(0)) @ K_c.T

    # ── Scan path (with optional batched D) ───────────────────────────
    if alpha_sorted:
        alpha_s, inv_perm = alpha, None
    else:
        alpha_s, perm = torch.sort(alpha)
        inv_perm = torch.argsort(perm)

    idx = torch.bucketize(beta, alpha_s, right=True)        # (k,)

    alpha_ext = torch.cat([beta.min().unsqueeze(0), alpha_s, beta.max().unsqueeze(0)])
    alpha_ip1 = alpha_ext[idx + 1]
    u_exp = torch.exp(2.0 * (beta - alpha_ip1)).to(out_dtype)
    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_exp = torch.exp(2.0 * (alpha_i - beta)).to(out_dtype)

    D_2d = D.unsqueeze(0) if not batched else D
    B_eff = D_2d.size(0)

    u_vals = D_2d * u_exp.unsqueeze(0)              # (B_eff, k)
    v_vals = D_2d * v_exp.unsqueeze(0)              # (B_eff, k)
    idx_e  = idx.unsqueeze(0).expand(B_eff, -1)

    u = torch.zeros(B_eff, n + 1, device=device, dtype=out_dtype) \
              .scatter_add(-1, idx_e, u_vals)
    v = torch.zeros(B_eff, n + 1, device=device, dtype=out_dtype) \
              .scatter_add(-1, idx_e, v_vals)
    w = torch.zeros(B_eff, n + 1, device=device, dtype=out_dtype) \
              .scatter_add(-1, idx_e, D_2d)

    alpha2 = 2.0 * alpha_s
    U = prefix(alpha2, u[:, :n])                    # (B_eff, n)
    V = suffix(alpha2, v[:, 1:n + 1])               # (B_eff, n)
    W = torch.cumsum(w[:, :n], dim=-1)              # (B_eff, n)

    ai = alpha_s.unsqueeze(1)
    aj = alpha_s.unsqueeze(0)
    diff = torch.exp(-torch.relu(ai - aj)).to(out_dtype)   # (n, n)

    M_raw = diff.unsqueeze(0) * (
        U.unsqueeze(-2)                                  # (B, 1, n)
        + (W.unsqueeze(-1) - W.unsqueeze(-2))            # (B, n, n)
        + V.unsqueeze(-1)                                # (B, n, 1)
    )                                                    # (B, n, n)
    M = torch.tril(M_raw) + torch.tril(M_raw, -1).transpose(-1, -2)

    if inv_perm is not None:
        M = M.index_select(-2, inv_perm).index_select(-1, inv_perm)

    return M if batched else M.squeeze(0)


# =============================================================================
# Fast multiplication on the PHASED Laplace kernel
#     A_phase[i, j] = exp(-|alpha_i - beta_j| / tau) * cos(gamma_i - delta_j)
# =============================================================================
def X_LaplexPhase(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: Optional[int] = None,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None,
                  dense_budget: Optional[int] = None,
                  thin_dim_limit: Optional[int] = None,
                  dense_safety_cap: Optional[int] = None,
                  B_chunk: Optional[int] = None) -> Tensor:
    """``X @ A_phase``.

    X : (B, n) or (n,)         (real)
    alpha, gamma : (n,)
    beta,  delta : (k,)
    tau   : optional positive scalar; alpha and beta are divided by it
    Returns (B, k) or (k,) with the same real dtype as `X`.

    Uses one complex ``X_Laplex``: cast `X` to complex, multiply row-wise by
    ``exp(i gamma)``, scan, multiply column-wise by ``exp(-i delta)``, take Re.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)

    cdtype = _complex_dtype_for(X.dtype)

    if tau is not None:
        tau_t = torch.as_tensor(tau, device=alpha.device, dtype=alpha.dtype)
        alpha = alpha / tau_t;  beta = beta / tau_t

    phase_in = torch.polar(torch.ones_like(gamma), gamma).to(cdtype)    # (n,)
    X_c = X.to(cdtype) * phase_in.unsqueeze(0)

    Y_c = X_Laplex(X_c, alpha, beta,
                   threshold=threshold,
                   a_sorted=alpha_sorted, b_sorted=beta_sorted,
                   dense_budget=dense_budget,
                   thin_dim_limit=thin_dim_limit,
                   dense_safety_cap=dense_safety_cap,
                   B_chunk=B_chunk)

    phase_out = torch.polar(torch.ones_like(delta), -delta).to(cdtype)  # (k,)
    Y = (Y_c * phase_out.unsqueeze(0)).real

    return Y.squeeze(0) if squeeze else Y


def LaplexPhase_X(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: Optional[int] = None,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None,
                  dense_budget: Optional[int] = None,
                  thin_dim_limit: Optional[int] = None,
                  dense_safety_cap: Optional[int] = None,
                  B_chunk: Optional[int] = None) -> Tensor:
    """``A_phase @ X``.

    X : (k, B) or (k,)     ->     (n, B) or (n,).
    Implemented via the transpose identity:
        A_phase @ X = (X^T @ A_phase^T)^T,
    and ``A_phase^T`` equals ``A_phase`` with ``(alpha, gamma)`` and
    ``(beta, delta)`` swapped.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(1)
    Y = X_LaplexPhase(X.T, beta, alpha, delta, gamma,
                      threshold=threshold,
                      alpha_sorted=beta_sorted, beta_sorted=alpha_sorted,
                      tau=tau,
                      dense_budget=dense_budget,
                      thin_dim_limit=thin_dim_limit,
                      dense_safety_cap=dense_safety_cap,
                      B_chunk=B_chunk).T
    return Y.squeeze(1) if squeeze else Y


def LaplexPhase_D_LaplexPhaseT(alpha: Tensor, beta: Tensor,
                               gamma: Tensor, delta: Tensor, D: Tensor,
                               threshold: Optional[int] = None,
                               alpha_sorted: bool = False,
                               tau: Optional[float] = None,
                               dense_budget: Optional[int] = None,
                               thin_dim_limit: Optional[int] = None,
                               dense_safety_cap: Optional[int] = None) -> Tensor:
    """``A_phase @ diag(D) @ A_phase^T``  ->  shape (n, n)."""
    real_dtype = alpha.dtype
    cdtype = _complex_dtype_for(real_dtype)
    n = alpha.numel()
    k = beta.numel()

    if tau is not None:
        tau_t = torch.as_tensor(tau, device=alpha.device, dtype=real_dtype)
        alpha_s, beta_s = alpha / tau_t, beta / tau_t
    else:
        alpha_s, beta_s = alpha, beta

    if threshold is not None and thin_dim_limit is None:
        thin_dim_limit = threshold

    if _should_use_dense(n, k, dense_budget, thin_dim_limit, dense_safety_cap):
        K = torch.exp(-(alpha_s.unsqueeze(1) - beta_s.unsqueeze(0)).abs())
        A = K * torch.cos(gamma.unsqueeze(1) - delta.unsqueeze(0))
        return (A * D.unsqueeze(0)) @ A.T

    G0 = Laplex_D_LaplexT(alpha_s, beta_s, D,
                          alpha_sorted=alpha_sorted,
                          dense_budget=dense_budget,
                          thin_dim_limit=thin_dim_limit,
                          dense_safety_cap=dense_safety_cap)

    D_complex = D.to(cdtype) * torch.polar(torch.ones_like(delta),
                                           -2.0 * delta).to(cdtype)
    Gc = Laplex_D_LaplexT(alpha_s, beta_s, D_complex,
                          alpha_sorted=alpha_sorted,
                          dense_budget=dense_budget,
                          thin_dim_limit=thin_dim_limit,
                          dense_safety_cap=dense_safety_cap)

    gi = gamma.unsqueeze(1)
    gj = gamma.unsqueeze(0)
    cos_diff = torch.cos(gi - gj)
    cos_sum  = torch.cos(gi + gj)
    sin_sum  = torch.sin(gi + gj)

    re_T1 = cos_sum * Gc.real - sin_sum * Gc.imag
    return 0.5 * (cos_diff * G0 + re_T1)


def LaplexPhaseT_D_LaplexPhase(alpha: Tensor, beta: Tensor,
                               gamma: Tensor, delta: Tensor, D: Tensor,
                               threshold: Optional[int] = None,
                               beta_sorted: bool = False,
                               tau: Optional[float] = None,
                               dense_budget: Optional[int] = None,
                               thin_dim_limit: Optional[int] = None,
                               dense_safety_cap: Optional[int] = None) -> Tensor:
    """``A_phase^T @ diag(D) @ A_phase``  ->  shape (k, k).
    Same construction with the two sides swapped.
    """
    return LaplexPhase_D_LaplexPhaseT(beta, alpha, delta, gamma, D,
                                      threshold=threshold,
                                      alpha_sorted=beta_sorted, tau=tau,
                                      dense_budget=dense_budget,
                                      thin_dim_limit=thin_dim_limit,
                                      dense_safety_cap=dense_safety_cap)


# =============================================================================
# Self-test  (run: python pure.py)
# =============================================================================
def _smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"pure.py self-test on device='{device}'")
    print("=" * 60)

    passed = failed = 0

    def chk(name, got, ref, tol):
        nonlocal passed, failed
        err = (got.detach().double() - ref.detach().double()).abs().max().item() / \
              (ref.detach().double().abs().max().item() + 1e-30)
        ok = err < tol
        passed += ok; failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:50s} err={err:.2e}")

    # ---- X_Laplex / Laplex_X ----
    print("\n--- X_Laplex / Laplex_X ---")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100)]:
        a = torch.randn(n, device=device)
        b = torch.randn(k, device=device)
        X = torch.randn(B, n, device=device)
        Kfull = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        chk(f"X_Laplex B={B}, n={n}, k={k}",
            X_Laplex(X, a, b, thin_dim_limit=0, dense_budget=0), X @ Kfull, 1e-4)

        Y = torch.randn(k, B, device=device)
        chk(f"Laplex_X B={B}, n={n}, k={k}",
            Laplex_X(Y, a, b, thin_dim_limit=0, dense_budget=0), Kfull @ Y, 1e-4)

    # ---- B-chunking sanity: scan-path output equals dense @ K ----
    print("\n--- B-chunking (large B) ---")
    a = torch.randn(800, device=device)
    b = torch.randn(1200, device=device)
    X = torch.randn(9000, 800, device=device)
    Kfull = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
    # Force scan path AND a chunked B sweep:
    chk("X_Laplex B=9000 (chunked, scan)",
        X_Laplex(X, a, b, thin_dim_limit=0, dense_budget=0, B_chunk=2048),
        X @ Kfull, 1e-4)

    # ---- Laplex_D_LaplexT ----
    print("\n--- Laplex_D_LaplexT ---")
    for n, k in [(50, 100), (200, 300)]:
        alpha = torch.randn(n, device=device)
        beta  = torch.randn(k, device=device)
        D     = torch.randn(k, device=device)
        Kfull = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        chk(f"real D, n={n}, k={k}",
            Laplex_D_LaplexT(alpha, beta, D,
                             thin_dim_limit=0, dense_budget=0),
            (Kfull * D.unsqueeze(0)) @ Kfull.T, 1e-3)

    # ---- Batched D: D shape (B, k) -> M shape (B, n, n) ----
    print("\n--- Laplex_D_LaplexT (batched D) ---")
    for n, k, B in [(50, 100, 4), (200, 300, 8)]:
        alpha = torch.randn(n, device=device)
        beta  = torch.randn(k, device=device)
        Db    = torch.randn(B, k, device=device)
        Kfull = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        M_ref = torch.einsum('ic,bc,lc->bil', Kfull, Db, Kfull)
        chk(f"batched D n={n}, k={k}, B={B}",
            Laplex_D_LaplexT(alpha, beta, Db,
                             thin_dim_limit=0, dense_budget=0), M_ref, 1e-3)

    # ---- gradient sanity ----
    print("\n--- gradient sanity ---")
    a = torch.randn(40, device=device, requires_grad=True)
    b = torch.randn(60, device=device, requires_grad=True)
    X = torch.randn(2, 40, device=device, requires_grad=True)
    y = X_Laplex(X, a, b, thin_dim_limit=0, dense_budget=0)
    y.sum().backward()
    ok = a.grad is not None and b.grad is not None and X.grad is not None
    passed += ok; failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  X_Laplex backward populates a.grad, b.grad, X.grad")

    print(f"\n{'='*60}\nResult: {passed} PASS, {failed} FAIL\n{'='*60}")
    return failed == 0


if __name__ == "__main__":
    _smoke_test()
