"""
laplex_noCUDA.py
================

Pure-PyTorch reference implementation of the LAPLEX operations on the
real and phased Laplace kernels.  Extracted from `laplex_pytorch.ipynb`
and `laplex_phase_pytorch.ipynb` and consolidated into one importable
module — no custom CUDA kernels, no `load_inline`, runs anywhere PyTorch
runs (CPU, CUDA, MPS, ROCm, ...).

API
---
Real Laplace kernel  K(a, b)[i, j] = exp(-|a_i - b_j|):

    X_Laplex(X, a, b)                  = X @ K(a, b)
    Laplex_X(X, a, b)                  = K(a, b) @ X
    Laplex_D_LaplexT(alpha, beta, D)   = K(alpha, beta) @ diag(D) @ K(alpha, beta)^T

Phased Laplace kernel  A_phase[i, j] = exp(-|alpha_i - beta_j| / tau)
                                       * cos(gamma_i - delta_j):

    X_LaplexPhase(X, alpha, beta, gamma, delta, tau=None)        = X @ A_phase
    LaplexPhase_X(X, alpha, beta, gamma, delta, tau=None)        = A_phase @ X
    LaplexPhase_D_LaplexPhaseT(alpha, beta, gamma, delta, D, tau=None)
                                                                  = A_phase @ diag(D) @ A_phase^T
    LaplexPhaseT_D_LaplexPhase(alpha, beta, gamma, delta, D, tau=None)
                                                                  = A_phase^T @ diag(D) @ A_phase

Stability trick
---------------
The exponential-weighted prefix scan
    y_i = sum_{j<=i} exp(alpha_j - alpha_i) * x_j
is rewritten as
    y_i = exp(-alpha_i + logcumsumexp_{j<=i}(alpha_j + log x_j)),
encoding the sign of `x` in the imaginary part of a complex logarithm
(`log|x| + i*pi*[x<0]`).  The scan then collapses to a single
`torch.logcumsumexp` over a complex tensor — parallel, differentiable,
numerically stable, and accepting genuine complex inputs for free.

Dtype support
-------------
Inputs may be float16 / bfloat16 / float32 / float64; complex64 /
complex128 are accepted by the underlying `prefix` / `suffix` and by the
phased operations.  The output dtype follows PyTorch's standard
promotion rules (i.e., the dtype of the matmul argument).

Self-test
---------
Run `python laplex_noCUDA.py`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor





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





def prefix(alpha: Tensor, x: Tensor) -> Tensor:
    """Stable left-cumulative weighted sum:

        y[..., i] = sum_{j <= i} exp(alpha[j] - alpha[i]) * x[..., j].

    One ``torch.logcumsumexp`` over a complex log; differentiable; works for
    real or complex `x`; output has the same shape and realness as `x`.
    """
    alpha_b = alpha.view(*([1] * (x.dim() - 1)), -1)
    log_x = _safe_complex_log(x)                            
    log_s = torch.logcumsumexp(alpha_b + log_x, dim=-1)     
    if x.is_complex():
        return torch.exp(-alpha_b + log_s)
    
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





def X_Laplex(X: Tensor, a: Tensor, b: Tensor,
             threshold: int = 1024,
             a_sorted: bool = False, b_sorted: bool = False) -> Tensor:
    """``X @ Laplex(a, b)`` where ``Laplex[i, j] = exp(-|a[i] - b[j]|)``.

    X : (B, n) or (n,)        (real or complex)
    a : (n,)        b : (k,)
    Returns (B, k) or (k,) with the same dtype as `X`.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)
    B, n = X.shape
    k = b.numel()
    real_dtype = X.real.dtype if X.is_complex() else X.dtype

    
    if min(n, k) <= threshold:
        K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        y = X @ K.to(X.dtype)

    
    elif n <= k:
        if a_sorted:
            a_s, X_s = a, X
        else:
            a_s, perm = torch.sort(a)
            X_s = X[:, perm]

        P, S = prefix_suffix_sep(a_s, X_s)                  

        c = torch.searchsorted(a_s, b, right=True) - 1      
        c_l = c.clamp(min=0)
        w_l = torch.exp(a_s[c_l] - b) * (c >= 0).to(real_dtype)
        c_r = (c + 1).clamp(max=n - 1)
        w_r = torch.exp(b - a_s[c_r]) * ((c + 1) < n).to(real_dtype)

        y = w_l.unsqueeze(0) * P.index_select(1, c_l) \
          + w_r.unsqueeze(0) * S.index_select(1, c_r)

    
    else:
        if b_sorted:
            b_s, inv_perm_b = b, None
        else:
            b_s, perm_b = torch.sort(b)
            inv_perm_b = torch.argsort(perm_b)

        idx = torch.searchsorted(b_s, a, right=True) - 1    

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

    return y.squeeze(0) if squeeze else y


def Laplex_X(X: Tensor, a: Tensor, b: Tensor,
             threshold: int = 1024,
             a_sorted: bool = False, b_sorted: bool = False) -> Tensor:
    """``Laplex(a, b) @ X``.  Implemented via the transpose identity
    ``Laplex(a,b) X = (X^T Laplex(b,a))^T``.

    X : (k, D) or (k,)     ->     (n, D) or (n,).
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(1)
    out = X_Laplex(X.T, b, a, threshold=threshold,
                   a_sorted=b_sorted, b_sorted=a_sorted).T
    return out.squeeze(1) if squeeze else out


def Laplex_D_LaplexT(alpha: Tensor, beta: Tensor, D: Tensor,
                     threshold: int = 1024,
                     alpha_sorted: bool = False) -> Tensor:
    """``Laplex(alpha, beta) @ diag(D) @ Laplex(alpha, beta)^T``  -> (n, n).
    Accepts real or complex `D` (the result has D's dtype).
    """
    n, k = alpha.numel(), beta.numel()
    device = alpha.device
    out_dtype = D.dtype

    if n <= threshold:
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        return (K.to(out_dtype) * D.unsqueeze(0)) @ K.to(out_dtype).T

    if alpha_sorted:
        alpha_s, inv_perm = alpha, None
    else:
        alpha_s, perm = torch.sort(alpha)
        inv_perm = torch.argsort(perm)

    idx = torch.bucketize(beta, alpha_s, right=True)        

    
    alpha_ext = torch.cat([beta.min().unsqueeze(0), alpha_s, beta.max().unsqueeze(0)])
    alpha_ip1 = alpha_ext[idx + 1]

    u_vals = D * torch.exp(2.0 * (beta - alpha_ip1)).to(out_dtype)
    u = torch.zeros(n + 1, device=device, dtype=out_dtype).scatter_add(0, idx, u_vals)

    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_vals = D * torch.exp(2.0 * (alpha_i - beta)).to(out_dtype)
    v = torch.zeros(n + 1, device=device, dtype=out_dtype).scatter_add(0, idx, v_vals)

    w = torch.zeros(n + 1, device=device, dtype=out_dtype).scatter_add(0, idx, D)

    alpha2 = 2.0 * alpha_s
    U = prefix(alpha2, u[:n].unsqueeze(0)).squeeze(0)
    V = suffix(alpha2, v[1:n + 1].unsqueeze(0)).squeeze(0)
    W = torch.cumsum(w[:n], dim=0)

    ai = alpha_s.unsqueeze(1)
    aj = alpha_s.unsqueeze(0)
    diff = torch.exp(-torch.relu(ai - aj)).to(out_dtype)

    M_raw = diff * (U.unsqueeze(0) + (W.unsqueeze(1) - W.unsqueeze(0)) + V.unsqueeze(1))
    M = torch.tril(M_raw) + torch.tril(M_raw, -1).T

    if inv_perm is not None:
        M = M[inv_perm][:, inv_perm]
    return M






def X_LaplexPhase(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: int = 1024,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None) -> Tensor:
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

    phase_in = torch.polar(torch.ones_like(gamma), gamma).to(cdtype)    
    X_c = X.to(cdtype) * phase_in.unsqueeze(0)

    Y_c = X_Laplex(X_c, alpha, beta,
                   threshold=threshold,
                   a_sorted=alpha_sorted, b_sorted=beta_sorted)

    phase_out = torch.polar(torch.ones_like(delta), -delta).to(cdtype)  
    Y = (Y_c * phase_out.unsqueeze(0)).real

    return Y.squeeze(0) if squeeze else Y


def LaplexPhase_X(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: int = 1024,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None) -> Tensor:
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
                      tau=tau).T
    return Y.squeeze(1) if squeeze else Y


def LaplexPhase_D_LaplexPhaseT(alpha: Tensor, beta: Tensor,
                               gamma: Tensor, delta: Tensor, D: Tensor,
                               threshold: int = 1024,
                               alpha_sorted: bool = False,
                               tau: Optional[float] = None) -> Tensor:
    """``A_phase @ diag(D) @ A_phase^T``  ->  shape (n, n).

    From  ``A = (Z + conj(Z)) / 2``,  with ``Z[i,j] = e^{i gamma_i} K[i,j]
    e^{-i delta_j}``, the Gram matrix decomposes as
        M = 0.5 * [ cos(gamma_i - gamma_j) * G_0
                  + Re( e^{i (gamma_i + gamma_j)} * G_c ) ],
    where  G_0 = Laplex diag(D) Laplex^T  (real)  and
           G_c = Laplex diag(D * e^{-2 i delta}) Laplex^T  (complex).
    """
    real_dtype = alpha.dtype
    cdtype = _complex_dtype_for(real_dtype)

    if tau is not None:
        tau_t = torch.as_tensor(tau, device=alpha.device, dtype=real_dtype)
        alpha_s, beta_s = alpha / tau_t, beta / tau_t
    else:
        alpha_s, beta_s = alpha, beta

    n = alpha.numel()

    if n <= threshold:
        K = torch.exp(-(alpha_s.unsqueeze(1) - beta_s.unsqueeze(0)).abs())
        A = K * torch.cos(gamma.unsqueeze(1) - delta.unsqueeze(0))
        return (A * D.unsqueeze(0)) @ A.T

    G0 = Laplex_D_LaplexT(alpha_s, beta_s, D,
                          threshold=threshold, alpha_sorted=alpha_sorted)

    D_complex = D.to(cdtype) * torch.polar(torch.ones_like(delta),
                                           -2.0 * delta).to(cdtype)
    Gc = Laplex_D_LaplexT(alpha_s, beta_s, D_complex,
                          threshold=threshold, alpha_sorted=alpha_sorted)

    gi = gamma.unsqueeze(1)
    gj = gamma.unsqueeze(0)
    cos_diff = torch.cos(gi - gj)
    cos_sum  = torch.cos(gi + gj)
    sin_sum  = torch.sin(gi + gj)

    re_T1 = cos_sum * Gc.real - sin_sum * Gc.imag
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





def _smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"laplex_noCUDA self-test on device='{device}'")
    print("=" * 60)

    passed = failed = 0

    def chk(name, got, ref, tol):
        nonlocal passed, failed
        err = (got.detach().double() - ref.detach().double()).abs().max().item() / \
              (ref.detach().double().abs().max().item() + 1e-30)
        ok = err < tol
        passed += ok; failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:50s} err={err:.2e}")

    
    print("\n--- prefix / suffix (real x) ---")
    n_, B_ = 200, 4
    alpha = torch.sort(torch.randn(n_, device=device))[0]
    x = torch.randn(B_, n_, device=device)
    diff = alpha.unsqueeze(1) - alpha.unsqueeze(0)
    K = torch.exp(-diff.abs())
    chk("prefix",  prefix(alpha, x),  x @ torch.triu(K), 1e-4)
    chk("suffix",  suffix(alpha, x),  x @ torch.tril(K), 1e-4)

    
    xc = torch.randn(B_, n_, device=device, dtype=torch.complex64)
    
    Kt = torch.triu(K).to(xc.dtype)
    chk("prefix (complex x)", prefix(alpha, xc), xc @ Kt, 1e-4)

    
    print("\n--- X_Laplex / Laplex_X ---")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100)]:
        a = torch.randn(n, device=device)
        b = torch.randn(k, device=device)
        X = torch.randn(B, n, device=device)
        Kfull = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        chk(f"X_Laplex B={B}, n={n}, k={k}",
            X_Laplex(X, a, b, threshold=0), X @ Kfull, 1e-4)

        Y = torch.randn(k, B, device=device)
        chk(f"Laplex_X B={B}, n={n}, k={k}",
            Laplex_X(Y, a, b, threshold=0), Kfull @ Y, 1e-4)

    
    print("\n--- Laplex_D_LaplexT ---")
    for n, k in [(50, 100), (200, 300)]:
        alpha = torch.randn(n, device=device)
        beta  = torch.randn(k, device=device)
        D     = torch.randn(k, device=device)
        Kfull = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        chk(f"real D, n={n}, k={k}",
            Laplex_D_LaplexT(alpha, beta, D, threshold=0),
            (Kfull * D.unsqueeze(0)) @ Kfull.T, 1e-3)

    Dc = torch.randn(120, device=device, dtype=torch.complex64)
    alpha = torch.randn(80, device=device); beta = torch.randn(120, device=device)
    Kfull = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs()).to(Dc.dtype)
    chk("complex D, n=80, k=120",
        Laplex_D_LaplexT(alpha, beta, Dc, threshold=0),
        (Kfull * Dc.unsqueeze(0)) @ Kfull.T, 1e-3)

    
    print("\n--- X_LaplexPhase / LaplexPhase_X ---")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100)]:
        a = torch.randn(n, device=device); b = torch.randn(k, device=device)
        g = torch.randn(n, device=device); d = torch.randn(k, device=device)
        X = torch.randn(B, n, device=device)
        K_ = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        A = K_ * torch.cos(g.unsqueeze(1) - d.unsqueeze(0))
        chk(f"X_LaplexPhase B={B}, n={n}, k={k}",
            X_LaplexPhase(X, a, b, g, d, threshold=0), X @ A, 1e-4)

        Y = torch.randn(k, B, device=device)
        chk(f"LaplexPhase_X B={B}, n={n}, k={k}",
            LaplexPhase_X(Y, a, b, g, d, threshold=0), A @ Y, 1e-4)

    
    a = torch.randn(80, device=device); b = torch.randn(60, device=device)
    g = torch.randn(80, device=device); d = torch.randn(60, device=device)
    X = torch.randn(3, 80, device=device); tau = 0.7
    K_ = torch.exp(-(a / tau).unsqueeze(1) - (b / tau).unsqueeze(0)).abs()
    K_ = torch.exp(-((a / tau).unsqueeze(1) - (b / tau).unsqueeze(0)).abs())
    A = K_ * torch.cos(g.unsqueeze(1) - d.unsqueeze(0))
    chk("X_LaplexPhase tau=0.7",
        X_LaplexPhase(X, a, b, g, d, tau=tau, threshold=0), X @ A, 1e-4)

    
    print("\n--- LaplexPhase_D_LaplexPhaseT ---")
    for n, k in [(50, 100), (200, 300)]:
        a = torch.randn(n, device=device); b = torch.randn(k, device=device)
        g = torch.randn(n, device=device); d = torch.randn(k, device=device)
        D = torch.randn(k, device=device)
        K_ = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        A = K_ * torch.cos(g.unsqueeze(1) - d.unsqueeze(0))
        chk(f"phase Gram n={n}, k={k}",
            LaplexPhase_D_LaplexPhaseT(a, b, g, d, D, threshold=0),
            (A * D.unsqueeze(0)) @ A.T, 5e-3)

    
    print("\n--- LaplexPhaseT_D_LaplexPhase ---")
    for n, k in [(40, 60), (100, 200)]:
        a = torch.randn(n, device=device); b = torch.randn(k, device=device)
        g = torch.randn(n, device=device); d = torch.randn(k, device=device)
        D = torch.randn(n, device=device)
        K_ = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        A = K_ * torch.cos(g.unsqueeze(1) - d.unsqueeze(0))
        chk(f"phase Gram^T n={n}, k={k}",
            LaplexPhaseT_D_LaplexPhase(a, b, g, d, D, threshold=0),
            (A.T * D.unsqueeze(0)) @ A, 5e-3)

    
    print("\n--- gradient sanity ---")
    a = torch.randn(40, device=device, requires_grad=True)
    b = torch.randn(60, device=device, requires_grad=True)
    X = torch.randn(2, 40, device=device, requires_grad=True)
    y = X_Laplex(X, a, b, threshold=0)
    y.sum().backward()
    ok = a.grad is not None and b.grad is not None and X.grad is not None
    passed += ok; failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  X_Laplex backward populates a.grad, b.grad, X.grad")

    print(f"\n{'='*60}\nWynik: {passed} PASS, {failed} FAIL\n{'='*60}")
    return failed == 0


if __name__ == "__main__":
    _smoke_test()
