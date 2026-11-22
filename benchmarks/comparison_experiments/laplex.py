"""
laplex.py
=========

Optimised CUDA implementation of the three core operations on the real
Laplace kernel

    K(a, b)[i, j] = exp(-|a[i] - b[j]|),       a in R^n, b in R^k.

API matches the pure-PyTorch reference in `laplex_noCUDA.py` so that the
two are drop-in replacements for each other:

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
Self-test (correctness against the dense `(B, n) @ K` reference): run
    python laplex.py
"""
from __future__ import annotations

import os, sys
from typing import Optional

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_suffix import prefix_suffix_sep, prefix_suffix, _GatherCombineFn





def X_Laplex(X: Tensor, a: Tensor, b: Tensor,
             threshold: int = 1024,
             a_sorted: bool = False, b_sorted: bool = False) -> Tensor:
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

    Returns
    -------
    (B, k) or (k,) tensor in the promoted floating dtype (any fp64 input
    promotes everything to fp64; otherwise fp32).
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)
    B, n = X.shape
    k = b.numel()

    
    
    
    if n * k <= 2**20:
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

    
    P, S = prefix_suffix_sep(a_s, X_s)

    
    
    c = torch.searchsorted(a_s, b, right=True) - 1            

    
    c_l = c.clamp(min=0)
    gate_l = (c >= 0).to(X.dtype)
    w_l = torch.exp(a_s[c_l] - b) * gate_l                    

    
    c_r = (c + 1).clamp(max=n - 1)
    gate_r = ((c + 1) < n).to(X.dtype)
    w_r = torch.exp(b - a_s[c_r]) * gate_r                    

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

    
    idx = torch.searchsorted(b_s, a, right=True) - 1          

    
    i_u = idx.clamp(min=0)
    gate_u = (idx >= 0).to(X.dtype)
    vals_u = torch.exp(b_s[i_u] - a) * gate_u
    u = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_u.unsqueeze(0).expand(B, -1), X * vals_u.unsqueeze(0))

    
    i_v = (idx + 1).clamp(max=k - 1)
    gate_v = ((idx + 1) < k).to(X.dtype)
    vals_v = torch.exp(a - b_s[i_v]) * gate_v
    v = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_v.unsqueeze(0).expand(B, -1), X * vals_v.unsqueeze(0))

    
    result_sorted = prefix_suffix(b_s, v, u)
    if inv_perm_b is not None:
        return result_sorted[:, inv_perm_b]
    return result_sorted





def Laplex_X(X: Tensor, a: Tensor, b: Tensor,
             threshold: int = 1024,
             a_sorted: bool = False, b_sorted: bool = False) -> Tensor:
    """``K(a, b) @ X``.  Implemented via ``K(a, b) X = (X^T K(b, a))^T``.

    X : (k, D) or (k,)     ->     (n, D) or (n,)
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
    """``K(alpha, beta) @ diag(D) @ K(alpha, beta)^T``  ->  (n, n) symmetric.

    Cost ``O(n^2 + k log n)`` via three segment sums on `D` and one scan
    on `2 * alpha`, replacing the naive ``O(n^2 k)`` matmul.
    """
    n, k = alpha.numel(), beta.numel()

    if n * k <= 2**20:
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        return (K * D.unsqueeze(0)) @ K.T

    if alpha_sorted:
        alpha_s, inv_perm = alpha, None
    else:
        alpha_s, perm = torch.sort(alpha)
        inv_perm = torch.argsort(perm)

    device, dtype = alpha.device, alpha.dtype

    
    idx = torch.bucketize(beta, alpha_s, right=True)

    
    alpha_ext = torch.cat([beta.min().unsqueeze(0),
                           alpha_s,
                           beta.max().unsqueeze(0)])

    
    alpha_ip1 = alpha_ext[idx + 1]
    u_vals = D * torch.exp(2.0 * (beta - alpha_ip1))
    u = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, u_vals)

    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_vals = D * torch.exp(2.0 * (alpha_i - beta))
    v = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, v_vals)

    w = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, D)

    
    
    
    alpha2 = 2.0 * alpha_s
    U, V = prefix_suffix_sep(alpha2,
                             u[:n].clone().unsqueeze(0),
                             v[1:n + 1].clone().unsqueeze(0))
    U, V = U.squeeze(0), V.squeeze(0)

    
    W = torch.cumsum(w[:n], dim=0)

    
    ai = alpha_s.unsqueeze(1)
    aj = alpha_s.unsqueeze(0)
    diff = torch.exp(-torch.relu(ai - aj))                    

    M_raw = diff * (U.unsqueeze(0) + (W.unsqueeze(1) - W.unsqueeze(0)) + V.unsqueeze(1))

    
    M = torch.tril(M_raw) + torch.tril(M_raw, -1).T

    if inv_perm is not None:
        M = M[inv_perm][:, inv_perm]
    return M





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

    
    print("\n--- X_Laplex / Laplex_X (fp64) ---")
    for B, n, k in [(2, 80, 60), (4, 200, 100), (1, 150, 250)]:
        a = torch.randn(n, device=device, dtype=torch.float64)
        b = torch.randn(k, device=device, dtype=torch.float64)
        X = torch.randn(B, n, device=device, dtype=torch.float64)
        chk(f"X_Laplex fp64 B={B}, n={n}, k={k}",
            X_Laplex(X, a, b, threshold=0), X @ K_full(a, b), 1e-12)

    
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
