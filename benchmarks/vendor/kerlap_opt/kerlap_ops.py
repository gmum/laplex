"""
kerlap_ops.py
=============

Fast CUDA implementations of operations on the Laplace kernel matrix

    K(a, b)[i, j] = exp(-|a[i] - b[j]|)

    X_KerLap(X, a, b)            =  X @ K(a, b)
    KerLap_X(X, a, b)            =  K(a, b) @ X
    Ker_D_KerT(alpha, beta, D)   =  K(alpha, beta) @ diag(D) @ K(alpha, beta)^T

Built on the differentiable DL prefix/suffix of prefix_suffix_dl.py.
For a scan dimension <= threshold it falls back to a plain matrix product.

COMPLEXITY (compared with the naive O(n*k) or O(n^2*k))
    X @ K / K @ X :   O(n log n + k)   (scan over the shorter dimension)
    Ker_D_KerT    :   O(n^2 + k log n) instead of O(n*k + n^2*k)
"""

import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_suffix_dl import (
    prefix_suffix as _prefix_suffix_sum,
    prefix_suffix_sep,
    _GatherCombineFn,
)


# ============================================================
# X_KerLap  =  X @ K(a, b)
# ============================================================
def X_KerLap(X, a, b, threshold=1024, a_sorted=False, b_sorted=False):
    """
    Computes X @ K(a, b) where K[i, j] = exp(-|a[i] - b[j]|).

    X : (B, n) or (n,)
    a : (n,)
    b : (k,)
    threshold : for min(n, k) <= threshold a matrix product is used
    a_sorted  : True if a is already sorted ascending (skips the sort)
    b_sorted  : True if b is already sorted ascending
    returns : (B, k) or (k,)
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)

    n = a.numel()
    k = b.numel()

    # Fallback: naive product (faster for small n, k)
    if min(n, k) <= threshold:
        K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())  # (n, k)
        y = X @ K
        return y.squeeze(0) if squeeze else y

    # DL scan over the shorter dimension
    if n <= k:
        y = _xA(X, a, b, a_sorted=a_sorted)
    else:
        y = _xAT(X, a, b, b_sorted=b_sorted)

    return y.squeeze(0) if squeeze else y


def _xA(X, a, b, a_sorted=False):
    """
    X @ K(a, b) when n <= k: scan along the (shorter) a axis, then for
    every b[j] combine the values of its two neighbours in the sorted a.
    """
    B, n = X.shape
    k = b.numel()

    if a_sorted:
        a_s = a
        X_s = X
    else:
        a_s, perm = torch.sort(a)
        X_s = X[:, perm]

    # prefix(a_s, X_s) i suffix(a_s, X_s) (te same u, v = X_s)
    P, S = prefix_suffix_sep(a_s, X_s)

    # For every b[j]: c = max i such that a_s[i] <= b[j] (or -1 if b[j] < a_s[0])
    c = torch.searchsorted(a_s, b, right=True) - 1  # (k,) w [-1, n-1]

    # Left end: exp(a_s[c] - b) * P[:, c]   (only when c >= 0)
    c_l = c.clamp(min=0)
    gate_l = (c >= 0).to(X.dtype)
    w_l = torch.exp(a_s[c_l] - b) * gate_l  # (k,)

    # Right end: exp(b - a_s[c+1]) * S[:, c+1]   (only when c+1 < n)
    c_r = (c + 1).clamp(max=n - 1)
    gate_r = ((c + 1) < n).to(X.dtype)
    w_r = torch.exp(b - a_s[c_r]) * gate_r  # (k,)

    return _GatherCombineFn.apply(P, S, w_l, w_r, c_l, c_r)


def _xAT(X, a, b, b_sorted=False):
    """
    X @ K(a, b) when n > k: instead of scanning over n, scatter the n elements
    into k buckets (of the sorted b), then scan over k, with no gather.
    """
    B, n = X.shape
    k = b.numel()

    if b_sorted:
        b_s = b
        inv_perm_b = None
    else:
        b_s, perm_b = torch.sort(b)
        inv_perm_b = torch.argsort(perm_b)

    # For every a[i]: its bucket in the sorted b (as in _xA, with the roles swapped)
    idx = torch.searchsorted(b_s, a, right=True) - 1  # (n,)

    # Left scatter: u[c] += X[:, i] * exp(b_s[c] - a[i])
    i_u = idx.clamp(min=0)
    gate_u = (idx >= 0).to(X.dtype)
    vals_u = torch.exp(b_s[i_u] - a) * gate_u  # (n,)
    u = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_u.unsqueeze(0).expand(B, -1), X * vals_u.unsqueeze(0)
    )

    # Right scatter: v[c+1] += X[:, i] * exp(a[i] - b_s[c+1])
    i_v = (idx + 1).clamp(max=k - 1)
    gate_v = ((idx + 1) < k).to(X.dtype)
    vals_v = torch.exp(a - b_s[i_v]) * gate_v  # (n,)
    v = torch.zeros(B, k, device=X.device, dtype=X.dtype).scatter_add(
        1, i_v.unsqueeze(0).expand(B, -1), X * vals_v.unsqueeze(0)
    )

    # prefix(b_s, v) + suffix(b_s, u) -> (B, k)
    result_sorted = _prefix_suffix_sum(b_s, v, u)

    if inv_perm_b is not None:
        return result_sorted[:, inv_perm_b]
    return result_sorted


# ============================================================
# KerLap_X  =  K(a, b) @ X  (implemented through the transpose)
# ============================================================
def KerLap_X(X, a, b, threshold=1024, a_sorted=False, b_sorted=False):
    """
    Computes K(a, b) @ X  where K[i, j] = exp(-|a[i] - b[j]|).

    X : (k, D) or (k,)
    a : (n,)
    b : (k,)
    returns : (n, D) or (n,)
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(1)

    # K(a, b) @ X = (X^T @ K(b, a))^T
    result = X_KerLap(X.T, b, a, threshold, a_sorted=b_sorted, b_sorted=a_sorted).T

    return result.squeeze(1) if squeeze else result


# ============================================================
# Ker_D_KerT  =  K(alpha, beta) @ diag(D) @ K(alpha, beta)^T
# ============================================================
def Ker_D_KerT(alpha, beta, D, threshold=1024, alpha_sorted=False):
    """
    Computes K(alpha, beta) @ diag(D) @ K(alpha, beta)^T.  Result: (n, n), symmetric.

    alpha : (n,)
    beta  : (k,)
    D     : (k,)
    threshold    : for n <= threshold the naive product is used
    alpha_sorted : True if alpha is already sorted ascending
    """
    n = alpha.numel()
    k = beta.numel()

    if n <= threshold:
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())  # (n, k)
        return (K * D.unsqueeze(0)) @ K.T

    return _Ker_D_KerT_scan(alpha, beta, D, alpha_sorted=alpha_sorted)


def _Ker_D_KerT_scan(alpha, beta, D, alpha_sorted=False):
    """
    Scan-based O(n^2 + k log n) construction of the matrix M[i,j] = sum_c D[c] K[i,c] K[j,c].

    Idea: for every beta[c] find its bucket idx[c] in the sorted alpha.
    Then M[i,j] = diff[i,j] * (U[j] + (W[i]-W[j]) + V[i]), where diff, U, V, W
    come from three segment sums + a prefix/suffix scan over alpha.
    """
    n = alpha.numel()
    k = beta.numel()
    device = alpha.device
    dtype = alpha.dtype

    if alpha_sorted:
        alpha_s = alpha
        inv_perm = None
    else:
        alpha_s, perm = torch.sort(alpha)
        inv_perm = torch.argsort(perm)

    # Bucket of every beta[c] in [0, n]
    idx = torch.bucketize(beta, alpha_s, right=True)  # (k,)

    # alpha_ext: sentinels at both ends (so that idx=0 and idx=n work)
    alpha_ext = torch.cat(
        [
            beta.min().unsqueeze(0),
            alpha_s,
            beta.max().unsqueeze(0),
        ]
    )  # (n+2,)

    # u[i] = sum_{c: idx[c]=i} D[c] * exp(2*(beta[c] - alpha_{i+1}))
    alpha_ip1 = alpha_ext[idx + 1]
    u_vals = D * torch.exp(2.0 * (beta - alpha_ip1))
    u = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, u_vals)

    # v[i] = sum_{c: idx[c]=i} D[c] * exp(2*(alpha_{i-1} - beta[c]))
    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_vals = D * torch.exp(2.0 * (alpha_i - beta))
    v = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, v_vals)

    # w[i] = sum_{c: idx[c]=i} D[c]
    w = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, D)

    # U = prefix(2*alpha_s, u[:n]),  V = suffix(2*alpha_s, v[1:n+1])
    # The slice v[1:n+1] has a non-zero storage offset -> .clone() forces
    # the 16-byte alignment needed by the float4 loads in the kernel.
    alpha2 = 2.0 * alpha_s
    U, V = prefix_suffix_sep(
        alpha2, u[:n].clone().unsqueeze(0), v[1 : n + 1].clone().unsqueeze(0)
    )
    U = U.squeeze(0)
    V = V.squeeze(0)

    # W = cumsum(w[:n])
    W = torch.cumsum(w[:n], dim=0)

    # Build (n, n): diff[i,j] = exp(-relu(alpha_i - alpha_j))
    ai = alpha_s.unsqueeze(1)
    aj = alpha_s.unsqueeze(0)
    diff = torch.exp(-torch.relu(ai - aj))

    M_raw = diff * (U.unsqueeze(0) + (W.unsqueeze(1) - W.unsqueeze(0)) + V.unsqueeze(1))

    # Symmetrization: M = tril(M_raw) + tril(M_raw, -1)^T
    M = torch.tril(M_raw) + torch.tril(M_raw, -1).T

    if inv_perm is not None:
        M = M[inv_perm][:, inv_perm]
    return M


# ============================================================
# Naive references (for the tests)
# ============================================================
def _naive_X_KerLap(X, a, b):
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
    return X @ K


def _naive_KerLap_X(X, a, b):
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
    return K @ X


def _naive_Ker_D_KerT(alpha, beta, D):
    K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
    return (K * D.unsqueeze(0)) @ K.T


# ============================================================
# Correctness tests
# ============================================================
def run_correctness_tests():
    device = "cuda"
    torch.manual_seed(42)
    passed = failed = 0

    def check(name, got, ref, tol=2e-3):
        nonlocal passed, failed
        err = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
        ok = err < tol
        passed += ok
        failed += not ok
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {name:40s}  err={err:.2e}")
        return ok

    print("=== X_KerLap ===")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100), (2, 30, 50)]:
        a = torch.sort(torch.randn(n, device=device))[0]
        b = torch.randn(k, device=device)
        X = torch.randn(B, n, device=device)
        got = X_KerLap(X, a, b, threshold=0)
        ref = _naive_X_KerLap(X, a, b)
        check(f"B={B}, n={n}, k={k}", got, ref)

    a = torch.sort(torch.randn(80, device=device))[0]
    b = torch.randn(60, device=device)
    x1d = torch.randn(80, device=device)
    check(
        "1D input",
        X_KerLap(x1d, a, b, threshold=0),
        _naive_X_KerLap(x1d.unsqueeze(0), a, b).squeeze(0),
    )

    print("\n=== KerLap_X ===")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100)]:
        a = torch.sort(torch.randn(n, device=device))[0]
        b = torch.randn(k, device=device)
        X = torch.randn(k, B, device=device)
        got = KerLap_X(X, a, b, threshold=0)
        ref = _naive_KerLap_X(X, a, b)
        check(f"B={B}, n={n}, k={k}", got, ref)

    a1 = torch.sort(torch.randn(80, device=device))[0]
    b1 = torch.randn(60, device=device)
    x1d = torch.randn(60, device=device)
    check(
        "1D input",
        KerLap_X(x1d, a1, b1, threshold=0),
        _naive_KerLap_X(x1d.unsqueeze(1), a1, b1).squeeze(1),
    )

    print("\n=== Ker_D_KerT ===")
    for n, k in [(20, 30), (50, 100), (100, 200), (200, 300)]:
        alpha = torch.sort(torch.randn(n, device=device))[0]
        beta = torch.randn(k, device=device)
        D = torch.randn(k, device=device)
        got = Ker_D_KerT(alpha, beta, D, threshold=0)
        ref = _naive_Ker_D_KerT(alpha, beta, D)
        check(f"n={n}, k={k}", got, ref, tol=5e-3)

    alpha_unsorted = torch.randn(60, device=device)
    beta = torch.randn(100, device=device)
    D = torch.randn(100, device=device)
    got = Ker_D_KerT(alpha_unsorted, beta, D, threshold=0)
    ref = _naive_Ker_D_KerT(alpha_unsorted, beta, D)
    check("unsorted alpha, n=60, k=100", got, ref, tol=5e-3)

    print(f"\nPoprawnosc: {passed} PASS, {failed} FAIL\n")
    return failed == 0


def run_gradient_tests():
    device = "cuda"
    torch.manual_seed(123)
    passed = failed = 0

    def re(t1, t2):
        return ((t1 - t2).abs() / t2.abs().clamp(min=1e-6)).max().item()

    print("=== prefix_suffix_sep gradcheck (fp64) ===")
    for T in [8, 16, 32]:
        a = torch.linspace(0, 2, T, dtype=torch.float64, device=device).requires_grad_(
            True
        )
        u = (
            0.3 * torch.randn(2, T, dtype=torch.float64, device=device)
        ).requires_grad_(True)
        v = (
            0.3 * torch.randn(2, T, dtype=torch.float64, device=device)
        ).requires_grad_(True)

        def fn_sep(a, u, v):
            P, S = prefix_suffix_sep(a, u, v)
            return P + S

        ok = torch.autograd.gradcheck(
            fn_sep, (a, u, v), eps=1e-3, atol=5e-2, rtol=5e-2, raise_exception=False
        )
        passed += ok
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  T={T}")

    print("\n=== X_KerLap gradient vs naive ===")
    for n, k in [(20, 30), (30, 20)]:
        a = torch.sort(torch.randn(n, device=device))[0].requires_grad_(True)
        b = torch.randn(k, device=device).requires_grad_(True)
        X = torch.randn(4, n, device=device).requires_grad_(True)

        y_dl = X_KerLap(X, a, b, threshold=0)
        y_dl.sum().backward()
        ga_dl, gb_dl, gx_dl = a.grad.clone(), b.grad.clone(), X.grad.clone()
        a.grad.zero_()
        b.grad.zero_()
        X.grad.zero_()

        y_nv = _naive_X_KerLap(X, a, b)
        y_nv.sum().backward()
        ga_nv, gb_nv, gx_nv = a.grad.clone(), b.grad.clone(), X.grad.clone()

        ea, eb, ex = re(ga_dl, ga_nv), re(gb_dl, gb_nv), re(gx_dl, gx_nv)
        ok = ea < 0.02 and eb < 0.02 and ex < 0.02
        passed += ok
        failed += not ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  n={n},k={k}  ea={ea:.2e} eb={eb:.2e} ex={ex:.2e}"
        )

    print("\n=== KerLap_X gradient vs naive ===")
    n, k, D_ = 30, 40, 8
    a = torch.sort(torch.randn(n, device=device))[0].requires_grad_(True)
    b = torch.randn(k, device=device).requires_grad_(True)
    X = torch.randn(k, D_, device=device).requires_grad_(True)

    y_dl = KerLap_X(X, a, b, threshold=0)
    y_dl.sum().backward()
    ga_dl, gb_dl, gx_dl = a.grad.clone(), b.grad.clone(), X.grad.clone()
    a.grad.zero_()
    b.grad.zero_()
    X.grad.zero_()

    y_nv = _naive_KerLap_X(X, a, b)
    y_nv.sum().backward()
    ga_nv, gb_nv, gx_nv = a.grad.clone(), b.grad.clone(), X.grad.clone()

    ea, eb, ex = re(ga_dl, ga_nv), re(gb_dl, gb_nv), re(gx_dl, gx_nv)
    ok = ea < 0.02 and eb < 0.02 and ex < 0.02
    passed += ok
    failed += not ok
    print(
        f"  {'PASS' if ok else 'FAIL'}  n={n},k={k},D={D_}  ea={ea:.2e} eb={eb:.2e} ex={ex:.2e}"
    )

    print("\n=== Ker_D_KerT gradient vs naive ===")
    for n, k in [(15, 25), (25, 40)]:
        alpha = torch.sort(torch.randn(n, device=device))[0].requires_grad_(True)
        beta = torch.randn(k, device=device).requires_grad_(True)
        Dv = torch.randn(k, device=device).requires_grad_(True)

        M_dl = Ker_D_KerT(alpha, beta, Dv, threshold=0)
        M_dl.sum().backward()
        ga_dl = alpha.grad.clone()
        gb_dl = beta.grad.clone()
        gd_dl = Dv.grad.clone()
        alpha.grad.zero_()
        beta.grad.zero_()
        Dv.grad.zero_()

        M_nv = _naive_Ker_D_KerT(alpha, beta, Dv)
        M_nv.sum().backward()
        ga_nv = alpha.grad.clone()
        gb_nv = beta.grad.clone()
        gd_nv = Dv.grad.clone()

        ea = re(ga_dl, ga_nv)
        eb = re(gb_dl, gb_nv)
        ed = re(gd_dl, gd_nv)
        ok = ea < 0.05 and eb < 0.05 and ed < 0.05
        passed += ok
        failed += not ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  n={n},k={k}  ea={ea:.2e} eb={eb:.2e} ed={ed:.2e}"
        )

    print(f"\nGradient: {passed} PASS, {failed} FAIL\n")
    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    run_correctness_tests()

    print("=" * 60)
    print("GRADIENT TESTS")
    print("=" * 60)
    run_gradient_tests()
