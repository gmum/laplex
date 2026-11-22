"""
kerlap_ops.py

Fast CUDA implementations of three operations on the kernel matrix
K(a,b)[i,j] = exp(-|a[i] - b[j]|):

  X_KerLap(X, a, b)              = X @ K(a,b)
  KerLap_X(X, a, b)              = K(a,b) @ X
  Ker_D_KerT(alpha, beta, D)     = K(alpha,beta) @ diag(D) @ K(alpha,beta)^T

Built on the DL prefix/suffix scan of prefix_suffix_dl.py.
For a scan dimension <= 1024 it falls back to a matrix product.
All functions are differentiable (autograd).

COMPLEXITY
---------
Naive: O(n*k) (building the matrix K)
DL:     O(n log n + k)  or  O(k log k + n)  (scan over the shorter dimension)

Ker_D_KerT: O(n^2 + k log n) instead of O(n*k + n^2*k)
"""

import os, sys, math, time
import torch
from torch.autograd import Function

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_suffix_dl import (
    _precompute,
    _ps_raw,
    _pad_pair,
    prefix_suffix as _prefix_suffix_sum,
)


# ============================================================
# Core autograd: prefix and suffix SEPARATELY (different u, v)
# ============================================================
class _PrefSuffDiffFn(Function):
    """
    Returns (prefix(alpha, u), suffix(alpha, v)) for (possibly) different u, v.

    Forward:  1x _ps_raw  (2 DL scans)
    Backward: 1x _ps_raw  (2 DL scans) + elementwise ops
    """

    @staticmethod
    def forward(ctx, alpha, u, v):
        orig_dtype = alpha.dtype
        a32 = alpha.float()
        u32 = u.float()
        v32 = v.float()
        ef, n_fwd, ef_neg, n_bwd = _precompute(a32)
        P, S = _ps_raw(ef, n_fwd, ef_neg, n_bwd, u32, v32)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u32, v32, P, S)
        ctx.orig_dtype = orig_dtype
        return P.to(orig_dtype), S.to(orig_dtype)

    @staticmethod
    def backward(ctx, gP, gS):
        ef, n_fwd, ef_neg, n_bwd, u, v, P, S = ctx.saved_tensors
        od = ctx.orig_dtype
        gP32 = gP.float()
        gS32 = gS.float()

        # prefix(a, gS)  and  suffix(a, gP)  -- one _ps_raw
        prefix_gS, suffix_gP = _ps_raw(ef, n_fwd, ef_neg, n_bwd, gS32, gP32)

        grad_u = suffix_gP.to(od) if ctx.needs_input_grad[1] else None
        grad_v = prefix_gS.to(od) if ctx.needs_input_grad[2] else None

        grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_alpha = (
                (u * suffix_gP - v * prefix_gS - gP32 * P + gS32 * S).sum(0).to(od)
            )

        return grad_alpha, grad_u, grad_v


def prefix_suffix_sep(alpha, u, v=None):
    """
    Returns (prefix(alpha, u), suffix(alpha, v)).
    If v is None, v = u.
    alpha: [T], u: [B, T] or [T], v: [B, T] or [T].
    """
    if v is None:
        v = u
    squeeze = u.dim() == 1
    if squeeze:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0) if v is not u else u  # avoid a double unsqueeze
    P, S = _PrefSuffDiffFn.apply(alpha, u, v)
    if squeeze:
        P = P.squeeze(0)
        S = S.squeeze(0)
    return P, S


# ============================================================
# X_KerLap  =  X @ K(a, b)
# ============================================================
def X_KerLap(X, a, b, threshold=1024):
    """
    Computes X @ K(a,b)  where K[i,j] = exp(-|a[i] - b[j]|).

    X : (B, n) or (n,)
    a : (n,)
    b : (k,)
    threshold : for min(n,k) <= threshold a matrix product is used
    returns : (B, k) or (k,)
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)

    n = a.numel()
    k = b.numel()

    # Fallback: direct matrix product
    if min(n, k) <= threshold:
        K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())  # (n, k)
        y = X @ K
        return y.squeeze(0) if squeeze else y

    # DL scan -- scan over the shorter dimension
    if n <= k:
        y = _xA(X, a, b)
    else:
        y = _xAT(X, a, b)

    return y.squeeze(0) if squeeze else y


def _xA(X, a, b):
    """X @ K(a,b) -- sort a (n elements), scan over n, gather for the k outputs."""
    B, n = X.shape
    k = b.numel()

    a_s, perm = torch.sort(a)
    X_s = X[:, perm]

    # prefix and suffix separately
    P, S = prefix_suffix_sep(a_s, X_s)

    # For every b[j]: c = max i such that a_s[i] <= b[j]
    c = torch.searchsorted(a_s, b, right=True) - 1  # (k,), values in [-1, n-1]

    # Left: exp(a_s[c] - b) * P[:, c]   (valid when c >= 0)
    c_l = c.clamp(min=0)
    gate_l = (c >= 0).to(X.dtype)
    w_l = torch.exp(a_s[c_l] - b) * gate_l  # (k,)

    # Right: exp(b - a_s[c+1]) * S[:, c+1]   (valid when c+1 < n)
    c_r = (c + 1).clamp(max=n - 1)
    gate_r = ((c + 1) < n).to(X.dtype)
    w_r = torch.exp(b - a_s[c_r]) * gate_r  # (k,)

    y = w_l.unsqueeze(0) * P[:, c_l] + w_r.unsqueeze(0) * S[:, c_r]
    return y


def _xAT(X, a, b):
    """X @ K(a,b) when n > k -- sort b (k elements), scatter, scan over k."""
    B, n = X.shape
    k = b.numel()

    b_s, perm_b = torch.sort(b)
    inv_perm_b = torch.argsort(perm_b)

    # For every a[i], find its bucket in the sorted b
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

    # prefix(b_s, v) + suffix(b_s, u) -- uses the existing differentiable API
    result_sorted = _prefix_suffix_sum(b_s, v, u)

    return result_sorted[:, inv_perm_b]


# ============================================================
# KerLap_X  =  K(a, b) @ X
# ============================================================
def KerLap_X(X, a, b, threshold=1024):
    """
    Computes K(a,b) @ X  where K[i,j] = exp(-|a[i] - b[j]|).

    X : (k, D) or (k,)
    a : (n,)
    b : (k,)
    threshold : for min(n,k) <= threshold a matrix product is used
    returns : (n, D) or (n,)
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(1)

    # K(a,b) @ X = (X^T @ K(b,a))^T
    result = X_KerLap(X.T, b, a, threshold).T

    return result.squeeze(1) if squeeze else result


# ============================================================
# Ker_D_KerT  =  K(alpha,beta) @ diag(D) @ K(alpha,beta)^T
# ============================================================
def Ker_D_KerT(alpha, beta, D, threshold=1024):
    """
    Computes K(alpha,beta) @ diag(D) @ K(alpha,beta)^T.
    Result: a symmetric (n, n) matrix.

    alpha : (n,)
    beta  : (k,)
    D     : (k,)
    threshold : for n <= threshold the naive product is used
    returns : (n, n)
    """
    n = alpha.numel()
    k = beta.numel()

    # Fallback: direct product
    if n <= threshold:
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())  # (n, k)
        return (K * D.unsqueeze(0)) @ K.T

    return _Ker_D_KerT_scan(alpha, beta, D)


def _Ker_D_KerT_scan(alpha, beta, D):
    """Scan-based Ker_D_KerT for n > threshold."""
    n = alpha.numel()
    k = beta.numel()
    device = alpha.device
    dtype = alpha.dtype

    # Sort alpha
    alpha_s, perm = torch.sort(alpha)
    inv_perm = torch.argsort(perm)

    # Bucket beta into the alpha intervals
    idx = torch.bucketize(beta, alpha_s, right=True)  # (k,) values in [0, n]

    # alpha_ext: sentinels at both ends
    alpha_ext = torch.cat(
        [
            beta.min().unsqueeze(0),
            alpha_s,
            beta.max().unsqueeze(0),
        ]
    )  # (n+2,)

    # --- u: segment_sum( D * exp(2*(beta - alpha_{i+1})), idx, n+1 ) ---
    alpha_ip1 = alpha_ext[idx + 1]
    u_vals = D * torch.exp(2.0 * (beta - alpha_ip1))
    u = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, u_vals)

    # --- v: segment_sum( D * exp(2*(alpha_i - beta)), idx, n+1 ) ---
    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_vals = D * torch.exp(2.0 * (alpha_i - beta))
    v = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, v_vals)

    # --- w: segment_sum( D, idx, n+1 ) ---
    w = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, D)

    # --- U = prefix(2*alpha_s, u[:n]),  V = suffix(2*alpha_s, v[1:n+1]) ---
    alpha2 = 2.0 * alpha_s
    U, V = prefix_suffix_sep(alpha2, u[:n].unsqueeze(0), v[1 : n + 1].unsqueeze(0))
    U = U.squeeze(0)  # (n,)
    V = V.squeeze(0)  # (n,)

    # --- W = cumsum(w[:n]) ---
    W = torch.cumsum(w[:n], dim=0)

    # --- Build the matrix M ---
    ai = alpha_s.unsqueeze(1)  # (n, 1)
    aj = alpha_s.unsqueeze(0)  # (1, n)

    # diff[i,j] = exp(-relu(ai - aj)):  exp(aj-ai) for i>=j, 1 for i<j
    diff = torch.exp(-torch.relu(ai - aj))

    M_raw = diff * (U.unsqueeze(0) + (W.unsqueeze(1) - W.unsqueeze(0)) + V.unsqueeze(1))

    # Symmetrization: M = tril(M_raw) + tril(M_raw, -1)^T
    M = torch.tril(M_raw) + torch.tril(M_raw, -1).T

    # Unsort
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

    # ---- X_KerLap ----
    print("=== X_KerLap ===")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100), (2, 30, 50)]:
        a = torch.sort(torch.randn(n, device=device))[0]
        b = torch.randn(k, device=device)
        X = torch.randn(B, n, device=device)
        # Force scan path (threshold=0)
        got = X_KerLap(X, a, b, threshold=0)
        ref = _naive_X_KerLap(X, a, b)
        check(f"B={B}, n={n}, k={k}", got, ref)

    # 1D test
    a = torch.sort(torch.randn(80, device=device))[0]
    b = torch.randn(60, device=device)
    x1d = torch.randn(80, device=device)
    check(
        "1D input",
        X_KerLap(x1d, a, b, threshold=0),
        _naive_X_KerLap(x1d.unsqueeze(0), a, b).squeeze(0),
    )

    # ---- KerLap_X ----
    print("\n=== KerLap_X ===")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100)]:
        a = torch.sort(torch.randn(n, device=device))[0]
        b = torch.randn(k, device=device)
        X = torch.randn(k, B, device=device)
        got = KerLap_X(X, a, b, threshold=0)
        ref = _naive_KerLap_X(X, a, b)
        check(f"B={B}, n={n}, k={k}", got, ref)

    # 1D test
    a1 = torch.sort(torch.randn(80, device=device))[0]
    b1 = torch.randn(60, device=device)
    x1d = torch.randn(60, device=device)  # size = k = b1.numel()
    check(
        "1D input",
        KerLap_X(x1d, a1, b1, threshold=0),
        _naive_KerLap_X(x1d.unsqueeze(1), a1, b1).squeeze(1),
    )

    # ---- Ker_D_KerT ----
    print("\n=== Ker_D_KerT ===")
    for n, k in [(20, 30), (50, 100), (100, 200), (200, 300)]:
        alpha = torch.sort(torch.randn(n, device=device))[0]
        beta = torch.randn(k, device=device)
        D = torch.randn(k, device=device)
        got = Ker_D_KerT(alpha, beta, D, threshold=0)
        ref = _naive_Ker_D_KerT(alpha, beta, D)
        check(f"n={n}, k={k}", got, ref, tol=5e-3)

    # Unsorted alpha
    alpha_unsorted = torch.randn(60, device=device)
    beta = torch.randn(100, device=device)
    D = torch.randn(100, device=device)
    got = Ker_D_KerT(alpha_unsorted, beta, D, threshold=0)
    ref = _naive_Ker_D_KerT(alpha_unsorted, beta, D)
    check("unsorted alpha, n=60, k=100", got, ref, tol=5e-3)

    print(f"\nPoprawnosc: {passed} PASS, {failed} FAIL\n")
    return failed == 0


# ============================================================
# Gradient tests
# ============================================================
def run_gradient_tests():
    device = "cuda"
    torch.manual_seed(123)
    passed = failed = 0

    def gtest(name, fn, args, tol_atol=5e-2, tol_rtol=5e-2):
        nonlocal passed, failed
        ok = torch.autograd.gradcheck(
            fn,
            args,
            eps=1e-3,
            atol=tol_atol,
            rtol=tol_rtol,
            raise_exception=False,
        )
        passed += ok
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    # --- prefix_suffix_sep gradcheck ---
    print("=== prefix_suffix_sep gradcheck ===")
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
            P, S = _PrefSuffDiffFn.apply(a, u, v)
            return P + S  # reduced to a scalar

        gtest(f"T={T}", fn_sep, (a, u, v))

    # --- X_KerLap gradient compared with naive ---
    print("\n=== X_KerLap gradient vs naive ===")
    for n, k in [(20, 30), (30, 20)]:
        a = torch.sort(torch.randn(n, device=device))[0].requires_grad_(True)
        b = torch.randn(k, device=device).requires_grad_(True)
        X = torch.randn(4, n, device=device).requires_grad_(True)

        # DL
        y_dl = X_KerLap(X, a, b, threshold=0)
        y_dl.sum().backward()
        ga_dl, gb_dl, gx_dl = a.grad.clone(), b.grad.clone(), X.grad.clone()
        a.grad.zero_()
        b.grad.zero_()
        X.grad.zero_()

        # Naive
        y_nv = _naive_X_KerLap(X, a, b)
        y_nv.sum().backward()
        ga_nv, gb_nv, gx_nv = a.grad.clone(), b.grad.clone(), X.grad.clone()

        def re(t1, t2):
            return ((t1 - t2).abs() / t2.abs().clamp(min=1e-6)).max().item()

        ea, eb, ex = re(ga_dl, ga_nv), re(gb_dl, gb_nv), re(gx_dl, gx_nv)
        ok = ea < 0.02 and eb < 0.02 and ex < 0.02
        passed += ok
        failed += not ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  n={n},k={k}  ea={ea:.2e} eb={eb:.2e} ex={ex:.2e}"
        )

    # --- KerLap_X gradient ---
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

    # --- Ker_D_KerT gradient ---
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
        tol = 0.05
        ok = ea < tol and eb < tol and ed < tol
        passed += ok
        failed += not ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  n={n},k={k}  ea={ea:.2e} eb={eb:.2e} ed={ed:.2e}"
        )

    print(f"\nGradient: {passed} PASS, {failed} FAIL\n")
    return failed == 0


# ============================================================
# Benchmarks
# ============================================================
def run_benchmarks():
    device = "cuda"
    torch.manual_seed(0)

    def bench_fn(fn, warmup=10, iters=50):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        mem = torch.cuda.max_memory_allocated() / 1e6
        return ms, mem

    def bench_fb(fn_create, warmup=10, iters=50):
        """Forward + backward benchmark."""

        def step():
            args = fn_create()
            y = args[0](*args[1:])
            if y.dim() > 1:
                y.sum().backward()
            else:
                y.sum().backward()

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        mem = torch.cuda.max_memory_allocated() / 1e6
        return ms, mem

    # ================================================================
    print("=" * 80)
    print("=== X_KerLap / KerLap_X  benchmarks ===")
    print("=" * 80)

    hdr = (
        f"  {'Config':30s}  {'naive fwd':>9s}  {'DL fwd':>8s}"
        f"  {'naive f+b':>9s}  {'DL f+b':>8s}"
        f"  {'speedup':>8s}  {'DL mem':>8s}"
    )
    sep = "  " + "-" * 95

    # --- Scaling in T (n=k, B=4) ---
    print("\n--- X_KerLap: scaling in T (n=k, B=4) ---")
    print(hdr)
    print(sep)
    B = 4
    for logT in range(10, 21):
        T = 1 << logT
        a = torch.sort(torch.randn(T, device=device))[0]
        b = torch.randn(T, device=device)
        X = torch.randn(B, T, device=device)

        # DL first (so that naive does not pollute the GPU cache)
        t_dl, m_dl = bench_fn(lambda: X_KerLap(X, a, b, threshold=0))

        def mk_dl():
            a_ = a.detach().requires_grad_(True)
            b_ = b.detach().requires_grad_(True)
            x_ = X.detach().requires_grad_(True)
            return lambda x, a, b: X_KerLap(x, a, b, threshold=0), x_, a_, b_

        t_dl_fb, m_dl_fb = bench_fb(mk_dl)

        # Naive after DL (allocates a T*T matrix, so only for small T)
        t_nv, _ = (
            bench_fn(lambda: _naive_X_KerLap(X, a, b))
            if T <= 2**14
            else (float("nan"), 0)
        )

        def mk_naive():
            a_ = a.detach().requires_grad_(True)
            b_ = b.detach().requires_grad_(True)
            x_ = X.detach().requires_grad_(True)
            return _naive_X_KerLap, x_, a_, b_

        t_nv_fb, _ = bench_fb(mk_naive) if T <= 2**14 else (float("nan"), 0)

        spd = t_nv_fb / t_dl_fb if not math.isnan(t_nv_fb) else float("nan")

        desc = f"B={B}, T=2^{logT:2d}={T:>8d}"
        print(
            f"  {desc:30s}  {t_nv:7.2f}ms  {t_dl:6.2f}ms"
            f"  {t_nv_fb:7.2f}ms  {t_dl_fb:6.2f}ms"
            f"  {spd:7.1f}x  {m_dl_fb:6.0f}MB"
        )

    # --- Scaling with n != k ---
    print("\n--- X_KerLap: n != k (B=4) ---")
    print(hdr)
    print(sep)
    configs = [
        (4, 1024, 1 << 18, "n=1K, k=256K (xA path)"),
        (4, 1 << 18, 1024, "n=256K, k=1K (xAT path)"),
        (4, 1 << 16, 1 << 16, "n=k=64K"),
        (4, 1 << 14, 1 << 20, "n=16K, k=1M"),
    ]
    for B_, n_, k_, desc in configs:
        a = torch.sort(torch.randn(n_, device=device))[0]
        b = torch.randn(k_, device=device)
        X = torch.randn(B_, n_, device=device)

        t_dl, m_dl = bench_fn(lambda: X_KerLap(X, a, b, threshold=0))

        def mk_dl2():
            a_ = a.detach().requires_grad_(True)
            b_ = b.detach().requires_grad_(True)
            x_ = X.detach().requires_grad_(True)
            return lambda x, a, b: X_KerLap(x, a, b, threshold=0), x_, a_, b_

        t_dl_fb, m_dl_fb = bench_fb(mk_dl2)

        print(
            f"  {desc:30s}  {'--':>9s}  {t_dl:6.2f}ms"
            f"  {'--':>9s}  {t_dl_fb:6.2f}ms"
            f"  {'--':>8s}  {m_dl_fb:6.0f}MB"
        )

    # ================================================================
    print("\n" + "=" * 80)
    print("=== Ker_D_KerT  benchmarks ===")
    print("=" * 80)

    hdr2 = (
        f"  {'Config':30s}  {'naive fwd':>9s}  {'DL fwd':>8s}"
        f"  {'naive f+b':>9s}  {'DL f+b':>8s}"
        f"  {'speedup':>8s}  {'DL mem':>8s}"
    )
    print(hdr2)
    print(sep)

    for n_, k_ in [
        (256, 1 << 14),
        (256, 1 << 16),
        (256, 1 << 18),
        (1024, 1 << 14),
        (1024, 1 << 16),
        (1024, 1 << 18),
        (4096, 1 << 16),
        (4096, 1 << 18),
        (4096, 1 << 20),
    ]:
        alpha = torch.sort(torch.randn(n_, device=device))[0]
        beta = torch.randn(k_, device=device)
        Dv = torch.randn(k_, device=device)

        can_naive = n_ * k_ <= 2**26

        t_nv, _ = (
            bench_fn(lambda: _naive_Ker_D_KerT(alpha, beta, Dv))
            if can_naive
            else (float("nan"), 0)
        )
        t_dl, m_dl = bench_fn(lambda: Ker_D_KerT(alpha, beta, Dv, threshold=0))

        def mk_naive_kdk():
            a_ = alpha.detach().requires_grad_(True)
            b_ = beta.detach().requires_grad_(True)
            d_ = Dv.detach().requires_grad_(True)
            return _naive_Ker_D_KerT, a_, b_, d_

        def mk_dl_kdk():
            a_ = alpha.detach().requires_grad_(True)
            b_ = beta.detach().requires_grad_(True)
            d_ = Dv.detach().requires_grad_(True)
            return lambda a, b, d: Ker_D_KerT(a, b, d, threshold=0), a_, b_, d_

        t_nv_fb, _ = bench_fb(mk_naive_kdk) if can_naive else (float("nan"), 0)
        t_dl_fb, m_dl_fb = bench_fb(mk_dl_kdk)

        spd = (
            t_nv_fb / t_dl_fb if can_naive and not math.isnan(t_nv_fb) else float("nan")
        )

        desc = f"n={n_:5d}, k={k_:>8d}"
        print(
            f"  {desc:30s}  {t_nv:7.2f}ms  {t_dl:6.2f}ms"
            f"  {t_nv_fb:7.2f}ms  {t_dl_fb:6.2f}ms"
            f"  {spd:7.1f}x  {m_dl_fb:6.0f}MB"
        )

    print()


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    run_correctness_tests()

    print("=" * 60)
    print("GRADIENT TESTS")
    print("=" * 60)
    run_gradient_tests()

    print("=" * 60)
    print("BENCHMARKI")
    print("=" * 60)
    run_benchmarks()
