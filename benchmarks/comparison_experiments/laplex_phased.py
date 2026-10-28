from __future__ import annotations

import os, sys
from typing import Optional

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laplex as _laplex   





def _scale(t: Optional[Tensor], tau, ref: Tensor):
    """Return t / tau if tau is not None; cast tau to ref.dtype if needed."""
    if tau is None:
        return t
    tau_t = torch.as_tensor(tau, device=ref.device, dtype=ref.dtype)
    return t / tau_t





def X_LaplexPhase(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: int = 1024,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None) -> Tensor:
    """``X @ A_phase``  with  A_phase[i,j] = exp(-|alpha_i-beta_j|/tau) * cos(gamma_i-delta_j).

    X : (B, n) lub (n,)  real
    alpha, gamma : (n,)
    beta,  delta : (k,)
    Returns (B, k) lub (k,)  same real dtype as X.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)
    B, n = X.shape
    k = beta.numel()

    alpha_s = _scale(alpha, tau, alpha)
    beta_s  = _scale(beta,  tau, alpha)

    cos_g = torch.cos(gamma);  sin_g = torch.sin(gamma)        
    cos_d = torch.cos(delta);  sin_d = torch.sin(delta)        

    
    X_cs = torch.cat([X * cos_g.unsqueeze(0),
                      X * sin_g.unsqueeze(0)], dim=0)          
    Y_cs = _laplex.X_Laplex(X_cs, alpha_s, beta_s,
                            threshold=threshold,
                            a_sorted=alpha_sorted, b_sorted=beta_sorted)  
    Y_real_part = Y_cs[:B]                                     
    Y_imag_part = Y_cs[B:]                                     

    
    Y = Y_real_part * cos_d.unsqueeze(0) + Y_imag_part * sin_d.unsqueeze(0)
    return Y.squeeze(0) if squeeze else Y


def LaplexPhase_X(X: Tensor, alpha: Tensor, beta: Tensor,
                  gamma: Tensor, delta: Tensor,
                  threshold: int = 1024,
                  alpha_sorted: bool = False, beta_sorted: bool = False,
                  tau: Optional[float] = None) -> Tensor:
    """``A_phase @ X``.

    X : (k, B) lub (k,)     ->     (n, B) lub (n,).
    Implemented via transpose identity:
        A_phase X = (X^T A_phase^T)^T,  A_phase^T = A_phase swap (alpha,gamma)<->(beta,delta).
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

    Trzy real Gram-scany:
        G0      = K diag(D)              K^T
        Gc_real = K diag(D cos(2 delta)) K^T
        Gc_imag = K diag(-D sin(2 delta)) K^T
    Re[e^{i(gamma_i+gamma_j)} (Gc_real + i Gc_imag)] = cos*Gcr - sin*Gci.
    """
    alpha_s = _scale(alpha, tau, alpha)
    beta_s  = _scale(beta,  tau, alpha)

    
    G0 = _laplex.Laplex_D_LaplexT(alpha_s, beta_s, D,
                                  threshold=threshold, alpha_sorted=alpha_sorted)

    
    cos_2d = torch.cos(2.0 * delta);  sin_2d = torch.sin(2.0 * delta)
    Gc_real = _laplex.Laplex_D_LaplexT(alpha_s, beta_s, D * cos_2d,
                                       threshold=threshold, alpha_sorted=alpha_sorted)
    Gc_imag = _laplex.Laplex_D_LaplexT(alpha_s, beta_s, -D * sin_2d,
                                       threshold=threshold, alpha_sorted=alpha_sorted)

    gi = gamma.unsqueeze(1)
    gj = gamma.unsqueeze(0)
    cos_diff = torch.cos(gi - gj)
    cos_sum  = torch.cos(gi + gj)
    sin_sum  = torch.sin(gi + gj)

    
    re_T1 = cos_sum * Gc_real - sin_sum * Gc_imag
    return 0.5 * (cos_diff * G0 + re_T1)


def LaplexPhaseT_D_LaplexPhase(alpha: Tensor, beta: Tensor,
                               gamma: Tensor, delta: Tensor, D: Tensor,
                               threshold: int = 1024,
                               beta_sorted: bool = False,
                               tau: Optional[float] = None) -> Tensor:
    """``A_phase^T @ diag(D) @ A_phase``  ->  shape (k, k).
    Symmetric counterpart: same construction with both sides swapped.
    """
    return LaplexPhase_D_LaplexPhaseT(beta, alpha, delta, gamma, D,
                                      threshold=threshold,
                                      alpha_sorted=beta_sorted, tau=tau)





def _smoke_test():
    assert torch.cuda.is_available(), "CUDA niedostepne"
    device = "cuda"
    torch.manual_seed(0)

    import laplex_noCUDA as _ref

    print("=" * 60)
    print("laplex_phased.py: smoke test (vs laplex_noCUDA reference)")
    print("=" * 60)

    passed = failed = 0

    def chk(name, got, ref, tol):
        nonlocal passed, failed
        ref64 = ref.detach().double()
        got64 = got.detach().double()
        err = (got64 - ref64).abs().max().item() / (ref64.abs().max().item() + 1e-30)
        ok = err < tol
        passed += ok; failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:55s} dtype={got.dtype} err={err:.2e}")

    
    print("\n--- X_LaplexPhase / LaplexPhase_X  (fp32) ---")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 300, 100)]:
        a = torch.randn(n, device=device, dtype=torch.float32)
        b = torch.randn(k, device=device, dtype=torch.float32)
        g = torch.randn(n, device=device, dtype=torch.float32)
        d = torch.randn(k, device=device, dtype=torch.float32)
        X = torch.randn(B, n, device=device, dtype=torch.float32)
        Y = torch.randn(k, B, device=device, dtype=torch.float32)
        chk(f"X_LaplexPhase B={B}, n={n}, k={k}",
            X_LaplexPhase(X, a, b, g, d, threshold=0),
            _ref.X_LaplexPhase(X, a, b, g, d, threshold=0), 5e-4)
        chk(f"LaplexPhase_X B={B}, n={n}, k={k}",
            LaplexPhase_X(Y, a, b, g, d, threshold=0),
            _ref.LaplexPhase_X(Y, a, b, g, d, threshold=0), 5e-4)

    
    a = torch.randn(80, device=device, dtype=torch.float32)
    b = torch.randn(60, device=device, dtype=torch.float32)
    g = torch.randn(80, device=device, dtype=torch.float32)
    d = torch.randn(60, device=device, dtype=torch.float32)
    X = torch.randn(3, 80, device=device, dtype=torch.float32)
    chk("X_LaplexPhase tau=0.7",
        X_LaplexPhase(X, a, b, g, d, tau=0.7, threshold=0),
        _ref.X_LaplexPhase(X, a, b, g, d, tau=0.7, threshold=0), 5e-4)

    
    print("\n--- fp64 ---")
    a = torch.randn(80, device=device, dtype=torch.float64)
    b = torch.randn(60, device=device, dtype=torch.float64)
    g = torch.randn(80, device=device, dtype=torch.float64)
    d = torch.randn(60, device=device, dtype=torch.float64)
    X = torch.randn(3, 80, device=device, dtype=torch.float64)
    chk("X_LaplexPhase fp64",
        X_LaplexPhase(X, a, b, g, d, threshold=0),
        _ref.X_LaplexPhase(X, a, b, g, d, threshold=0), 1e-10)

    
    print("\n--- LaplexPhase_D_LaplexPhaseT ---")
    for n, k in [(50, 100), (200, 300)]:
        a = torch.randn(n, device=device, dtype=torch.float32)
        b = torch.randn(k, device=device, dtype=torch.float32)
        g = torch.randn(n, device=device, dtype=torch.float32)
        d = torch.randn(k, device=device, dtype=torch.float32)
        D = torch.randn(k, device=device, dtype=torch.float32)
        chk(f"phase Gram fp32 n={n}, k={k}",
            LaplexPhase_D_LaplexPhaseT(a, b, g, d, D, threshold=0),
            _ref.LaplexPhase_D_LaplexPhaseT(a, b, g, d, D, threshold=0), 5e-3)

    
    a = torch.randn(120, device=device, dtype=torch.float64)
    b = torch.randn(180, device=device, dtype=torch.float64)
    g = torch.randn(120, device=device, dtype=torch.float64)
    d = torch.randn(180, device=device, dtype=torch.float64)
    D = torch.randn(180, device=device, dtype=torch.float64)
    chk("phase Gram fp64 tau=0.5 n=120, k=180",
        LaplexPhase_D_LaplexPhaseT(a, b, g, d, D, tau=0.5, threshold=0),
        _ref.LaplexPhase_D_LaplexPhaseT(a, b, g, d, D, tau=0.5, threshold=0), 1e-10)

    
    print("\n--- LaplexPhaseT_D_LaplexPhase ---")
    for n, k in [(40, 60), (100, 200)]:
        a = torch.randn(n, device=device, dtype=torch.float32)
        b = torch.randn(k, device=device, dtype=torch.float32)
        g = torch.randn(n, device=device, dtype=torch.float32)
        d = torch.randn(k, device=device, dtype=torch.float32)
        D = torch.randn(n, device=device, dtype=torch.float32)
        chk(f"phase Gram^T n={n}, k={k}",
            LaplexPhaseT_D_LaplexPhase(a, b, g, d, D, threshold=0),
            _ref.LaplexPhaseT_D_LaplexPhase(a, b, g, d, D, threshold=0), 5e-3)

    
    print("\n--- gradient sanity ---")
    a = torch.randn(40, device=device, requires_grad=True)
    b = torch.randn(60, device=device, requires_grad=True)
    g = torch.randn(40, device=device, requires_grad=True)
    d = torch.randn(60, device=device, requires_grad=True)
    X = torch.randn(2, 40, device=device, requires_grad=True)
    y = X_LaplexPhase(X, a, b, g, d, threshold=0)
    y.sum().backward()
    ok = all(t.grad is not None for t in (a, b, g, d, X))
    passed += ok; failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  X_LaplexPhase backward populates a,b,g,d,X grads")

    print(f"\n{'='*60}\nWynik: {passed} PASS, {failed} FAIL\n{'='*60}")
    return failed == 0


if __name__ == "__main__":
    _smoke_test()
