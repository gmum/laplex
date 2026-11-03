"""Correctness tests for GaussianBCCB (run before any big run).

Strategy: build the model's covariance *independently* from its
coloring map S (x = irfft(L_k rfft(e))), Sigma = S S^T, then check the model's
log-likelihood / logdet (which use a *different* code path: per-freq Cholesky solve
+ half-spectrum multiplicities) against a dense reference.  A wrong Hermitian
multiplicity or normalization makes these two paths disagree -> test fails.

Run:  python test_bccb.py
"""
from __future__ import annotations

import math

import torch

from _model_bccb import GaussianBCCB, mult_weights, self_conj_mask


def _randomize(model: GaussianBCCB, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))
        # keep diagonals in a sane positive-ish range
        if model.model_type == "scalar":
            model.theta.copy_(torch.randn(model.theta.shape, generator=g, dtype=torch.float64))
            model.sigma_raw.copy_(torch.tensor(-1.0, dtype=torch.float64))
        else:
            model.diag_raw.copy_(
                0.5 + 0.3 * torch.randn(model.diag_raw.shape, generator=g, dtype=torch.float64)
            )


def _dft1d(m: int) -> torch.Tensor:
    j = torch.arange(m, dtype=torch.float64)
    ang = -2.0 * math.pi * torch.outer(j, j) / m
    return torch.exp(1j * ang) / math.sqrt(m)     # unitary (m,m) complex128


def _rigorous_sigma(model: GaussianBCCB) -> torch.Tensor:
    """Sigma = sum_k B_full[k] (x) conj(f_k) f_k^T, exactly what quad/logdet define.

    Independent of rfft/irfft: explicit unitary 2D DFT + Hermitian conjugate
    expansion of the half-spectrum blocks to the full spectrum.
    """
    H, W, C = model.H, model.W, model.C
    HW = H * W
    Ffull = torch.kron(_dft1d(H), _dft1d(W))      # (HW,HW), row k=u*W+v, col p=h*W+w
    Bhalf = model._blocks()                        # (H,Wh,3,3) c128
    Sigma = torch.zeros(C * HW, C * HW, dtype=torch.complex128)
    for u in range(H):
        for v in range(W):
            if v <= W // 2:
                Bk = Bhalf[u, v]
            else:
                Bk = Bhalf[(-u) % H, W - v].conj()
            fk = Ffull[u * W + v]                   # (HW,)
            outer = torch.outer(fk.conj(), fk)     # (HW,HW): conj(f_p) f_p'
            Sigma += torch.kron(Bk, outer)
    imag = Sigma.imag.abs().max().item()
    assert imag < 1e-9, f"Sigma has imaginary part {imag:.2e}"
    return Sigma.real


def _materialize_sigma_via_sampler(model: GaussianBCCB) -> torch.Tensor:
    """Sigma = S S^T where S = model's coloring map (sampler path)."""
    n = model.n
    E = torch.eye(n, dtype=torch.float64).view(n, model.C, model.H, model.W)
    out = model._color(E).reshape(n, n)            # out[j] = S e_j = column j of S
    S = out.T.contiguous()
    return S @ S.T


def test_loglik_matches_dense(model_type: str, H=8, W=8, seed=0, tol=1e-6):
    torch.manual_seed(seed)
    m = 0.3 * torch.randn(3 * H * W, dtype=torch.float64)
    model = GaussianBCCB(H, W, model_type=model_type, mean_init=m)
    _randomize(model, seed)

    Sigma = _rigorous_sigma(model)

    # realness + PD
    assert torch.allclose(Sigma, Sigma.T, atol=1e-9), "Sigma not symmetric"
    evals = torch.linalg.eigvalsh(Sigma)
    assert evals.min() > 0, f"Sigma not PD (min eig {evals.min():.2e})"

    # logdet: model vs dense slogdet(Sigma_rigorous)
    err_ld = (model._logdet() - torch.linalg.slogdet(Sigma)[1]).abs().item()
    print(f"[{model_type}] logdet: model vs dense err = {err_ld:.3e}")
    assert err_ld < 1e-6, f"logdet mismatch {err_ld:.3e}"

    # reference LL via MultivariateNormal on the rigorous Sigma
    mvn = torch.distributions.MultivariateNormal(m, covariance_matrix=Sigma)
    xb = m.unsqueeze(0) + torch.randn(5, 3 * H * W, dtype=torch.float64) @ torch.linalg.cholesky(Sigma).T
    ll_ref = mvn.log_prob(xb)
    ll_model = model.log_likelihood(xb)
    err = (ll_ref - ll_model).abs().max().item()
    print(f"[{model_type}] LL vs dense MVN: max abs err = {err:.3e}  "
          f"(ref {ll_ref[0].item():.4f}, model {ll_model[0].item():.4f})")
    assert err < tol, f"LL mismatch {err:.3e} > {tol}"

    # sampler consistency: S S^T should equal the rigorous Sigma
    SST = _materialize_sigma_via_sampler(model)
    rel = (SST - Sigma).norm() / Sigma.norm()
    print(f"[{model_type}] sampler cov S S^T vs Sigma: rel err = {rel:.3e}")
    assert rel < 1e-6, f"sampler covariance mismatch {rel:.3e}"


def test_parseval(H=8, W=8):
    """sum_half mult |x_tilde|^2 == ||x||^2  (unitary rfft2 with multiplicities)."""
    x = torch.randn(3, H, W, dtype=torch.float64)
    xf = torch.fft.rfft2(x, norm="ortho")
    mult = mult_weights(H, W)
    lhs = (mult[None] * xf.abs() ** 2).sum().item()
    rhs = (x ** 2).sum().item()
    print(f"[parseval] sum mult|X|^2 = {lhs:.6f}  ||x||^2 = {rhs:.6f}")
    assert abs(lhs - rhs) < 1e-8


def test_sample_cov(model_type: str, H=6, W=6, N=400_000, seed=1):
    torch.manual_seed(seed)
    model = GaussianBCCB(H, W, model_type=model_type,
                         mean_init=torch.zeros(3 * H * W, dtype=torch.float64))
    _randomize(model, seed)
    Sigma = _rigorous_sigma(model)
    xs = model.sample(N) - model.m
    emp = (xs.T @ xs) / N
    rel = (emp - Sigma).norm() / Sigma.norm()
    print(f"[{model_type}] empirical-cov rel Frobenius err ({N} samples) = {rel:.4f}")
    assert rel < 0.05


def test_order_sensitivity(H=8, W=8, seed=2):
    """BCCB LL should change under a coordinate permutation of the data (it is a
    fixed-geometry model); this is the mechanism behind the permutation control."""
    torch.manual_seed(seed)
    n = 3 * H * W
    model = GaussianBCCB(H, W, model_type="full",
                         mean_init=torch.zeros(n, dtype=torch.float64))
    _randomize(model, seed)
    x = model.sample(64)                        # data the model likes (raster order)
    perm = torch.randperm(n)
    ll_raster = model.log_likelihood(x).mean().item()
    ll_perm = model.log_likelihood(x[:, perm]).mean().item()
    print(f"[order] mean LL raster = {ll_raster:.1f}  permuted = {ll_perm:.1f}  "
          f"(drop = {ll_raster - ll_perm:.1f})")
    assert ll_perm < ll_raster, "BCCB should prefer its native raster order"


def test_grad_finite(model_type: str, H=8, W=8):
    model = GaussianBCCB(H, W, model_type=model_type,
                         mean_init=torch.zeros(3 * H * W, dtype=torch.float64))
    _randomize(model, 0)
    x = torch.randn(4, 3 * H * W, dtype=torch.float64)
    loss = -model.log_likelihood(x).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    print(f"[{model_type}] backward OK, all grads finite")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("=== self-conjugate mask (8x8) ===")
    print(self_conj_mask(8, 8).nonzero().tolist())
    test_parseval()
    for mt in ("scalar", "full"):
        test_loglik_matches_dense(mt)
        test_grad_finite(mt)
    for mt in ("scalar", "full"):
        test_sample_cov(mt)
    test_order_sensitivity()
    print("\nALL TESTS PASSED")
