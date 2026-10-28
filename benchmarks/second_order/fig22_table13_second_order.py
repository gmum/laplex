"""
fig22_table13_second_order.py  (App. K: Fig. 22, Tab. 13; correctness claim in the text)
=============================
Paper mapping: --mode correctness -> max. relative error vs dense in App. K;
--mode convergence -> Fig. 22 (left); --mode scaling -> Fig. 22 (right) and Tab. 13.

Second-order optimization with an EXACT LAPLEX curvature. This directly exercises Theorem 2 — the exact weighted
Gram M = A diag(D) A^T with A = LAPLEX(a, b) — *inside a real second-order
optimizer* (Fisher scoring / natural gradient / Gauss-Newton for a GLM).

Task. Logistic regression with a structured Laplace-kernel design:
    A = LAPLEX(a, b) in R^{n x k},   u in R^n,   logits ell = A^T u in R^k,
    labels y_j ~ Bernoulli(sigmoid(ell_j)).
    L(u) = (1/k) sum_j BCE(ell_j, y_j) + (lambda/2) ||u||^2.

For this GLM the Fisher information / generalized-Gauss-Newton curvature in u is
exactly
    H(u) = (1/k) A diag(s) A^T + lambda I,   s_j = p_j (1 - p_j),  p = sigmoid(ell),
i.e. precisely the Theorem-2 object. The natural-gradient / Fisher-scoring step is
    u <- u - H(u)^{-1} grad L(u).
Every quantity is obtained from LAPLEX primitives WITHOUT materializing A:
    ell   = A^T u        = X_Laplex(u, a, b)
    A r   = A @ r        = Laplex_X(r, a, b)
    H     = A diag(s) A^T = Laplex_D_LaplexT(a, b, s) (+ lambda I)

Three modes (--mode) answer the three things the paper's claim implies:
  correctness : LAPLEX Fisher step == dense Fisher step (rel err ~ machine eps).
  convergence : Fisher scoring (2nd order, exact LAPLEX curvature) vs Adam/GD
                (1st order) on the SAME objective -> far fewer iterations.
  scaling     : fix n, grow the inner dimension k. The naive dense curvature
                needs the n x k matrix A and O(n^2 k) work (OOM/slow); the LAPLEX
                curvature is exact in O(n^2 + k log k), enabling second-order
                steps where the dense one is infeasible.

Comments in English (repo convention).

Example
-------
    python fig22_table13_second_order.py --mode correctness
    python fig22_table13_second_order.py --mode convergence --n 512 --k 50000
    python fig22_table13_second_order.py --mode scaling --k_max_exp 25
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# laplex.py and prefix_suffix.py of ../comparison_experiments (the released operators)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "comparison_experiments"))
import laplex  # CUDA kernels: X_Laplex, Laplex_X, Laplex_D_LaplexT


# =============================================================================
# Problem + LAPLEX-based GLM primitives
# =============================================================================
def build_problem(n, k, device, dtype, seed=0):
    """Random Laplace-kernel design + a Bernoulli target drawn from a planted u*."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(n, generator=g, dtype=dtype).to(device)
    b = torch.randn(k, generator=g, dtype=dtype).to(device)
    u_star = (torch.randn(n, generator=g, dtype=dtype) / (n ** 0.5)).to(device)
    with torch.no_grad():
        ell = laplex.X_Laplex(u_star, a, b, threshold=0)          # A^T u*  in R^k
        p = torch.sigmoid(ell)
        y = torch.bernoulli(p.clamp(1e-6, 1 - 1e-6)).to(dtype)
    return a, b, y, u_star


def loss_and_grad(u, a, b, y, lam):
    """L(u), grad L(u), and per-sample curvature weights s = p(1-p). LAPLEX-only."""
    k = y.numel()
    ell = laplex.X_Laplex(u, a, b, threshold=0)                   # A^T u  in R^k
    p = torch.sigmoid(ell)
    # BCE-with-logits: softplus(ell) - y*ell   (stable, y in {0,1})
    loss = (F.softplus(ell) - y * ell).mean() + 0.5 * lam * (u @ u)
    grad = laplex.Laplex_X(p - y, a, b, threshold=0) / k + lam * u  # A(p-y)/k + lam u
    s = p * (1.0 - p)
    return loss, grad, s


def fisher_curvature_laplex(a, b, s, lam, n):
    """H = (1/k) A diag(s) A^T + lambda I  via the exact LAPLEX Gram (Thm 2)."""
    k = s.numel()
    M = laplex.Laplex_D_LaplexT(a, b, s / k, threshold=0)          # (n, n), no A formed
    M = M + lam * torch.eye(n, device=M.device, dtype=M.dtype)
    return M


def fisher_curvature_dense(a, b, s, lam, n):
    """Same H but by explicitly materializing A = K(a, b) (the naive baseline)."""
    k = s.numel()
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())        # (n, k)  <-- materialised
    M = (K * (s / k).unsqueeze(0)) @ K.T
    return M + lam * torch.eye(n, device=M.device, dtype=M.dtype)


def newton_step(u, a, b, y, lam, curvature_fn):
    """One Fisher-scoring / natural-gradient step; returns (delta, loss)."""
    loss, grad, s = loss_and_grad(u, a, b, y, lam)
    H = curvature_fn(a, b, s, lam, u.numel())
    delta = torch.linalg.solve(H, grad)                            # H^{-1} grad
    return delta, loss.item()


# =============================================================================
# Mode: correctness  (LAPLEX curvature/step == dense curvature/step)
# =============================================================================
def run_correctness(args, out):
    print("=" * 66)
    print("CORRECTNESS: LAPLEX vs dense Fisher-scoring step (fp64)")
    print("=" * 66)
    dtype = torch.float64
    rows = []
    for (n, k) in [(64, 512), (128, 2000), (256, 8000), (400, 20000)]:
        a, b, y, _ = build_problem(n, k, args.device, dtype, seed=args.seed)
        u = torch.zeros(n, device=args.device, dtype=dtype)
        # run a few Newton iters and compare the two curvatures/steps each time
        max_rel = 0.0
        for _ in range(5):
            _, grad, s = loss_and_grad(u, a, b, y, args.lam)
            H_l = fisher_curvature_laplex(a, b, s, args.lam, n)
            H_d = fisher_curvature_dense(a, b, s, args.lam, n)
            d_l = torch.linalg.solve(H_l, grad)
            d_d = torch.linalg.solve(H_d, grad)
            relH = (H_l - H_d).abs().max().item() / H_d.abs().max().item()
            reld = (d_l - d_d).norm().item() / d_d.norm().item()
            max_rel = max(max_rel, relH, reld)
            u = u - d_l
        rows.append(dict(n=n, k=k, max_rel_err=max_rel))
        print(f"  n={n:4d} k={k:6d}  max rel-err (curvature & step) = {max_rel:.2e}")
    json.dump(rows, open(out / "correctness.json", "w"), indent=2)


# =============================================================================
# Mode: convergence  (2nd-order LAPLEX vs 1st-order Adam, same objective)
# =============================================================================
def run_convergence(args, out):
    print("=" * 66)
    print(f"CONVERGENCE: Fisher scoring (LAPLEX) vs Adam  (n={args.n}, k={args.k})")
    print("=" * 66)
    dtype = torch.float64
    a, b, y, _ = build_problem(args.n, args.k, args.device, dtype, seed=args.seed)

    # --- second order: Fisher scoring with the exact LAPLEX curvature ---
    u = torch.zeros(args.n, device=args.device, dtype=dtype)
    hist_newton = []
    for it in range(args.newton_iters):
        loss, grad, s = loss_and_grad(u, a, b, y, args.lam)
        hist_newton.append((it, loss.item(), grad.norm().item()))
        H = fisher_curvature_laplex(a, b, s, args.lam, args.n)
        u = u - torch.linalg.solve(H, grad)
    loss, grad, _ = loss_and_grad(u, a, b, y, args.lam)
    hist_newton.append((args.newton_iters, loss.item(), grad.norm().item()))
    L_star = hist_newton[-1][1]

    # --- first order: Adam on the same L(u) (gradients via LAPLEX matvecs) ---
    u2 = torch.zeros(args.n, device=args.device, dtype=dtype, requires_grad=False)
    m = torch.zeros_like(u2); v = torch.zeros_like(u2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist_adam = []
    for it in range(args.adam_iters):
        loss, grad, _ = loss_and_grad(u2, a, b, y, args.lam)
        hist_adam.append((it, loss.item(), grad.norm().item()))
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
        u2 = u2 - args.adam_lr * mh / (vh.sqrt() + eps)

    # iterations Adam needs to reach Newton's final loss (within 1e-6)
    reach = next((it for it, l, _ in hist_adam if l <= L_star + 1e-6), None)
    print(f"  Fisher scoring: reached L*={L_star:.6f} in "
          f"{len(hist_newton) - 1} iters (grad_norm "
          f"{hist_newton[0][2]:.2e} -> {hist_newton[-1][2]:.2e})")
    print(f"  Adam(lr={args.adam_lr}): "
          + (f"reached L* at iter {reach}" if reach is not None
             else f"did NOT reach L* in {args.adam_iters} iters "
                  f"(final L={hist_adam[-1][1]:.6f})"))
    json.dump(dict(newton=hist_newton, adam=hist_adam, L_star=L_star,
                   adam_reach_iter=reach, n=args.n, k=args.k),
              open(out / "convergence.json", "w"), indent=2)
    _plot_convergence(hist_newton, hist_adam, L_star, out / "convergence.png")


# =============================================================================
# Mode: scaling  (grow inner dim k; dense curvature OOMs, LAPLEX does not)
# =============================================================================
def run_scaling(args, out):
    print("=" * 66)
    print(f"SCALING: cost of one Fisher-scoring step vs inner dim k  (n={args.n})")
    print("=" * 66)
    dtype = torch.float32
    ks = [2 ** e for e in range(args.k_min_exp, args.k_max_exp + 1)]
    rows = []
    for k in ks:
        a, b, y, _ = build_problem(args.n, k, args.device, dtype, seed=args.seed)
        u = torch.zeros(args.n, device=args.device, dtype=dtype)
        _, _, s = loss_and_grad(u, a, b, y, args.lam)

        # LAPLEX curvature timing + peak memory
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time()
        _ = fisher_curvature_laplex(a, b, s, args.lam, args.n)
        torch.cuda.synchronize()
        t_lap = time.time() - t0
        mem_lap = torch.cuda.max_memory_allocated() / 1e9

        # dense curvature timing + peak memory (guard OOM)
        t_dense, mem_dense, dense_ok = float("nan"), float("nan"), True
        try:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = time.time()
            _ = fisher_curvature_dense(a, b, s, args.lam, args.n)
            torch.cuda.synchronize()
            t_dense = time.time() - t0
            mem_dense = torch.cuda.max_memory_allocated() / 1e9
        except torch.cuda.OutOfMemoryError:
            dense_ok = False
            torch.cuda.empty_cache()
        rows.append(dict(k=k, t_laplex_s=t_lap, mem_laplex_gb=mem_lap,
                         t_dense_s=t_dense, mem_dense_gb=mem_dense,
                         dense_ok=dense_ok))
        print(f"  k=2^{k.bit_length()-1:<2d}={k:>10d} | LAPLEX {t_lap*1e3:8.2f} ms "
              f"{mem_lap:6.3f} GB | dense "
              + (f"{t_dense*1e3:8.2f} ms {mem_dense:6.3f} GB"
                 if dense_ok else "OOM / infeasible"))
        del a, b, y, u, s; torch.cuda.empty_cache()
    json.dump(rows, open(out / "scaling.json", "w"), indent=2)
    _plot_scaling(rows, out / "scaling.png")


# =============================================================================
# Plots
# =============================================================================
def _plot_convergence(hn, ha, L_star, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy([it for it, _, _ in hn], [l - L_star + 1e-12 for _, l, _ in hn],
                "o-", label="Fisher scoring (LAPLEX, 2nd order)")
    ax.semilogy([it for it, _, _ in ha], [l - L_star + 1e-12 for _, l, _ in ha],
                "-", label="Adam (1st order)")
    ax.set_xlabel("iteration"); ax.set_ylabel("L(u) - L*   (log)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"  [plot] {path}")


def _plot_scaling(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ks, [r["t_laplex_s"] * 1e3 for r in rows], "o-", label="LAPLEX Gram")
    dk = [r["k"] for r in rows if r["dense_ok"]]
    dt = [r["t_dense_s"] * 1e3 for r in rows if r["dense_ok"]]
    if dk:
        ax.loglog(dk, dt, "s-", label="dense Gram (until OOM)")
    ax.set_xlabel("inner dimension k"); ax.set_ylabel("one curvature build (ms)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"  [plot] {path}")


# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser("LAPLEX second-order optimization (App. K)")
    p.add_argument("--mode", required=True,
                   choices=["correctness", "convergence", "scaling"])
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=20000)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--newton_iters", type=int, default=15)
    p.add_argument("--adam_iters", type=int, default=5000)
    p.add_argument("--adam_lr", type=float, default=0.05)
    p.add_argument("--k_min_exp", type=int, default=12)
    p.add_argument("--k_max_exp", type=int, default=24)
    p.add_argument("--outputs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    args.device = torch.device("cuda")

    out = Path(args.outputs) / (args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    json.dump({k: str(v) for k, v in vars(args).items()},
              open(out / "params.json", "w"), indent=2)
    print("outputs ->", out)

    if args.mode == "correctness":
        run_correctness(args, out)
    elif args.mode == "convergence":
        run_convergence(args, out)
    elif args.mode == "scaling":
        run_scaling(args, out)
