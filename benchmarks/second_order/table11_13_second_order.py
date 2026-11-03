"""
table11_13_second_order.py  (App. K: Tabs. 11, 12, 13)
==========================
Paper mapping: Table A -> Tab. 11, Table B -> Tab. 12, Table C -> Tab. 13.

Tables of the second-order-optimization experiment. Reuses the primitives of
`fig22_table14_second_order.py` (exact LAPLEX Fisher/GGN curvature) and adds:

  Table A  Quadratic convergence trace: per-iteration loss gap and gradient norm
           of Fisher scoring (exact LAPLEX curvature).
  Table B  Optimizer comparison: iterations AND wall-clock to reach a gradient
           tolerance for the exact-LAPLEX second-order method vs first-order
           Adam / gradient descent (mean over seeds); shows the benefit is real
           in wall-clock, not only in iteration count.
  Table C  Approximate-curvature baselines: Fisher scoring with a rank-r
           Nystrom (sub-sampled) curvature vs the exact LAPLEX curvature. This
           is the concrete "...would otherwise require Nystrom / low-rank
           approximations" comparison: the approximation needs large r (hence a
           materialized n x r slice and O(n^2 r) work) to match the exact step,
           whereas LAPLEX gives the exact curvature at O(n^2 + k log k).

All tables are printed as Markdown and saved to the output dir. CUDA, fp64.

Usage:
    python table11_13_second_order.py --n 512 --k 50000 --seeds 0 1 2
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# laplex.py and prefix_suffix.py of ../comparison_experiments (the released operators)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "comparison_experiments"))
import laplex
from fig22_table14_second_order import (
    build_problem, loss_and_grad, fisher_curvature_laplex,
)


# =============================================================================
# Approximate (Nystrom / sub-sampled) curvature baseline
# =============================================================================
def fisher_curvature_nystrom(a, b, u, lam, n, r, gen):
    """Rank-r Nystrom curvature: sub-sample r of the k feature-anchors, form the
    n x r dense slice A_S = K(a, b[S]) explicitly (what a non-LAPLEX method must
    do), and use the scaled sub-sample estimator
        H_hat = (1/r) A_S diag(s_S) A_S^T + lambda I.
    This is the standard way to get a second-order step without the exact Gram.
    """
    k = b.numel()
    ell = laplex.X_Laplex(u, a, b, threshold=0)          # A^T u (full, cheap)
    p = torch.sigmoid(ell)
    s = p * (1.0 - p)
    S = torch.randperm(k, generator=gen, device="cpu")[:r].to(a.device)
    bS, sS = b[S], s[S]
    A_S = torch.exp(-(a.unsqueeze(1) - bS.unsqueeze(0)).abs())   # n x r  (materialised)
    H = (A_S * (sS / r).unsqueeze(0)) @ A_S.T
    return H + lam * torch.eye(n, device=a.device, dtype=a.dtype)


# =============================================================================
# Generic optimiser to a gradient tolerance (returns trace + summary)
# =============================================================================
def run_optimizer(kind, a, b, y, lam, n, tol, max_iters, device, dtype,
                  adam_lr=0.05, gd_lr=None, curv_fn=None):
    u = torch.zeros(n, device=device, dtype=dtype)
    m = torch.zeros_like(u); v = torch.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    trace = []  # (iter, loss, grad_norm)
    torch.cuda.synchronize(); t0 = time.time()
    it_reached, t_reached = None, None
    it = 0
    while it < max_iters:
        loss, grad, s = loss_and_grad(u, a, b, y, lam)
        gn = grad.norm().item()
        trace.append((it, loss.item(), gn))
        if gn < tol and it_reached is None:
            torch.cuda.synchronize(); t_reached = time.time() - t0; it_reached = it
            break
        if kind == "newton":
            H = curv_fn(a, b, s, lam, n)
            u = u - torch.linalg.solve(H, grad)
        elif kind == "adam":
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
            u = u - adam_lr * mh / (vh.sqrt() + eps)
        elif kind == "gd":
            u = u - gd_lr * grad
        it += 1
    torch.cuda.synchronize(); t_total = time.time() - t0
    loss, grad, _ = loss_and_grad(u, a, b, y, lam)
    final_gn = grad.norm().item()
    return dict(trace=trace, it_reached=it_reached, t_reached=t_reached,
                t_total=t_total, final_loss=loss.item(), final_grad=final_gn)


# =============================================================================
# Tables
# =============================================================================
def table_A_trace(args, out, md):
    a, b, y, _ = build_problem(args.n, args.k, args.device, torch.float64, seed=0)
    res = run_optimizer("newton", a, b, y, args.lam, args.n, tol=1e-12,
                        max_iters=args.newton_iters, device=args.device,
                        dtype=torch.float64, curv_fn=fisher_curvature_laplex)
    L_star = res["final_loss"]
    md.append(f"\n### Table A — Fisher scoring (exact LAPLEX curvature): "
              f"quadratic convergence (n={args.n}, k={args.k})\n")
    md.append("| iter | L(u) − L\\* | ‖grad‖ |")
    md.append("|---|---|---|")
    for (it, loss, gn) in res["trace"]:
        md.append(f"| {it} | {max(loss - L_star, 0.0):.2e} | {gn:.2e} |")
    print("\n".join(md[-(len(res['trace'])+3):]))


def table_B_compare(args, out, md):
    tol = args.tol
    rows = {"newton": [], "adam": [], "gd": []}
    for seed in args.seeds:
        a, b, y, _ = build_problem(args.n, args.k, args.device, torch.float64, seed=seed)
        rows["newton"].append(run_optimizer(
            "newton", a, b, y, args.lam, args.n, tol, args.newton_iters,
            args.device, torch.float64, curv_fn=fisher_curvature_laplex))
        # first-order: pick the best Adam lr per seed (fair to the baseline)
        best = None
        for lr in (0.03, 0.1, 0.3):
            r = run_optimizer("adam", a, b, y, args.lam, args.n, tol,
                              args.first_order_iters, args.device, torch.float64,
                              adam_lr=lr)
            if best is None or r["final_grad"] < best["final_grad"]:
                best = r
        rows["adam"].append(best)
        rows["gd"].append(run_optimizer(
            "gd", a, b, y, args.lam, args.n, tol, args.first_order_iters,
            args.device, torch.float64, gd_lr=args.gd_lr))

    def agg(key, field, fmt):
        vals = [r[field] for r in rows[key] if r[field] is not None]
        if not vals:
            return "—"
        t = torch.tensor(vals, dtype=torch.float64)
        return (fmt.format(t.mean().item()) if len(vals) == 1
                else f"{fmt.format(t.mean().item())} ± {fmt.format(t.std().item())}")

    md.append(f"\n### Table B — iterations and wall-clock to reach ‖grad‖ < {tol:g} "
              f"(n={args.n}, k={args.k}, mean over {len(args.seeds)} seeds)\n")
    md.append("| method | order | iters to tol | wall-clock to tol (s) | final ‖grad‖ (after budget) |")
    md.append("|---|---|---|---|---|")
    labels = {"newton": ("Fisher scoring (exact LAPLEX)", "2nd"),
              "adam": ("Adam (best of 3 lr)", "1st"),
              "gd": ("Gradient descent", "1st")}
    for key in ("newton", "adam", "gd"):
        name, order = labels[key]
        reached = [r for r in rows[key] if r["it_reached"] is not None]
        it_s = agg(key, "it_reached", "{:.0f}") if reached else "not reached"
        wc_s = (agg(key, "t_reached", "{:.3f}") if reached else
                f">{args.first_order_iters if key!='newton' else args.newton_iters} it")
        fg_s = agg(key, "final_grad", "{:.1e}")
        md.append(f"| {name} | {order} | {it_s} | {wc_s} | {fg_s} |")
    print("\n".join(md[-(len(labels)+3):]))


def table_C_nystrom(args, out, md):
    tol = args.tol
    seed = args.seeds[0]
    a, b, y, _ = build_problem(args.n, args.k, args.device, torch.float64, seed=seed)
    gen = torch.Generator(device="cpu").manual_seed(1234)

    md.append(f"\n### Table C — exact LAPLEX curvature vs rank-r Nyström "
              f"(sub-sampled) curvature inside Fisher scoring "
              f"(n={args.n}, k={args.k}, tol {tol:g})\n")
    md.append("| curvature | rank r | materialised slice | iters to tol | final ‖grad‖ |")
    md.append("|---|---|---|---|---|")

    # rank-r Nystrom curvatures. The Nystrom curvature depends on u directly
    # (it re-samples and re-evaluates s each step), so we use a dedicated loop.
    for r in args.nystrom_ranks:
        u = torch.zeros(args.n, device=args.device, dtype=torch.float64)
        it_reached = None
        for it in range(args.newton_iters):
            loss, grad, _ = loss_and_grad(u, a, b, y, args.lam)
            gn = grad.norm().item()
            if gn < tol:
                it_reached = it; break
            H = fisher_curvature_nystrom(a, b, u, args.lam, args.n, r, gen)
            u = u - torch.linalg.solve(H, grad)
        _, grad, _ = loss_and_grad(u, a, b, y, args.lam)
        md.append(f"| Nyström | {r} | n×{r} dense | "
                  f"{it_reached if it_reached is not None else 'not reached'} | "
                  f"{grad.norm().item():.1e} |")

    # exact LAPLEX curvature (rank = k, no slice materialised)
    u = torch.zeros(args.n, device=args.device, dtype=torch.float64)
    it_reached = None
    for it in range(args.newton_iters):
        loss, grad, s = loss_and_grad(u, a, b, y, args.lam)
        if grad.norm().item() < tol:
            it_reached = it; break
        H = fisher_curvature_laplex(a, b, s, args.lam, args.n)
        u = u - torch.linalg.solve(H, grad)
    _, grad, _ = loss_and_grad(u, a, b, y, args.lam)
    md.append(f"| **exact (LAPLEX)** | **{args.k} (full)** | **none** | "
              f"**{it_reached if it_reached is not None else 'not reached'}** | "
              f"**{grad.norm().item():.1e}** |")
    print("\n".join(md[-(len(args.nystrom_ranks)+4):]))


if __name__ == "__main__":
    p = argparse.ArgumentParser("LAPLEX second-order: table-only evidence")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=50000)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--newton_iters", type=int, default=40)
    p.add_argument("--first_order_iters", type=int, default=20000)
    p.add_argument("--gd_lr", type=float, default=0.1)
    p.add_argument("--nystrom_ranks", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    p.add_argument("--outputs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()
    assert torch.cuda.is_available()
    args.device = torch.device("cuda")
    out = Path(args.outputs) / ("tables_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    md = [f"# Second-order optimization with an exact LAPLEX curvature — tables",
          f"(n={args.n}, k={args.k}, lambda={args.lam}, fp64)"]
    table_A_trace(args, out, md)
    table_B_compare(args, out, md)
    table_C_nystrom(args, out, md)
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print("\n[saved]", out / "tables.md")
