"""
Exp 4: asymmetric  n vs k.
Two configurations:
   (a) n = 1024 fixed, k in {2^10, ..., 2^20}
   (b) k = 1024 fixed, n in {2^10, ..., 2^20}
B = 8, fp32, CUDA.  Dense reference up to OOM.
Output: 2-panel figure  exp4_asymm.{pdf,png}  (forward time).
"""
from __future__ import annotations

import os, gc
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import bench_fwd, save, COL_OURS, COL_DENSE, DenseGate

import laplex as lap

DEVICE = "cuda"
DTYPE  = torch.float32
B      = 8

HN = 21
HN = 29

OUT_DIR = "./figures/exp_4"
os.makedirs(OUT_DIR, exist_ok=True)


def call_ours(X, a, b):
    return lap.X_Laplex(X, a, b, threshold=0)


def call_dense(X, a, b):
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
    return X @ K


def sweep_axis(fixed_dim, fixed_size, vary_logs):
    """fixed_dim in {'n', 'k'}.  Returns (sizes, t_ours, t_dense_or_nan)."""
    print(f"\n--- {fixed_dim} = {fixed_size}, varying the other axis ---")
    print(f"{'other':>9s}  |  {'ours [ms]':>10s}  {'dense [ms]':>11s}")
    print("-" * 45)

    dense_gate = DenseGate(time_limit_ms=500.0)
    out = []
    for logV in vary_logs:
        v = 1 << logV
        if fixed_dim == "n":
            n, k = fixed_size, v
        else:
            n, k = v, fixed_size

        torch.manual_seed(n + k)
        a = torch.sort(torch.randn(n, device=DEVICE, dtype=DTYPE))[0]
        b = torch.randn(k, device=DEVICE, dtype=DTYPE)
        X = torch.randn(B, n, device=DEVICE, dtype=DTYPE)

        t_o = bench_fwd(lambda: call_ours(X, a, b))

        
        
        if dense_gate.alive and dense_gate.predicted_oom(n * k * 4):
            print(f"  [dense disabled: {dense_gate.reason}]")
        t_d = dense_gate.try_time(lambda: bench_fwd(lambda: call_dense(X, a, b)))
        if not dense_gate.alive and t_d != t_d and dense_gate.reason:
            
            pass

        print(f"2^{logV:<2d}={v:<6d}|   {t_o:8.3f}    {t_d:9.3f}")
        out.append((v, t_o, t_d))

        del a, b, X
        torch.cuda.empty_cache(); gc.collect()
        
        df = pd.DataFrame(out, columns=["sizes", "ours", "dense"])
        df.to_csv(f"{OUT_DIR}/exp4_{fixed_dim}-results.csv", index=False)
        
    df = pd.DataFrame(out, columns=["sizes", "ours", "dense"])
    df.to_csv(f"{OUT_DIR}/exp4_{fixed_dim}-results.csv", index=False)

    if dense_gate.reason:
        print(f"  (dense gate tripped: {dense_gate.reason})")
    return out


def main():
    print(f"GPU = {torch.cuda.get_device_name(0)}, B={B}, fp32")

    res_n = sweep_axis("n", 1024, list(range(10, HN)))
    res_k = sweep_axis("k", 1024, list(range(10, HN)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    for ax, res, label_axis, fixed_lbl in [
        (ax1, res_n, "k", r"$n = 1024$ fixed"),
        (ax2, res_k, "n", r"$k = 1024$ fixed"),
    ]:
        sizes  = [r[0] for r in res]
        ours   = [r[1] for r in res]
        dense  = [r[2] for r in res]

        ax.loglog(sizes, ours,  "o-", color=COL_OURS, label="ours", linewidth=2)
        ns_d = [s for s, v in zip(sizes, dense) if v == v]
        v_d  = [v for v in dense if v == v]
        if ns_d:
            ax.loglog(ns_d, v_d, "s-", color=COL_DENSE, label="explicit dense", linewidth=2)
            n_oom = ns_d[-1] * 2
            if n_oom <= sizes[-1]:
                ax.axvline(n_oom, color=COL_DENSE, linestyle=":", alpha=0.5)
                ax.text(n_oom, ax.get_ylim()[0] * 1.5, "  OOM",
                        color=COL_DENSE, va="bottom", ha="left", fontsize=9)

        ax.set_xlabel(f"${label_axis}$")
        ax.set_ylabel("time [ms]")
        ax.set_title(fixed_lbl)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left")

    save(fig, f"{OUT_DIR}/exp4_asymm")


if __name__ == "__main__":
    main()
