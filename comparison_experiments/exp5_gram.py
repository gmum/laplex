"""
Exp 5: weighted Gram matrix scaling   K diag(D) K^T.
Output dimension fixed at n = 1024 so that the (n, n) result has constant
size (4 MB in fp32) — this isolates the scaling of the inner dimension k.
k in {2^10, ..., 2^20}.   fp32, CUDA.

Output: 2-panel figure  exp5_gram.{pdf,png}  (time + peak memory).
"""
from __future__ import annotations

import os, gc
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import bench_fwd, peak_mem_mb, save, COL_OURS, COL_DENSE, DenseGate

import laplex as lap

DEVICE = "cuda"
DTYPE  = torch.float32
N      = 1024

HN = 21
HN = 31

OUT_DIR = "./figures/exp_5"
os.makedirs(OUT_DIR, exist_ok=True)


def call_ours(alpha, beta, D):
    return lap.Laplex_D_LaplexT(alpha, beta, D, threshold=0, alpha_sorted=True)


def call_dense(alpha, beta, D):
    K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
    return (K * D.unsqueeze(0)) @ K.T


def main():
    print(f"GPU = {torch.cuda.get_device_name(0)},  n = {N} (fixed),  fp32")
    print(f"{'k':>9s}  |  {'ours [ms]':>10s}  {'dense [ms]':>11s}  | "
          f"{'ours mem':>9s}  {'dense mem':>10s}")
    print("-" * 70)

    rows = []
    alpha = torch.sort(torch.randn(N, device=DEVICE, dtype=DTYPE))[0]
    dense_gate = DenseGate(time_limit_ms=500.0)
    for logK in range(10, HN):
        k = 1 << logK
        torch.manual_seed(k)
        beta = torch.randn(k, device=DEVICE, dtype=DTYPE)
        D    = torch.randn(k, device=DEVICE, dtype=DTYPE)

        t_o   = bench_fwd(lambda: call_ours(alpha, beta, D))
        mem_o = peak_mem_mb(lambda: call_ours(alpha, beta, D))

        
        
        if dense_gate.alive:
            dense_gate.predicted_oom(N * k * 4 * 3)
        t_d = dense_gate.try_time(lambda: bench_fwd(lambda: call_dense(alpha, beta, D)))
        if t_d == t_d:
            try:
                mem_d = peak_mem_mb(lambda: call_dense(alpha, beta, D))
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache(); gc.collect()
                mem_d = float("nan")
        else:
            mem_d = float("nan")

        print(f"2^{logK:<2d}={k:<6d}|   {t_o:8.3f}    {t_d:9.3f}  |   "
              f"{mem_o:7.1f}    {mem_d:8.1f}")
        rows.append((k, t_o, t_d, mem_o, mem_d))

        del beta, D
        torch.cuda.empty_cache(); gc.collect()
        
        df = pd.DataFrame(rows, columns=["ks", "t_o", "t_d", "mem_o", "mem_d"])
        df.to_csv(f"{OUT_DIR}/exp5-results.csv", index=False)

    if dense_gate.reason:
        print(f"  (dense gate tripped: {dense_gate.reason})")
        
    df = pd.DataFrame(rows, columns=["ks", "t_o", "t_d", "mem_o", "mem_d"])
    df.to_csv(f"{OUT_DIR}/exp5-results.csv", index=False)

    ks      = [r[0] for r in rows]
    t_o     = [r[1] for r in rows]
    t_d     = [r[2] for r in rows]
    mem_o   = [r[3] for r in rows]
    mem_d   = [r[4] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    for ax, ours, dense, ylab, title in [
        (ax1, t_o,   t_d,   "time [ms]",        "Forward time"),
        (ax2, mem_o, mem_d, "peak memory [MB]", "Peak memory"),
    ]:
        ax.loglog(ks, ours, "o-", color=COL_OURS, label="ours", linewidth=2)
        ks_d = [k for k, v in zip(ks, dense) if v == v]
        v_d  = [v for v in dense if v == v]
        if ks_d:
            ax.loglog(ks_d, v_d, "s-", color=COL_DENSE, label="explicit dense", linewidth=2)
            k_oom = ks_d[-1] * 2
            if k_oom <= ks[-1]:
                ax.axvline(k_oom, color=COL_DENSE, linestyle=":", alpha=0.5)
                ax.text(k_oom, ax.get_ylim()[0] * 1.5, "  OOM",
                        color=COL_DENSE, va="bottom", ha="left", fontsize=9)
        ax.set_xlabel(r"$k$  (inner dim, $n = 1024$ fixed)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left")

    save(fig, f"{OUT_DIR}/exp5_gram")


if __name__ == "__main__":
    main()
