"""
Exp 1 (GPU): time + peak memory + forward+backward, ours vs explicit dense.
n = k,  B = 8,  fp32,  CUDA.
Our: n in {2^10, ..., 2^20}.   Dense: n in {2^10, ..., 2^14} (then OOM).
Output: 3-panel figure  exp1_gpu.{pdf,png}.
"""
from __future__ import annotations

import os, gc
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import (
    bench_fwd, bench_fb, peak_mem_mb, safe_call, save,
    COL_OURS, COL_DENSE,
)
import laplex as lap

DEVICE = "cuda"
DTYPE  = torch.float32
B      = 8

HN_O = 21
HN_D = 15

HN_O = 31 
HN_D = 16

LOG_NS_OURS  = list(range(10, HN_O))   
LOG_NS_DENSE = list(range(10, HN_D))   

OUT_DIR = "./figures/exp_1"
os.makedirs(OUT_DIR, exist_ok=True)


def make_inputs(n):
    a = torch.sort(torch.randn(n, device=DEVICE, dtype=DTYPE))[0]
    b = torch.randn(n, device=DEVICE, dtype=DTYPE)
    X = torch.randn(B, n, device=DEVICE, dtype=DTYPE)
    return a, b, X



def call_ours_fwd(X, a, b):
    return lap.X_Laplex(X, a, b, threshold=0, a_sorted=True)


def step_ours_fb(X_, a_, b_):
    a = a_.detach().clone().requires_grad_(True)
    b = b_.detach().clone().requires_grad_(True)
    X = X_.detach().clone().requires_grad_(True)
    y = lap.X_Laplex(X, a, b, threshold=0, a_sorted=True)
    y.sum().backward()



def call_dense_fwd(X, a, b):
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
    return X @ K


def step_dense_fb(X_, a_, b_):
    a = a_.detach().clone().requires_grad_(True)
    b = b_.detach().clone().requires_grad_(True)
    X = X_.detach().clone().requires_grad_(True)
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
    (X @ K).sum().backward()


def main():
    print(f"GPU = {torch.cuda.get_device_name(0)}, B={B}, dtype={DTYPE}\n")

    rows = []
    print(f"{'n':>9s}  | {'ours fwd':>9s}  {'dense fwd':>9s} | "
          f"{'ours fb':>9s}  {'dense fb':>9s} | "
          f"{'ours mem':>9s}  {'dense mem':>9s}")
    print("-" * 90)

    for logN in LOG_NS_OURS:
        n = 1 << logN
        try:
            a, b, X = make_inputs(n)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache(); gc.collect()
            continue
        
        t_ours = bench_fwd(lambda: call_ours_fwd(X, a, b))

        
        tfb_ours = bench_fb(lambda: step_ours_fb(X, a, b))

        
        mem_ours = peak_mem_mb(lambda: call_ours_fwd(X, a, b))

        
        if logN in LOG_NS_DENSE:
            t_dense   = bench_fwd(lambda: call_dense_fwd(X, a, b))
            tfb_dense = bench_fb(lambda: step_dense_fb(X, a, b))
            mem_dense = peak_mem_mb(lambda: call_dense_fwd(X, a, b))
        else:
            t_dense = tfb_dense = mem_dense = float("nan")

        rows.append(dict(n=n, t_ours=t_ours, t_dense=t_dense,
                         tfb_ours=tfb_ours, tfb_dense=tfb_dense,
                         mem_ours=mem_ours, mem_dense=mem_dense))
        print(f"2^{logN:<2d}={n:<6d}|  "
              f"{t_ours:7.3f}    {t_dense:7.3f}  | "
              f"{tfb_ours:7.3f}    {tfb_dense:7.3f}  | "
              f"{mem_ours:7.1f}    {mem_dense:7.1f}")
        del a, b, X
        torch.cuda.empty_cache(); gc.collect()
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT_DIR}/exp1-results.csv", index=False)
        
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/exp1-results.csv", index=False)

    
    ns         = [r["n"]         for r in rows]
    t_ours     = [r["t_ours"]    for r in rows]
    t_dense    = [r["t_dense"]   for r in rows]
    tfb_ours   = [r["tfb_ours"]  for r in rows]
    tfb_dense  = [r["tfb_dense"] for r in rows]
    mem_ours   = [r["mem_ours"]  for r in rows]
    mem_dense  = [r["mem_dense"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.0))

    for ax, ours, dense, ylab, title in [
        (ax1, t_ours,    t_dense,    "time [ms]",          "Forward"),
        (ax2, tfb_ours,  tfb_dense,  "time [ms]",          "Forward + backward"),
        (ax3, mem_ours,  mem_dense,  "peak memory [MB]",   "Peak GPU memory (forward)"),
    ]:
        ax.loglog(ns, ours,  "o-", color=COL_OURS,  label="ours (LAPLEX)", linewidth=2)
        
        ns_d   = [n_ for n_, v in zip(ns, dense) if v == v]   
        v_d    = [v  for v in dense if v == v]
        if ns_d:
            ax.loglog(ns_d, v_d, "s-", color=COL_DENSE, label="explicit dense", linewidth=2)
            
            n_oom = ns_d[-1] * 2
            if n_oom <= ns[-1]:
                ax.axvline(n_oom, color=COL_DENSE, linestyle=":", alpha=0.5)
                ax.text(n_oom, ax.get_ylim()[0] * 1.5, "  OOM",
                        color=COL_DENSE, va="bottom", ha="left", fontsize=9)
        ax.set_xlabel(r"$n = k$  (square kernel)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left")

    save(fig, f"{OUT_DIR}/exp1_gpu")


if __name__ == "__main__":
    main()
