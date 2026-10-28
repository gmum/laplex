"""
Exp 6: CPU benchmark — pure-PyTorch laplex_noCUDA vs explicit dense.
n = k,  B = 1,  fp32,  device = CPU.
n in {2^10, ..., 2^14}.

Output: 1-panel figure  exp6_cpu.{pdf,png}.
"""
from __future__ import annotations

import os
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import save, COL_NOCUDA, COL_DENSE

import laplex_noCUDA as lapnc

DEVICE = "cpu"
DTYPE  = torch.float32
B      = 1

HN = 16
# HN = 31

OUT_DIR = "./figures/exp_6"
os.makedirs(OUT_DIR, exist_ok=True)


def bench_cpu(call, warmup=3, iters=10):
    """Best-of-3 CPU timing."""
    def _one():
        for _ in range(warmup):
            call()
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        return (time.perf_counter() - t0) / iters * 1e3
    return min(_one() for _ in range(3))


def main():
    print(f"device = CPU,  B = {B}, dtype = fp32")
    print(f"{'n':>9s}  |  {'noCUDA [ms]':>11s}  {'dense [ms]':>11s}")
    print("-" * 50)

    rows = []
    for logN in range(10, HN):
        n = 1 << logN
        torch.manual_seed(n)
        a = torch.sort(torch.randn(n, device=DEVICE, dtype=DTYPE))[0]
        b = torch.randn(n, device=DEVICE, dtype=DTYPE)
        X = torch.randn(B, n, device=DEVICE, dtype=DTYPE)

        def call_ours():
            with torch.no_grad():
                return lapnc.X_Laplex(X, a, b, threshold=0)

        def call_dense():
            with torch.no_grad():
                K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
                return X @ K

        t_o = bench_cpu(call_ours)
        t_d = bench_cpu(call_dense)
        print(f"2^{logN:<2d}={n:<6d}|    {t_o:8.3f}      {t_d:8.3f}")
        rows.append((n, t_o, t_d))
        
        df = pd.DataFrame(rows, columns=["ns", "t_o", "t_d"])
        df.to_csv(f"{OUT_DIR}/exp6-results.csv", index=False)

    df = pd.DataFrame(rows, columns=["ns", "t_o", "t_d"])
    df.to_csv(f"{OUT_DIR}/exp6-results.csv", index=False)

    ns  = [r[0] for r in rows]
    t_o = [r[1] for r in rows]
    t_d = [r[2] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    ax.loglog(ns, t_o, "o-", color=COL_NOCUDA, label="ours (laplex_noCUDA)", linewidth=2)
    ax.loglog(ns, t_d, "s-", color=COL_DENSE, label="explicit dense", linewidth=2)
    ax.set_xlabel(r"$n = k$")
    ax.set_ylabel("time [ms]")
    ax.set_title("CPU benchmark  ($B = 1$, fp32)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save(fig, f"{OUT_DIR}/exp6_cpu")


if __name__ == "__main__":
    main()
