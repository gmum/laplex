"""
Exp 3: batch-size scaling at fixed n.
Fixed n = k = 2^14, B in {2^0, 2^1, ..., 2^8}.
Output: 2-panel figure  exp3_batch.{pdf,png}  (forward, forward+backward).
"""
from __future__ import annotations

import os, gc
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import bench_fwd, bench_fb, save, COL_OURS, COL_DENSE

import laplex as lap

DEVICE = "cuda"
DTYPE  = torch.float32
N      = 1 << 14         

OUT_DIR = "./figures/exp_3"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print(f"GPU = {torch.cuda.get_device_name(0)},  n = k = {N}, dtype = fp32\n")
    print(f"{'B':>4s}   |  {'ours fwd':>9s}  {'dense fwd':>9s}  | "
          f"{'ours fb':>9s}  {'dense fb':>9s}")
    print("-" * 70)

    rows = []
    a = torch.sort(torch.randn(N, device=DEVICE, dtype=DTYPE))[0]
    b = torch.randn(N, device=DEVICE, dtype=DTYPE)

    
    SKIP_DENSE_FB_AT_LARGE_B = True
    for logB in range(0, 9):
        B = 1 << logB
        X = torch.randn(B, N, device=DEVICE, dtype=DTYPE)

        def call_ours(): return lap.X_Laplex(X, a, b, threshold=0, a_sorted=True)

        def step_ours():
            a_ = a.detach().clone().requires_grad_(True)
            b_ = b.detach().clone().requires_grad_(True)
            X_ = X.detach().clone().requires_grad_(True)
            lap.X_Laplex(X_, a_, b_, threshold=0, a_sorted=True).sum().backward()

        def call_dense():
            K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
            return X @ K

        def step_dense():
            a_ = a.detach().clone().requires_grad_(True)
            b_ = b.detach().clone().requires_grad_(True)
            X_ = X.detach().clone().requires_grad_(True)
            K = torch.exp(-(a_.unsqueeze(1) - b_.unsqueeze(0)).abs())
            (X_ @ K).sum().backward()

        t_o   = bench_fwd(call_ours)
        t_d   = bench_fwd(call_dense)
        tfb_o = bench_fb(step_ours)
        try:
            tfb_d = bench_fb(step_dense) if not (SKIP_DENSE_FB_AT_LARGE_B and B > 32) else float("nan")
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache(); gc.collect()
            tfb_d = float("nan")

        print(f"{B:>4d}   |   {t_o:7.3f}    {t_d:7.3f}  |  {tfb_o:7.3f}    {tfb_d:7.3f}")
        rows.append((B, t_o, t_d, tfb_o, tfb_d))

        del X
        torch.cuda.empty_cache(); gc.collect()
        
        df = pd.DataFrame(rows, columns=["Bs", "t_o", "t_d", "tfb_o", "tfb_d"])
        df.to_csv(f"{OUT_DIR}/exp3-results.csv", index=False)
        
    df = pd.DataFrame(rows, columns=["Bs", "t_o", "t_d", "tfb_o", "tfb_d"])
    df.to_csv(f"{OUT_DIR}/exp3-results.csv", index=False)

    Bs    = [r[0] for r in rows]
    t_o   = [r[1] for r in rows]
    t_d   = [r[2] for r in rows]
    tfb_o = [r[3] for r in rows]
    tfb_d = [r[4] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, ours, dense, title in [
        (ax1, t_o,   t_d,   "Forward"),
        (ax2, tfb_o, tfb_d, "Forward + backward"),
    ]:
        ax.loglog(Bs, ours,  "o-", color=COL_OURS,  label="ours",          linewidth=2)
        ax.loglog(Bs, dense, "s-", color=COL_DENSE, label="explicit dense", linewidth=2)
        ax.set_xlabel("batch size $B$")
        ax.set_ylabel("time [ms]")
        ax.set_title(f"{title}   ($n = k = 2^{{14}}$)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    save(fig, f"{OUT_DIR}/exp3_batch")


if __name__ == "__main__":
    main()
