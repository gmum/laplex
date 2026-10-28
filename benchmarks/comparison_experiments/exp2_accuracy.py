"""
Exp 2: numerical accuracy of fp32 X_Laplex (ours) and fp32 explicit dense
multiplication, both compared against an fp64 dense ground truth.

n = k,   B = 8,   n in {2^10, ..., 2^14}.
Output: 2-panel figure  exp2_accuracy.{pdf,png}  (rel L2 + rel Linf).
"""
from __future__ import annotations

import os, torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import save, COL_OURS, COL_DENSE

import laplex as lap

DEVICE = "cuda"
B = 8

HN = 15
# HN = 31

OUT_DIR = "./figures/exp_2"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print(f"GPU = {torch.cuda.get_device_name(0)}\n")
    print(f"{'n':>9s}   ||  {'ours rel L2':>11s}  {'ours rel Linf':>13s}"
          f"  ||  {'dense32 rel L2':>14s}  {'dense32 rel Linf':>16s}")
    print("-" * 88)

    rows = []
    for logN in range(10, HN):
        n = 1 << logN
        torch.manual_seed(42 + logN)

        a32 = torch.sort(torch.randn(n, device=DEVICE, dtype=torch.float32))[0]
        b32 = torch.randn(n, device=DEVICE, dtype=torch.float32)
        X32 = torch.randn(B, n, device=DEVICE, dtype=torch.float32)

        
        a64, b64, X64 = a32.double(), b32.double(), X32.double()
        K64 = torch.exp(-(a64.unsqueeze(1) - b64.unsqueeze(0)).abs())
        Y_gt = X64 @ K64                                            

        
        Y_ours = lap.X_Laplex(X32, a32, b32, threshold=0, a_sorted=True)
        err_o = (Y_ours.double() - Y_gt).abs()
        l2_o   = err_o.norm().item() / Y_gt.norm().item()
        linf_o = err_o.max().item()  / Y_gt.abs().max().item()

        
        K32 = torch.exp(-(a32.unsqueeze(1) - b32.unsqueeze(0)).abs())
        Y_d  = X32 @ K32
        err_d = (Y_d.double() - Y_gt).abs()
        l2_d   = err_d.norm().item() / Y_gt.norm().item()
        linf_d = err_d.max().item()  / Y_gt.abs().max().item()

        print(f"2^{logN:<2d}={n:<6d} ||  {l2_o:9.2e}    {linf_o:9.2e}"
              f"  ||  {l2_d:9.2e}      {linf_d:9.2e}")
        rows.append((n, l2_o, linf_o, l2_d, linf_d))
        
        df = pd.DataFrame(rows, columns=["ns", "l2_o", "linf_o", "l2_d", "linf_d"])
        df.to_csv(f"{OUT_DIR}/exp2-results.csv", index=False)
        
    df = pd.DataFrame(rows, columns=["ns", "l2_o", "linf_o", "l2_d", "linf_d"])
    df.to_csv(f"{OUT_DIR}/exp2-results.csv", index=False)

    ns      = [r[0] for r in rows]
    l2_o    = [r[1] for r in rows]
    linf_o  = [r[2] for r in rows]
    l2_d    = [r[3] for r in rows]
    linf_d  = [r[4] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    for ax, our, dense, title in [
        (ax1, l2_o,   l2_d,   r"Relative $\ell_2$ error"),
        (ax2, linf_o, linf_d, r"Relative $\ell_\infty$ error"),
    ]:
        ax.loglog(ns, our,   "o-", color=COL_OURS,  label="ours fp32",  linewidth=2)
        ax.loglog(ns, dense, "s-", color=COL_DENSE, label="dense fp32", linewidth=2)
        eps = torch.finfo(torch.float32).eps
        ax.axhline(eps, color="k", linestyle="--", alpha=0.4, label=r"fp32 $\varepsilon$")
        ax.set_xlabel(r"$n = k$")
        ax.set_ylabel("relative error (vs fp64 dense reference)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    save(fig, f"{OUT_DIR}/exp2_accuracy")


if __name__ == "__main__":
    main()
