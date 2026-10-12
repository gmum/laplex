"""
Exp 7: Numerical accuracy on a Toeplitz problem (uniform grid).

For uniform anchors a_i = b_i = i * dx, the Laplace kernel
K[i, j] = exp(-|a_i - b_j|) = exp(-|i-j| * dx) is symmetric Toeplitz,
so the matrix-vector product X @ K can be computed exactly by:
  - dense matmul                       (the explicit reference)
  - Toeplitz embedding + 3 FFTs        (the canonical fast method)
  - LAPLEX prefix-suffix scan          (our method)

We compare the relative error of the *fp32* outputs of all three methods
against an *fp64* dense reference.  The experiment shows that even on
the home turf of FFT (uniform grid) our scan-based method matches or
beats the FFT path in numerical accuracy, while requiring no uniformity
assumption.

n in {2^10, ..., 2^14}.   B = 8.   fp32.
Output: 2-panel figure  exp7_toeplitz_accuracy.{pdf,png}.
"""
from __future__ import annotations

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import save, COL_OURS, COL_DENSE, COL_DENSE64, COL_NOCUDA

import laplex as lap

DEVICE = "cuda"
B      = 8

HN = 16
# HN = 31

OUT_DIR = "./figures/exp_7"
os.makedirs(OUT_DIR, exist_ok=True)


def toeplitz_pre_fft(n, dx, dtype, device):
    """Return the FFT of the 2n-vector that embeds the symmetric Toeplitz
    kernel K[i,j] = exp(-|i-j|*dx) into a circulant matrix of size 2n."""
    idx = torch.arange(n, device=device, dtype=dtype)
    c = torch.exp(-idx * dx)                    
    emb = torch.zeros(2 * n, device=device, dtype=dtype)
    emb[:n] = c
    emb[n + 1:] = c[1:].flip(0)
    cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
    return torch.fft.fft(emb.to(cdt))


def toeplitz_matvec_batched(X, C_fft):
    """Y = X @ K (symmetric Toeplitz) via circulant embedding + FFT.
    X : (B, n) -> Y : (B, n)."""
    B_, n = X.shape
    cdt = torch.complex64 if X.dtype == torch.float32 else torch.complex128
    X_pad = torch.zeros(B_, 2 * n, device=X.device, dtype=cdt)
    X_pad[:, :n] = X.to(cdt)
    Y = torch.fft.ifft(torch.fft.fft(X_pad, dim=-1) * C_fft.unsqueeze(0), dim=-1)
    return Y[:, :n].real


def main():
    print(f"GPU = {torch.cuda.get_device_name(0)},  uniform grid, B = {B}\n")
    print(f"{'n':>9s}  ||  {'dense fp32':>10s}  {'Toep-FFT':>10s}  {'LAPLEX':>10s}"
          f"  ||  {'dense fp32':>10s}  {'Toep-FFT':>10s}  {'LAPLEX':>10s}")
    print(f"{'':>9s}  ||  {'rel L2':>10s}  {'rel L2':>10s}  {'rel L2':>10s}"
          f"  ||  {'rel Linf':>10s}  {'rel Linf':>10s}  {'rel Linf':>10s}")
    print("-" * 110)

    rows = []
    for logN in range(10, HN):
        n = 1 << logN
        dx = 1.0 / n         

        torch.manual_seed(42 + logN)
        a32 = torch.arange(n, device=DEVICE, dtype=torch.float32) * dx
        b32 = a32                                                          
        X32 = torch.randn(B, n, device=DEVICE, dtype=torch.float32)

        
        a64, b64, X64 = a32.double(), b32.double(), X32.double()
        K64 = torch.exp(-(a64.unsqueeze(1) - b64.unsqueeze(0)).abs())
        Y_ref = X64 @ K64

        
        K32 = torch.exp(-(a32.unsqueeze(1) - b32.unsqueeze(0)).abs())
        Y_dense = X32 @ K32

        
        C_fft = toeplitz_pre_fft(n, dx, torch.float32, DEVICE)
        Y_fft = toeplitz_matvec_batched(X32, C_fft)

        
        Y_lap = lap.X_Laplex(X32, a32, b32, threshold=0, a_sorted=True)

        
        def errs(Y):
            d = (Y.double() - Y_ref).abs()
            return (d.norm().item() / Y_ref.norm().item(),
                    d.max().item()  / Y_ref.abs().max().item())

        l2_d, li_d = errs(Y_dense)
        l2_f, li_f = errs(Y_fft)
        l2_l, li_l = errs(Y_lap)

        print(f"2^{logN:<2d}={n:<6d} ||  "
              f"{l2_d:8.2e}    {l2_f:8.2e}    {l2_l:8.2e}  ||  "
              f"{li_d:8.2e}    {li_f:8.2e}    {li_l:8.2e}")
        rows.append(dict(n=n, l2_d=l2_d, l2_f=l2_f, l2_l=l2_l,
                              li_d=li_d, li_f=li_f, li_l=li_l))

        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT_DIR}/exp7_toeplitz_acc-results.csv", index=False)
        
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/exp7_toeplitz_acc-results.csv", index=False)

    
    ns   = [r["n"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    for ax, key_pre, title in [
        (ax1, "l2",  r"Relative $\ell_2$ error"),
        (ax2, "li", r"Relative $\ell_\infty$ error"),
    ]:
        ours    = [r[key_pre + "_l"] for r in rows]
        toep    = [r[key_pre + "_f"] for r in rows]
        dense32 = [r[key_pre + "_d"] for r in rows]
        ax.loglog(ns, ours,    "o-", color=COL_OURS,    label="LAPLEX (ours)",     linewidth=2)
        ax.loglog(ns, toep,    "^-", color=COL_DENSE64, label="Toeplitz--FFT",      linewidth=2)
        ax.loglog(ns, dense32, "s-", color=COL_DENSE,   label="dense fp32",         linewidth=2)
        eps = torch.finfo(torch.float32).eps
        ax.axhline(eps, color="k", linestyle="--", alpha=0.4, label=r"fp32 $\varepsilon$")
        ax.set_xlabel(r"$n = k$  (uniform grid)")
        ax.set_ylabel("relative error vs fp64 dense reference")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    save(fig, f"{OUT_DIR}/exp7_toeplitz_accuracy")


if __name__ == "__main__":
    main()
