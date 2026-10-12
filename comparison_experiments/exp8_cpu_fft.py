"""
Exp 8: CPU runtime — LAPLEX (pure-PyTorch) vs Toeplitz-FFT (CPU) vs single FFT.

The point of this experiment is to demonstrate that the equal-complexity-class
claim from Sec.~4.2 (ours vs FFT-based methods) does not depend on the
specialised CUDA kernel we ship: the same scaling shows up on CPU using only
PyTorch's built-in `torch.logcumsumexp` and `torch.fft`.

Square kernel, n = k, B = 1, fp32, CPU.
n in {2^10, ..., 2^20}.

Output: 1-panel figure  exp8_cpu_fft.{pdf,png}.
"""
from __future__ import annotations

import os
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import save, COL_NOCUDA, COL_DENSE64, COL_OURS

import laplex_noCUDA as lapnc

DEVICE = "cpu"
DTYPE  = torch.float32
B      = 1

HN = 21

OUT_DIR = "./figures/exp_8"
os.makedirs(OUT_DIR, exist_ok=True)


def bench_cpu(call, warmup=3, iters=10):
    """Best-of-3 CPU timing in milliseconds."""
    def _one():
        with torch.no_grad():
            for _ in range(warmup):
                call()
            t0 = time.perf_counter()
            for _ in range(iters):
                call()
            return (time.perf_counter() - t0) / iters * 1e3
    return min(_one() for _ in range(3))


def toeplitz_pre_fft(n, dx, dtype, device):
    idx = torch.arange(n, device=device, dtype=dtype)
    c = torch.exp(-idx * dx)
    emb = torch.zeros(2 * n, device=device, dtype=dtype)
    emb[:n] = c
    emb[n + 1:] = c[1:].flip(0)
    cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
    return torch.fft.fft(emb.to(cdt))


def toeplitz_matvec(x, C_fft):
    """x : (n,)  ->  y : (n,)  via circulant embedding."""
    n = x.numel()
    cdt = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    x_pad = torch.zeros(2 * n, device=x.device, dtype=cdt)
    x_pad[:n] = x.to(cdt)
    return torch.fft.ifft(torch.fft.fft(x_pad) * C_fft)[:n].real


def main():
    print(f"device = CPU,  B = {B},  fp32\n")
    print(f"{'n':>9s}   |  {'LAPLEX-noCUDA':>14s}  {'Toep-FFT':>11s}  {'pure FFT':>10s}")
    print("-" * 60)

    rows = []
    
    for logN in range(10, HN):
        n = 1 << logN
        dx = 1.0 / n
        torch.manual_seed(n)

        
        a = torch.sort(torch.randn(n, device=DEVICE, dtype=DTYPE))[0]
        b = torch.randn(n, device=DEVICE, dtype=DTYPE)
        x = torch.randn(n, device=DEVICE, dtype=DTYPE)
        x_cplx = x.to(torch.complex64)

        
        C_fft = toeplitz_pre_fft(n, dx, DTYPE, DEVICE)

        t_lap  = bench_cpu(lambda: lapnc.X_Laplex(x.unsqueeze(0), a, b, threshold=0,
                                                  a_sorted=True))
        t_toep = bench_cpu(lambda: toeplitz_matvec(x, C_fft))
        t_fft  = bench_cpu(lambda: torch.fft.fft(x_cplx))

        print(f"2^{logN:<2d}={n:<6d}|  {t_lap:11.3f}     {t_toep:8.3f}     {t_fft:8.3f}")
        rows.append((n, t_lap, t_toep, t_fft))
        
        df = pd.DataFrame(rows, columns=["ns", "t_lap", "t_toep", "t_fft"])
        df.to_csv(f"{OUT_DIR}/exp8_cpu_fft-results.csv", index=False)
        
    df = pd.DataFrame(rows, columns=["ns", "t_lap", "t_toep", "t_fft"])
    df.to_csv(f"{OUT_DIR}/exp8_cpu_fft-results.csv", index=False)

    ns      = [r[0] for r in rows]
    t_lap   = [r[1] for r in rows]
    t_toep  = [r[2] for r in rows]
    t_fft   = [r[3] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.5))
    ax.loglog(ns, t_lap,  "o-", color=COL_OURS,    label="LAPLEX (laplex_noCUDA)", linewidth=2)
    ax.loglog(ns, t_toep, "^-", color=COL_DENSE64, label="Toeplitz--FFT (CPU)",     linewidth=2)
    ax.loglog(ns, t_fft,  "s-", color="k",         label="pure FFT (CPU)",          linewidth=2)
    
    n_ref = torch.tensor(ns, dtype=torch.float64)
    nlogn = (n_ref * torch.log2(n_ref)).tolist()
    scale = t_fft[-1] / nlogn[-1]
    ref = [v * scale for v in nlogn]
    ax.loglog(ns, ref, "k--", alpha=0.4, label=r"$\mathcal{O}(n \log n)$ reference")
    ax.set_xlabel(r"$n = k$  ($B = 1$)")
    ax.set_ylabel("time [ms]")
    ax.set_title("CPU runtime: LAPLEX vs FFT-based baselines")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, f"{OUT_DIR}/exp8_cpu_fft")


if __name__ == "__main__":
    main()
