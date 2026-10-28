"""
Generate the RFF comparison figure for the main paper:
  - left panel : wall-clock time vs the number of features D
                 (RFF, with LAPLEX shown as a constant baseline)
  - right panel: relative L2 value error vs D
                 (LINEAR y-axis, with [min, max] band over seeds and a
                  1/sqrt(D) reference curve)

n = k = 2^16,  B = 1,  fp32,  CUDA.
D in {1, 2, 4, ..., 2048},  10 random seeds per D.

Output:  rff_combined.{pdf,png}  inside  NeurIPS_2026/images/.
"""
from __future__ import annotations

import os, sys, time, math
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import COL_OURS, COL_DENSE, COL_DENSE64
import laplex as lap

DEVICE  = "cuda"
DTYPE   = torch.float32
N       = 1 << 16
N_REPS  = 10

HN = 12
HN = 31

OUT_DIR = "./figures/exp_rff_values"
os.makedirs(OUT_DIR, exist_ok=True)


def rff_matvec(x, a, b, D, seed):
    """Random Fourier Features approximation of  x^T K(a, b)
    for the Laplace kernel exp(-|a-b|), spectral density Cauchy(0,1).

    x : (n,),  a : (n,),  b : (k,)
    Returns (k,).
    """
    device, dtype = x.device, x.dtype
    scale = math.sqrt(2.0 / D)
    gen = torch.Generator(device=device).manual_seed(seed)

    
    U = torch.rand(D, generator=gen, device=device, dtype=dtype)
    w = torch.tan(math.pi * (U - 0.5))                        
    c = 2 * math.pi * torch.rand(D, generator=gen,
                                  device=device, dtype=dtype)  

    Phi_a = scale * torch.cos(a.unsqueeze(1) * w + c)         
    proj  = x @ Phi_a                                         
    Phi_b = scale * torch.cos(b.unsqueeze(1) * w + c)         
    return Phi_b @ proj                                       



def time_fn(call, warmup=3, iters=10):
    def _one():
        with torch.no_grad():
            for _ in range(warmup):
                call()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                call()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3
    return min(_one() for _ in range(3))


def main():
    print(f"GPU = {torch.cuda.get_device_name(0)}")
    print(f"n = k = 2^16 = {N},  B = 1,  fp32,  {N_REPS} seeds per D\n")

    torch.manual_seed(0)
    a = torch.sort(torch.randn(N, device=DEVICE, dtype=DTYPE))[0]
    b = torch.randn(N, device=DEVICE, dtype=DTYPE)
    x = torch.randn(N, device=DEVICE, dtype=DTYPE)

    
    y_gt = lap.X_Laplex(x, a, b, threshold=0, a_sorted=True).detach()
    y_gt_norm = y_gt.norm().item()
    print(f"||y_gt||_2 = {y_gt_norm:.4e}\n")

    
    t_lap = time_fn(lambda: lap.X_Laplex(x, a, b, threshold=0, a_sorted=True),
                    warmup=10, iters=30)
    print(f"LAPLEX (exact) runtime: {t_lap:.3f} ms  (independent of D)\n")

    Ds = [1 << i for i in range(0, HN)]            

    print(f"{'D':>5s}  | {'time mean [ms]':>14s} {'time min':>10s} {'time max':>10s} | "
          f"{'err median':>11s} {'err min':>9s} {'err max':>9s}")
    print("-" * 86)

    rows = []
    for D in Ds:
        times, errs = [], []
        for rep in range(N_REPS):
            seed = 1000 * D + rep

            try:
                with torch.no_grad():
                    _ = rff_matvec(x, a, b, D, seed=seed)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    y_rff = rff_matvec(x, a, b, D, seed=seed)
                    torch.cuda.synchronize()
                    t_ms = (time.perf_counter() - t0) * 1e3
                err = (y_rff - y_gt).norm().item() / y_gt_norm
                del y_rff
            except (torch.AcceleratorError, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                err = float("nan")
                t_ms = float("nan")

            times.append(t_ms); errs.append(err)
            torch.cuda.empty_cache()

        t_mean = sum(times) / N_REPS
        t_min, t_max = min(times), max(times)
        
        e_sorted = sorted(errs)
        e_med = e_sorted[N_REPS // 2]
        e_min, e_max = e_sorted[0], e_sorted[-1]

        print(f"{D:5d}  |  {t_mean:11.3f}    {t_min:7.3f}    {t_max:7.3f}  |  "
              f"{e_med:9.3e}   {e_min:7.3e}   {e_max:7.3e}")
        rows.append(dict(D=D, t_mean=t_mean, t_min=t_min, t_max=t_max,
                              e_med=e_med, e_min=e_min, e_max=e_max))
        
        df = pd.DataFrame(rows)
        df["t_lap"] = t_lap
        df.to_csv(f"{OUT_DIR}/exp_rff_val-results.csv", index=False)
    
    df = pd.DataFrame(rows)
    df["t_lap"] = t_lap
    df.to_csv(f"{OUT_DIR}/exp_rff_val-results.csv", index=False)
    
    
    Ds_arr  = [r["D"]      for r in rows]
    t_means = [r["t_mean"] for r in rows]
    t_mins  = [r["t_min"]  for r in rows]
    t_maxs  = [r["t_max"]  for r in rows]
    e_med   = [r["e_med"]  for r in rows]
    e_min   = [r["e_min"]  for r in rows]
    e_max   = [r["e_max"]  for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    
    
    ax1.fill_between(Ds_arr, t_mins, t_maxs, color=COL_DENSE, alpha=0.20)
    ax1.loglog(Ds_arr, t_means, "o-", color=COL_DENSE,
               label=r"RFF  (mean over $10$ seeds)", linewidth=2)
    ax1.axhline(t_lap, color=COL_OURS, linestyle="--", linewidth=2,
                label=fr"LAPLEX (exact) = ${t_lap:.2f}$\,ms")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"$D$  (number of RFF features)")
    ax1.set_ylabel("time [ms]")
    ax1.set_title(r"(a)  Runtime  ($n = k = 2^{16}$, $B = 1$)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    
    ax2.fill_between(Ds_arr, e_min, e_max, color=COL_DENSE, alpha=0.20,
                     label=r"RFF $[\min, \max]$ over $10$ seeds")
    ax2.plot(Ds_arr, e_med, "o-", color=COL_DENSE, linewidth=2,
             label="RFF median")
    
    ref = [e_med[0] * math.sqrt(Ds_arr[0] / D) for D in Ds_arr]
    ax2.plot(Ds_arr, ref, "k--", alpha=0.6, linewidth=1.5,
             label=r"$\propto 1/\sqrt{D}$ reference")
    ax2.axhline(0.0, color=COL_OURS, linestyle="--", linewidth=2,
                label="LAPLEX (exact, error = 0)")
    ax2.set_xscale("log", base=2)
    
    ax2.set_ylim(bottom=-0.2, top=max(e_max) * 1.05)
    ax2.set_xlabel(r"$D$  (number of RFF features)")
    ax2.set_ylabel(r"relative $\ell_2$ error  $\|\hat y - y\|_2 / \|y\|_2$")
    ax2.set_title(r"(b)  Approximation accuracy")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "rff_combined.pdf")
    png = os.path.join(OUT_DIR, "rff_combined.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"\nsaved: {pdf}\n       {png}")


if __name__ == "__main__":
    main()
