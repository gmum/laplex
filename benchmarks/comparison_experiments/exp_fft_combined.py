"""
Generate the 2-panel main-paper figure for the FFT comparison subsection:
  - left  panel: GPU runtime  (LAPLEX, Toeplitz-FFT, cuFFT)  vs  n
  - right panel: relative L2 error on uniform grid  (LAPLEX, Toeplitz-FFT, dense fp32)
                 vs  n,  measured against an fp64 dense reference.

Output:  fft_combined.{pdf,png}  inside  NeurIPS_2026/images/.
"""
from __future__ import annotations

import os, sys, time
import torch
import matplotlib.pyplot as plt
import pandas as pd

from _common import COL_OURS, COL_DENSE, COL_DENSE64, report_nan_when_error
import laplex as lap


DEVICE = "cuda"
DTYPE  = torch.float32

HN = 21
HA = 15

HN = 29
HA = 16

OUT_DIR = "./figures/fft_combined"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------- shared FFT helpers (symmetric Toeplitz) -----------------
def toeplitz_pre_fft(n, dx, dtype, device):
    idx = torch.arange(n, device=device, dtype=dtype)
    c = torch.exp(-idx * dx)
    emb = torch.zeros(2 * n, device=device, dtype=dtype)
    emb[:n] = c
    emb[n + 1:] = c[1:].flip(0)
    cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
    return torch.fft.fft(emb.to(cdt))


def toeplitz_matvec(x, C_fft):
    """x : (n,) -> y : (n,) via circulant embedding."""
    n = x.numel()
    cdt = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    x_pad = torch.zeros(2 * n, device=x.device, dtype=cdt)
    x_pad[:n] = x.to(cdt)
    return torch.fft.ifft(torch.fft.fft(x_pad) * C_fft)[:n].real


def toeplitz_matvec_batched(X, C_fft):
    B, n = X.shape
    cdt = torch.complex64 if X.dtype == torch.float32 else torch.complex128
    X_pad = torch.zeros(B, 2 * n, device=X.device, dtype=cdt)
    X_pad[:, :n] = X.to(cdt)
    Y = torch.fft.ifft(torch.fft.fft(X_pad, dim=-1) * C_fft.unsqueeze(0), dim=-1)
    return Y[:, :n].real


# ----------------- timing helper -----------------
@report_nan_when_error
def bench(fn, warmup=10, iters=50):
    def _one():
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3
    return min(_one() for _ in range(3))


# ----------------- runtime sweep (irregular anchors, B=1) -----------------
def runtime_sweep():
    print("--- Runtime sweep: irregular anchors, B=1 ---")
    print(f"{'n':>9s}  | {'LAPLEX [ms]':>12s} {'Toep [ms]':>10s} {'cuFFT [ms]':>11s}")
    rows = []
    for logN in range(10, HN):
        n = 1 << logN
        torch.manual_seed(n)
        a = torch.sort(torch.randn(n, device=DEVICE, dtype=DTYPE))[0]
        b = torch.randn(n, device=DEVICE, dtype=DTYPE)
        x = torch.randn(n, device=DEVICE, dtype=DTYPE)
        x_cplx = x.to(torch.complex64)

        # For Toeplitz-FFT we need a uniform grid; pick dx = 1/n so the
        # circulant embedding is well-conditioned.  This is just the
        # baseline-applicability setup; the runtime comparison itself
        # cares only about complexity, not about the kernel values.
        try:
            C_fft = toeplitz_pre_fft(n, 1.0 / n, DTYPE, DEVICE)
            t_toep = bench(lambda: toeplitz_matvec(x, C_fft))
            t_fft  = bench(lambda: torch.fft.fft(x_cplx))
        except (torch.AcceleratorError, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            t_fft = float("nan")


        t_lap  = bench(lambda: lap.X_Laplex(x.unsqueeze(0), a, b,
                                            threshold=0, a_sorted=True))

        print(f"2^{logN:<2d}={n:<6d}|  {t_lap:9.3f}    {t_toep:8.3f}    {t_fft:8.4f}")
        rows.append((n, t_lap, t_toep, t_fft))
        
        df = pd.DataFrame(rows, columns=["ns_r", "t_lap", "t_toep", "t_fft"])
        df.to_csv(f"{OUT_DIR}/runtime-results.csv", index=False)
        
    df = pd.DataFrame(rows, columns=["ns_r", "t_lap", "t_toep", "t_fft"])
    df.to_csv(f"{OUT_DIR}/runtime-results.csv", index=False)
    return rows


# ----------------- accuracy sweep (uniform grid) -----------------
def accuracy_sweep():
    print("\n--- Accuracy sweep: uniform grid, B=8 ---")
    print(f"{'n':>9s}  | {'dense fp32':>11s} {'Toep-FFT':>10s} {'LAPLEX':>10s}  (rel L2 vs fp64)")
    B = 8
    rows = []
    for logN in range(10, HA):
        n = 1 << logN
        dx = 1.0 / n
        torch.manual_seed(42 + logN)
        a = torch.arange(n, device=DEVICE, dtype=DTYPE) * dx
        X = torch.randn(B, n, device=DEVICE, dtype=DTYPE)

        # fp64 reference
        K64 = torch.exp(-(a.double().unsqueeze(1) - a.double().unsqueeze(0)).abs())
        Y_ref = X.double() @ K64

        # dense fp32
        K32 = torch.exp(-(a.unsqueeze(1) - a.unsqueeze(0)).abs())
        Y_d = X @ K32

        # Toeplitz-FFT fp32
        C_fft = toeplitz_pre_fft(n, dx, DTYPE, DEVICE)
        Y_t = toeplitz_matvec_batched(X, C_fft)

        # LAPLEX fp32
        Y_l = lap.X_Laplex(X, a, a, threshold=0, a_sorted=True)

        def rl2(Y): return (Y.double() - Y_ref).norm().item() / Y_ref.norm().item()
        e_d, e_t, e_l = rl2(Y_d), rl2(Y_t), rl2(Y_l)
        print(f"2^{logN:<2d}={n:<6d}|  {e_d:9.2e}   {e_t:8.2e}   {e_l:8.2e}")
        rows.append((n, e_d, e_t, e_l))
        
        df = pd.DataFrame(rows, columns=["ns_a", "e_d", "e_t", "e_l"])
        df.to_csv(f"{OUT_DIR}/accuracy-results.csv", index=False)
        
    df = pd.DataFrame(rows, columns=["ns_a", "e_d", "e_t", "e_l"])
    df.to_csv(f"{OUT_DIR}/accuracy-results.csv", index=False)
    return rows


# ----------------- combined figure -----------------
def main():
    print(f"GPU = {torch.cuda.get_device_name(0)}\n")
    accuracy = accuracy_sweep()
    runtime = runtime_sweep()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # --- Panel 1: runtime ---
    ns_r   = [r[0] for r in runtime]
    t_lap  = [r[1] for r in runtime]
    t_toep = [r[2] for r in runtime]
    t_fft  = [r[3] for r in runtime]
    ax1.loglog(ns_r, t_lap,  "o-", color=COL_OURS,    label=r"$\mathrm{LAPLEX}$ (irregular $a, b$)", linewidth=2)
    ax1.loglog(ns_r, t_toep, "^-", color=COL_DENSE,   label="Toeplitz--FFT (uniform grid only)",     linewidth=2)
    ax1.loglog(ns_r, t_fft,  "s-", color=COL_DENSE64, label="cuFFT (single FFT, no kernel)",         linewidth=2)
    # O(n log n) reference fitted to the last cuFFT point
    n_ref = torch.tensor(ns_r, dtype=torch.float64)
    nlogn = (n_ref * torch.log2(n_ref)).tolist()
    scale = t_fft[-1] / nlogn[-1]
    ax1.loglog(ns_r, [v * scale for v in nlogn], "k--", alpha=0.4,
               label=r"$\mathcal{O}(n \log n)$")
    ax1.set_xlabel(r"$n = k$  (square kernel, $B = 1$)")
    ax1.set_ylabel("time [ms]")
    ax1.set_title("(a) GPU runtime")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9, loc="upper left")

    # --- Panel 2: accuracy ---
    ns_a    = [r[0] for r in accuracy]
    e_d     = [r[1] for r in accuracy]
    e_t     = [r[2] for r in accuracy]
    e_l     = [r[3] for r in accuracy]
    ax2.loglog(ns_a, e_l, "o-", color=COL_OURS,    label="LAPLEX fp32",        linewidth=2)
    ax2.loglog(ns_a, e_t, "^-", color=COL_DENSE,   label="Toeplitz--FFT fp32", linewidth=2)
    ax2.loglog(ns_a, e_d, "s-", color=COL_DENSE64, label="dense fp32",         linewidth=2)
    eps = torch.finfo(torch.float32).eps
    ax2.axhline(eps, color="k", linestyle="--", alpha=0.4, label=r"fp32 $\varepsilon$")
    ax2.set_xlabel(r"$n = k$  (uniform grid)")
    ax2.set_ylabel(r"relative $\ell_2$ error vs fp64 dense")
    ax2.set_title("(b) Accuracy on a Toeplitz problem")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "fft_combined.pdf")
    png = os.path.join(OUT_DIR, "fft_combined.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"\nsaved: {pdf}\n       {png}")


if __name__ == "__main__":
    main()
