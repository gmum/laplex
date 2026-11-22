import os, sys, time, math
import torch
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laplex import X_Laplex as X_KerLap   


OUT_DIR = "./figures/exp_rff_grad"
os.makedirs(OUT_DIR, exist_ok=True)


def rff_matvec_diff(x, a, b, D, seed):
    
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


def rel_err(g, g_ref):
    return (g - g_ref).norm().item() / g_ref.norm().item()


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    median = xs[n // 2]
    mean   = sum(xs) / n
    p95    = xs[int(0.95 * (n - 1))]
    return dict(mean=mean, median=median, min=xs[0], max=xs[-1], p95=p95)


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(0)

    print(f"device = {torch.cuda.get_device_name(0)}")
    print(f"dtype  = {dtype},  B = 1,  n = k = 2^16 = 65536\n")

    n = 1 << 16
    a0 = torch.sort(torch.randn(n, device=device, dtype=dtype))[0]
    b0 = torch.randn(n, device=device, dtype=dtype)
    x0 = torch.randn(n, device=device, dtype=dtype)

    
    print("Computing ground truth (KerLap forward + backward) ...")
    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    y_gt = X_KerLap(x0, a, b, threshold=0, a_sorted=True)
    L = y_gt.sum()
    L.backward()
    ga_gt = a.grad.detach().clone()
    gb_gt = b.grad.detach().clone()
    y_gt_d = y_gt.detach()
    del y_gt, L, a, b
    torch.cuda.empty_cache()

    print(f"  ||y||   = {y_gt_d.norm().item():.4e}")
    print(f"  ||g_a|| = {ga_gt.norm().item():.4e}")
    print(f"  ||g_b|| = {gb_gt.norm().item():.4e}\n")

    
    Ds = [1 << i for i in range(0, 12)]   
    N_REPS = 10

    hdr = (f"  {'D':>5s}  "
           f"{'val med':>9s}  {'val max':>9s}  "
           f"{'grad_a med':>11s}  {'grad_a max':>11s}  "
           f"{'grad_b med':>11s}  {'grad_b max':>11s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for D in Ds:
        val_errs, ga_errs, gb_errs = [], [], []
        for rep in range(N_REPS):
            seed = 1000 * D + rep
            a = a0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            y = rff_matvec_diff(x0, a, b, D, seed=seed)
            L = y.sum()
            L.backward()
            val_errs.append(rel_err(y.detach(), y_gt_d))
            ga_errs.append(rel_err(a.grad.detach(), ga_gt))
            gb_errs.append(rel_err(b.grad.detach(), gb_gt))
            del y, L, a, b
            torch.cuda.empty_cache()

        sv = stats(val_errs); sa = stats(ga_errs); sb = stats(gb_errs)
        rows.append((D, sv, sa, sb))
        print(f"  {D:5d}  "
              f"{sv['median']:7.2e}  {sv['max']:7.2e}  "
              f"{sa['median']:9.2e}  {sa['max']:9.2e}  "
              f"{sb['median']:9.2e}  {sb['max']:9.2e}")
              
        df = pd.DataFrame(rows, columns=["D", "sv", "sa", "sb"])
        df.to_csv(f"{OUT_DIR}/exp_rff_grad-results.csv", index=False)

    df = pd.DataFrame(rows, columns=["D", "sv", "sa", "sb"])
    df.to_csv(f"{OUT_DIR}/exp_rff_grad-results.csv", index=False)
    
    
    
    
    Ds_arr = [r[0] for r in rows]
    val_med = [r[1]['median'] for r in rows]
    val_min = [r[1]['min']    for r in rows]
    val_max = [r[1]['max']    for r in rows]

    ga_med = [r[2]['median'] for r in rows]
    ga_min = [r[2]['min']    for r in rows]
    ga_max = [r[2]['max']    for r in rows]

    gb_med = [r[3]['median'] for r in rows]
    gb_min = [r[3]['min']    for r in rows]
    gb_max = [r[3]['max']    for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    def plot_band(ax, xs, med, lo, hi, color, label, marker):
        ax.plot(xs, med, marker + "-", color=color, label=label, linewidth=2)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.20)

    
    plot_band(ax1, Ds_arr, val_med, val_min, val_max, "C0", "value  err", "o")
    plot_band(ax1, Ds_arr, ga_med,  ga_min,  ga_max,  "C1", r"grad $a$ err", "s")
    plot_band(ax1, Ds_arr, gb_med,  gb_min,  gb_max,  "C2", r"grad $b$ err", "^")
    
    ref_inv_sqrt = [val_med[0] * math.sqrt(Ds_arr[0] / D) for D in Ds_arr]
    ax1.plot(Ds_arr, ref_inv_sqrt, "k--", alpha=0.4, label=r"$\propto 1/\sqrt{D}$")

    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xlabel("D  (number of RFF features)")
    ax1.set_ylabel("relative error (median; band = [min, max] across 10 seeds)")
    ax1.set_title(r"RFF vs KerLap: value and gradient  ($n = k = 2^{16}$)")
    ax1.grid(True, which="both", alpha=0.3); ax1.legend()

    
    
    all_D, all_ga = [], []
    for D in Ds:
        for rep in range(N_REPS):
            all_D.append(D)
            all_ga.append(float('nan'))  

    ratio_max_med = [m / (me + 1e-30) for m, me in zip(ga_max, ga_med)]
    ratio_val_med = [m / (me + 1e-30) for m, me in zip(val_max, val_med)]
    ax2.semilogx(Ds_arr, ratio_val_med, "o-", color="C0",
                label="value:  max / median", linewidth=2)
    ax2.semilogx(Ds_arr, ratio_max_med, "s-", color="C1",
                label=r"grad $a$: max / median", linewidth=2)

    ratio_b_max_med = [m / (me + 1e-30) for m, me in zip(gb_max, gb_med)]
    ax2.semilogx(Ds_arr, ratio_b_max_med, "^-", color="C2",
                label=r"grad $b$: max / median", linewidth=2)

    ax2.set_xlabel("D")
    ax2.set_ylabel("max / median error (across 10 seeds)")
    ax2.set_title("Heavy tail: worst-case to median ratio")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    ax2.axhline(1.0, color="k", linewidth=0.5)

    fig.tight_layout()

    out_png = os.path.join(OUT_DIR, "exp_rff_gradient.png")
    fig.savefig(out_png, dpi=130)
    fig.savefig(out_png.replace(".png", ".pdf"))

    print(f"\nsaved: {out_png}\n       {out_png.replace('.png', '.pdf')}")

if __name__ == "__main__":
    main()
