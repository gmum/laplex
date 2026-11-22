"""Fig. 17 (App. E.3): CPU runtime of LAPLEX (no CUDA), Toeplitz-FFT and one FFT.

B = 1, fp32, n = 2^10 ... 2^20 (range of the published figure and script; the
text of App. E.3 says 2^18), single CPU thread as stated in the caption
(--threads to change).  LAPLEX runs on irregular anchors, Toeplitz-FFT on the
uniform grid, the FFT on a complex64 copy of the input.  Timing as in the
original CPU scripts: best of 3 runs of 10 calls after 3 warm-up calls.

    python fig17_cpu_fft.py [--threads 1]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C
import impls
import plotting as P

FIG = "fig17_cpu_fft"


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=20, warmup=3, iters=10, device="cpu")
    p.add_argument("--threads", type=int, default=1)
    return p


def run(args):
    C.set_threads(args.threads)
    proto, tag, info = C.init(args)
    lap = impls.x_laplex_cpu()

    def make_inputs(n, seed):
        torch.manual_seed(seed)
        a = torch.sort(torch.randn(n))[0]
        X = torch.randn(1, n)
        return dict(a=a, b=torch.randn(n), X=X, x_c=X[0].to(torch.complex64),
                    C_fft=impls.toeplitz_pre_fft(n, 1.0 / n, torch.float32, "cpu"))

    ops = [C.Op("laplex_cpu", fwd=lambda d: (lambda: lap(d["X"], d["a"], d["b"])),
                predict_bytes=lambda n: 200 * n),
           C.Op("toeplitz", fwd=lambda d: (lambda: impls.toeplitz_matvec(d["X"], d["C_fft"])),
                predict_bytes=lambda n: 64 * n),
           C.Op("fft", fwd=lambda d: (lambda: torch.fft.fft(d["x_c"])), predict_bytes=lambda n: 32 * n)]
    sweep = C.Sweep(FIG, tag, "cpu", proto, axis="n",
                    extra=dict(B=1, threads=torch.get_num_threads()))
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                       modes=("fwd",), seed=args.seed,
                       inputs_bytes=lambda n: 64 * n)   # a, b, X, x_c, C_fft (2n complex) + FFT temporaries
    sweep.save(meta=dict(env=info))
    a = P.aggregate(C.load_results(FIG, [tag]))
    last = a["x"].max()
    t = {m: a[(a.method == m) & (a.x == last)]["min"].iloc[0] for m in a.method.unique()}
    C.write_summary(FIG, tag, dict(n=int(last), ms=t,
                                   ratio_to_fft={m: v / t["fft"] for m, v in t.items()}))


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig17")
    ax = axes[0]
    for i, t in enumerate(tags):
        sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
        for m, lbl, col, mk in (("laplex_cpu", "LAPLEX (no CUDA)", P.BLUE, "o"),
                                ("toeplitz", "Toeplitz-FFT (CPU)", P.ORANGE, "^"),
                                ("fft", "Pure FFT (CPU)", "black", "s")):
            xy = P.draw(ax, df, agg, method=m, mode="fwd", metric="time_ms", device=t,
                        color=col, marker=mk, label=lbl + sfx, linestyle=P.DEVICE_STYLES[i % 4])
            if m == "fft" and xy is not None and i == 0:
                x, y = xy
                ax.plot(x, x * np.log2(x) * y[-1] / (x[-1] * np.log2(x[-1])), "--", color="0.5",
                        linewidth=2.5, label=r"$\mathcal{O}(n\log n)$ Reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Values of $n$")
    ax.set_ylabel("Time (ms)")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig17", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(df)))


if __name__ == "__main__":
    C.main(parser(), run, plot)
