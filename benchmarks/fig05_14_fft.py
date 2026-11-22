"""Fig. 5 (Sec. 6) and Fig. 14 (App. E.1): LAPLEX vs Toeplitz-FFT vs cuFFT.

(a) GPU runtime of one matvec, B = 1, n = k = 2^10 ... 2^28, LAPLEX on
    irregular anchors a, b ~ N(0, 1), Toeplitz-FFT on its uniform-grid
    specialisation (circulant embedding + 3 FFTs), and a single cuFFT call
    as a hardware lower bound.  Every method runs until it does not fit.
    Fig. 5 is panel (a) alone.
(b) Relative l2 error against an fp64 dense reference on the uniform-grid
    problem, B = 8, n = 2^10 ... 2^15 (same data as the l2 panel of Fig. 16).

    python fig05_14_fft.py [--quick] [--acc-max-log 15]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C
import impls
import plotting as P

FIG_RT = "fig05_14a_fft_runtime"
FIG_ACC = "fig14b_fft_accuracy"


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=28)
    p.add_argument("--acc-max-log", type=int, default=15)
    p.add_argument("--acc-batch", type=int, default=8)
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.PAPER_COMPARISON_IMPL,
                   help="the paper used the released laplex.py here")
    return p


def run(args):
    proto, tag, info = C.init(args)
    lap = impls.x_laplex(args.impl)

    def make_inputs(n, seed):
        torch.manual_seed(seed)
        a = torch.randn(n, device="cuda")
        # torch.sort takes at most INT_MAX elements; beyond that LAPLEX stops with LIMIT before
        # it reads a, and the other methods do not need it sorted
        a = torch.sort(a)[0] if n <= C.INT32_MAX else a
        b = torch.randn(n, device="cuda")
        X = torch.randn(1, n, device="cuda")
        return dict(a=a, b=b, X=X)

    def toeplitz(d):
        n = d["X"].shape[1]
        C_fft = impls.toeplitz_pre_fft(n, 1.0 / n, torch.float32, "cuda")
        fn = lambda: impls.toeplitz_matvec(d["X"], C_fft)  # noqa: E731
        fn.input_bytes = C.tensor_bytes([d["X"], C_fft])
        return fn

    def cufft(d):
        x_c = d["X"][0].to(torch.complex64)
        fn = lambda: torch.fft.fft(x_c)  # noqa: E731
        fn.input_bytes = C.tensor_bytes(x_c)
        return fn

    ops = [C.Op("laplex", limit=C.int32_limit(lambda n: n),
                fwd=lambda d: (lambda: lap(d["X"], d["a"], d["b"]))),
           C.Op("toeplitz", fwd=toeplitz),
           C.Op("cufft", fwd=cufft)]
    sweep = C.Sweep(FIG_RT, tag, "cuda", proto, axis="n", extra=dict(B=1, impl=args.impl))
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                       modes=("fwd",), seed=args.seed)
    sweep.save(meta=dict(env=info))

    acc_proto = C.Protocol(repeats=1)
    sweep = C.Sweep(FIG_ACC, tag, "cuda", acc_proto, axis="n",
                    extra=dict(B=args.acc_batch, impl=args.impl))
    C.run_accuracy_sweep(
        sweep, [1 << l for l in range(args.min_log, args.acc_max_log + 1)],
        lambda n, s: impls.toeplitz_problem(n, args.acc_batch, s), impls.toeplitz_methods(lap),
        seed_of=lambda n, r: args.seed + 42 + n.bit_length() - 1)
    sweep.save(meta=dict(env=info))
    summarize(tag)


def summarize(tag):
    a = P.aggregate(C.load_results(FIG_RT, [tag]))
    t = {m: a[(a.method == m) & (a.metric == "time_ms")].set_index("x")["min"]
         for m in ("laplex", "toeplitz", "cufft")}
    xs = sorted(set.intersection(*[set(v.index) for v in t.values()]))
    out = {}
    if xs:
        x = 1 << 20 if (1 << 20) in xs else xs[-1]
        out["ratio_laplex_toeplitz_cufft"] = dict(
            n=int(x), ms={m: v[x] for m, v in t.items()},
            normalised_to_toeplitz={m: v[x] / t["toeplitz"][x] for m, v in t.items()})
    C.write_summary(FIG_RT, tag, out)


def _runtime(ax, df, agg, tags, suffix=""):
    for i, t in enumerate(tags):
        sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
        for m, lbl, col, mk in (("laplex", "LAPLEX", P.BLUE, "o"), ("toeplitz", "Toeplitz-FFT", P.ORANGE, "^"),
                                ("cufft", "cuFFT" + suffix, P.GREEN, "s")):
            xy = P.draw(ax, df, agg, method=m, mode="fwd", metric="time_ms", device=t, color=col, marker=mk,
                        label=lbl + sfx, linestyle=P.DEVICE_STYLES[i % 4])
            if m == "cufft" and xy is not None and i == 0:
                x, y = xy
                nlogn = x * np.log2(x)
                ax.plot(x, nlogn * y[-1] / nlogn[-1], "--", color="0.3", linewidth=2,
                        label=r"$\mathcal{O}(n\log n)$" + suffix)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Values of $n$")
    ax.set_ylabel("Time (ms)")


def plot(args):
    tags = C.plot_tags(args)
    df_rt = C.load_results(FIG_RT, tags)
    agg_rt = P.aggregate(df_rt)
    out = C.figure_dir(FIG_RT, tags)

    fig, axes = P.figure("fig05")
    _runtime(axes[0], df_rt, agg_rt, tags)
    P.finish(fig, axes, out, "fig05", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(df_rt)))

    fig, axes = P.figure("fig14", ncols=2)
    _runtime(axes[0], df_rt, agg_rt, tags, suffix=" (runtime only)")
    df_acc = C.load_results(FIG_ACC, tags)
    agg_acc = P.aggregate(df_acc)
    for i, t in enumerate(tags):
        sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
        for m, lbl, col, mk in (("laplex", "LAPLEX", P.BLUE, "o"), ("toeplitz", "Toeplitz-FFT", P.ORANGE, "^"),
                                ("dense", "Dense (accuracy only)", P.VERMILLION, "D")):
            P.draw(axes[1], df_acc, agg_acc, method=m, mode="value", metric="rel_l2", device=t, color=col,
                   marker=mk, label=lbl + sfx, stat="median", linestyle=P.DEVICE_STYLES[i % 4])
    axes[1].axhline(torch.finfo(torch.float32).eps, color="black", linestyle=":", linewidth=2,
                    label=r"FP32 $\varepsilon$ (accuracy only)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Values of $n$")
    axes[1].set_ylabel(r"Relative $\ell_2$ Error")
    import pandas as pd
    P.finish(fig, axes, out, "fig14", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(pd.concat([df_rt, df_acc]))))


if __name__ == "__main__":
    C.main(parser(), run, plot)
