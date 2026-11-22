"""Fig. 16 (App. E.2): accuracy on a uniform-grid (Toeplitz) problem.

a_i = b_i = i / n, so K(a, a) is symmetric Toeplitz and Toeplitz-FFT applies.
B = 8, n = 2^10 ... 2^15 (as in the original script), fp32.  Relative l2 and
l_inf error of dense fp32, Toeplitz-FFT fp32 and LAPLEX fp32 against an fp64
dense reference.  The fp64 reference bounds the range (--max-log continues
until it no longer fits).

    python fig16_toeplitz_accuracy.py [--max-log 16]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig16_toeplitz_accuracy"
LBL = {"laplex": "LAPLEX (ours)", "toeplitz": "Toeplitz-FFT", "dense": "Dense FP32"}


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=15, repeats=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.PAPER_COMPARISON_IMPL,
                   help="the paper used the released laplex.py here")
    return p


def run(args):
    proto, tag, info = C.init(args)
    methods = impls.toeplitz_methods(impls.x_laplex(args.impl))
    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="n", extra=dict(B=args.batch, impl=args.impl))
    C.run_accuracy_sweep(
        sweep, [1 << l for l in C.log_range(args)],
        lambda n, s: impls.toeplitz_problem(n, args.batch, s), methods,
        seed_of=lambda n, r: args.seed + 42 + n.bit_length() - 1 + 1000 * r)
    sweep.save(meta=dict(env=info))


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig16", ncols=2)
    axes[1].sharey(axes[0])
    axes[1].tick_params(which="both", labelleft=False)
    for ax, metric in zip(axes, ("rel_l2", "rel_linf")):
        for i, t in enumerate(tags):
            sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
            for m, col, mk in (("laplex", P.BLUE, "o"), ("toeplitz", P.ORANGE, "^"), ("dense", P.GREEN, "s")):
                P.draw(ax, df, agg, method=m, mode="value", metric=metric, device=t, color=col, marker=mk,
                       label=LBL[m] + sfx, stat="median", linestyle=P.DEVICE_STYLES[i % 4])
        ax.axhline(torch.finfo(torch.float32).eps, color="0.5", linestyle="--", linewidth=1.5,
                   label=r"FP32 $\varepsilon$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Values of $n$")
    axes[0].set_ylabel("Relative Error")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig16", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(df)))


if __name__ == "__main__":
    C.main(parser(), run, plot)
