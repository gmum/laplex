"""Fig. 9 (App. D.3): relative error of fp32 X K(a, b) against an fp64 dense reference.

Square kernel, B = 8, n = 2^10 ... 2^14 (published range), irregular anchors.
LAPLEX fp32 vs explicit-dense fp32; the same fp32 inputs are cast to fp64 for
the reference.  The fp64 dense reference bounds the range: with --max-log the
sweep continues until the reference itself no longer fits.

With the released code (``--impl public-raw``) the point n = 2^10 is a dense
fp32 matmul (n*k <= 2^20 shortcut), which is why both published curves start
at the same value.

    python fig09_accuracy.py [--max-log 16]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig09_accuracy"


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=14, repeats=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.DEFAULT_IMPL)
    return p


def run(args):
    proto, tag, info = C.init(args)
    lap = impls.x_laplex(args.impl)
    B = args.batch

    def make_problem(n, seed):
        torch.manual_seed(seed)
        a = torch.sort(torch.randn(n, device="cuda"))[0]
        b = torch.randn(n, device="cuda")
        X = torch.randn(B, n, device="cuda")
        Y_ref = X.double() @ impls.dense_kernel(a.double(), b.double())
        return dict(a=a, b=b, X=X, Y_ref=Y_ref)

    methods = {"laplex": lambda p: lap(p["X"], p["a"], p["b"]),
               "dense": lambda p: impls.x_dense(p["X"], p["a"], p["b"])}
    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="n", extra=dict(B=B, impl=args.impl))
    # Seeds as in the original script (42 + log2 n), shifted by repeat.
    C.run_accuracy_sweep(sweep, [1 << l for l in C.log_range(args)], make_problem, methods,
                         seed_of=lambda n, r: args.seed + 42 + n.bit_length() - 1 + 1000 * r)
    sweep.save(meta=dict(env=info))


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig09", ncols=2)
    axes[1].sharey(axes[0])
    axes[1].tick_params(which="both", labelleft=False)
    for ax, (metric, title) in zip(axes, [("rel_l2", r"Relative $\ell_2$ Error"),
                                          ("rel_linf", r"Relative $\ell_\infty$ Error")]):
        for i, t in enumerate(tags):
            sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
            for m, lbl in (("laplex", "LAPLEX (ours) FP32"), ("dense", "Dense FP32")):
                P.draw(ax, df, agg, method=m, mode="value", metric=metric, device=t,
                       label=lbl + sfx, stat="median", linestyle=P.DEVICE_STYLES[i % 4])
        ax.axhline(torch.finfo(torch.float32).eps, color="0.5", linestyle="--", linewidth=1.5,
                   label=r"FP32 $\varepsilon$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Values of $n$")
    axes[0].set_ylabel("Relative Error")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig09", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(df)))


if __name__ == "__main__":
    C.main(parser(), run, plot)
