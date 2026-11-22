"""Fig. 18 (App. F.2): RFF gradient accuracy on the Laplace kernel, n = k = 2^16.

Back-propagates L = 1^T y through the RFF estimator and through LAPLEX (the
exact reference) and reports the relative l2 error of y, dL/da and dL/db for
D = 2^0 ... 2^11 (published range; --max-log continues until RFF does not
fit), 10 seeds per D.  Left: median and [min, max]; right: max / median.

    python fig18_rff_gradient.py [--max-log 14]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C
import impls
import plotting as P

FIG = "fig18_rff_gradient"
SERIES = [("value", "Value", P.BLUE, "o"), ("grad_a", "Grad $a$", P.ORANGE, "s"),
          ("grad_b", "Grad $b$", P.GREEN, "^")]


def parser():
    p = C.make_parser(__doc__, min_log=0, max_log=11)
    p.add_argument("--log-n", type=int, default=16)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.PAPER_COMPARISON_IMPL,
                   help="the paper used the released laplex.py here")
    return p


def rel(g, ref):
    return (g - ref).norm().item() / ref.norm().item()


def run(args):
    proto, tag, info = C.init(args)
    lap = impls.x_laplex(args.impl)
    n = 1 << args.log_n
    seeds = 2 if args.quick else args.seeds
    torch.manual_seed(args.seed)
    a0 = torch.sort(torch.randn(n, device="cuda"))[0]
    b0 = torch.randn(n, device="cuda")
    x0 = torch.randn(n, device="cuda")

    a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
    y_gt = lap(x0.unsqueeze(0), a, b)[0]
    y_gt.sum().backward()
    ref = dict(value=y_gt.detach(), grad_a=a.grad.detach().clone(), grad_b=b.grad.detach().clone())
    del a, b, y_gt
    print({k: f"{v.norm().item():.4e}" for k, v in ref.items()}, flush=True)

    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="D",
                    extra=dict(n=n, impl=args.impl, seeds=seeds))
    ser = {k: C.Series("rff", k, "rel_l2") for k, *_ in SERIES}
    for D in [1 << l for l in C.log_range(args)]:
        if not sweep.alive(("rff", "all")):
            break

        def body(D=D):
            for s in range(seeds):
                a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
                y = impls.rff_matvec(x0, a, b, D, seed=1000 * D + s)
                y.sum().backward()
                got = dict(value=y.detach(), grad_a=a.grad, grad_b=b.grad)
                for k in ser:
                    sweep.record(ser[k], D, s, rel(got[k], ref[k]))
                del y, a, b, got
        body.series = list(ser.values())
        sweep.measure(("rff", "all"), D, 0, body)
        C.clear("cuda")
        C.print_point(sweep, D, 0)
        sweep.save()
    sweep.save(meta=dict(env=info, ref_norms={k: v.norm().item() for k, v in ref.items()}))


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig18", ncols=2)
    ref = None
    for i, t in enumerate(tags):
        ls = P.DEVICE_STYLES[i % 4]
        sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
        for k, lbl, col, mk in SERIES:
            r = agg[(agg.device == t) & (agg["mode"] == k)].sort_values("x")
            if r.empty:
                continue
            axes[0].fill_between(r.x, r["min"], r["max"], color=col, alpha=0.15, lw=0)
            P.draw(axes[0], df, agg, method="rff", mode=k, metric="rel_l2", device=t,
                   stat="median", color=col, marker=mk, label=lbl + sfx, linestyle=ls)
            axes[1].plot(r.x, r["max"] / r["median"], ls, color=col, marker=mk, label=lbl + sfx)
            if k == "value" and i == 0:
                ref = r.x.to_numpy(float), r["median"].iloc[0]
    if ref is not None:   # drawn last so that it ends the legend, as in the paper
        D, m0 = ref
        axes[0].plot(D, m0 * np.sqrt(D[0] / D), "--", color="0.5", linewidth=2, label=r"$\propto 1/\sqrt{D}$")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of RFF Features $D$")
        h, l = ax.get_legend_handles_labels()
        extra = P.limit_legend_handles(df) if ax is axes[0] else []   # the OOM marker of the left panel
        ax.legend(h + extra, l + [x.get_label() for x in extra], loc="upper right", frameon=True, edgecolor="0.8")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title("Value and Gradient")
    axes[1].set_ylabel("Error Max/Median")
    axes[1].set_title("Heavy Tail: Worst-Case vs. Median Ratio")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig18", args.ylim_json)


if __name__ == "__main__":
    C.main(parser(), run, plot)
