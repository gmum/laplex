"""Fig. 12 (App. D.6): weighted Gram matrix K(a, b) diag(D) K(a, b)^T.

Output dimension fixed at n = 1024 (the result is always 1024 x 1024), inner
dimension k = 2^10 ... 2^30, fp32.  Forward time and peak memory of LAPLEX vs
the explicit-dense (K * D) @ K^T.  The original script stopped the dense
baseline at a 500 ms / 1.5 GB budget; here it runs until it does not fit.

Writes the speed-up and memory ratio at the largest k where dense still runs
(App. D.6/D.8, "149x at k = 2^17") to summary.json.

    python fig12_gram.py [--quick]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig12_gram"


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=30)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.DEFAULT_IMPL)
    return p


def run(args):
    proto, tag, info = C.init(args)
    gram = impls.gram_laplex(args.impl)
    n = args.n

    def make_inputs(k, seed):
        torch.manual_seed(seed)
        alpha = torch.sort(torch.randn(n, device="cuda"))[0]
        beta = torch.randn(k, device="cuda")
        D = torch.randn(k, device="cuda")
        return dict(alpha=alpha, beta=beta, D=D)

    ops = [C.Op("laplex", limit=C.int32_limit(lambda k: k),
                fwd=lambda d: (lambda: gram(d["alpha"], d["beta"], d["D"]))),
           C.Op("dense", fwd=lambda d: (lambda: impls.gram_dense(d["alpha"], d["beta"], d["D"])))]
    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="k", extra=dict(n=n, impl=args.impl))
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                       modes=("fwd",), seed=args.seed)
    sweep.save(meta=dict(env=info))
    summarize(tag)


def summarize(tag):
    df = C.load_results(FIG, [tag])
    a = P.aggregate(df)

    def best(m, metric):
        return a[(a.method == m) & (a.metric == metric)].set_index("x")["min"]

    out = {}
    for metric in ("time_ms", "mem_mib"):
        o, d = best("laplex", metric), best("dense", metric)
        xs = sorted(set(o.index) & set(d.index))
        if xs:
            x = xs[-1]
            out[metric] = dict(k=int(x), log2k=int(x).bit_length() - 1, laplex=o[x], dense=d[x],
                               ratio=d[x] / o[x])
    out["dense_stop"] = P.stop_of(df, method="dense", mode="fwd", metric="time_ms")
    out["laplex_stop"] = P.stop_of(df, method="laplex", mode="fwd", metric="time_ms")
    C.write_summary(FIG, tag, out)


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig12", ncols=2)
    for ax, (metric, ylab) in zip(axes, [("time_ms", "Time (ms)"), ("mem_mib", "Peak Memory (MiB)")]):
        for i, t in enumerate(tags):
            sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
            for m in ("laplex", "dense"):
                P.draw(ax, df, agg, method=m, mode="fwd", metric=metric, device=t,
                       label=P.LABELS[m] + sfx, linestyle=P.DEVICE_STYLES[i % 4])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Values of $k$")
        ax.set_ylabel(ylab)
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig12", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(df)))


if __name__ == "__main__":
    C.main(parser(), run, plot)
