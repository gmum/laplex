"""Fig. 11 (App. D.5): asymmetric kernels with one axis fixed at 1024, B = 8.

Left : n = 1024 fixed, k = 2^10 ... 2^28 (branch A: scan along a, gather per b).
Right: k = 1024 fixed, n = 2^10 ... 2^28 (branch B: scatter into sorted b, scan).
Forward time (and peak memory) of LAPLEX vs explicit dense, fp32.  The
original script stopped the dense baseline at a 500 ms / 1.5 GB budget; here it
runs until it does not fit in GPU memory.

Writes the speed-up at the largest size where dense still runs (App. D.5/D.8,
"~37-47x") to summary.json.

    python fig11_asymmetric.py [--quick] [--panel n|k|both]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig11_asymmetric"
PANELS = {"n": ("fig11_asymmetric_nfixed", "k", "$n = 1024$ fixed"),
          "k": ("fig11_asymmetric_kfixed", "n", "$k = 1024$ fixed")}


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=28)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--fixed", type=int, default=1024)
    p.add_argument("--panel", choices=["n", "k", "both"], default="both")
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.DEFAULT_IMPL)
    return p


def run(args):
    proto, tag, info = C.init(args)
    lap = impls.x_laplex(args.impl)
    B, F = args.batch, args.fixed
    ops = [C.Op("laplex", limit=C.int32_limit(lambda v: B * v),   # X or output is (B, v)
                fwd=lambda d: (lambda: lap(d["X"], d["a"], d["b"]))),
           C.Op("dense", fwd=lambda d: (lambda: impls.x_dense(d["X"], d["a"], d["b"])))]
    summary = {}
    for panel in (["n", "k"] if args.panel == "both" else [args.panel]):
        fig_id, axis, _ = PANELS[panel]

        def make_inputs(v, seed, panel=panel):
            n, k = (F, v) if panel == "n" else (v, F)
            torch.manual_seed(seed)
            a = torch.sort(torch.randn(n, device="cuda"))[0]
            b = torch.randn(k, device="cuda")
            X = torch.randn(B, n, device="cuda")
            return dict(a=a, b=b, X=X)

        print(f"--- {panel} = {F} fixed, sweeping {axis} ---", flush=True)
        sweep = C.Sweep(fig_id, tag, "cuda", proto, axis=axis,
                        extra=dict(B=B, fixed=F, panel=panel, impl=args.impl))
        C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                           modes=("fwd",), seed=args.seed)
        sweep.save(meta=dict(env=info))
        summary[panel + "_fixed"] = speedup_at_last_dense(C.load_results(fig_id, [tag]))
    C.write_summary(FIG, tag, summary)


def speedup_at_last_dense(df):
    a = P.aggregate(df)
    o = a[(a.method == "laplex") & (a.metric == "time_ms")].set_index("x")["min"]
    d = a[(a.method == "dense") & (a.metric == "time_ms")].set_index("x")["min"]
    xs = sorted(set(o.index) & set(d.index))
    if not xs:
        return None
    x = xs[-1]
    return dict(size=int(x), log2=int(x).bit_length() - 1, laplex_ms=o[x], dense_ms=d[x],
                speedup=d[x] / o[x], dense_stop=P.stop_of(df, method="dense", mode="fwd",
                                                          metric="time_ms"))


def plot(args):
    tags = C.plot_tags(args)
    fig, axes = P.figure("fig11", ncols=2)
    import pandas as pd
    frames = []
    for ax, panel in zip(axes, ["n", "k"]):
        fig_id, axis, title = PANELS[panel]
        df = C.load_results(fig_id, tags)
        frames.append(df)
        agg = P.aggregate(df)
        for i, t in enumerate(tags):
            sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
            for m in ("laplex", "dense"):
                P.draw(ax, df, agg, method=m, mode="fwd", metric="time_ms", device=t,
                       label=P.LABELS[m] + sfx, linestyle=P.DEVICE_STYLES[i % 4])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(f"Values of ${axis}$")
    axes[0].set_ylabel("Time (ms)")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig11", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(pd.concat(frames))))


if __name__ == "__main__":
    C.main(parser(), run, plot)
