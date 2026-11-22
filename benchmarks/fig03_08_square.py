"""Fig. 3 (Sec. 4.1) = Fig. 8 (App. D.2): X K(a, b) for a square kernel, B = 8.

Forward time, forward+backward time and peak memory of the forward call for
LAPLEX and the explicit-dense baseline, n = k = 2^10 ... 2^20, fp32.  Both
methods run until they no longer fit in GPU memory.

Also writes the headline ratios used in Sec. 4.1, App. D.2 and D.8 (forward
speed-up, forward+backward speed-up and peak-memory reduction at the largest n
where both methods run) to summary.json.

    python fig03_08_square.py                 # run + plot, paper protocol
    python fig03_08_square.py --quick         # development preset
    python fig03_08_square.py --stage plot --tags H100 RTX4090
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig03_08_square"


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.DEFAULT_IMPL)
    return p


def run(args):
    proto, tag, info = C.init(args)
    lap = impls.x_laplex(args.impl)
    B = args.batch

    def make_inputs(n, seed):
        torch.manual_seed(seed)
        a = torch.sort(torch.randn(n, device="cuda"))[0]
        b = torch.randn(n, device="cuda")
        X = torch.randn(B, n, device="cuda")
        return dict(a=a, b=b, X=X)

    ops = [C.Op("laplex", limit=C.int32_limit(lambda n: B * n),
                fwd=lambda d: (lambda: lap(d["X"], d["a"], d["b"])),
                fb=lambda d: C.fb_step(lap, d["X"], d["a"], d["b"])),
           C.Op("dense", fwd=lambda d: (lambda: impls.x_dense(d["X"], d["a"], d["b"])),
                fb=lambda d: C.fb_step(impls.x_dense, d["X"], d["a"], d["b"]))]
    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="n", extra=dict(B=B, impl=args.impl))
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                       seed=args.seed)
    sweep.save(meta=dict(env=info))
    summarize(tag)


def summarize(tag):
    df = C.load_results(FIG, [tag])
    a = P.aggregate(df)

    def best(method, mode, metric):
        s = a[(a.method == method) & (a["mode"] == mode) & (a.metric == metric)]
        return s.set_index("x")["min"]

    out = {}
    for mode, metric, name in [("fwd", "time_ms", "forward_speedup"),
                               ("fb", "time_ms", "fwd_bwd_speedup"),
                               ("fwd", "mem_mib", "fwd_peak_memory_reduction")]:
        o, d = best("laplex", mode, metric), best("dense", mode, metric)
        common_x = sorted(set(o.index) & set(d.index))
        if common_x:
            x = common_x[-1]
            out[name] = dict(n=int(x), log2n=int(x).bit_length() - 1, laplex=o[x], dense=d[x],
                             ratio=d[x] / o[x])
    for mode in ("fwd", "fb"):
        st = P.stop_of(df, method="dense", mode=mode, metric="time_ms")
        out[f"dense_{mode}_first_failure"] = None if st is None else dict(status=st[0], n=int(st[1]))
    o = best("laplex", "fb", "time_ms")
    if len(o):
        out["laplex_fb_at_largest_n"] = dict(n=int(o.index.max()), ms=o[o.index.max()])
    C.write_summary(FIG, tag, out)


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    out = C.figure_dir(FIG, tags)
    panels = [("fwd", "time_ms", "Forward", "Time (ms)"),
              ("fb", "time_ms", "Forward + Backward", None),
              ("fwd", "mem_mib", "Peak Memory (forward)", "Peak Memory (MiB)")]
    for stem, titles in (("fig03", False), ("fig08", True)):
        fig, axes = P.figure(stem, ncols=3)
        axes[1].sharey(axes[0])
        axes[1].tick_params(which="both", labelleft=False)
        for ax, (mode, metric, title, ylab) in zip(axes, panels):
            for i, t in enumerate(tags):
                sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
                for m in ("laplex", "dense"):
                    P.draw(ax, df, agg, method=m, mode=mode, metric=metric, device=t,
                           label=P.LABELS[m] + sfx, linestyle=P.DEVICE_STYLES[i % 4])
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Values of $n$")
            if ylab:
                ax.set_ylabel(ylab)
            if titles:
                ax.set_title(title)
        P.finish(fig, axes, out, stem, args.ylim_json,
                 legend=P.collect_legend(axes, P.limit_legend_handles(df)))


if __name__ == "__main__":
    C.main(parser(), run, plot)
