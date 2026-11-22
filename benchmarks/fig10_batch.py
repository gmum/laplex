"""Fig. 10 (App. D.4): batch-size scaling of X K(a, b) at n = k = 2^14.

B = 2^0 ... 2^8, forward and forward+backward, LAPLEX vs explicit dense, fp32.
The original script skipped the dense forward+backward for every B > 32
(``SKIP_DENSE_FB_AT_LARGE_B``) without trying it; here it runs until it does
not fit in GPU memory, like every other series.

    python fig10_batch.py [--quick] [--stage plot --tags H100 ...]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig10_batch"


def parser():
    p = C.make_parser(__doc__, min_log=0, max_log=8)
    p.add_argument("--log-n", type=int, default=14, help="n = k = 2**log_n")
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.DEFAULT_IMPL)
    return p


def run(args):
    proto, tag, info = C.init(args)
    lap = impls.x_laplex(args.impl)
    n = 1 << args.log_n

    def make_inputs(B, seed):
        torch.manual_seed(seed)
        a = torch.sort(torch.randn(n, device="cuda"))[0]
        b = torch.randn(n, device="cuda")
        X = torch.randn(B, n, device="cuda")
        return dict(a=a, b=b, X=X)

    ops = [C.Op("laplex", limit=C.int32_limit(lambda b_: b_ * n),
                fwd=lambda d: (lambda: lap(d["X"], d["a"], d["b"])),
                fb=lambda d: C.fb_step(lap, d["X"], d["a"], d["b"])),
           C.Op("dense", fwd=lambda d: (lambda: impls.x_dense(d["X"], d["a"], d["b"])),
                fb=lambda d: C.fb_step(impls.x_dense, d["X"], d["a"], d["b"]))]
    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="B", extra=dict(n=n, impl=args.impl))
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                       seed=args.seed)
    sweep.save(meta=dict(env=info))
    df = C.load_results(FIG, [tag])
    C.write_summary(FIG, tag, {f"{m}_{mode}_stop": P.stop_of(df, method=m, mode=mode,
                                                             metric="time_ms")
                               for m in ("laplex", "dense") for mode in ("fwd", "fb")})


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig10", ncols=2)
    axes[1].sharey(axes[0])
    axes[1].tick_params(which="both", labelleft=False)
    for ax, (mode, title) in zip(axes, [("fwd", "Forward"), ("fb", "Forward + Backward")]):
        for i, t in enumerate(tags):
            sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
            for m in ("laplex", "dense"):
                P.draw(ax, df, agg, method=m, mode=mode, metric="time_ms", device=t,
                       label=P.LABELS[m] + sfx, linestyle=P.DEVICE_STYLES[i % 4])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Batch Size")
    axes[0].set_ylabel("Time (ms)")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig10", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(df)))


if __name__ == "__main__":
    C.main(parser(), run, plot)
