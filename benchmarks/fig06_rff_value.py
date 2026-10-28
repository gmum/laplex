"""Fig. 6 (Sec. 6): Random Fourier Features vs exact LAPLEX, n = k = 2^16, B = 1.

D = 2^0, 2^1, ... features (until RFF no longer fits in GPU memory), 10 seeds
per D.  As in the original script, each seed gets one warm-up and one timed
RFF call; the line is the mean time over seeds, the band [min, max].  The
value error is relative l2 against the LAPLEX output (exact kernel, fp32).
LAPLEX itself is timed with the standard protocol.

    python fig06_rff_value.py [--quick] [--seeds 10]
"""
from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C
import impls
import plotting as P

FIG = "fig06_rff_value"


def parser():
    p = C.make_parser(__doc__, min_log=0, max_log=30)
    p.add_argument("--log-n", type=int, default=16)
    p.add_argument("--seeds", type=int, default=10, help="RFF feature draws per D")
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.PAPER_COMPARISON_IMPL,
                   help="the paper used the released laplex.py here")
    return p


def run(args):
    proto, tag, info = C.init(args)
    lap = impls.x_laplex(args.impl)
    n = 1 << args.log_n
    seeds = 1 if args.quick else args.seeds
    torch.manual_seed(args.seed)
    a = torch.sort(torch.randn(n, device="cuda"))[0]
    b = torch.randn(n, device="cuda")
    x = torch.randn(n, device="cuda")
    with torch.no_grad():
        y_gt = lap(x.unsqueeze(0), a, b)[0]
    y_norm = y_gt.norm().item()

    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="D",
                    extra=dict(n=n, impl=args.impl, seeds=seeds))
    s_lap = C.Series("laplex", "fwd", "time_ms")
    sweep.measure(("laplex", "fwd"), 0, 0, C.timed_body(
        sweep, s_lap, None, lambda: (lambda: lap(x.unsqueeze(0), a, b)), 0, 0, grad=False))
    print(f"  LAPLEX (exact): {sweep.rows[-1]['value']:.3f} ms", flush=True)

    s_t, s_e = C.Series("rff", "fwd", "time_ms"), C.Series("rff", "fwd", "rel_l2")
    for D in [1 << l for l in C.log_range(args)]:
        if not sweep.alive(("rff", "fwd")):
            break

        def body(D=D):
            for s in range(seeds):
                seed = 1000 * D + s
                with torch.no_grad():
                    impls.rff_matvec(x, a, b, D, seed)
                    C.sync("cuda")
                    t0 = time.perf_counter()
                    y = impls.rff_matvec(x, a, b, D, seed)
                    C.sync("cuda")
                    t = (time.perf_counter() - t0) * 1e3
                if proto.time_limit_ms is not None and t > proto.time_limit_ms:
                    raise C.TimeLimitExceeded(f"{t:.0f} ms > {proto.time_limit_ms:.0f} ms")
                sweep.record(s_t, D, s, t)
                sweep.record(s_e, D, s, (y - y_gt).norm().item() / y_norm)
                del y
        body.series = [s_t, s_e]
        sweep.measure(("rff", "fwd"), D, 0, body)
        C.clear("cuda")
        C.print_point(sweep, D, 0)
        sweep.save()
    sweep.save(meta=dict(env=info))
    summarize(tag)


def summarize(tag):
    df = C.load_results(FIG, [tag])
    a = P.aggregate(df, keys=("method", "metric", "x"))
    t_lap = a[(a.method == "laplex")]["min"].iloc[0]
    t = a[(a.method == "rff") & (a.metric == "time_ms")].set_index("x")["mean"]
    e = a[(a.method == "rff") & (a.metric == "rel_l2")].set_index("x")["median"]
    cross = [D for D in t.index if t[D] >= t_lap]
    below10 = [D for D in e.index if e[D] < 0.10]
    out = dict(laplex_ms=t_lap,
               runtime_crossover=None if not cross else dict(D=int(cross[0]),
                                                             median_err=e.get(cross[0])),
               first_D_below_10pct=None if not below10 else dict(
                   D=int(below10[0]), slowdown_vs_laplex=t[below10[0]] / t_lap),
               rff_stop=P.stop_of(df, method="rff", mode="fwd", metric="time_ms"))
    C.write_summary(FIG, tag, out)


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig06", ncols=2)
    for i, t in enumerate(tags):
        ls = P.DEVICE_STYLES[i % 4]
        sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
        r = agg[(agg.device == t) & (agg.method == "rff")]
        re_ = r[r.metric == "rel_l2"].sort_values("x")
        t_lap = agg[(agg.device == t) & (agg.method == "laplex")]["min"].iloc[0]
        n_seeds = int(re_["count"].max()) if len(re_) else 0
        P.draw(axes[0], df, agg, method="rff", mode="fwd", metric="time_ms", device=t, stat="mean",
               color=P.ORANGE, marker="o", label=f"RFF (mean over ${n_seeds}$ seeds){sfx}", linestyle=ls)
        axes[0].axhline(t_lap, color=P.BLUE, linestyle="--", label=f"LAPLEX (exact) = ${t_lap:.2f}$\\,ms{sfx}")
        axes[1].fill_between(re_.x, re_["min"], re_["max"], color=P.ORANGE, alpha=0.2, lw=0,
                             label=r"RFF $[\min, \max]$ over " + f"${n_seeds}$ seeds{sfx}")
        P.draw(axes[1], df, agg, method="rff", mode="fwd", metric="rel_l2", device=t, stat="median",
               color=P.ORANGE, marker="o", label=f"RFF median{sfx}", linestyle=ls)
        if i == 0 and len(re_):
            D = re_.x.to_numpy(float)
            axes[1].plot(D, re_["median"].iloc[0] * np.sqrt(D[0] / D), "--", color="0.2", linewidth=1.5,
                         label=r"$\propto 1/\sqrt{D}$ reference")
    axes[1].axhline(0.0, color=P.BLUE, linestyle="--", label="LAPLEX (exact, error = 0)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of RFF Features $D$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Time (ms)")
    axes[1].set_ylabel(r"Relative $\ell_2$ Error")
    h, l = axes[0].get_legend_handles_labels()   # + the OOM marker of the RFF curves
    axes[0].legend(h + P.limit_legend_handles(df), l + [x.get_label() for x in P.limit_legend_handles(df)],
                   loc="upper left", frameon=True, edgecolor="0.8")
    axes[1].legend(loc="upper right", frameon=True, edgecolor="0.8")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig06", args.ylim_json)


if __name__ == "__main__":
    C.main(parser(), run, plot)
