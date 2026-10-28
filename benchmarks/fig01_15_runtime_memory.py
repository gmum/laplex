"""Fig. 1 (Sec. 1) and Fig. 15 (App. E.1): runtime and peak memory vs n, B = 1.

LAPLEX, the explicit-dense baseline and a single torch.fft.fft (cuFFT) call on
square problems n = k = 2^10 ... 2^30, fp32, irregular anchors a, b ~ N(0, 1).
Every method runs until it no longer fits in GPU memory; the camera-ready version
applies this protocol to every benchmark.

As in the published figures, LAPLEX and Dense are timed forward+backward and
cuFFT forward only (it is not a kernel matvec and has no backward here); the
forward times of LAPLEX and Dense are stored too and plotted in
fig01_fwd.{pdf,png}.  Lines are means over seeds, bands +- one std.

Overlay several GPUs (H100 solid, RTX 2080 dashed in the paper):
    python fig01_15_runtime_memory.py --stage plot --tags H100 RTX2080Ti
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig01_15_runtime_memory"


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=30)
    p.add_argument("--impl", choices=impls.IMPLS, default=impls.DEFAULT_IMPL,
                   help="the published Figs. 1/15 used 'fused' (capped at n < 2^26 as the file is now)")
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

    def cufft(d):
        x_c = d["X"][0].to(torch.complex64)

        def fn():
            return torch.fft.fft(x_c)
        fn.input_bytes = C.tensor_bytes(x_c)
        return fn

    ops = [C.Op("laplex", limit=C.int32_limit(lambda n: n),
                fwd=lambda d: (lambda: lap(d["X"], d["a"], d["b"])),
                fb=lambda d: C.fb_step(lap, d["X"], d["a"], d["b"])),
           C.Op("dense", fwd=lambda d: (lambda: impls.x_dense(d["X"], d["a"], d["b"])),
                fb=lambda d: C.fb_step(impls.x_dense, d["X"], d["a"], d["b"])),
           C.Op("cufft", fwd=cufft)]
    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="n", extra=dict(B=1, impl=args.impl))
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                       seed=args.seed)
    sweep.save(meta=dict(env=info))
    summarize(tag)


def summarize(tag):
    df = C.load_results(FIG, [tag])
    a = P.aggregate(df)
    out = {}
    for m, mode in [("laplex", "fb"), ("dense", "fb"), ("cufft", "fwd"), ("laplex", "fwd"),
                    ("dense", "fwd")]:
        for metric in ("time_ms", "mem_mib"):
            s = a[(a.method == m) & (a["mode"] == mode) & (a.metric == metric)].sort_values("x")
            if len(s):
                last = s.iloc[-1]
                out[f"{m}_{mode}_{metric}"] = dict(
                    first=dict(n=int(s.iloc[0].x), mean=s.iloc[0]["mean"]),
                    last_that_fits=dict(n=int(last.x), log2n=int(last.x).bit_length() - 1,
                                        mean=last["mean"]))
        st = P.stop_of(df, method=m, mode=mode, metric="time_ms")
        out[f"{m}_{mode}_stop"] = None if st is None else dict(status=st[0], n=int(st[1]))
    C.write_summary(FIG, tag, out)


SERIES = [("laplex", "fb", "LAPLEX"), ("dense", "fb", "Dense"), ("cufft", "fwd", "cuFFT")]


def _panel(ax, df, agg, tags, metric, series):
    """Lines without markers, one line style per GPU (H100 solid, RTX dashed), as in Fig. 1."""
    for m, mode, name in series:
        for i, t in enumerate(tags):
            P.draw(ax, df, agg, method=m, mode=mode, metric=metric, device=t, label=f"{name} - {P.device_label(t)}",
                   stat="mean", band=True, linestyle=P.DEVICE_STYLES[i % 4], markevery=False)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Values of $n$")


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    out = C.figure_dir(FIG, tags)

    for stem, key, panels in [("fig01", "fig01", [("time_ms", "Time (ms)", SERIES)]),
                              ("fig15", "fig15", [("time_ms", "Time (ms)", SERIES),
                                                  ("mem_mib", "Peak Memory (MiB)", SERIES)]),
                              ("fig01_fwd", "fig01", [("time_ms", "Forward time (ms)",
                                                       [("laplex", "fwd", "LAPLEX"), ("dense", "fwd", "Dense"),
                                                        ("cufft", "fwd", "cuFFT")])])]:
        fig, axes = P.figure(key, ncols=len(panels))
        for ax, (metric, ylab, series) in zip(axes, panels):
            _panel(ax, df, agg, tags, metric, series)
            ax.set_ylabel(ylab)
        # paper layout: one column per method, GPUs below each other; the limit marker under LAPLEX
        # (Fig. 1, as published) or in a column of its own (Fig. 15, which keeps two legend rows)
        own = stem == "fig15" or len(tags) == 1
        legend = P.collect_legend(axes, P.limit_legend_handles(df), insert_extra_at=None if own else len(tags))
        P.finish(fig, axes, out, stem, args.ylim_json, legend=legend, legend_ncol=4 if own else 3)


if __name__ == "__main__":
    C.main(parser(), run, plot)
