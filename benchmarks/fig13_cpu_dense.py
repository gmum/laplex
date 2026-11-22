"""Fig. 13 (App. D.7): CPU, pure-PyTorch LAPLEX (laplex_noCUDA) vs explicit dense.

Square kernel, B = 1, n = 2^10 ... 2^14 (published range), fp32, forward.
Timing as in the original CPU script: best of 3 runs of 10 calls after 3
warm-up calls.  On CPU the OOM point cannot be caught (the cgroup OOM killer
ends the process), so it is predicted from the dense footprint (3 n^2 fp32
buffers) against the job's memory limit; see common.cpu_guard.

Thread count is not stated in the paper; by default torch uses all CPUs
granted to the job (logged in meta.json).  --threads 1 matches Fig. 17.

    python fig13_cpu_dense.py [--threads 1] [--max-log 16]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

import common as C
import impls
import plotting as P

FIG = "fig13_cpu_dense"


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=14, warmup=3, iters=10, device="cpu")
    p.add_argument("--threads", type=int, default=None)
    return p


def run(args):
    C.set_threads(args.threads)
    proto, tag, info = C.init(args)
    lap = impls.x_laplex_cpu()

    def make_inputs(n, seed):
        torch.manual_seed(seed)
        a = torch.sort(torch.randn(n))[0]
        return dict(a=a, b=torch.randn(n), X=torch.randn(1, n))

    ops = [C.Op("laplex_cpu", fwd=lambda d: (lambda: lap(d["X"], d["a"], d["b"])),
                predict_bytes=lambda n: 200 * n),   # complex log/logcumsumexp buffers, generous
           C.Op("dense", fwd=lambda d: (lambda: impls.x_dense(d["X"], d["a"], d["b"])),
                predict_bytes=lambda n: 3 * n * n * 4)]
    sweep = C.Sweep(FIG, tag, "cpu", proto, axis="n",
                    extra=dict(B=1, threads=torch.get_num_threads()))
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], make_inputs, ops,
                       modes=("fwd",), seed=args.seed,
                       inputs_bytes=lambda n: 24 * n)   # a, b, X + the sort's values and int64 indices
    sweep.save(meta=dict(env=info))
    a = P.aggregate(C.load_results(FIG, [tag]))
    o = a[a.method == "laplex_cpu"].set_index("x")["min"]
    d = a[a.method == "dense"].set_index("x")["min"]
    xs = sorted(set(o.index) & set(d.index))
    C.write_summary(FIG, tag, {} if not xs else dict(
        n=int(xs[-1]), laplex_ms=o[xs[-1]], dense_ms=d[xs[-1]], speedup=d[xs[-1]] / o[xs[-1]],
        dense_first_n=int(xs[0]), dense_first_ms=d[xs[0]]))


def plot(args):
    tags = C.plot_tags(args)
    df = C.load_results(FIG, tags)
    agg = P.aggregate(df)
    fig, axes = P.figure("fig13")
    ax = axes[0]
    for i, t in enumerate(tags):
        sfx = f" - {P.device_label(t)}" if len(tags) > 1 else ""
        P.draw(ax, df, agg, method="laplex_cpu", mode="fwd", metric="time_ms", device=t,
               label="LAPLEX (no CUDA)" + sfx, linestyle=P.DEVICE_STYLES[i % 4])
        P.draw(ax, df, agg, method="dense", mode="fwd", metric="time_ms", device=t,
               label="Dense" + sfx, linestyle=P.DEVICE_STYLES[i % 4])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Values of $n$")
    ax.set_ylabel("Time (ms)")
    P.finish(fig, axes, C.figure_dir(FIG, tags), "fig13", args.ylim_json,
             legend=P.collect_legend(axes, P.limit_legend_handles(df)))


if __name__ == "__main__":
    C.main(parser(), run, plot)
