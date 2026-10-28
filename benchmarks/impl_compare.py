"""Compare every LAPLEX implementation in the setting of Fig. 1 (n = k, fp32).

Each implementation runs in its own process (several of them import modules
with the same name, e.g. ``prefix_suffix``).  A worker first checks the
values and the gradients w.r.t. X, a, b against an fp64 dense reference at
n = 2^12, then times forward and forward+backward (and peak memory) over
n = 2^min ... 2^max with the stopping rule of common.py.  The plot stage
draws all implementations together and writes, for every n, which
implementation is fastest and by how much (summary.json, dominance.md).

    python impl_compare.py --max-log 22                 # all implementations, B = 1
    python impl_compare.py --batch 8 --impls lib2026 fused public
    python impl_compare.py --stage plot --tag H100
"""
from __future__ import annotations

import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

import common as C
import impls
import plotting as P

COLORS = {"lib2026": "#0072B2", "lib2026-auto": "#56B4E9", "public": "#009E73",
          "public-raw": "#7FC97F", "fused": "#D55E00", "merged": "#CC79A7", "dl": "#E69F00",
          "dl-opt": "#F0E442", "mix-fused": "#000000", "nocuda": "#999999"}
MARKERS = dict(zip(COLORS, "osD^vP*Xhp"))


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=22)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--impls", nargs="+", default=list(impls.ALL_IMPLS), choices=impls.ALL_IMPLS)
    p.add_argument("--worker", default=None, help=argparse_hidden())
    return p


def argparse_hidden():
    import argparse
    return argparse.SUPPRESS


def fig_id(B, impl):
    return f"impl_compare_B{B}/{impl}"


def check(f, B, n=1 << 12):
    """Relative max error of y, dX, da, db against fp64 dense."""
    torch.manual_seed(123)
    a = torch.sort(torch.randn(n, device="cuda"))[0]
    b = torch.randn(n, device="cuda")
    X = torch.randn(B, n, device="cuda")
    out = {}
    for prec, (Xs, as_, bs) in (("fp32", (X, a, b)), ("fp64", (X.double(), a.double(), b.double()))):
        leaves = [t.clone().requires_grad_(True) for t in (Xs, as_, bs)]
        y = (f if prec == "fp32" else impls.x_dense)(*leaves)
        y.sum().backward()
        out[prec] = [y.detach().double()] + [t.grad.double() for t in leaves]
    return {k: ((u - r).abs().max() / r.abs().max()).item()
            for k, u, r in zip(("y", "dX", "da", "db"), out["fp32"], out["fp64"])}


def worker(args):
    proto, tag, info = C.init(args)
    impl, B = args.worker, args.batch
    sweep = C.Sweep(fig_id(B, impl), tag, "cuda", proto, axis="n", extra=dict(B=B, impl=impl))
    try:
        f = impls.x_laplex(impl)
        err = check(f, B)
        print(f"  [{impl}] max rel error vs fp64 dense at n=2^12: {err}", flush=True)
    except Exception as e:  # noqa: BLE001 - recorded, the implementation is skipped
        sweep.save(meta=dict(env=info, check_error=f"{type(e).__name__}: {e}"))
        raise
    sweep.save(meta=dict(env=info, check=err))

    def make_inputs(n, seed):
        torch.manual_seed(seed)
        return dict(a=torch.sort(torch.randn(n, device="cuda"))[0], b=torch.randn(n, device="cuda"),
                    X=torch.randn(B, n, device="cuda"))

    ops = [C.Op(impl, fwd=lambda d: (lambda: f(d["X"], d["a"], d["b"])),
                fb=lambda d: C.fb_step(f, d["X"], d["a"], d["b"]))]
    C.run_timing_sweep(sweep, [1 << l for l in C.log_range(args)], ops=ops,
                       make_inputs=make_inputs, seed=args.seed)
    sweep.save(meta=dict(env=info, check=err))


def run(args):
    passthrough = [a for a in sys.argv[1:]]
    if "--impls" in passthrough:          # drop the implementation list
        i = passthrough.index("--impls")
        j = i + 1
        while j < len(passthrough) and not passthrough[j].startswith("--"):
            j += 1
        del passthrough[i:j]
    for impl in args.impls:
        print(f"=== {impl}", flush=True)
        r = subprocess.run([sys.executable, __file__, *passthrough, "--stage", "run",
                            "--worker", impl])
        if r.returncode:
            print(f"  [{impl}] worker failed with exit code {r.returncode}", flush=True)


def load(tag, B, impl_list):
    frames, checks = [], {}
    for impl in impl_list:
        d = C.RESULTS / tag / fig_id(B, impl)
        if (d / "raw.csv").exists() and (d / "raw.csv").stat().st_size > 1:
            try:
                frames.append(pd.read_csv(d / "raw.csv"))
            except pd.errors.EmptyDataError:
                pass
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            checks[impl] = m.get("check") or m.get("check_error")
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), checks


def plot(args):
    tag = C.plot_tags(args)[0]
    B = args.batch
    df, checks = load(tag, B, args.impls)
    if df.empty:
        raise SystemExit("no results")
    agg = P.aggregate(df)
    present = [i for i in args.impls if i in set(df.method)]
    fig, axes = P.figure("impl_compare", ncols=3, figsize=(13, 3.8))
    for ax, (mode, metric, title, ylab) in zip(axes, [
            ("fwd", "time_ms", "Forward", "Time (ms)"),
            ("fb", "time_ms", "Forward + Backward", "Time (ms)"),
            ("fb", "mem_mib", "Peak memory (fwd+bwd)", "Peak Memory (MiB)")]):
        for impl in present:
            P.draw(ax, df, agg, method=impl, mode=mode, metric=metric, device=tag,
                   color=COLORS[impl], marker=MARKERS[impl], label=impl)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Values of $n$")
        ax.set_ylabel(ylab)
    out = C.figure_dir(f"impl_compare_B{B}", [tag])
    P.finish(fig, axes, out, f"impl_compare_B{B}",
             legend=P.collect_legend(axes, P.limit_legend_handles(df)), legend_ncol=6)

    # Fastest implementation per n; best-of-repeats times.
    summary, lines = dict(B=B, tag=tag, checks=checks), []
    for mode in ("fwd", "fb"):
        t = agg[(agg["mode"] == mode) & (agg.metric == "time_ms")].pivot(
            index="x", columns="method", values="min")[[i for i in present if i in set(agg.method)]]
        rows = {}
        lines += [f"\n### {mode}, B = {B} (best of repeats, ms; * = fastest)\n",
                  "| log2 n | " + " | ".join(t.columns) + " | fastest | 2nd / 1st |",
                  "|---" * (len(t.columns) + 3) + "|"]
        for x, r in t.iterrows():
            r = r.dropna().sort_values()
            if r.empty:
                continue
            best = r.index[0]
            ratio = r.iloc[1] / r.iloc[0] if len(r) > 1 else float("nan")
            rows[int(x)] = dict(fastest=best, ms=r.iloc[0], second=r.index[1] if len(r) > 1 else None,
                                ratio_second_to_first=ratio)
            cells = [(f"**{t.loc[x, c]:.3g}**" if c == best else f"{t.loc[x, c]:.3g}")
                     if pd.notna(t.loc[x, c]) else "–" for c in t.columns]
            lines.append(f"| {int(x).bit_length() - 1} | " + " | ".join(cells)
                         + f" | {best} | {ratio:.2f} |")
        summary[mode] = rows
    stops = {impl: {m: P.stop_of(df, method=impl, mode=m, metric="time_ms") for m in ("fwd", "fb")}
             for impl in present}
    summary["stops"] = stops
    lines += ["\n### correctness at n = 2^12 (max relative error vs fp64 dense)\n"]
    lines += [f"- {k}: {v}" for k, v in checks.items()]
    lines += ["\n### first failing size\n"] + [f"- {k}: {v}" for k, v in stops.items()]
    (out / "dominance.md").write_text("\n".join(lines) + "\n")
    C.write_summary(f"impl_compare_B{B}", tag, summary)
    print("\n".join(lines))


def main():
    args = parser().parse_args()
    if args.worker:
        worker(args)
        return
    if args.stage in ("run", "all"):
        run(args)
    if args.stage in ("plot", "all"):
        plot(args)


if __name__ == "__main__":
    main()
