"""Tab. 3 (App. D.1): relative cost of anchor gradients, sorting and permutation.

Same procedure as kerlap_bench_ablation.py --method dl-fused (not in this repository),
which produced the published table: forward+backward of
X K(a, b), n = k = 2^10 ... 2^25, B = 8, fp32, four variants

    full      -- fused kernel, sorts a internally (sort + X[:, perm]), grads w.r.t. X, a, b
    nosort    -- a_sorted=True: no sort, no permutation
    nopermute -- pays torch.sort(a) but skips X[:, perm] (a is already sorted)
    noanchor  -- as full, gradient w.r.t. X only

measured round-robin (--rr times, median) for each seed, 8 warm-up + 30 timed
calls each.  Per seed: sorting = nopermute - nosort, permutation = full - nopermute,
anchor gradients = full - noanchor, remaining = full - the three; percentages of
full, then mean +- std (population) over seeds.

    python table03_anchor_cost.py [--quick]
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

import common as C
import impls

FIG = "table03_anchor_cost"
VARIANTS = ("full", "nosort", "nopermute", "noanchor")
PAPER_ROWS = (10, 13, 16, 19, 22, 25)


def parser():
    p = C.make_parser(__doc__, min_log=10, max_log=25, repeats=3, warmup=8, iters=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rr", type=int, default=5, help="round-robin repetitions per seed (median)")
    p.add_argument("--impl", choices=("fused", "fused-orig"), default="fused")
    return p


def run(args):
    proto, tag, info = C.init(args)
    kf = impls.fused(orig=args.impl == "fused-orig")
    B, rr = args.batch, (1 if args.quick else args.rr)

    def full(x, a, b):
        return kf.X_KerLap_fused(x, a, b, threshold=0)

    def nosort(x, a, b):
        return kf.X_KerLap_fused(x, a, b, threshold=0, a_sorted=True)

    def nopermute(x, a, b):
        return nosort(x, torch.sort(a)[0], b)

    def step(fn, X, a, b, anchor_grad=True):
        def s():
            x_ = X.detach().requires_grad_(True)
            a_ = a.detach().requires_grad_(anchor_grad)
            b_ = b.detach().requires_grad_(anchor_grad)
            fn(x_, a_, b_).sum().backward()
        return s

    sweep = C.Sweep(FIG, tag, "cuda", proto, axis="n", extra=dict(B=B, impl=args.impl, rr=rr))
    series = {v: C.Series(v, "fb", "time_ms") for v in VARIANTS}
    for log_n in C.log_range(args):
        n = 1 << log_n
        if not sweep.any_alive([(v, "fb") for v in VARIANTS]):
            break
        if B * n > C.INT32_MAX:
            for v in VARIANTS:
                sweep.stop((v, "fb"), [series[v]], n, C.LIMIT, "B*n > int32 indexing of the custom kernels")
            break
        for seed in range(args.seed, args.seed + proto.repeats):
            torch.manual_seed(seed)
            d = C.try_inputs(sweep, lambda: dict(a=torch.sort(torch.randn(n, device="cuda"))[0],
                                                b=torch.randn(n, device="cuda"),
                                                X=torch.randn(B, n, device="cuda")),
                             n, {(v, "fb"): [series[v]] for v in VARIANTS})
            if d is None:
                break
            fns = {"full": step(full, d["X"], d["a"], d["b"]),
                   "nosort": step(nosort, d["X"], d["a"], d["b"]),
                   "nopermute": step(nopermute, d["X"], d["a"], d["b"]),
                   "noanchor": step(full, d["X"], d["a"], d["b"], anchor_grad=False)}
            samples = {v: [] for v in VARIANTS}
            for _ in range(rr):
                for v in VARIANTS:
                    def body(v=v):
                        samples[v].append(C.time_ms(fns[v], proto, "cuda", grad=True))
                    body.series = [series[v]]
                    sweep.measure((v, "fb"), n, seed, body)
            for v in VARIANTS:
                if sweep.alive((v, "fb")) and samples[v]:
                    sweep.record(series[v], n, seed, float(np.median(samples[v])))
            del d, fns
            C.clear("cuda")
        C.print_point(sweep, n, 0)
        sweep.save()
    sweep.save(meta=dict(env=info))
    summarize(tag)


def breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds of the percentage columns, per n."""
    ok = df[(df.status == "ok") & (df.repeat >= 0)]
    t = ok.pivot_table(index=["x", "repeat"], columns="method", values="value")
    parts = pd.DataFrame(index=t.index)
    parts["sorting"] = t["nopermute"] - t["nosort"]
    parts["permutation"] = t["full"] - t["nopermute"]
    parts["anchor gradients"] = t["full"] - t["noanchor"]
    parts["remaining"] = t["full"] - parts.sum(axis=1)
    pct = 100.0 * parts.div(t["full"], axis=0)
    pct["full_ms"] = t["full"]
    g = pct.groupby(level="x")
    out = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std"))
    out.index = [int(x).bit_length() - 1 for x in out.index]
    return out


def summarize(tag):
    b = breakdown(C.load_results(FIG, [tag]))
    C.write_summary(FIG, tag, json.loads(b.to_json(orient="index")))


def plot(args):
    """No figure: print the table."""
    b = breakdown(C.load_results(FIG, C.plot_tags(args)))
    cols = ["anchor gradients", "sorting", "permutation", "remaining"]
    print("log2 n | " + " | ".join(cols) + " | full [ms]")
    for n, r in b.iterrows():
        print(f"{n:6d} | " + " | ".join(f"{r[c + '_mean']:5.1f} ± {r[c + '_std']:4.1f}" for c in cols)
              + f" | {r['full_ms_mean']:.3f}")


if __name__ == "__main__":
    C.main(parser(), run, plot)
