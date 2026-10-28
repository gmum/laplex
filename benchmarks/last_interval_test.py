"""Do the LAPLEX implementations handle anchors beyond the ends of the sorted ``a``?

Thm. 1/4 (matvec) and Thm. 6 (weighted Gram) define the per-interval vectors only for
the n intervals P_1..P_n, and the proofs use e^{a_r - b_j} with a_{n+1} = +inf, so as
written they do not cover b_j >= a_n (the interval P_{n+1}). This script checks whether
the code does: it compares each implementation with a float64 dense reference on anchor
sets that put b entirely above max(a), below min(a), on both sides, exactly on the end
anchors, or inside [min(a), max(a)) as a control. Values and gradients (matvec) are checked.

One implementation per process (``lib2026``/``public``/``fused`` import different
modules called ``prefix_suffix``). Implementations come from ``impls.py``
with the scan path forced, plus ``pure`` = ``vendor/lib2026/pure.py`` (dense dispatch
disabled).

    bash sif.sh python last_interval_test.py --impl pure --device cpu
    bash sif.sh python last_interval_test.py --impl lib2026          # needs a GPU
    bash sif.sh python last_interval_test.py --summarize             # results/last_interval/summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "last_interval"

MATVEC_IMPLS = ("lib2026", "public", "fused", "nocuda", "pure")
GRAM_IMPLS = ("lib2026", "public", "fused", "pure")
CASES = ("inside", "random", "above", "below", "both_sides", "end_ties")
MATVEC_SHAPES = ((1024, 4096), (4096, 1024), (2048, 2048))   # (n, k): branch A, branch B, square
GRAM_SHAPES = ((256, 4096), (1024, 1024), (2048, 256))
# Rounding-level agreement (``ok``) and the level that indicates missing or wrong
# contributions (``WRONG``); anything in between is reported as ``imprecise``.
TOL = {"float32": 1e-5, "float64": 1e-10}
WRONG = {"float32": 1e-2, "float64": 1e-6}
GRADS = ("grad_X", "grad_a", "grad_b")
# Implementations that cast float64 inputs to float32 internally and cast the result back
# (kerlap_fused: ``a32 = alpha.float()``). Their float64 rows are judged, and their ties
# counted, at float32 precision.
INTERNAL_FP32 = {"fused"}


def load_impl(name: str):
    """Return (matvec f(X, a, b) or None, gram f(alpha, beta, D) or None); a/alpha sorted."""
    import common
    common.setup_extensions_dir()
    if name == "pure":
        sys.path.insert(0, str(HERE / "vendor" / "lib2026"))
        import pure
        scan = dict(dense_budget=0, thin_dim_limit=0, dense_safety_cap=None)
        return (lambda X, a, b: pure.X_Laplex(X, a, b, a_sorted=True, **scan),
                lambda al, be, D: pure.Laplex_D_LaplexT(al, be, D, alpha_sorted=True, **scan))
    import impls
    matvec = impls.x_laplex(name) if name in MATVEC_IMPLS else None
    gram = impls.gram_laplex(name) if name in GRAM_IMPLS else None
    return matvec, gram


def make_anchors(case: str, n: int, k: int, gen):
    """Sorted a (n,) and b (k,) in float64 on the CPU."""
    import torch
    f64 = torch.float64
    a = torch.randn(n, generator=gen, dtype=f64).sort().values
    lo, hi = a[0], a[-1]
    u = torch.rand(k, generator=gen, dtype=f64)
    if case == "random":
        b = torch.randn(k, generator=gen, dtype=f64)
    elif case == "inside":                       # control: every b_j in some P_i, i <= n
        b = lo + (hi - lo) * u
    elif case == "above":                        # every b_j in P_{n+1}
        b = hi + 2.0 * u + 1e-3
    elif case == "below":                        # every b_j in P_1 = [-inf, a_1)
        b = lo - 2.0 * u - 1e-3
    elif case == "both_sides":
        b = torch.where(torch.arange(k) % 2 == 0, hi + 2.0 * u + 1e-3, lo - 2.0 * u - 1e-3)
    elif case == "end_ties":                     # b_j exactly on a_1 / a_n, rest inside
        b = lo + (hi - lo) * u
        b[0::3] = hi
        b[1::3] = lo
    else:
        raise ValueError(case)
    return a, b


def rel_err(y, ref) -> float:
    return float((y.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def dense_k(a, b):
    import torch
    return torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())


def check_matvec(f, case, n, k, B, dtype, device, gen, tie_dtype=None):
    import torch
    a64, b64 = make_anchors(case, n, k, gen)
    X64 = torch.randn(B, n, generator=gen, dtype=torch.float64)
    G64 = torch.randn(B, k, generator=gen, dtype=torch.float64)
    # Reference from the rounded inputs, so only the algorithm's error is measured.
    a, b, X = (t.to(dtype).to(device) for t in (a64, b64, X64))
    ta, tb = (a, b) if tie_dtype is None else (a.to(tie_dtype), b.to(tie_dtype))
    ties = int((ta.unsqueeze(1) == tb.unsqueeze(0)).sum())
    ra, rb, rX = (t.detach().double().requires_grad_(True) for t in (a, b, X))
    ref = rX @ dense_k(ra, rb)
    (ref * G64.to(device)).sum().backward()
    row = dict(value=None, grad_X=None, grad_a=None, grad_b=None, ties=ties)
    Xg, ag, bg = (t.detach().clone().requires_grad_(True) for t in (X, a, b))
    y = f(Xg, ag, bg)
    row["value"] = rel_err(y.detach(), ref.detach())
    try:
        (y * G64.to(device=device, dtype=y.dtype)).sum().backward()
        for name, t, r in (("grad_X", Xg, rX), ("grad_a", ag, ra), ("grad_b", bg, rb)):
            row[name] = None if t.grad is None else rel_err(t.grad, r.grad)
    except Exception as e:                       # value checked even if backward is unsupported
        row["grad_error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def check_gram(f, case, n, k, dtype, device, gen):
    import torch
    a64, b64 = make_anchors(case, n, k, gen)
    D64 = 0.1 + torch.rand(k, generator=gen, dtype=torch.float64)
    a, b, D = (t.to(dtype).to(device) for t in (a64, b64, D64))
    K = dense_k(a.double(), b.double())
    ref = (K * D.double().unsqueeze(0)) @ K.T
    with torch.no_grad():
        M = f(a, b, D)
    return dict(value=rel_err(M, ref))


def checked_keys(row) -> list[str]:
    """Errors that enter the verdict. At exact ties a_i = b_j (case ``end_ties``, or an
    accidental tie after rounding to float32) the kernel has a kink, so the anchor
    gradients depend on the subgradient each code picks; they are reported, not judged."""
    keys = ["value"] + ([] if row["op"] == "gram" else list(GRADS))
    if row.get("ties", 0) > 0:
        keys = [k for k in keys if k not in ("grad_a", "grad_b")]
    return [k for k in keys if row.get(k) is not None]


def verdict(row, dtype) -> str:
    if "error" in row:
        return "ERROR"
    dtype = row.get("computes_in", dtype)
    errs = [row[k] for k in checked_keys(row)]
    if not errs or max(errs) > WRONG[dtype]:
        return "WRONG"
    return "imprecise" if max(errs) > TOL[dtype] else "ok"


def run(args) -> int:
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    matvec, gram = load_impl(args.impl)
    rows = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for op, f, shapes in (("matvec", matvec, MATVEC_SHAPES), ("gram", gram, GRAM_SHAPES)):
            if f is None:
                continue
            for (n, k) in shapes:
                for case in CASES:
                    gen = torch.Generator().manual_seed(args.seed)
                    row = dict(impl=args.impl, device=device, dtype=dtype_name, op=op, n=n, k=k, case=case)
                    lowered = args.impl in INTERNAL_FP32 and dtype_name == "float64"
                    if lowered:
                        row["computes_in"] = "float32"
                    try:
                        if op == "matvec":
                            row.update(check_matvec(f, case, n, k, args.batch, dtype, device, gen,
                                                    tie_dtype=torch.float32 if lowered else None))
                        else:
                            row.update(check_gram(f, case, n, k, dtype, device, gen))
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {e}"[:300]
                        if args.verbose:
                            traceback.print_exc()
                    row["verdict"] = verdict(row, dtype_name)
                    rows.append(row)
                    print(fmt_row(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.impl}_{device}.json"
    path.write_text(json.dumps(dict(impl=args.impl, device=device, torch=torch.__version__,
                                    tolerances=TOL, wrong=WRONG, rows=rows), indent=1))
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("ok", "imprecise", "WRONG", "ERROR")}
    print(f"\n{len(rows)} checks: {counts} -> {path}")
    return 0


def fmt(x) -> str:
    return "—" if x is None else f"{x:.1e}"


def fmt_row(r) -> str:
    if "error" in r:
        return f"{r['impl']:8s} {r['dtype']:8s} {r['op']:6s} n={r['n']:5d} k={r['k']:5d} {r['case']:10s} ERROR {r['error']}"
    grads = "" if r["op"] == "gram" else \
        f" dX {fmt(r.get('grad_X'))} da {fmt(r.get('grad_a'))} db {fmt(r.get('grad_b'))}" + \
        (f" ({r['ties']} ties)" if r.get("ties") else "")
    return (f"{r['impl']:8s} {r['dtype']:8s} {r['op']:6s} n={r['n']:5d} k={r['k']:5d} {r['case']:10s} "
            f"value {fmt(r['value'])}{grads}  {r['verdict']}")


def rejudge(impl: str, rows: list[dict]) -> None:
    """Apply the current rules to stored rows. Float64 rows of INTERNAL_FP32 impls written
    before that rule existed take their tie counts from the float32 rows with the same
    anchors (same seed, same rounding)."""
    if impl in INTERNAL_FP32:
        fp32_ties = {(r["op"], r["n"], r["k"], r["case"]): r.get("ties", 0)
                     for r in rows if r["dtype"] == "float32"}
        for r in rows:
            if r["dtype"] == "float64" and "computes_in" not in r:
                r["computes_in"] = "float32"
                if r["op"] == "matvec":
                    r["ties"] = fp32_ties.get((r["op"], r["n"], r["k"], r["case"]), r.get("ties", 0))
    for r in rows:
        r["verdict"] = verdict(r, r["dtype"])


def summarize() -> int:
    files = sorted(OUT.glob("*.json"))
    if not files:
        print(f"no results in {OUT}", file=sys.stderr)
        return 1
    head = "| impl | device | dtype | op | " + " | ".join(CASES) + " | verdict |"
    sep = "|---|---|---|---|" + "---|" * len(CASES) + "---|"
    values, grads, kinks = [head, sep], [head, sep], []
    for fpath in files:
        d = json.loads(fpath.read_text())
        rejudge(d["impl"], d["rows"])
        groups = {}
        for r in d["rows"]:
            groups.setdefault((r["dtype"], r["op"]), []).append(r)
        for (dtype, op), rs in sorted(groups.items()):
            for table, keys in ((values, ("value",)), (grads, GRADS)):
                if op == "gram" and table is grads:
                    continue
                cells = []
                for case in CASES:
                    cr = [r for r in rs if r["case"] == case]
                    if any("error" in r for r in cr):
                        cells.append("ERROR")
                        continue
                    errs = [r[k] for r in cr for k in keys if k in checked_keys(r)]
                    cells.append(fmt(max(errs, default=None)))
                verd = sorted({r["verdict"] for r in rs if r["verdict"] != "ok"}) or ["ok"]
                table.append(f"| {d['impl']} | {d['device']} | {dtype} | {op} | " + " | ".join(cells)
                             + f" | {', '.join(verd)} |")
            for r in rs:
                if r.get("ties", 0) > 0 and "error" not in r:
                    kinks.append(f"| {d['impl']} | {dtype} | {r['case']} | {r['n']}x{r['k']} | {r['ties']} "
                                 f"| {fmt(max(r.get(k) or 0 for k in ('grad_a', 'grad_b')))} |")
    lines = ["# Last-interval test (b beyond the ends of sorted a)", "",
             "Relative L2 error against a float64 dense reference built from the same rounded",
             "inputs; worst case over the tested shapes (n, k): matvec "
             f"{list(MATVEC_SHAPES)}, Gram {list(GRAM_SHAPES)}.",
             f"Verdict: ok <= {TOL}; WRONG > {WRONG} (missing or wrong contributions);"
             " imprecise in between. Cases above/below/both_sides put every b_j outside"
             " [min a, max a], i.e. in P_{n+1} or P_1.",
             f"{', '.join(sorted(INTERNAL_FP32))} computes in float32 internally; its float64 rows are"
             " judged at float32 precision.", "",
             "## Values", ""] + values + ["", "## Gradients (matvec; worst of dX, da, db)", ""] + grads
    if kinks:
        lines += ["", "## Anchor gradients with exact ties a_i = b_j (not judged)", "",
                  "The kernel has a kink there; the codes and the dense autograd reference may pick",
                  "different subgradients. end_ties has ties by construction; other cases only by",
                  "accident after rounding to float32.", "",
                  "| impl | dtype | case | n x k | ties | worst da/db |", "|---|---|---|---|---|---|"] + kinks
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=sorted(set(MATVEC_IMPLS) | set(GRAM_IMPLS)))
    p.add_argument("--device", choices=("cuda", "cpu"), default=None)
    p.add_argument("--dtypes", nargs="+", default=["float32", "float64"], choices=("float32", "float64"))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summarize", action="store_true", help="collect results/last_interval/*.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.summarize:
        return summarize()
    if args.impl is None:
        p.error("--impl is required (or --summarize)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
