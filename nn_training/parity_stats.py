"""Statistics for the from-scratch re-run of the trainable columns of Table 1 (App. G).

App. G of the submitted version (l. 976) states: "An independent from-scratch re-run (5
fresh seeds) gives the same picture." This collects the re-run made by sbatch/parity.sbatch with the released,
unchanged scripts ``CountSketch_LogisticRegression/{laplex,dense}.py`` and compares
it with the published table (Table 1 and the App. G table).

Expected layout (one directory per method and seed, as written by the scripts):
    results/parity/<laplex|dense>/seed<S>/<timestamp>/results.csv

The accuracy of a run at a given m is the best validation accuracy over its epochs, as
the scripts' own "Best Acc" summary.

    bash sif.sh python parity_stats.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "parity"
METHODS = ("laplex", "dense")
# Unnumbered table in App. G (mean +- std over 5 seeds, Welch p).
PUBLISHED = {
    25: dict(laplex=(43.52, 0.77), dense=(43.52, 1.52), p=1.000),
    100: dict(laplex=(43.56, 0.46), dense=(43.04, 0.55), p=0.146),
    500: dict(laplex=(43.84, 0.89), dense=(43.00, 0.79), p=0.153),
    1000: dict(laplex=(44.16, 1.51), dense=(43.80, 0.47), p=0.634),
}


def collect(root: Path) -> pd.DataFrame:
    """One row per (method, seed, m): best validation accuracy in %."""
    rows = []
    for method in METHODS:
        for seed_dir in sorted((root / method).glob("seed*")):
            runs = sorted(seed_dir.glob("*/results.csv"))
            if not runs:
                print(f"[warn] no results.csv in {seed_dir}", file=sys.stderr)
                continue
            if len(runs) > 1:
                print(f"[warn] {len(runs)} runs in {seed_dir}; using the last one", file=sys.stderr)
            df = pd.read_csv(runs[-1])
            for m, g in df.groupby("m"):
                rows.append(dict(method=method, seed=seed_dir.name, m=int(m), acc=100 * g["acc"].max(),
                                 epochs=int(g["epoch"].max()) + 1, run=str(runs[-1].parent.name)))
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=OUT)
    args = p.parse_args()
    df = collect(args.root)
    if df.empty:
        print(f"no results under {args.root}", file=sys.stderr)
        return 1
    df.to_csv(args.root / "per_seed.csv", index=False)

    out, lines = [], ["# Re-run of the trainable projection columns (Table 1, App. G)", "",
                      "Best validation accuracy (%) per run; mean +- std (ddof=1) over seeds; "
                      "Welch t-test and Mann-Whitney U, two-sided.", "",
                      "| m | LAPLEX re-run | Dense re-run | diff | Welch p | MWU p | n seeds | "
                      "LAPLEX paper | Dense paper | Welch p paper |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(df["m"].unique()):
        x = df[(df.m == m) & (df.method == "laplex")]["acc"].to_numpy()
        y = df[(df.m == m) & (df.method == "dense")]["acc"].to_numpy()
        rec = dict(m=int(m), n_laplex=len(x), n_dense=len(y),
                   laplex_mean=float(x.mean()) if len(x) else None, laplex_std=float(x.std(ddof=1)) if len(x) > 1 else None,
                   dense_mean=float(y.mean()) if len(y) else None, dense_std=float(y.std(ddof=1)) if len(y) > 1 else None)
        if len(x) > 1 and len(y) > 1:
            rec["welch_p"] = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
            rec["mwu_p"] = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        pub = PUBLISHED.get(int(m))
        rec["published"] = pub
        out.append(rec)

        def ms(mean, std):
            return "—" if mean is None else f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"
        diff = "—" if rec["laplex_mean"] is None or rec["dense_mean"] is None else \
            f"{rec['laplex_mean'] - rec['dense_mean']:+.2f}"
        pp = (lambda k: f"{rec[k]:.3f}" if k in rec else "—")
        lines.append(f"| {m} | {ms(rec['laplex_mean'], rec['laplex_std'])} | {ms(rec['dense_mean'], rec['dense_std'])} "
                     f"| {diff} | {pp('welch_p')} | {pp('mwu_p')} | {len(x)}/{len(y)} "
                     + (f"| {pub['laplex'][0]:.2f} ± {pub['laplex'][1]:.2f} | {pub['dense'][0]:.2f} ± {pub['dense'][1]:.2f} "
                        f"| {pub['p']:.3f} |" if pub else "| — | — | — |"))
    sig = [r["m"] for r in out if r.get("welch_p") is not None and r["welch_p"] < 0.05]
    lines += ["", "Same picture as the paper (no significant difference at any m, Welch p >= 0.05): "
              + ("**yes**" if not sig and all("welch_p" in r for r in out) else
                 f"**no** (p < 0.05 at m = {sig})" if sig else "incomplete")]
    found = {mth: sorted(df[df.method == mth]["seed"].unique()) for mth in METHODS}
    lines += ["", "Seeds with results (a task that did not finish leaves no results.csv): "
              + "; ".join(f"{mth}: {', '.join(s) or 'none'} ({len(s)})" for mth, s in found.items())]
    lines += ["", "Per-seed values: per_seed.csv."]
    (args.root / "summary.json").write_text(json.dumps(out, indent=1))
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
