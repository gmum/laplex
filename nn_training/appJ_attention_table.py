"""
appJ_attention_table.py  (App. J: the table of best configurations)
=======================
Aggregates the attention-replacement runs into the table of App. J.

Setup (produced by appJ_attention.py): a standard pre-norm
Transformer on pixel-token sequences in which ONLY the attention sub-layer is
swapped, everything else (architecture, optimiser, schedule, sweep) held
identical. Four attention operators are compared:
    Softmax    - standard softmax attention (primary baseline)
    RFF        - Performer-style random-feature linear attention
    RFF_tau    - RFF with a learnable temperature
    Structured - LAPLEX Laplace-kernel attention:  A_ij = exp(-|a_i - b_j|/tau)
Datasets: MNIST, SVHN, CIFAR-10. Sweep: att_dim in {64,128,256} x
att_depth in {4,8} x 3 repeats.

For each dataset and model we select the configuration with the best mean
test accuracy over its 3 repeats and report mean +/- std there. Absolute
accuracies are low by design (tiny pixel-token transformer); the comparison
is relative. No training is run.

Usage:
    python appJ_attention_table.py --root results/attention
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def collect(root):
    rows = []
    for ds in ("mnist", "svhn", "cifar10"):
        for d in glob.glob(os.path.join(root, ds, "*")):
            pj, cj = os.path.join(d, "params.json"), os.path.join(d, "results.csv")
            if not os.path.exists(pj):
                continue
            pr = json.load(open(pj))
            best = np.nan
            if os.path.exists(cj):
                try:
                    best = pd.read_csv(cj)["acc_test"].max()
                except Exception:
                    pass
            rows.append(
                dict(
                    ds=ds,
                    model=pr.get("model"),
                    att_dim=pr.get("att_dim"),
                    att_depth=pr.get("att_depth"),
                    best=best,
                    ok=not np.isnan(best),
                )
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/attention")
    args = ap.parse_args()

    df = collect(args.root)
    n_missing = int((~df.ok).sum())
    print(
        f"Loaded {len(df)} runs; {n_missing} crashed without results.csv.\n"
    )

    order = ["Softmax", "RFF", "RFF_tau", "Structured"]
    for ds in ("mnist", "svhn", "cifar10"):
        sub = df[(df.ds == ds) & df.ok]
        g = (
            sub.groupby(["model", "att_dim", "att_depth"])["best"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        best = g.loc[g.groupby("model")["mean"].idxmax()].set_index("model")
        print(f"=== {ds} (best config per model, mean +/- std over repeats) ===")
        for mdl in order:
            if mdl in best.index:
                r = best.loc[mdl]
                print(
                    f"  {mdl:>10}: {r['mean']:5.2f} +/- {r['std']:4.2f}  "
                    f"(att_dim={int(r['att_dim'])}, depth={int(r['att_depth'])}, "
                    f"n={int(r['count'])})"
                )
        print()


if __name__ == "__main__":
    main()
