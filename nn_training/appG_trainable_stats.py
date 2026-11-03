"""
appG_trainable_stats.py  (App. G: statistics of the trainable columns of Table 1)
======================
Standard deviations and significance tests for the trainable columns of Table 1
(trainable LAPLEX vs. trainable dense projection on ImageNet-10). Reads the per-seed
results and aggregates them; no training is run.

Inputs (5 independent-seed runs each, one timestamped subfolder per seed):
    <results>/<laplex_dir>/*/results.csv   # trainable LAPLEX
    <results>/<dense_dir>/*/results.csv    # trainable dense

Each results.csv has per-epoch rows with columns [m, epoch, acc, ...]; the reported
accuracy is the best (max-over-epoch) validation accuracy per m, as in Table 1. Seeds are
independent, so LAPLEX and dense are not paired: unpaired Welch t-test and Mann-Whitney U.

Usage:
    python appG_trainable_stats.py --results <root> --dims 25 100 500 1000
"""

import argparse
import glob
import os

import pandas as pd

try:
    from scipy import stats

    HAVE_SCIPY = True
except ImportError:  # significance tests are optional
    HAVE_SCIPY = False


def per_seed_best(folder, dims):
    """Return a (n_seeds x n_dims) DataFrame of best-epoch accuracy (%)."""
    runs = sorted(glob.glob(os.path.join(folder, "*", "results.csv")))
    if not runs:
        raise FileNotFoundError(f"no results.csv under {folder}")
    table = {}
    for r in runs:
        df = pd.read_csv(r)
        table[os.path.basename(os.path.dirname(r))] = {
            m: df[df.m == m]["acc"].max() * 100.0 for m in dims
        }
    return pd.DataFrame(table).T[list(dims)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[25, 100, 500, 1000])
    ap.add_argument("--laplex_dir", default="experiments_II")
    ap.add_argument("--dense_dir", default="experiments_III")
    args = ap.parse_args()

    lap = per_seed_best(os.path.join(args.results, args.laplex_dir), args.dims)
    den = per_seed_best(os.path.join(args.results, args.dense_dir), args.dims)

    print(f"Trainable LAPLEX per-seed best acc (%) [{len(lap)} seeds]:")
    print(lap.round(2).to_string())
    print(f"\nTrainable dense per-seed best acc (%) [{len(den)} seeds]:")
    print(den.round(2).to_string())

    print("\n=== Table 1 trainable columns: mean +/- std (%) ===")
    hdr = f"{'m':>6} | {'LAPLEX':>16} | {'Dense':>16}"
    if HAVE_SCIPY:
        hdr += f" | {'Welch p':>9} | {'MWU p':>9}"
    print(hdr)
    print("-" * len(hdr))
    for m in args.dims:
        a, b = lap[m].values, den[m].values
        line = (
            f"{m:>6} | {a.mean():6.2f} +/- {a.std(ddof=1):4.2f} "
            f"| {b.mean():6.2f} +/- {b.std(ddof=1):4.2f}"
        )
        if HAVE_SCIPY:
            p_t = stats.ttest_ind(a, b, equal_var=False).pvalue
            p_u = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            line += f" | {p_t:9.3f} | {p_u:9.3f}"
        print(line)

    print(
        "\nNote: seeds are independent across the two experiments, so the "
        "comparison is unpaired (Welch / Mann-Whitney). Accuracy is the "
        "best-epoch value, consistent with the frozen columns."
    )


if __name__ == "__main__":
    main()
