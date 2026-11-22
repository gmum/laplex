"""Poster figure: Fig. 1 and Fig. 5 of the paper merged into one runtime-vs-n plot.

LAPLEX, Dense, Toeplitz-FFT and one cuFFT call on an H100, all under the Fig. 5
protocol: one mat-vec X K(a, b), B = 1, fp32, forward only, n = k = 2^10, 2^11, ...,
irregular anchors a, b ~ N(0, 1), every method run until it does not fit.

* LAPLEX, Toeplitz-FFT, cuFFT: the Fig. 5 sweep (benchmarks/fig05_14_fft.py).
* Dense: the forward timings of the Fig. 1 sweep (benchmarks/fig01_15_runtime_memory.py).
  Its inputs are built by the same code as in Fig. 5 and the protocol, B, dtype and
  LAPLEX implementation are checked to be the same; Fig. 1 itself plots fwd+bwd.

Each point is the fastest of the repeats (as in Fig. 5). The dot at the end of each
curve is the largest n that fits ("memory limit"): the next size is OOM for Dense,
Toeplitz-FFT and cuFFT. LAPLEX stops at 2^31 on the int32 indexing guard, but its
forward peak memory is linear in n (52 GiB at 2^30), so 2^31 would not fit in 80 GB
either.

    python scripts/make_scaling_fig.py [--results benchmarks/results]

Fonts: TeX Gyre Heros from scripts/fonts/ (fonts/tex-gyre/opentype on CTAN), else
Helvetica / Arial / DejaVu Sans, whichever is installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG05 = "fig05_14a_fft_runtime"
FIG01 = "fig01_15_runtime_memory"

# dense as in the projection panel of the poster (red is kept for fixed projections there)
GREEN, DARK, ORANGE, GREY = "#2e7d32", "#303030", "#e65100", "#8c8c8c"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=ROOT / "benchmarks" / "results",
                   help="results/ of benchmarks/common.py")
    p.add_argument("--fft-tag", default="H100-max-fused", help="results tag of the Fig. 5 sweep")
    p.add_argument("--dense-tag", default="H100-max", help="results tag of the Fig. 1 sweep")
    p.add_argument("--fonts", type=Path, default=ROOT / "scripts" / "fonts")
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "scaling_combined")
    return p


def load(results: Path, tag: str, figure: str):
    d = results / tag / figure
    meta = json.loads((d / "meta.json").read_text())
    df = pd.read_csv(d / "raw.csv")
    return meta, df


def check_protocol(m5, m1):
    """Both sweeps must be the same measurement up to the method list."""
    for m in (m5, m1):
        assert "H100" in m["env"]["gpu"], m["env"]["gpu"]
        assert m["B"] == 1, m["B"]
    for k in ("protocol", "impl", "axis"):
        assert m5[k] == m1[k], (k, m5[k], m1[k])
    assert m5["env"]["laplex_git"] == m1["env"]["laplex_git"]


def series(df, method):
    """n -> fastest forward time (ms) over the repeats, successful sizes only."""
    s = df[(df.method == method) & (df["mode"] == "fwd") & (df.metric == "time_ms") & (df.status == "ok")]
    assert s.dtypes["value"] == float and len(s), method
    return s.groupby("x")["value"].min().sort_index()


def stop(meta, method):
    s = meta["stopped"].get(f"{method}/fwd")
    return (s["status"], s["x"]) if s else None


def set_style(font_dir: Path):
    for f in sorted(font_dir.glob("*.otf")):
        font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans") if f in names))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family],
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "0.35", "axes.linewidth": 1.0,
        "xtick.color": "0.35", "ytick.color": "0.35", "xtick.labelcolor": "0.15", "ytick.labelcolor": "0.15",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200,
    })
    return family


def plot(t, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.3))
    fig.subplots_adjust(left=0.10, right=0.77, bottom=0.14, top=0.97)

    lap, toe, fft, den = t["laplex"], t["toeplitz"], t["cufft"], t["dense"]
    curves = [(lap, "LAPLEX", GREEN, 4.0),
              (den, "dense", DARK, 2.6),
              (toe, "Toeplitz–FFT", ORANGE, 2.6),
              (fft, "one FFT call", GREY, 1.8)]
    handles = []
    for z, (s, lbl, col, lw) in enumerate(curves):
        h, = ax.plot(s.index, s.values, color=col, lw=lw, zorder=4 - z, solid_capstyle="round", label=lbl)
        ax.plot(s.index[-1], s.iloc[-1], "o", color=col, ms=9, zorder=5)
        handles.append(h)
    handles.append(ax.plot([], [], "o", color="0.25", ms=9, label="memory limit")[0])

    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=15, handlelength=1.8, labelspacing=0.7, borderaxespad=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(2 ** 9.5, 2 ** 31.5)
    ax.xaxis.set_major_locator(FixedLocator([2.0 ** k for k in range(10, 32, 5)]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"$2^{{{round(np.log2(v))}}}$"))
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(-3, 4)]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_locator(NullLocator())
    ax.set_ylim(fft.min() / 1.6, lap.max() * 2.2)
    ax.grid(axis="y", color="0.9", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=4, width=1.0)
    ax.set_xlabel(r"Problem size $n$")
    ax.set_ylabel("Time of operation (ms)")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def report(t, m5, m1):
    lap, toe, den = t["laplex"], t["toeplitz"], t["dense"]
    print(f"protocol: {m5['protocol']}, B={m5['B']}, fwd, fp32, impl={m5['impl']}, {m5['env']['gpu']}")
    for m, meta in (("laplex", m5), ("toeplitz", m5), ("cufft", m5), ("dense", m1)):
        n = int(t[m].index[-1])
        print(f"{m:9s} largest n = 2^{n.bit_length() - 1} = {n:,}   {t[m].iloc[-1]:.4g} ms   "
              f"next: {stop(meta, m)}")
    x = 1 << 20
    print(f"LAPLEX / Toeplitz-FFT at n=2^20: {lap[x]:.4g} / {toe[x]:.4g} ms = {lap[x] / toe[x]:.2f}")
    x = den.index[-1]
    print(f"Dense / LAPLEX at n=2^{int(x).bit_length() - 1}: {den[x]:.4g} / {lap[x]:.4g} ms = "
          f"{den[x] / lap[x]:.0f}x")


def main():
    args = parser().parse_args()
    m5, d5 = load(args.results, args.fft_tag, FIG05)
    m1, d1 = load(args.results, args.dense_tag, FIG01)
    check_protocol(m5, m1)
    t = {m: series(d5, m) for m in ("laplex", "toeplitz", "cufft")}
    t["dense"] = series(d1, "dense")
    # the shared methods of the two sweeps should agree; a large gap means a busy GPU
    for m in ("laplex", "cufft"):
        r = (series(d1, m) / t[m]).dropna()
        print(f"{m}: Fig. 1 / Fig. 5 sweep time ratio {r.min():.2f}-{r.max():.2f} over {len(r)} sizes")
    print("font:", set_style(args.fonts))
    plot(t, args.out)
    report(t, m5, m1)
    print("wrote", args.out.with_suffix(".pdf"), args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
