"""Shared figure style, layout and OOM marking.

Style.  The published benchmark figures are matplotlib .pgf files in a seaborn
"whitegrid"-like style (light grid and frame, no tick marks), sans-serif text
typeset by LaTeX, the seaborn "colorblind" palette (blue, orange, green, ...)
assigned in legend order.  The reproduction uses the same style and is saved as
.pgf (typeset with the paper's fonts when included) and as .pdf through the pgf
backend; the .png preview is rendered by Agg and only approximates the fonts.

Layout.  ``figure(key)`` takes the size, the axes rectangles and the font sizes
of the published figure (manuscript.geometry), so both versions look alike.
``finish()`` keeps the published scale of every panel (inches per decade): the
y range covers the published range, the plotted data and the ranges recorded by
other versions of the same figure (``--ylim-json``), and the plot area grows by
the same height above and below the published range in every panel, so a value
sits where it sits in the manuscript figure, shifted by that height.
<stem>.layout.json records the saved width and where the top of the published
range lies, which the HTML report uses to align both versions.  The x axis is
always base 2 (2^k), the y axis base 10.  Legends name the GPU as the manuscript
does (device_label), not the results tag, and take one row where it fits.

PAPER (``--paper``): the version for the manuscript itself, <stem>_paper.{pgf,pdf}:
the published figure size (so it takes the same space on the page; taller only
by a legend row the published figure did not have) and a y range fitted to this
version's data only (no growth, no other versions).

Stop markers, on the last size measured of each series (no vertical lines): red
dot = the next size does not fit ("GPU Memory Limit", as in Fig. 1); open square
= the next size is outside the implementation (int32 indexing of the custom
kernels, LIMIT) or failed; open circle = accuracy figures, the fp64 reference
does not fit at the next size, so no error can be computed (REF_OOM).
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogFormatterSciNotation, LogLocator,  # noqa: E402
                               NullFormatter)

import manuscript  # noqa: E402

# seaborn "colorblind", as in the published figures
BLUE, ORANGE, GREEN, VERMILLION = "#0173B2", "#DE8F05", "#029E73", "#D55E00"
GRAY = "0.35"
LIMIT_COLOR = "#E8000B"
COLORS = {"laplex": BLUE, "laplex_cpu": BLUE, "dense": ORANGE, "cufft": GREEN, "toeplitz": ORANGE,
          "rff": ORANGE, "fft": "black"}
MARKERS = {"laplex": "o", "laplex_cpu": "o", "dense": "s", "cufft": "s", "toeplitz": "^", "rff": "o",
           "fft": "s"}
LABELS = {"laplex": "LAPLEX (ours)", "dense": "Dense", "cufft": "cuFFT", "toeplitz": "Toeplitz-FFT",
          "rff": "RFF", "laplex_cpu": "LAPLEX (no CUDA)", "fft": "Pure FFT (CPU)"}
DEVICE_STYLES = ["-", "--", ":", "-."]
PAPER = False   # set by common.main for --paper
STOP_GRAY = "0.25"


def device_label(tag):
    """GPU name for legends, as in the manuscript: 'H100-max-fused' -> 'H100', 'RTX2080Ti' -> 'RTX2080'."""
    base = re.sub(r"-max(-fused)?$|-fused$", "", tag)
    return {"RTX2080Ti": "RTX2080"}.get(base, base)


def style(tick_size=12.32, label_size=13.44):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": tick_size, "axes.labelsize": label_size, "axes.titlesize": label_size,
        "xtick.labelsize": tick_size, "ytick.labelsize": tick_size, "legend.fontsize": tick_size,
        "axes.edgecolor": "0.8", "axes.linewidth": 1.25, "axes.facecolor": "white",
        "axes.grid": True, "axes.grid.which": "both", "grid.color": "0.8", "grid.linewidth": 0.8,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
        "axes.formatter.use_mathtext": True,
        "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "pgf.preamble": manuscript.FONT_PREAMBLE,
        "savefig.dpi": 200,
    })


def figure(key, ncols=1, nrows=1, figsize=None, sharey=False):
    """Figure with the size and axes rectangles of the published figure ``key``."""
    g = manuscript.geometry(key)
    if g and len(g.axes) == ncols * nrows:
        style(g.tick_size, g.label_size)
        fig = plt.figure(figsize=g.figsize)
        W, H = g.figsize
        axes = []
        for a in g.axes:
            x0, y0, w, h = a.rect
            kw = {"sharey": axes[-1]} if sharey and axes else {}
            axes.append(fig.add_axes((x0 / W, y0 / H, w / W, h / H), **kw))
        fig._paper = g
    else:
        style()
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5.5 * ncols, 4), sharey=sharey)
        axes = list(np.atleast_1d(axes).ravel())
        fig._paper = None
    fig._key = key
    return fig, axes


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ok_rows(df):
    return df[(df["status"] == "ok") & (df["repeat"] >= 0)]


def aggregate(df, keys=("device", "method", "mode", "metric", "x")):
    """min / mean / std / median / max / count over repeats of successful rows."""
    g = ok_rows(df).groupby(list(keys))["value"]
    out = g.agg(["min", "mean", "std", "median", "max", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def stop_of(df, **sel):
    """(status, x) of the first failing size of a series, or None."""
    m = df["repeat"] < 0
    for k, v in sel.items():
        m &= df[k] == v
    r = df[m & (df["status"] != "ok")]
    if r.empty:
        return None
    r = r.sort_values("x").iloc[0]
    return r["status"], r["x"]


def draw(ax, df, agg, *, method, mode, metric, device, color=None, marker=None, label=None,
         stat="min", band=False, linestyle="-", markevery=True):
    """Draw one series and its stopping marker; returns the plotted (x, y)."""
    sel = dict(device=device, method=method, mode=mode, metric=metric)
    a = agg
    for k, v in sel.items():
        a = a[a[k] == v]
    a = a.sort_values("x")
    color = color or COLORS.get(method, "k")
    marker = marker if marker is not None else (MARKERS.get(method, "o") if markevery else None)
    if a.empty:
        return None
    x, y = a["x"].to_numpy(float), a[stat].to_numpy(float)
    ax.plot(x, y, linestyle=linestyle, marker=marker, color=color, label=label)
    if band:
        lo = np.maximum(a["mean"] - a["std"], 1e-12)
        ax.fill_between(x, lo, a["mean"] + a["std"], color=color, alpha=0.2, linewidth=0)
    st = stop_of(df, **sel)
    if st is not None:
        ax.plot([x[-1]], [y[-1]], **STOP_MARKERS[_stop_kind(st[0])], linestyle="", zorder=5)
    return x, y


# last size before a stop: red dot = OOM; open square = implementation limit (int32) or error;
# open circle = the fp64 reference does not fit (accuracy)
STOP_MARKERS = {
    "oom": dict(marker="o", color=LIMIT_COLOR, markersize=8),
    "limit": dict(marker="s", markerfacecolor="none", markeredgecolor=STOP_GRAY, markeredgewidth=1.6,
                  markersize=11),
    "ref": dict(marker="o", markerfacecolor="none", markeredgecolor=STOP_GRAY, markeredgewidth=1.6,
                markersize=11),
}
STOP_LABELS = {"oom": "GPU Memory Limit", "limit": "Implementation limit", "ref": "Reference OOM"}


def _stop_kind(status):
    return {"OOM": "oom", "REF_OOM": "ref"}.get(status, "limit")


def limit_legend_handles(df):
    """Legend proxies for the stopping markers that actually appear in ``df``."""
    stat = set(df.loc[(df["repeat"] < 0) & (df["status"] != "ok"), "status"])
    kinds = {_stop_kind(s) for s in stat}
    h = [Line2D([], [], linestyle="", label=STOP_LABELS[k], **STOP_MARKERS[k])
         for k in ("oom", "limit", "ref") if k in kinds]
    if "limit" in kinds and stat - {"OOM", "REF_OOM", "LIMIT"}:
        h[[k for k in ("oom", "limit", "ref") if k in kinds].index("limit")].set_label("Stopped (limit / error)")
    return h


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
def log2_x(ax, max_labels=9):
    """Base-2 log x axis with 2^k labels (integer k, step chosen to keep <= max_labels)."""
    ax.set_xscale("log", base=2)
    lo, hi = ax.get_xlim()
    k0, k1 = math.ceil(math.log2(lo) - 1e-9), math.floor(math.log2(hi) + 1e-9)
    step = max(1, math.ceil((k1 - k0 + 1) / max_labels))
    start = k0 + (-k0) % step if k0 % step else k0
    ax.xaxis.set_major_locator(FixedLocator([2.0 ** k for k in range(start, k1 + 1, step)]))
    ax.xaxis.set_minor_locator(FixedLocator([2.0 ** k for k in range(k0, k1 + 1)]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$2^{{{int(round(math.log2(v)))}}}$"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def log10_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=30))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=30))
    lo, hi = ax.get_ylim()
    narrow = math.log10(hi / lo) < 1.0   # under one decade: label 2x10^k etc., as the paper does
    ax.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 1))
                                 if narrow else NullFormatter())


def _data_range(ax):
    """Range of the plotted data (lines and bands) on the y axis."""
    ys = []
    for ln in ax.get_lines():
        if not ln.get_transform().contains_branch_seperately(ax.transData)[1]:
            continue   # y in axes coordinates (axvline: stop markers)
        y = np.asarray(ln.get_ydata(), float)
        if y.size > 1 or ln.get_marker() not in ("", "None"):
            ys.append(y)
    for c in ax.collections:
        for p in c.get_paths():
            ys.append(p.vertices[:, 1])
    if not ys:
        return None
    y = np.concatenate(ys)
    y = y[np.isfinite(y)]
    if ax.get_yscale() == "log":
        y = y[y > 0]
    return (float(y.min()), float(y.max())) if y.size else None


def _pad(lo, hi, log, frac=0.04):
    if log:
        a, b = math.log10(lo), math.log10(hi)
        d = (b - a) * frac or 0.1
        return 10 ** (a - d), 10 ** (b + d)
    d = (hi - lo) * frac or 1.0
    return lo - d, hi + d


def _yscale_of(pa):
    """(transform, inches per transformed unit) of a published panel: log10 for log axes."""
    f = math.log10 if pa.yscale == "log" else (lambda v: v)
    return f, pa.rect[3] / (f(pa.ylim[1]) - f(pa.ylim[0]))


def finish(fig, axes, out_dir: Path, stem: str, ylim_json=(), legend=None, legend_ncol=None,
           legend_axes=None, legend_loc="upper right"):
    """Apply axis conventions, harmonised y ranges and the paper layout; save .pgf/.pdf/.png.

    Records the data y range of each panel in <stem>.yrange.json for other versions.
    ``legend_ncol``: fixed columns filled column by column (Figs. 1/15: one column per method);
    by default one row, or the fewest rows that fit the figure width, entries in reading order.
    """
    g = getattr(fig, "_paper", None)
    if PAPER:
        ylim_json = ()
    data = []
    for ax in axes:
        if ax.get_xscale() == "log":
            log2_x(ax)
        data.append(_data_range(ax))
    others = []
    for f in ylim_json:
        if Path(f).name != f"{stem}.yrange.json":
            continue
        try:
            others.append(json.loads(Path(f).read_text())["data"])
        except (OSError, KeyError, ValueError):
            pass
    limits = {}
    for i, ax in enumerate(axes):
        cand = [r for r in [data[i]] + [o[i] for o in others if i < len(o)] if r]
        if cand:
            limits[i] = _pad(min(r[0] for r in cand), max(r[1] for r in cand), ax.get_yscale() == "log")
    for i, ax in enumerate(axes):   # panels sharing a y axis get one range
        sib = [j for j, b in enumerate(axes) if j in limits and b in ax.get_shared_y_axes().get_siblings(ax)]
        if i in limits:
            limits[i] = (min(limits[j][0] for j in sib), max(limits[j][1] for j in sib))
    # Published panels keep their scale (inches per decade / per unit); every one grows by the same
    # height `up` above and `down` below its published range, so one shift aligns all panels.
    pub = {i: pa for i, (ax, pa) in enumerate(zip(axes, g.axes if g else []))
           if pa.ylim and pa.yscale == ax.get_yscale() and not PAPER}
    up = down = 0.0
    for i, pa in pub.items():
        if i in limits:
            f, s = _yscale_of(pa)
            up = max(up, s * (f(limits[i][1]) - f(pa.ylim[1])))
            down = max(down, s * (f(pa.ylim[0]) - f(limits[i][0])))
    for i, ax in enumerate(axes):
        if i in pub:
            pa = pub[i]
            f, s = _yscale_of(pa)
            lo, hi = f(pa.ylim[0]) - down / s, f(pa.ylim[1]) + up / s
            ax.set_ylim(*((10 ** lo, 10 ** hi) if pa.yscale == "log" else (lo, hi)))
        elif i in limits:
            ax.set_ylim(*limits[i])
    for ax in axes:
        if ax.get_yscale() == "log":
            log10_y(ax)
        ax.grid(True, which="major", color="0.8", linewidth=0.8)
        ax.grid(True, which="minor", color="0.93", linewidth=0.5)
        ax.set_axisbelow(True)
    grow = up + down
    if g and grow > 1e-3:
        W, H = fig.get_size_inches()
        rows = sorted({round(pa.rect[1], 3) for pa in g.axes})
        Hn = H + grow * len(rows)
        fig.set_size_inches(W, Hn)
        for ax, pa in zip(axes, g.axes):
            x0, y0, w, h = pa.rect
            # margins and the space above the axes (legend, titles) keep their size in inches
            ax.set_position((x0 / W, (y0 + rows.index(round(y0, 3)) * grow) / Hn, w / W, (h + grow) / Hn))
    if legend is not None:
        handles, labels = legend
        if legend_axes is not None:
            legend_axes.legend(handles, labels, loc=legend_loc, frameon=True, framealpha=0.9,
                               edgecolor="0.8")
        else:
            # above everything drawn on top of the axes (titles included)
            r = fig.canvas.get_renderer()
            inv = fig.transFigure.inverted()
            top = max(max(inv.transform(ax.title.get_window_extent(r))[1, 1] if ax.get_title() else 0,
                          ax.get_position().y1) for ax in axes)
            for nc in [legend_ncol] if legend_ncol else range(len(handles), 0, -1):
                hl = (handles, labels) if legend_ncol else _reading_order(handles, labels, nc)
                leg = fig.legend(*hl, loc="lower center", bbox_to_anchor=(0.5, top + 0.01), ncol=nc,
                                 frameon=False, columnspacing=1.5, handlelength=2.2)
                if legend_ncol or nc == 1 or \
                        leg.get_window_extent(r).width <= 0.98 * fig.get_size_inches()[0] * fig.dpi:
                    break
                leg.remove()
            W, H = fig.get_size_inches()
            over = (inv.transform(leg.get_window_extent(r))[1, 1] - 1) * H + 0.02   # inches above the top
            if PAPER and over > 0:
                # more legend rows than the published figure had: the figure grows by them, the axes
                # keep their size and place (instead of the legend covering the plot)
                boxes = [ax.get_position() for ax in axes]
                fig.set_size_inches(W, H + over)
                for ax, b in zip(axes, boxes):
                    ax.set_position((b.x0, b.y0 * H / (H + over), b.width, b.height * H / (H + over)))
                leg.set_bbox_to_anchor((0.5, (top + 0.01) * H / (H + over)), transform=fig.transFigure)
    out_dir.mkdir(parents=True, exist_ok=True)
    if PAPER:
        # the published size exactly: the figure takes the same space in the manuscript
        bb = fig.get_tightbbox(fig.canvas.get_renderer())   # inches
        W, H = fig.get_size_inches()
        if bb.x0 < -0.02 or bb.y0 < -0.02 or bb.x1 > W + 0.02 or bb.y1 > H + 0.02:
            print(f"  WARNING {stem}_paper: content outside the published size {W:.2f}x{H:.2f} in: "
                  f"x {bb.x0:.2f}..{bb.x1:.2f}, y {bb.y0:.2f}..{bb.y1:.2f}")
        exts = save_vector(fig, out_dir / f"{stem}_paper")
        plt.close(fig)
        print(f"  saved {out_dir / stem}_paper.{{{','.join(exts)}}}")
        return
    (out_dir / f"{stem}.yrange.json").write_text(json.dumps({"data": data}))
    # an explicit tight box, so that the layout of the saved file is known
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.05)
    p0 = axes[0].get_position()
    top = p0.y0 * fig.get_size_inches()[1] + (down + g.axes[0].rect[3] if 0 in pub else p0.height * fig.get_size_inches()[1])
    (out_dir / f"{stem}.layout.json").write_text(json.dumps(
        dict(width_in=bb.width, height_in=bb.height, anchor_in=bb.y1 - top,
             note="anchor_in: inches from the top of the saved figure to the top of the published y range "
                  "of the first panel (its axes top if it has none)")))
    exts = save_vector(fig, out_dir / stem, bbox_inches=bb)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches=bb)
    plt.close(fig)
    print(f"  saved {out_dir / stem}.{{{','.join(exts)},png}}")


def save_vector(fig, base: Path, **kw):
    """<base>.pgf and <base>.pdf typeset by LaTeX (pgf backend); without pdflatex only an Agg
    <base>.pdf, whose fonts only approximate the paper's.  Returns the extensions written."""
    if shutil.which(plt.rcParams["pgf.texsystem"]):
        for ext in ("pgf", "pdf"):
            fig.savefig(f"{base}.{ext}", backend="pgf", **kw)
        return ["pgf", "pdf"]
    print(f"  WARNING: {plt.rcParams['pgf.texsystem']} not found, {base.name}.pdf without LaTeX fonts, no .pgf")
    fig.savefig(f"{base}.pdf", **kw)
    return ["pdf"]


def _reading_order(handles, labels, ncol):
    """Reorder for a legend filled column by column so that it reads row by row."""
    nrows, nlarge = divmod(len(handles), ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrows + (c < nlarge))]
    return [handles[i] for i in order], [labels[i] for i in order]


def collect_legend(axes, extra=(), insert_extra_at=None):
    handles, labels, seen = [], [], set()
    for ax in np.atleast_1d(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and not l.startswith("_"):
                seen.add(l)
                handles.append(h)
                labels.append(l)
    pos = len(handles) if insert_extra_at is None else insert_extra_at
    for h in extra:
        handles.insert(pos, h)
        labels.insert(pos, h.get_label())
        pos += 1
    return handles, labels
