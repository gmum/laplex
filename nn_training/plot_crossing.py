"""Figure and summary for crossing_trace.py runs (App. M, Fig. 23).

For every run directory under results/crossing/ (or the ones given) writes
``<run>/fig_crossing.{pdf,png}`` and ``<run>/summary.json``, and collects all runs in
``results/crossing/summary.md`` next to the values the paper reports.

    bash sif.sh python plot_crossing.py
    bash sif.sh python plot_crossing.py results/crossing/aug_m500_e6_seed0
    bash sif.sh python plot_crossing.py results/crossing/noaug_m500_e6_seed0 --paper FIG.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "crossing"
# App. M, non-augmented run (Fig. 23)
PAPER = dict(input_rates="0.49 - 0.50 per epoch; 0.44 - 0.48 per step after the first",
             output_rates="0.23 in the first epoch, about 0.10 later; 0.00 - 0.06 per step",
             anchor_grad_norm="0.13 - 1.49 (joint); 0.31 - 0.84 after the first five steps",
             loss="epoch-level loss decreases monotonically, 2.13 -> 1.60; "
                  "minibatch loss increases in 68 of 143 steps")


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def running_mean(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, min(w, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    out[:w - 1] = c[1:w] / np.arange(1, w)
    return out


def summarize_run(run: Path) -> dict:
    s, e = read_csv(run / "steps.csv"), read_csv(run / "epochs.csv")
    meta = json.loads((run / "meta.json").read_text())
    loss = s["loss"]
    inc = np.diff(loss) > 0
    return dict(
        run=run.name, steps=int(len(loss)), epochs=int(len(e["epoch"])), augment=meta["args"].get("augment"),
        input_epoch_rates=[round(x, 4) for x in e["cross_a_epoch"]],
        output_epoch_rates=[round(x, 4) for x in e["cross_b_epoch"]],
        input_step_rate_mean=float(s["cross_a_step"].mean()), output_step_rate_mean=float(s["cross_b_step"].mean()),
        grad_a=[float(s["grad_a"].min()), float(s["grad_a"].max())],
        grad_b=[float(s["grad_b"].min()), float(s["grad_b"].max())],
        grad_anchor=[float(s["grad_anchor"].min()), float(s["grad_anchor"].max())],
        loss_step_increases=int(inc.sum()), loss_step_increase_fraction=float(inc.mean()) if len(inc) else 0.0,
        epoch_train_loss=[round(x, 4) for x in e["train_loss"]],
        epoch_loss_monotone=bool(np.all(np.diff(e["train_loss"]) < 0)),
        val_acc=[round(100 * x, 2) for x in e["val_acc"]],
        tau=[float(s["tau"][0]), float(s["tau"][-1])],
    )


def plot_run(run: Path, paper_out: Path | None = None) -> None:
    """Diagnostic figure in the run directory, or with ``paper_out`` a compact version for
    the paper: panels (a)-(d), no run name, sized for the full text width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, e = read_csv(run / "steps.csv"), read_csv(run / "epochs.csv")
    step, ep = s["step"], s["epoch"]
    spe = max(1, int(round(len(step) / max(1, len(e["epoch"])))))
    bounds = [step[ep == k][0] for k in np.unique(ep)][1:]
    paper = paper_out is not None
    fs = 8 if paper else 10
    plt.rcParams.update({"font.size": fs, "axes.titlesize": fs + 1, "legend.fontsize": fs - 1})
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.6) if paper else (11, 7), constrained_layout=True)
    titles = (("(a) Training loss per step", "(b) Anchor-gradient norm per step",
               "(c) Crossing rate per step", "(d) Crossing rate per epoch") if paper else
              ("Loss per step", "Anchor-gradient norm per step", "Crossing rate per step", "Per epoch (as Fig. 23)"))

    ax = axes[0, 0]
    ax.plot(step, s["loss"], lw=0.6, color="0.6", label="minibatch")
    ax.plot(step, running_mean(s["loss"], spe), lw=1.6, color="C3",
            label="running mean (1 epoch)" if paper else f"running mean ({spe} steps)")
    ax.set(xlabel="step", ylabel="training loss", title=titles[0])
    ax.legend()

    ax = axes[0, 1]
    ax.plot(step, s["grad_a"], lw=0.8, label=r"$\|\nabla_a L\|$ (input anchors)")
    ax.plot(step, s["grad_b"], lw=0.8, label=r"$\|\nabla_b L\|$ (output anchors)")
    ax.set(xlabel="step", ylabel="gradient norm", yscale="log", title=titles[1])
    ax.legend()

    ax = axes[1, 0]
    ax.plot(step, s["cross_a_step"], lw=0.8, label="input anchors $a$")
    ax.plot(step, s["cross_b_step"], lw=0.8, label="output anchors $b$")
    ax.set(xlabel="step", ylabel="crossing rate", title=titles[2])
    ax.legend()

    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        for b in bounds:
            a.axvline(b, color="0.85", lw=0.6, zorder=0)

    ax = axes[1, 1]
    ax.plot(e["epoch"], e["cross_a_epoch"], "o-", label="input anchors $a$")
    ax.plot(e["epoch"], e["cross_b_epoch"], "s-", label="output anchors $b$")
    ax.set(xlabel="epoch", ylabel="crossing rate" if paper else "crossing rate (start vs. end of epoch)",
           title=titles[3])
    tw = ax.twinx()
    tw.plot(e["epoch"], e["train_loss"], "--", color="C3", label="training loss")
    tw.set_ylabel("training loss")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")

    if paper:
        paper_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_out)
    else:
        fig.suptitle(f"Anchor crossings, {run.name}")
        for ext in ("pdf", "png"):
            fig.savefig(run / f"fig_crossing.{ext}", dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", type=Path)
    p.add_argument("--paper", type=Path, default=None, metavar="OUT.pdf",
                   help="only write the compact paper figure of the single run given")
    args = p.parse_args()
    if args.paper:
        if len(args.runs) != 1:
            p.error("--paper needs exactly one run directory")
        plot_run(args.runs[0], paper_out=args.paper)
        print(f"written {args.paper}")
        return 0
    runs = args.runs or sorted(d for d in OUT.iterdir() if (d / "steps.csv").is_file())
    if not runs:
        print(f"no runs in {OUT}", file=sys.stderr)
        return 1
    summaries = []
    for run in runs:
        sm = summarize_run(run)
        (run / "summary.json").write_text(json.dumps(sm, indent=1))
        plot_run(run)
        summaries.append(sm)
        print(json.dumps(sm))

    def rng(v):
        return f"{v[0]:.3g} - {v[1]:.3g}"
    lines = ["# Anchor-crossing diagnostic (crossing_trace.py)", "",
             "Paper values (App. M, Fig. 23) for comparison:", ""]
    lines += [f"* {k}: {v}" for k, v in PAPER.items()]
    lines += ["", "| run | aug | epochs | input rate per epoch | output rate per epoch | mean step rate in/out "
              "| anchor grad norm (joint) | da | db | loss increases step-to-step | epoch loss monotone | val acc % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sm in summaries:
        lines.append(
            f"| {sm['run']} | {sm['augment']} | {sm['epochs']} | {', '.join(f'{x:.3f}' for x in sm['input_epoch_rates'])} "
            f"| {', '.join(f'{x:.3f}' for x in sm['output_epoch_rates'])} "
            f"| {sm['input_step_rate_mean']:.4f} / {sm['output_step_rate_mean']:.4f} | {rng(sm['grad_anchor'])} "
            f"| {rng(sm['grad_a'])} | {rng(sm['grad_b'])} | {sm['loss_step_increases']}/{sm['steps'] - 1} "
            f"| {sm['epoch_loss_monotone']} | {', '.join(f'{x:.1f}' for x in sm['val_acc'])} |")
    lines += ["", "Crossing rate per epoch: adjacent pairs of the order at the start of the epoch that are",
              "reversed at its end. Per step: the same between consecutive optimiser steps."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
