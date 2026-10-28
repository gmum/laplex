"""Anchor-crossing diagnostic with per-step and per-epoch traces (App. M, Fig. 23).

Trains the trainable LAPLEX projection of Table 1 (``Net`` / ``ExpKernelLayer`` from
``CountSketch_LogisticRegression/laplex.py``, m = 500) with the optimiser,
scheduler and loss of its ``train_model`` (AdamW lr 1e-3 wd 1e-4, ReduceLROnPlateau on
the validation loss, cross-entropy) and logs, at every optimiser step:

* the minibatch loss,
* ||grad a||, ||grad b|| (input anchors ``alpha`` in R^d, output anchors ``beta`` in R^m)
  and their joint norm, taken after backward and before the step,
* the step crossing rate of each anchor set: the fraction of adjacent pairs in the sorted
  order before the step whose order is reversed after the step,
* tau and the learning rate.

Per epoch it logs the epoch crossing rate (adjacent pairs in the order at the start of the
epoch that are reversed at its end), the sample-weighted mean training loss, validation
loss/accuracy and tau. The paper does not say which crossing definition Fig. 23 used;
both are recorded. ``ExpKernelLayer`` materialises K = exp(-|alpha - beta| / tau) densely,
so crossings never pass through the scan in this experiment (gradients are the same).

    bash sif.sh python crossing_trace.py                       # augmented data, 6 epochs (Fig. 23 shows 0-5)
    bash sif.sh python crossing_trace.py --no-augment --seed 1
    bash sif.sh python crossing_trace.py --fake-data 64 --fake-dim 3072 --batch-size 16 --epochs 2 --device cpu

Writes results/crossing/<tag>/{steps.csv, epochs.csv, meta.json}; plot with plot_crossing.py.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
CS_DIR = HERE / "CountSketch_LogisticRegression"
DEFAULT_DATA = HERE / "data" / "imagenet10"
OUT = HERE / "results" / "crossing"


def load_original():
    """The released model and loader, imported unchanged."""
    sys.path.insert(0, str(CS_DIR))
    import imagenet10_dataset
    spec = importlib.util.spec_from_file_location("cs_laplex", CS_DIR / "laplex.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, imagenet10_dataset


def crossing_rate(x_now: torch.Tensor, order_before: torch.Tensor) -> float:
    """Fraction of pairs adjacent in ``order_before`` whose order is reversed in ``x_now``."""
    xs = x_now[order_before]
    return (xs[1:] < xs[:-1]).float().mean().item()


def make_loaders(args, ds_mod):
    if args.fake_data:
        g = torch.Generator().manual_seed(args.seed)
        def fake(n):
            return TensorDataset(torch.randn(n, args.fake_dim, generator=g), torch.randint(0, 10, (n,), generator=g))
        train_ds, val_ds = fake(args.fake_data), fake(max(args.fake_data // 4, args.batch_size))
    else:
        train_tf = ds_mod.DEFAULT_TRAIN_TRANSFORM if args.augment else ds_mod.DEFAULT_VAL_TRANSFORM
        train_ds = ds_mod.ImageNet10(root=args.root_dataset, split="train", transform=train_tf)
        val_ds = ds_mod.ImageNet10(root=args.root_dataset, split="val")
    pin = args.device == "cuda"
    # Same loader settings as get_imagenet10_loaders (shuffle, drop_last on train).
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=pin, drop_last=True)
    val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                     pin_memory=pin)
    return train, val


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--m", type=int, default=500)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512, help="run_experiments.sh uses 512")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--augment", dest="augment", action="store_true", default=True,
                   help="released training transform (RandomResizedCrop, flip, ColorJitter; default)")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="validation transform for training too (App. M ablations used non-augmented features)")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tag", default=None, help="output subdirectory (default: derived from the settings)")
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    p.add_argument("--fake-data", type=int, default=0, metavar="N", help="random inputs instead of ImageNet10")
    p.add_argument("--fake-dim", type=int, default=3 * 224 * 224)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cs, ds_mod = load_original()
    train_loader, val_loader = make_loaders(args, ds_mod)
    d = train_loader.dataset[0][0].numel()
    n_classes = 10
    model = cs.Net(d=d, m=args.m, num_classes=n_classes).to(args.device)
    proj = model.proj
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=7, min_lr=1e-5)

    tag = args.tag or (f"{'fake' if args.fake_data else ('aug' if args.augment else 'noaug')}"
                       f"_m{args.m}_e{args.epochs}_seed{args.seed}")
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(args={k: str(v) for k, v in vars(args).items()}, d=d, n_train=len(train_loader.dataset),
                n_val=len(val_loader.dataset), steps_per_epoch=len(train_loader), host=socket.gethostname(),
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                model_source=str(CS_DIR / "laplex.py"),
                crossing_definitions=dict(
                    step="pairs adjacent in the sorted order before an optimiser step, reversed after it",
                    epoch="pairs adjacent in the sorted order at the start of an epoch, reversed at its end"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    step_fields = ["step", "epoch", "loss", "grad_a", "grad_b", "grad_anchor", "grad_w", "grad_log_tau",
                   "cross_a_step", "cross_b_step", "tau", "lr"]
    epoch_fields = ["epoch", "steps", "train_loss", "val_loss", "val_acc", "tau", "lr",
                    "cross_a_epoch", "cross_b_epoch", "cross_a_step_mean", "cross_b_step_mean",
                    "grad_anchor_min", "grad_anchor_max", "loss_step_increases", "seconds"]
    fs, fe = open(out / "steps.csv", "w", newline=""), open(out / "epochs.csv", "w", newline="")
    ws, we = csv.DictWriter(fs, step_fields), csv.DictWriter(fe, epoch_fields)
    ws.writeheader()
    we.writeheader()

    alpha, beta = proj.alpha, proj.beta
    step, prev_loss, stop = 0, None, False
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        a_start, b_start = alpha.detach().clone(), beta.detach().clone()
        loss_sum, n_seen, rows = 0.0, 0, []
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            order_a, order_b = torch.argsort(alpha.detach()), torch.argsort(beta.detach())
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            ga, gb = alpha.grad.norm().item(), beta.grad.norm().item()
            row = dict(step=step, epoch=epoch, loss=loss.item(), grad_a=ga, grad_b=gb,
                       grad_anchor=math.hypot(ga, gb), grad_w=proj.w.grad.norm().item(),
                       grad_log_tau=proj.log_tau.grad.abs().item(),
                       tau=(nn.functional.softplus(proj.log_tau) + 1e-6).item(),
                       lr=optimizer.param_groups[0]["lr"])
            optimizer.step()
            row["cross_a_step"] = crossing_rate(alpha.detach(), order_a)
            row["cross_b_step"] = crossing_rate(beta.detach(), order_b)
            ws.writerow(row)
            rows.append(row)
            loss_sum += row["loss"] * x.size(0)
            n_seen += x.size(0)
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        fs.flush()
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        scheduler.step(val_loss)
        losses = [r["loss"] for r in rows]
        if prev_loss is not None:
            losses = [prev_loss] + losses
        erow = dict(epoch=epoch, steps=len(rows), train_loss=loss_sum / max(n_seen, 1), val_loss=val_loss,
                    val_acc=val_acc, tau=(nn.functional.softplus(proj.log_tau) + 1e-6).item(),
                    lr=optimizer.param_groups[0]["lr"],
                    cross_a_epoch=crossing_rate(alpha.detach(), torch.argsort(a_start)),
                    cross_b_epoch=crossing_rate(beta.detach(), torch.argsort(b_start)),
                    cross_a_step_mean=float(np.mean([r["cross_a_step"] for r in rows])),
                    cross_b_step_mean=float(np.mean([r["cross_b_step"] for r in rows])),
                    grad_anchor_min=min(r["grad_anchor"] for r in rows),
                    grad_anchor_max=max(r["grad_anchor"] for r in rows),
                    loss_step_increases=int(sum(b > a for a, b in zip(losses, losses[1:]))),
                    seconds=time.time() - t0)
        prev_loss = rows[-1]["loss"]
        we.writerow(erow)
        fe.flush()
        print(f"epoch {epoch}: train {erow['train_loss']:.4f} val {val_loss:.4f} acc {100 * val_acc:.2f}% "
              f"cross a/b epoch {erow['cross_a_epoch']:.3f}/{erow['cross_b_epoch']:.3f} "
              f"step-mean {erow['cross_a_step_mean']:.4f}/{erow['cross_b_step_mean']:.4f} "
              f"|grad| [{erow['grad_anchor_min']:.3g}, {erow['grad_anchor_max']:.3g}] "
              f"loss up {erow['loss_step_increases']}/{len(rows)} tau {erow['tau']:.4f} "
              f"({erow['seconds']:.0f} s)", flush=True)
        if stop:
            break
    fs.close()
    fe.close()
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
