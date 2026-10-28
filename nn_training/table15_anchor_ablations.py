"""
table15_anchor_ablations.py  (App. M: Tab. 15)
===========================
Anchor initialization and number of kernel components for the *trainable* LAPLEX
projection on ImageNet-10 (m = 500). Same task and model as
`CountSketch_LogisticRegression/laplex.py` (the trainable-LAPLEX column of Table 1);
heads, optimiser, schedule and early stopping match it.

Two modes (choose with --mode):

  init_sweep  (Tab. 15, left)
      Sweeps the anchor init distribution (normal / uniform / linspace), its scale,
      and the initial temperature tau, each over several seeds, and reports best
      accuracy mean +/- std.

  components  (Tab. 15, right)
      Sweeps the number of components I of a multi-component projection (sum of I
      independent Laplace kernels with per-component mixing weights) over several
      seeds; reports best accuracy and the mean anchor crossing rate per I.

The paper's runs use cached, non-augmented features (--features_on_gpu).

Example
-------
    python table15_anchor_ablations.py --mode init_sweep \
        --root_dataset data/imagenet10 --m 500 --epochs 30 \
        --seeds 1 2 3 4 5 --features_on_gpu

    python table15_anchor_ablations.py --mode components \
        --root_dataset data/imagenet10 --m 500 --epochs 30 --batch_size 128 \
        --components 1 2 3 4 --seeds 1 2 3 --features_on_gpu
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys

# imagenet10_dataset.py of the released Table 1 experiments
sys.path.insert(0, str(Path(__file__).resolve().parent / "CountSketch_LogisticRegression"))
from imagenet10_dataset import get_imagenet10_loaders  # noqa: E402


# =============================================================================
# Models
# =============================================================================
def _init_anchor(n: int, dist: str, scale: float) -> torch.Tensor:
    """Initialise an anchor vector of length n under the requested scheme.

    dist:
        "normal"   -> scale * N(0, 1)                (laplex.py default)
        "uniform"  -> Uniform(-scale, scale)
        "linspace" -> evenly spaced on [-scale, scale] (deterministic grid)
    """
    if dist == "normal":
        return scale * torch.randn(n)
    if dist == "uniform":
        return (torch.rand(n) * 2.0 - 1.0) * scale
    if dist == "linspace":
        return torch.linspace(-scale, scale, n)
    raise ValueError(f"unknown anchor dist: {dist}")


class ExpKernelLayer(nn.Module):
    """Single-component trainable LAPLEX projection (identical to CountSketch_LogisticRegression/laplex.py,
    with configurable initialisation for the init-sensitivity sweep).

    z = (x * w) @ K,   K[i, j] = exp(-|alpha_i - beta_j| / tau),  tau > 0.
    Only d + m + 1 coordinates are learned (alpha, beta, w share d/m, log_tau).
    """

    def __init__(
        self,
        d: int,
        m: int,
        init_tau: float = 0.1,
        alpha_dist: str = "normal",
        beta_dist: str = "normal",
        alpha_scale: float = 1.0,
        beta_scale: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.m = m
        self.alpha = nn.Parameter(_init_anchor(d, alpha_dist, alpha_scale))
        self.beta = nn.Parameter(_init_anchor(m, beta_dist, beta_scale))
        self.w = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_tau))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 1e-6
        diff = torch.abs(self.alpha.unsqueeze(1) - self.beta.unsqueeze(0))
        K = torch.exp(-diff / tau)
        return (x * self.w) @ K


class MultiExpKernelLayer(nn.Module):
    """Multi-component trainable LAPLEX projection: a sum of I independent
    Laplace kernels with per-component mixing weights.

        z = sum_{c=1..I} g_c * ((x * w_c) @ K_c),
        K_c[i, j] = exp(-|alpha^c_i - beta^c_j| / tau_c).

    This is the natural "number of anchor components" knob for a projection
    layer (cf. the multi-component variant I in {1,2,3} used in the appendix
    density model). I = 1 reduces exactly to ExpKernelLayer.
    """

    def __init__(self, d: int, m: int, n_comp: int = 1, init_tau: float = 0.1):
        super().__init__()
        self.d = d
        self.m = m
        self.n_comp = n_comp
        self.alpha = nn.Parameter(torch.randn(n_comp, d))
        self.beta = nn.Parameter(torch.randn(n_comp, m))
        self.w = nn.Parameter(torch.randn(n_comp, d) / math.sqrt(d))
        self.log_tau = nn.Parameter(torch.log(torch.full((n_comp,), float(init_tau))))
        # Per-component mixing weights (start balanced).
        self.g = nn.Parameter(torch.ones(n_comp) / n_comp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 1e-6  # (I,)
        z = 0.0
        for c in range(self.n_comp):
            diff = torch.abs(self.alpha[c].unsqueeze(1) - self.beta[c].unsqueeze(0))
            K = torch.exp(-diff / tau[c])
            z = z + self.g[c] * ((x * self.w[c]) @ K)
        return z


class Net(nn.Module):
    def __init__(self, proj: nn.Module, m: int, num_classes: int = 10):
        super().__init__()
        self.proj = proj
        self.classifier = nn.Linear(m, num_classes)

    def forward(self, x):
        return self.classifier(self.proj(x))


# =============================================================================
# Anchor-ordering metrics
# =============================================================================
def _order_and_rank(values: torch.Tensor):
    """Return (sorted_order, rank) for a 1-D tensor.

    sorted_order[k] = index of the k-th smallest anchor.
    rank[i]         = position of anchor i in the sorted order.
    """
    order = torch.argsort(values)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=values.device)
    return order, rank


@torch.no_grad()
def ordering_stats(
    prev_order: torch.Tensor, prev_rank: torch.Tensor, values: torch.Tensor
):
    """Order-stability metrics between the previous checkpoint and now.

    crossing_rate: fraction of adjacent pairs in the PREVIOUS sorted order
        whose values are now out of order (a "crossing"). Cheap: O(n).
    rank_change:   fraction of anchors whose absolute sorted rank changed.

    Returns (crossing_rate, rank_change, new_order, new_rank).
    """
    n = values.numel()
    left = values[prev_order[:-1]]
    right = values[prev_order[1:]]
    crossings = (left > right).sum().item()
    crossing_rate = crossings / max(n - 1, 1)

    new_order, new_rank = _order_and_rank(values)
    rank_change = (new_rank != prev_rank).float().mean().item()
    return crossing_rate, rank_change, new_order, new_rank


# =============================================================================
# Early stopping (inlined to avoid the wandb import in utils.py)
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# =============================================================================
# Training loop (shared) with optional anchor-ordering instrumentation
# =============================================================================
def anchor_grad_norm(layer: nn.Module) -> float:
    """L2 norm of the gradient on the anchor parameters (alpha, beta)."""
    g = 0.0
    for name in ("alpha", "beta"):
        p = getattr(layer, name, None)
        if p is not None and p.grad is not None:
            g += p.grad.detach().pow(2).sum().item()
    return math.sqrt(g)


def train_model(
    model,
    trn_loader,
    tst_loader,
    opts,
    track_ordering=False,
    track_first_epoch_steps=False,
):
    """Train one model. Returns (best_acc, history_df, ordering_df).

    ordering_df is empty unless track_ordering=True and the projection has
    `alpha`/`beta` parameters (i.e. a LAPLEX projection).
    """
    device = opts.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.8, patience=7, min_lr=1e-5
    )
    early = EarlyStopping(patience=opts.patience)

    layer = model.proj
    has_anchors = hasattr(layer, "alpha") and hasattr(layer, "beta")
    track_ordering = track_ordering and has_anchors

    # Snapshot the initial sorted order (state = t0). For the multi-component
    # layer we track the first component's anchors (representative).
    def _anchor_vec(attr):
        v = getattr(layer, attr).detach()
        return v[0] if v.dim() > 1 else v

    # Two INDEPENDENT reference states: one advanced per step (fine-grained,
    # first epoch), one advanced per epoch. Keeping them separate avoids the
    # per-step updates polluting the per-epoch comparison.
    if track_ordering:
        a_order_s, a_rank_s = _order_and_rank(_anchor_vec("alpha"))
        b_order_s, b_rank_s = _order_and_rank(_anchor_vec("beta"))
        a_order_e, a_rank_e = _order_and_rank(_anchor_vec("alpha"))
        b_order_e, b_rank_e = _order_and_rank(_anchor_vec("beta"))

    best_acc, res, order_rows = 0.0, [], []
    global_step = 0

    for epoch in (pbar := tqdm(range(opts.epochs), desc=f"m={layer.m} ep")):
        model.train()
        train_loss, total = 0.0, 0
        for x, y in trn_loader:
            x = x.view(x.size(0), -1).to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            # Per-step ordering trace during the first epoch (fine-grained view
            # of whether early training is where crossings concentrate).
            if track_ordering and track_first_epoch_steps and epoch == 0:
                gnorm = anchor_grad_norm(layer)
                optimizer.step()
                a_cr, a_rc, a_order_s, a_rank_s = ordering_stats(
                    a_order_s, a_rank_s, _anchor_vec("alpha")
                )
                b_cr, b_rc, b_order_s, b_rank_s = ordering_stats(
                    b_order_s, b_rank_s, _anchor_vec("beta")
                )
                order_rows.append(
                    dict(
                        scope="step",
                        epoch=epoch,
                        step=global_step,
                        loss=loss.item(),
                        anchor_grad_norm=gnorm,
                        alpha_crossing_rate=a_cr,
                        alpha_rank_change=a_rc,
                        beta_crossing_rate=b_cr,
                        beta_rank_change=b_rc,
                    )
                )
            else:
                optimizer.step()

            train_loss += loss.item() * x.size(0)
            total += x.size(0)
            global_step += 1

        avg_train_loss = train_loss / total

        # Evaluation
        model.eval()
        test_loss, correct, total_te = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tst_loader:
                x = x.view(x.size(0), -1).to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True)
                out = model(x)
                test_loss += criterion(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total_te += x.size(0)
        acc = correct / total_te
        avg_test_loss = test_loss / total_te
        scheduler.step(avg_test_loss)
        best_acc = max(best_acc, acc)

        # Per-epoch ordering trace (always, when tracking): compares against
        # the end of the previous epoch (epoch 0 -> against t0 init).
        if track_ordering:
            a_cr, a_rc, a_order_e, a_rank_e = ordering_stats(
                a_order_e, a_rank_e, _anchor_vec("alpha")
            )
            b_cr, b_rc, b_order_e, b_rank_e = ordering_stats(
                b_order_e, b_rank_e, _anchor_vec("beta")
            )
            order_rows.append(
                dict(
                    scope="epoch",
                    epoch=epoch,
                    step=global_step,
                    loss=avg_train_loss,
                    anchor_grad_norm=float("nan"),
                    alpha_crossing_rate=a_cr,
                    alpha_rank_change=a_rc,
                    beta_crossing_rate=b_cr,
                    beta_rank_change=b_rc,
                )
            )

        res.append(
            dict(
                m=layer.m,
                epoch=epoch,
                train_loss=avg_train_loss,
                test_loss=avg_test_loss,
                acc=acc,
                lr=optimizer.param_groups[0]["lr"],
            )
        )
        pbar.write(
            f"ep {epoch + 1}: loss {avg_train_loss:.4f}/{avg_test_loss:.4f} "
            f"acc {acc * 100:.2f}%"
        )
        early(avg_test_loss)
        if early.early_stop:
            pbar.write("Early stopping.")
            break

    return best_acc, pd.DataFrame(res), pd.DataFrame(order_rows)


# =============================================================================
# Mode runners
# =============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_init_sweep(d, n_classes, trn, tst, args):
    """(B) Initialisation-sensitivity sweep -> best_acc mean +/- std."""
    # (alpha_dist, beta_dist, scale, init_tau) configurations.
    configs = [
        ("normal", "normal", 1.0, 0.1),  # laplex.py default
        ("normal", "normal", 3.0, 0.1),  # wider spread
        ("uniform", "uniform", 1.0, 0.1),
        ("uniform", "uniform", 3.0, 0.1),
        ("linspace", "linspace", 3.0, 0.1),  # deterministic grid init
        ("normal", "normal", 1.0, 0.05),  # sharper kernel
        ("normal", "normal", 1.0, 0.5),  # smoother kernel
    ]
    rows = []
    for ad, bd, sc, tau in configs:
        for seed in args.seeds:
            set_seed(seed)
            proj = ExpKernelLayer(
                d,
                args.m,
                init_tau=tau,
                alpha_dist=ad,
                beta_dist=bd,
                alpha_scale=sc,
                beta_scale=sc,
            )
            model = Net(proj, args.m, n_classes)
            best, _, _ = train_model(model, trn, tst, args)
            rows.append(
                dict(
                    alpha_dist=ad,
                    beta_dist=bd,
                    scale=sc,
                    init_tau=tau,
                    seed=seed,
                    best_acc=best,
                )
            )
            print(f"[init] {ad}/{bd} scale={sc} tau={tau} seed={seed}: {best*100:.2f}%")
    df = pd.DataFrame(rows)
    df.to_csv(args.outputs / "init_sweep_raw.csv", index=False)
    agg = (
        df.groupby(["alpha_dist", "beta_dist", "scale", "init_tau"])["best_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg[["mean", "std"]] *= 100
    agg.to_csv(args.outputs / "init_sweep_summary.csv", index=False)
    print("\n[init_sweep] best_acc (%) mean +/- std:\n", agg.to_string(index=False))


def run_components(d, n_classes, trn, tst, args):
    """(C) Number-of-anchor-components sweep -> best_acc + mean crossing_rate."""
    rows = []
    for I in args.components:
        for seed in args.seeds:
            set_seed(seed)
            proj = MultiExpKernelLayer(d, args.m, n_comp=I, init_tau=args.init_tau)
            model = Net(proj, args.m, n_classes)
            best, _, order = train_model(model, trn, tst, args, track_ordering=True)
            ep = order[order.scope == "epoch"]
            mcr = float(ep.alpha_crossing_rate.mean()) if len(ep) else float("nan")
            n_params = sum(p.numel() for p in proj.parameters())
            rows.append(
                dict(
                    n_comp=I,
                    seed=seed,
                    best_acc=best,
                    mean_alpha_crossing_rate=mcr,
                    proj_params=n_params,
                )
            )
            print(
                f"[comp] I={I} seed={seed}: {best*100:.2f}%  "
                f"crossing={mcr:.4f}  params={n_params}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(args.outputs / "components_raw.csv", index=False)
    agg = (
        df.groupby("n_comp")
        .agg(
            best_acc_mean=("best_acc", "mean"),
            best_acc_std=("best_acc", "std"),
            mean_crossing=("mean_alpha_crossing_rate", "mean"),
            proj_params=("proj_params", "first"),
            count=("best_acc", "count"),
        )
        .reset_index()
    )
    agg[["best_acc_mean", "best_acc_std"]] *= 100
    agg.to_csv(args.outputs / "components_summary.csv", index=False)
    print("\n[components] summary:\n", agg.to_string(index=False))


def build_cached_loaders(root, batch_size, num_workers, on_gpu=False):
    """Precompute flattened features ONCE (deterministic val transform, no
    augmentation) and serve them from RAM (or GPU), so each epoch is a fast
    matmul instead of re-decoding ~12.6k JPEGs. Used to make the init/component
    sweeps tractable; augmentation is intentionally OFF here (documented),
    as in the paper's runs.

    on_gpu=True keeps the feature matrices resident on the GPU as fp16 (cast to
    fp32 in the forward), eliminating the per-batch host->device transfer that
    otherwise dominates; num_workers is then forced to 0.

    Returns (train_loader, test_loader, n_classes, d).
    """
    from torch.utils.data import DataLoader, TensorDataset
    from imagenet10_dataset import ImageNet10, DEFAULT_VAL_TRANSFORM

    def _flatten(split):
        ds = ImageNet10(root=root, split=split, transform=DEFAULT_VAL_TRANSFORM)
        loader = DataLoader(ds, batch_size=128, num_workers=num_workers)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.view(x.size(0), -1))
            ys.append(y)
        return torch.cat(xs), torch.cat(ys), ds.classes

    Xtr, ytr, classes = _flatten("train")
    Xte, yte, _ = _flatten("val")
    if on_gpu:
        dev = torch.device("cuda")
        Xtr, ytr = Xtr.half().to(dev), ytr.to(dev)
        Xte, yte = Xte.half().to(dev), yte.to(dev)
        num_workers = 0
    tr = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    te = DataLoader(
        TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False, num_workers=0
    )
    return tr, te, len(classes), Xtr.shape[1]


# =============================================================================
# Entry point
# =============================================================================
def get_params():
    p = argparse.ArgumentParser("LAPLEX anchor ablations (App. M, Tab. 15)")
    p.add_argument(
        "--mode", required=True, choices=["init_sweep", "components"]
    )
    p.add_argument("--root_dataset", type=str, default="data/imagenet10")
    p.add_argument("--m", type=int, default=500, help="projection / output dim")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--init_tau", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="seeds for the sweeps",
    )
    p.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument(
        "--cache_features",
        action="store_true",
        help="precompute flattened features once (no augmentation); "
        "much faster, recommended for the sweeps",
    )
    p.add_argument(
        "--features_on_gpu",
        action="store_true",
        help="keep cached features resident on GPU as fp16 (removes "
        "the per-batch host->device transfer); implies "
        "--cache_features",
    )
    p.add_argument("--outputs", default="results/anchor_dynamics")
    return p.parse_args()


if __name__ == "__main__":
    args = get_params()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.outputs = Path(args.outputs) / (
        args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    args.outputs.mkdir(parents=True, exist_ok=True)
    (args.outputs / "params.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )
    print(vars(args))

    if args.cache_features or args.features_on_gpu:
        print("Caching flattened features (val transform, no augmentation)...")
        trn, tst, n_classes, d = build_cached_loaders(
            args.root_dataset,
            args.batch_size,
            args.num_workers,
            on_gpu=args.features_on_gpu,
        )
    else:
        trn, tst = get_imagenet10_loaders(
            root=args.root_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        n_classes = len(trn.dataset.classes)
        d = trn.dataset[0][0].numel()
    print(f"d={d}, n_classes={n_classes}, m={args.m}")

    if args.mode == "init_sweep":
        run_init_sweep(d, n_classes, trn, tst, args)
    elif args.mode == "components":
        run_components(d, n_classes, trn, tst, args)
