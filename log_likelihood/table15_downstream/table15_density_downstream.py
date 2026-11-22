"""
table15_density_downstream.py  (App. L: Tab. 15)
Structured covariance baselines on raster-order ImageNette-320 and MAP denoising.
Models in density_models.py.

    python table15_density_downstream.py --max_per_class 800 --rank 3 \\
        --imagenet_root ../../nn_training/data/imagenet10

Data: ImageNette-320 built from the local full ImageNet (the 10 synsets of the attention
and projection experiments), n = 307,200.

Models (Gaussian x ~ N(m, Sigma)): LAPLEX and dense low-rank factor analysis (Woodbury),
PPCA, Kronecker/tensor-normal, closed-form stationary FFT-circulant. Criteria: test
log-likelihood (total nats, nats/dim, bits/dim) and MAP-denoising PSNR/SSIM; with
--ood_root also an OOD-detection AUROC (ID = ImageNette val, OOD = e.g. CelebA; not in the
paper).
"""

import argparse
import glob
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from density_models import FAModel, PPCAModel, KroneckerModel, CirculantModel

IMAGENET10_SYNSETS = ["n01440764", "n01774384", "n01882714", "n02102040",
                      "n02979186", "n03000684", "n03028079", "n03788195",
                      "n03877845", "n03944341"]

ALL_MODELS = ["LAPLEX", "dense-LR", "PPCA", "Kronecker", "FFT-circulant"]

try:
    from sklearn.metrics import roc_auc_score
    def auroc(id_scores, ood_scores):
        y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
        s = np.r_[id_scores, ood_scores]
        return float(roc_auc_score(y, s))
except ImportError:
    def auroc(id_scores, ood_scores):  # rank-based fallback
        s = np.r_[id_scores, ood_scores]
        r = s.argsort().argsort()
        n1, n0 = len(id_scores), len(ood_scores)
        return float((r[:n1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

try:
    from skimage.metrics import structural_similarity as _ssim
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def _load_image(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (C,H,W) in [0,1]


def load_imagenette(root, split, size, max_per_class=None):
    xs = []
    sub = "train" if split == "train" else "val"
    for syn in IMAGENET10_SYNSETS:
        files = sorted(glob.glob(os.path.join(root, sub, syn, "*")))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            try:
                xs.append(_load_image(f, size).reshape(-1))
            except Exception:
                pass
    return torch.from_numpy(np.stack(xs))  # (N, C*H*W) fp32 in [0,1]


def load_ood(root, size, max_n):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPEG"):
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    files = sorted(files)[:max_n]
    xs = [_load_image(f, size).reshape(-1) for f in files]
    return torch.from_numpy(np.stack(xs))


def chunked_mean(X, chunk=256):
    m = torch.zeros(X.shape[1], dtype=torch.float64)
    for s in range(0, X.shape[0], chunk):
        m += X[s:s + chunk].double().sum(0)
    return m / X.shape[0]


# -----------------------------------------------------------------------------
# Training by maximizing the exact log-likelihood (Adam)
# -----------------------------------------------------------------------------
def train_model(model, Xtr, n, args, tag):
    dev = args.device
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    N = Xtr.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for s in range(0, N, args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb = Xtr[idx].to(dev).double()
            opt.zero_grad()
            loss = -model.loglik(xb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"  [{tag}] epoch {ep+1}/{args.epochs}  mean -LL/dim "
              f"{tot/nb/n:.4f}", flush=True)
    return model


def train_fa(kind, Xtr, n, k, mean, args):
    model = FAModel(n, k, kind, mean, dtype=torch.float64).to(args.device)
    return train_model(model, Xtr, n, args, kind)


def laplex_param_count(n, k):
    """FAModel(kind='laplex'): mean, log_d, a, w (n each) + b (k) + L (k x k)."""
    return 4 * n + k + k * k


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def eval_ll(model, X, dev, bs=64):
    out = []
    for s in range(0, X.shape[0], bs):
        out.append(model.loglik(X[s:s+bs].to(dev).double()).cpu())
    return torch.cat(out)


@torch.no_grad()
def eval_map_psnr(model, X, dev, noise_std, size, bs=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    psnrs, ssims = [], []
    for s in range(0, X.shape[0], bs):
        xb = X[s:s+bs].double()
        noise = torch.randn(xb.shape, generator=g).double() * noise_std
        yb = (xb + noise).to(dev)
        xhat = model.map_denoise(yb, noise_std ** 2).cpu().clamp(0, 1)
        mse = ((xhat - xb) ** 2).mean(1)
        psnrs.append(10.0 * torch.log10(1.0 / mse.clamp(min=1e-12)))
        if HAVE_SSIM:
            for i in range(xb.shape[0]):
                a = xhat[i].reshape(3, size, size).permute(1, 2, 0).numpy()
                b = xb[i].reshape(3, size, size).permute(1, 2, 0).numpy()
                ssims.append(_ssim(b, a, channel_axis=2, data_range=1.0))
    psnr = float(torch.cat(psnrs).mean())
    ssim = float(np.mean(ssims)) if ssims else float("nan")
    return psnr, ssim


def n_params(m):
    if isinstance(m, PPCAModel): return int(m.V.numel() + m.mean.numel() + 1)
    if isinstance(m, KroneckerModel): return int(sum(s.numel() for s in m.S) + m.mean.numel())
    if isinstance(m, CirculantModel): return int(m.psd.numel() + m.mean.numel())
    return sum(p.numel() for p in m.parameters())


# -----------------------------------------------------------------------------
# One coordinate order: fit every model, evaluate it, free it
# -----------------------------------------------------------------------------
def run_order(order, data, cfg, args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    Xtr, Xva, Xood = data
    n, dims, size, k, r = cfg["n"], cfg["dims"], cfg["size"], cfg["k"], cfg["r"]
    downstream = True
    mean = chunked_mean(Xtr)

    sub = Xtr[torch.randperm(Xtr.shape[0])[:args.subsample]].double()  # CPU fp64

    def build(name):
        # Closed-form baselines are fitted on a CPU subsample (the flip-flop
        # intermediates do not fit on a 10 GB GPU); only small tensors move.
        if name == "PPCA":          return PPCAModel(sub, r).to(dev)
        if name == "Kronecker":     return KroneckerModel(sub, dims, iters=args.kron_iters).to(dev)
        if name == "FFT-circulant": return CirculantModel(sub, dims).to(dev)
        if name == "dense-LR":      return train_fa("dense", Xtr, n, r, mean, args)
        if name == "LAPLEX":        return train_fa("laplex", Xtr, n, k, mean, args)
        raise ValueError(name)

    rows = []
    for name in [m for m in ALL_MODELS if m in args.models]:
        label = name
        print(f"[{order}] fitting {label} ...", flush=True)
        t0 = time.time()
        m = build(name)
        t_fit = time.time() - t0
        ll_id = eval_ll(m, Xva, dev, bs=args.eval_bs)
        row = dict(order=order, model=label, params=n_params(m),
                   test_LL_total=float(ll_id.mean()),
                   test_LL_per_dim=float(ll_id.mean()) / n,
                   bits_per_dim=-float(ll_id.mean()) / n / math.log(2.0),
                   fit_seconds=t_fit)
        if downstream and hasattr(m, "map_denoise"):
            row["map_psnr"], row["map_ssim"] = eval_map_psnr(
                m, Xva, dev, args.noise_std, size)
        if downstream and Xood is not None:
            ll_ood = eval_ll(m, Xood, dev, bs=args.eval_bs)
            row["ood_auroc"] = auroc(ll_id.numpy(), ll_ood.numpy())
        rows.append(row)
        print(f"[{order}] [{label}] params {row['params']:,} | test LL "
              f"{row['test_LL_total']:+,.0f} nats | {row['test_LL_per_dim']:.4f} nats/dim"
              f" | {row['bits_per_dim']:+.3f} bits/dim"
              + (f" | PSNR {row['map_psnr']:.2f} | SSIM {row['map_ssim']:.3f}"
                 if "map_psnr" in row else "")
              + (f" | OOD-AUROC {row['ood_auroc']:.3f}" if "ood_auroc" in row else ""),
              flush=True)
        del m; torch.cuda.empty_cache()
    del sub
    return rows


def markdown_table(rows, title):
    def f(x, fmt):
        return format(x, fmt) if x is not None else "—"
    md = [title, "",
          "| model | params | test LL (nats) | LL/dim | bits/dim | MAP PSNR (dB) | MAP SSIM | OOD AUROC |",
          "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['model']} | {x['params']:,} | {x['test_LL_total']:+,.0f} | "
                  f"{x['test_LL_per_dim']:.4f} | {x['bits_per_dim']:+.3f} | "
                  f"{f(x.get('map_psnr'), '.2f')} | {f(x.get('map_ssim'), '.3f')} | "
                  f"{f(x.get('ood_auroc'), '.3f')} |")
    return md


# -----------------------------------------------------------------------------
def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.outputs) / f"imagenette_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    size = args.image_size
    Xtr = load_imagenette(args.imagenet_root, "train", size, max_per_class=args.max_per_class)
    Xva = load_imagenette(args.imagenet_root, "val", size)
    Xood = load_ood(args.ood_root, size, max_n=Xva.shape[0]) if args.ood_root else None
    n, dims = 3 * size * size, (3, size, size)
    print(f"outputs -> {out}\nimagenette: n={n}; train {tuple(Xtr.shape)}, "
          f"val {tuple(Xva.shape)}, ood {None if Xood is None else tuple(Xood.shape)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # budget-match every baseline to the actual LAPLEX parameter count
    k = args.k_laplex
    P_laplex = laplex_param_count(n, k)
    r = args.rank or max(1, round((P_laplex - 2 * n) / n))  # dense-LR: (2 + r) n
    cfg = dict(size=size, n=n, dims=dims, k=k, r=r, P_laplex=P_laplex,
               epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
               subsample=args.subsample, noise_std=args.noise_std, seed=args.seed,
               n_train=int(Xtr.shape[0]), n_val=int(Xva.shape[0]),
               n_ood=None if Xood is None else int(Xood.shape[0]))
    print(f"LAPLEX params {P_laplex:,}; dense/PPCA rank r={r}", flush=True)

    rows = run_order("raster", (Xtr, Xva, Xood), cfg, args)
    md = markdown_table(rows, f"## imagenette (n={n:,}), raster coordinates") + [""]
    json.dump(dict(config={**cfg, "dims": list(dims)}, rows=rows),
              open(out / "results.json", "w"), indent=2)

    header = ["# Density models on raster-order ImageNette-320",
              f"k_laplex={k} (LAPLEX params {P_laplex:,}), dense/PPCA rank r={r}, "
              f"{cfg['n_train']} train / {cfg['n_val']} test, "
              f"{args.epochs} epochs, lr {args.lr}, batch {args.batch_size}; "
              f"bits/dim = -LL/(n ln 2) for x in [0,1]^n", ""]
    (out / "tables.md").write_text("\n".join(header + md) + "\n")
    print("\n".join(header + md))
    print("\n[saved]", out / "tables.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--k_laplex", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--noise_std", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_per_class", type=int, default=None,
                   help="cap train images per class (None = all)")
    p.add_argument("--subsample", type=int, default=4000,
                   help="#train images for the closed-form PPCA/Kronecker fits")
    p.add_argument("--kron_iters", type=int, default=6,
                   help="flip-flop iterations for the Kronecker MLE")
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--rank", type=int, default=None,
                   help="dense-LR and PPCA rank (None = match the LAPLEX parameter count)")
    p.add_argument("--eval_bs", type=int, default=64)
    p.add_argument("--imagenet_root", default="../../nn_training/data/imagenet10")
    p.add_argument("--ood_root", default=None,
                   help="folder of OOD images for the AUROC (optional, not in the paper)")
    p.add_argument("--outputs", default="results/density")
    args = p.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
