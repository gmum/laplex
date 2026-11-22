"""ImageNette-320 experiment for GaussianKerLapComplexWoodbury.

The same as `_run_imagenette.py` (160x160), only with n=320^2*3=307200.

Usage:
    python _run_imagenette320.py --complex --k 1000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAPLEX_DIR = _HERE
if _LAPLEX_DIR not in sys.path:
    sys.path.insert(0, _LAPLEX_DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from _run_mnist import (
    GaussianKerLapWoodbury,
    evaluate,
    empirical_mean,
)
from _model_complex import GaussianKerLapComplexWoodbury
from _model_dense import GaussianDenseLowRankWoodbury


IMG_SIZE = 320
N_IMG = IMG_SIZE * IMG_SIZE * 3                     # 307,200
IMG_SHAPE = (3, IMG_SIZE, IMG_SIZE)

_IMAGENETTE_ROOT = os.environ.get("IMAGENETTE_ROOT", os.path.join(_HERE, "data", "imagenette2-320"))
_RESULTS = os.path.join(_HERE, "results")


def auto_band_width(n, k, target_multiple=10):
    target = target_multiple * n
    fixed = 4 * n + 3 * k + 1
    remaining = target - fixed
    if remaining <= 0:
        return 1
    disc = (2 * k - 1) ** 2 - 8 * remaining
    if disc < 0:
        return None
    b = int(round(((2 * k - 1) - disc ** 0.5) / 2))
    b = max(1, b)
    if b >= k // 2:
        return None
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-tau", action="store_true")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--complex", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--k-dense", type=int, default=44)
    parser.add_argument("--learn-mean", action="store_true")
    parser.add_argument("--band-width", type=int, default=None)
    parser.add_argument("--param-multiple", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Smaller batch for n=307200 because of memory.")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    if args.complex and args.dense:
        parser.error("--complex and --dense are mutually exclusive.")

    if args.param_multiple is not None:
        args.band_width = auto_band_width(N_IMG, args.k, args.param_multiple)

    single_tau = not args.two_tau
    K_LATENT = args.k
    use_complex = args.complex
    if args.dense:
        suffix = f"320_dense_kd{args.k_dense}"
    else:
        kernel = "complex" if use_complex else "real"
        tau_str = "1tau" if single_tau else "2tau"
        band_str = f"_band{args.band_width}" if args.band_width is not None else ""
        suffix = f"320_{kernel}_{tau_str}_k{K_LATENT}{band_str}"
    tag = "imagenette"

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(_RESULTS, exist_ok=True)
    progress_path = os.path.join(_RESULTS, f"progress_{suffix}.log")
    samples_path = os.path.join(_RESULTS, f"samples_{tag}_{suffix}.png")
    ckpt_path = os.path.join(_RESULTS, f"model_{tag}_{suffix}.pt")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    open(progress_path, "w").close()
    log(f"device={device}  torch={torch.__version__}  dataset={tag}-320  mode={suffix}")
    if args.param_multiple is not None:
        b_info = args.band_width if args.band_width is not None else "None (full-tril)"
        log(f"auto-band: target {args.param_multiple}·n={args.param_multiple * N_IMG}  → band_width={b_info}")

    # ImageNette transform: short-edge resize 320 + center-crop 320 → flat
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    train_ds = datasets.ImageFolder(os.path.join(_IMAGENETTE_ROOT, "train"), transform=tf)
    test_ds  = datasets.ImageFolder(os.path.join(_IMAGENETTE_ROOT, "val"),   transform=tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=0, pin_memory=(device == "cuda"))

    log(f"train={len(train_ds)}  test={len(test_ds)}  n={N_IMG}  k={K_LATENT}")

    if args.learn_mean:
        mean_init = None
        log("mean: learnable (init=0)")
    else:
        mean_init = empirical_mean(train_loader, N_IMG, device=device)
        log(f"mean: Welford-frozen  ||m||={mean_init.norm().item():.3f}")

    if args.dense:
        model = GaussianDenseLowRankWoodbury(N_IMG, args.k_dense, mean_init=mean_init).to(device)
    elif use_complex:
        model = GaussianKerLapComplexWoodbury(N_IMG, K_LATENT,
                                              single_tau=single_tau,
                                              mean_init=mean_init,
                                              band_width=args.band_width).to(device)
    else:
        model = GaussianKerLapWoodbury(N_IMG, K_LATENT,
                                       single_tau=single_tau,
                                       mean_init=mean_init).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model={suffix}  n_trainable={n_params}  "
        f"(m {'learnable' if args.learn_mean else 'frozen'})")

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    num_epochs = args.epochs
    eval_every = 5
    train_lls, test_lls = [], []

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            ll = model.log_likelihood(x)
            loss = -ll.mean()
            loss.backward()
            opt.step()
            total += ll.mean().item()
            n_batches += 1
        train_ll = total / max(n_batches, 1)
        train_lls.append((epoch, train_ll))

        do_eval = (epoch % eval_every == 0) or (epoch == num_epochs) or (epoch == 1)
        if do_eval:
            test_ll = evaluate(model, test_loader, device)
            test_lls.append((epoch, test_ll))
            elapsed = time.time() - t0
            eta = elapsed * (num_epochs - epoch) / max(epoch, 1)
            extra = ""
            if not single_tau and not args.dense:
                extra = f"  tau_L={model.tau_L().item():.3f}  tau_R={model.tau_R().item():.3f}"
            log(f"epoch {epoch:3d}/{num_epochs}  "
                f"train_LL={train_ll:.2f}  test_LL={test_ll:.2f}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s{extra}")
        else:
            elapsed = time.time() - t0
            eta = elapsed * (num_epochs - epoch) / max(epoch, 1)
            log(f"epoch {epoch:3d}/{num_epochs}  "
                f"train_LL={train_ll:.2f}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    total_time = time.time() - t0
    log(f"done in {total_time:.1f}s")
    log(f"final train_LL = {train_lls[-1][1]:.2f}")
    log(f"final test_LL  = {test_lls[-1][1]:.2f}")
    log("eval points: " + ", ".join(f"({e},{v:.2f})" for e, v in test_lls))

    if args.dense:
        pass
    elif single_tau:
        log(f"tau = {model.tau_L().item():.4f}")
    else:
        log(f"tau_L = {model.tau_L().item():.4f}   tau_R = {model.tau_R().item():.4f}")

    torch.save(
        {"state_dict": model.state_dict(),
         "train_lls": train_lls, "test_lls": test_lls,
         "single_tau": single_tau, "n": N_IMG, "k": K_LATENT, "dataset": f"{tag}-320"},
        ckpt_path,
    )
    log(f"saved checkpoint -> {ckpt_path}")

    # Samples
    model.eval()
    with torch.no_grad():
        samples = model.sample(10).cpu()

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        img = samples[i].view(*IMG_SHAPE).clamp(0, 1)
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].axis("off")
    kernel_label = "complex" if use_complex else ("real" if not args.dense else "dense")
    if args.dense:
        title = f"ImageNette-320  dense k_d={args.k_dense}  test_LL={test_lls[-1][1]:.1f}"
    elif single_tau:
        title = (f"ImageNette-320  [{kernel_label}] 1-tau  k={K_LATENT}  "
                 f"test_LL={test_lls[-1][1]:.1f}  tau={model.tau_L().item():.3f}")
    else:
        title = (f"ImageNette-320  [{kernel_label}] 2-tau  k={K_LATENT}")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(samples_path, dpi=120, bbox_inches="tight")
    log(f"saved samples -> {samples_path}")


if __name__ == "__main__":
    main()
