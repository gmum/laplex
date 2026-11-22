"""Trening LAPLEX / dense LR Gaussian density modelu na DIV2K-1024.

DIV2K: 800 train / 100 val, random crop 1024x1024 -> n = 3,145,728
(identyczne n jak MetFaces-1024, dzieki czemu modele sa porownywalne).

Komendy:
    python _run_div2k.py --complex --k 1000 --batch-size 8 --epochs 20
    python _run_div2k.py --dense --k-dense 4 --batch-size 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from _run_mnist import GaussianKerLapWoodbury, evaluate, empirical_mean
from _model_complex import GaussianKerLapComplexWoodbury
from _model_dense import GaussianDenseLowRankWoodbury
from _div2k_dataset import make_div2k_datasets, _DIV2K_ROOT

IMG_SIZE  = 1024
N_IMG     = IMG_SIZE * IMG_SIZE * 3      # 3,145,728
IMG_SHAPE = (3, IMG_SIZE, IMG_SIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-tau", action="store_true")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--complex", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--k-dense", type=int, default=4)
    parser.add_argument("--learn-mean", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--div2k-root", type=str, default=None,
                        help=f"Root katalogu DIV2K. Default: {_DIV2K_ROOT}")
    args = parser.parse_args()

    if args.complex and args.dense:
        parser.error("--complex i --dense wzajemnie wykluczajace.")

    single_tau  = not args.two_tau
    K_LATENT    = args.k
    use_complex = args.complex

    if args.dense:
        suffix = f"1024_dense_kd{args.k_dense}_b{args.batch_size}"
    else:
        kernel  = "complex" if use_complex else "real"
        tau_str = "1tau" if single_tau else "2tau"
        suffix  = f"1024_{kernel}_{tau_str}_k{K_LATENT}_b{args.batch_size}"
    tag = "div2k"

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = os.path.join(_HERE, "results")
    os.makedirs(results, exist_ok=True)
    progress_path = os.path.join(results, f"progress_{suffix}.log")
    samples_path  = os.path.join(results, f"samples_{tag}_{suffix}.png")
    ckpt_path     = os.path.join(results, f"model_{tag}_{suffix}.pt")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    open(progress_path, "w").close()
    log(f"device={device}  torch={torch.__version__}  dataset={tag}-{IMG_SIZE}  mode={suffix}")
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        log(f"GPU={torch.cuda.get_device_name()}  cap=sm_{cap[0]}{cap[1]}")

    # resize krotszy bok do 1024, potem crop (niektore obrazy maja high < 1024)
    tf_train = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    train_ds, val_ds, train_root, val_root = make_div2k_datasets(
        train_transform=tf_train,
        val_transform=tf_val,
        root=args.div2k_root,
    )
    log(f"div2k train: {train_root}")
    log(f"div2k val:   {val_root}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(val_ds,   batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))

    log(f"train={len(train_ds)}  val={len(val_ds)}  n={N_IMG}  k={K_LATENT}")

    if args.learn_mean:
        mean_init = None
        log("mean: learnable (init=0)")
    else:
        mean_init = empirical_mean(train_loader, N_IMG, device=device)
        log(f"mean: Welford-frozen  ||m||={mean_init.norm().item():.3f}")

    if args.dense:
        model = GaussianDenseLowRankWoodbury(N_IMG, args.k_dense,
                                             mean_init=mean_init).to(device)
    elif use_complex:
        model = GaussianKerLapComplexWoodbury(N_IMG, K_LATENT,
                                              single_tau=single_tau,
                                              mean_init=mean_init).to(device)
    else:
        model = GaussianKerLapWoodbury(N_IMG, K_LATENT,
                                       single_tau=single_tau,
                                       mean_init=mean_init).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model={suffix}  n_trainable={n_params}  "
        f"(m {'learnable' if args.learn_mean else 'frozen'})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

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
         "single_tau": single_tau, "n": N_IMG, "k": K_LATENT, "dataset": f"{tag}-{IMG_SIZE}"},
        ckpt_path,
    )
    log(f"saved checkpoint -> {ckpt_path}")

    model.eval()
    with torch.no_grad():
        samples = model.sample(10).cpu()

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        img = samples[i].view(*IMG_SHAPE).clamp(0, 1)
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].axis("off")
    if args.dense:
        title = f"DIV2K-{IMG_SIZE}  dense k_d={args.k_dense}  test_LL={test_lls[-1][1]:.1f}"
    elif single_tau:
        kernel_label = "complex" if use_complex else "real"
        title = (f"DIV2K-{IMG_SIZE}  [{kernel_label}] 1-tau  k={K_LATENT}  "
                 f"test_LL={test_lls[-1][1]:.1f}  tau={model.tau_L().item():.3f}")
    else:
        title = f"DIV2K-{IMG_SIZE}  [real] 2-tau  k={K_LATENT}"
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(samples_path, dpi=120, bbox_inches="tight")
    log(f"saved samples -> {samples_path}")


if __name__ == "__main__":
    main()
