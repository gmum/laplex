"""App. I.2 (Tabs. 9 and 10): train + evaluate the stationary block-circulant (BCCB)
baselines on ImageNette-320 or DIV2K-1024, raster or permuted coordinate order. Protocol as
in the paper (Adam lr 2e-3, seed 0, frozen Welford mean, fp64 linear algebra; ImageNette
batch 64, 25 epochs; DIV2K batch 4, 20 epochs). run.sh runs every row of the two tables.

Usage:
    python _run_bccb.py --model full --order raster
    python _run_bccb.py --model full --order permuted
    python _run_bccb.py --model diag --order raster      # closed-form floor
    python _run_bccb.py --model mix --n-components 5 --order permuted --lr 5e-4 \\
        --warm-start results/model_imagenette_full_permuted.pt
    python _run_bccb.py --dataset div2k ...              # same, DIV2K-1024

Writes results/results.json (merged by "<model>_<order>") and a sample panel PNG.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402  (torch before numpy/matplotlib, per project setup)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:
    plt = None
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from _model_bccb import GaussianBCCB, GaussianBCCBMixture  # noqa: E402
from _perm_dataset import load_or_build_perm, inverse_perm, PermutedFlatDataset  # noqa: E402
from _diag_floor import diag_gaussian_bits_per_dim  # noqa: E402

from PIL import Image  # noqa: E402


class FlatImageFolder(torch.utils.data.Dataset):
    """Folder of images (no class subdirs) -> (flat 3*H*W tensor, 0).  For DIV2K."""

    def __init__(self, root, transform):
        exts = (".png", ".jpg", ".jpeg")
        self.root, self.transform = root, transform
        self.files = sorted(f for f in os.listdir(root) if f.lower().endswith(exts))
        if not self.files:
            raise RuntimeError(f"No images in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("RGB")
        return self.transform(img), 0


DATASETS = {
    "imagenette": dict(img_size=320, epochs=25, batch=64,
                       roots=[os.environ.get("IMAGENETTE_ROOT", "data/imagenette2-320")]),
    "div2k": dict(img_size=1024, epochs=20, batch=4,
                  roots=[os.environ.get("DIV2K_ROOT", "data/div2k")]),
}


# --- shared helpers (identical convention to paper's _run_mnist) ----------- #
def empirical_mean(loader, n, device):
    acc = torch.zeros(n, dtype=torch.float64, device=device)
    total = 0
    for x, _ in loader:
        x = x.view(x.shape[0], -1).to(device=device, dtype=torch.float64)
        acc += x.sum(dim=0)
        total += x.shape[0]
    return (acc / total).to(torch.float32)


def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            total += model.log_likelihood(x).sum().item()
            count += x.shape[0]
    model.train()
    return total / count


def resolve_root(dataset) -> str:
    for r in DATASETS[dataset]["roots"]:
        if os.path.isdir(r):
            return r
    raise FileNotFoundError(f"{dataset} not found; tried {DATASETS[dataset]['roots']}")


def make_loaders(dataset, root, img_size, n, batch_size, order, perm_seed):
    to_flat = transforms.Lambda(lambda x: x.view(-1))
    if dataset == "imagenette":
        tf = transforms.Compose([transforms.Resize(img_size),
                                 transforms.CenterCrop(img_size),
                                 transforms.ToTensor(), to_flat])
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
        test_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=tf)
    else:  # div2k: flat PNG folders, random-crop train / center-crop val (paper protocol)
        tf_train = transforms.Compose([transforms.Resize(img_size),
                                       transforms.RandomCrop(img_size),
                                       transforms.ToTensor(), to_flat])
        tf_val = transforms.Compose([transforms.Resize(img_size),
                                     transforms.CenterCrop(img_size),
                                     transforms.ToTensor(), to_flat])
        train_ds = FlatImageFolder(os.path.join(root, "DIV2K_train_HR"), tf_train)
        test_ds = FlatImageFolder(os.path.join(root, "DIV2K_valid_HR"), tf_val)

    perm = None
    if order == "permuted":
        perm = load_or_build_perm(n, perm_seed)
        train_ds = PermutedFlatDataset(train_ds, perm)
        test_ds = PermutedFlatDataset(test_ds, perm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader, perm


def save_results(entry: dict, dataset: str):
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    fname = "results.json" if dataset == "imagenette" else f"results_{dataset}.json"
    path = os.path.join(_HERE, "results", fname)
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[f"{entry['model']}_{entry['order']}"] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imagenette", "div2k"], default="imagenette")
    ap.add_argument("--model", choices=["scalar", "full", "diag", "mix"], required=True)
    ap.add_argument("--order", choices=["raster", "permuted"], default="raster")
    ap.add_argument("--n-components", type=int, default=3, help="mixture components (--model mix)")
    ap.add_argument("--mix-type", choices=["scalar", "full"], default="full",
                    help="per-component BCCB type for --model mix")
    ap.add_argument("--warm-start", type=str, default=None,
                    help="single-BCCB checkpoint to warm-start each mixture component from "
                         "(each component = trained single covariance + jitter -> mix starts = single)")
    ap.add_argument("--warm-jitter", type=float, default=0.02,
                    help="relative noise added to warm-started block params (symmetry breaking)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    img_size = cfg["img_size"]
    n = img_size * img_size * 3
    img_shape = (3, img_size, img_size)
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch_size if args.batch_size is not None else cfg["batch"]
    model_name = f"mix{args.n_components}" if args.model == "mix" else args.model

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.dataset}_{model_name}_{args.order}"

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    log_path = os.path.join(_HERE, "results", f"log_{tag}.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    open(log_path, "w").close()
    root = resolve_root(args.dataset)
    log(f"device={device} torch={torch.__version__} dataset={args.dataset} "
        f"model={args.model} order={args.order} n={n} epochs={epochs} batch={batch}")

    train_loader, test_loader, perm = make_loaders(
        args.dataset, root, img_size, n, batch, args.order, args.perm_seed
    )
    log(f"train={len(train_loader.dataset)} test={len(test_loader.dataset)}")

    # ---- diagonal floor: closed form, no training ------------------------- #
    if args.model == "diag":
        t0 = time.time()
        mean_ll, bpd = diag_gaussian_bits_per_dim(
            train_loader, test_loader, n, device=device
        )
        log(f"diag floor: test_LL={mean_ll:.2f}  bits/dim={bpd:.4f}  "
            f"({time.time()-t0:.1f}s)")
        entry = dict(model="diag", order=args.order, n_params=2 * n,
                     final_test_ll=mean_ll, final_train_ll=None,
                     bits_per_dim=bpd, train_time_s=time.time() - t0,
                     peak_mem_mb=None, epochs=0, test_ll_curve=[])
        log(f"saved -> {save_results(entry, args.dataset)}")
        return

    # ---- BCCB models ------------------------------------------------------ #
    mean_init = empirical_mean(train_loader, n, device=device)
    log(f"mean: Welford-frozen  ||m||={mean_init.norm().item():.3f}")

    if args.model == "mix":
        model = GaussianBCCBMixture(img_size, img_size, K=args.n_components,
                                    model_type=args.mix_type, mean_init=mean_init).to(device)
    else:
        model = GaussianBCCB(img_size, img_size, model_type=args.model,
                             mean_init=mean_init).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model={model_name}  n_trainable={n_params}")

    # warm-start each mixture component from a trained single-BCCB covariance, so the
    # mixture starts == single Gaussian and training can only match/improve it (mix >= single).
    if args.model == "mix" and args.warm_start:
        ck = torch.load(args.warm_start, map_location=device)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        torch.manual_seed(args.seed)
        n_copied = 0
        with torch.no_grad():
            for comp in model.components:
                for name, p in comp.named_parameters():
                    if name in sd:
                        p.copy_(sd[name].to(device=p.device, dtype=p.dtype)
                                + args.warm_jitter * torch.randn_like(p))
                        n_copied += 1
        log(f"warm-start from {os.path.basename(args.warm_start)}: "
            f"copied {n_copied} params ({n_copied // max(len(model.components),1)}/component), "
            f"jitter={args.warm_jitter}, lr={args.lr}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    eval_every = 5
    test_curve = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            ll = model.log_likelihood(x)
            (-ll.mean()).backward()
            opt.step()
            total += ll.mean().item()
            nb += 1
        train_ll = total / max(nb, 1)

        if epoch % eval_every == 0 or epoch in (1, epochs):
            test_ll = evaluate(model, test_loader, device)
            test_curve.append((epoch, test_ll))
            el = time.time() - t0
            log(f"epoch {epoch:3d}/{epochs}  train_LL={train_ll:.2f}  "
                f"test_LL={test_ll:.2f}  elapsed={el:.1f}s  "
                f"eta={el*(epochs-epoch)/max(epoch,1):.1f}s")

    train_time = time.time() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else None
    final_test_ll = evaluate(model, test_loader, device)
    bpd = -final_test_ll / (n * math.log(2.0))
    log(f"DONE {tag}: test_LL={final_test_ll:.2f}  bits/dim={bpd:.4f}  "
        f"params={n_params}  time={train_time:.1f}s  peak_mem={peak_mem}")

    entry = dict(model=model_name, order=args.order, n_params=n_params,
                 final_test_ll=final_test_ll, final_train_ll=train_ll,
                 bits_per_dim=bpd, train_time_s=train_time,
                 peak_mem_mb=peak_mem, epochs=epochs,
                 test_ll_curve=test_curve)
    log(f"saved -> {save_results(entry, args.dataset)}")

    torch.save({"state_dict": model.state_dict(), "entry": entry},
               os.path.join(_HERE, "results", f"model_{tag}.pt"))

    # ---- sample panel (un-permute for viewing if permuted); never fail on it -- #
    try:
        model.eval()
        with torch.no_grad():
            samples = model.sample(8).cpu().float()
        if args.order == "permuted":
            samples = samples[:, inverse_perm(perm).cpu()]
        imgs = samples.clamp(0, 1).view(-1, *img_shape).permute(0, 2, 3, 1).numpy()
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))
        for ax, im in zip(axes, imgs):
            ax.imshow(im)
            ax.axis("off")
        fig.suptitle(f"BCCB {model_name} ({args.order}) samples  bits/dim={bpd:.3f}")
        fig.savefig(os.path.join(_HERE, "results", f"samples_{tag}.png"),
                    dpi=110, bbox_inches="tight")
        log("samples saved")
    except Exception as e:
        log(f"sample panel skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
