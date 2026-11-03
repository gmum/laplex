"""Demo: LAPLEX density model on DIV2K-1024 (pretrained checkpoint).

Loads model_div2k_1024_complex_1tau_k1000.pt, evaluates test log-likelihood
on the DIV2K validation set (100 images), generates 8 samples, and saves:
  - demo_samples.png  : 8 generated images
  - demo_results.txt  : test LL value and run info

Usage:
    python demo.py --div2k-root /path/to/div2k
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

from _model_complex import GaussianKerLapComplexWoodbury
from _div2k_dataset import make_div2k_datasets, _DIV2K_ROOT

IMG_SIZE  = 1024
N_IMG     = IMG_SIZE * IMG_SIZE * 3      # 3,145,728
IMG_SHAPE = (3, IMG_SIZE, IMG_SIZE)

CKPT_NAME = "model_div2k_1024_complex_1tau_k1000.pt"


def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            ll = model.log_likelihood(x)
            total += ll.sum().item()
            count += x.shape[0]
    return total / count


def main():
    parser = argparse.ArgumentParser(description="LAPLEX DIV2K demo")
    parser.add_argument(
        "--div2k-root", type=str,
        default=_DIV2K_ROOT,
        help=f"Root of the DIV2K dataset directory. Default: {_DIV2K_ROOT}",
    )
    parser.add_argument(
        "--ckpt", type=str,
        default=os.path.join(_HERE, CKPT_NAME),
        help=f"Path to checkpoint .pt file. Default: ./{CKPT_NAME}",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=4,
        help="Batch size for evaluation (default: 4, adjust for available VRAM).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}", flush=True)
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU={torch.cuda.get_device_name()}  sm_{cap[0]}{cap[1]}", flush=True)

    # --- Transforms (identical to training) ---
    tf_val = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    _, val_ds, _, val_root = make_div2k_datasets(
        train_transform=tf_val,  # unused but required by API
        val_transform=tf_val,
        root=args.div2k_root,
    )
    print(f"val set: {len(val_ds)} images  ({val_root})", flush=True)

    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    model = GaussianKerLapComplexWoodbury(N_IMG, k=1000, single_tau=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded  n_trainable={n_params}", flush=True)

    # --- Evaluate test LL ---
    print("Evaluating test log-likelihood on val set (100 images)...", flush=True)
    t0 = time.time()
    test_ll = evaluate(model, val_loader, device)
    elapsed = time.time() - t0
    print(f"test_LL = {test_ll:.2f}  (elapsed {elapsed:.1f}s)", flush=True)

    # --- Generate 8 samples ---
    print("Generating 8 samples...", flush=True)
    with torch.no_grad():
        samples = model.sample(8).cpu()

    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        img = samples[i].view(*IMG_SHAPE).clamp(0, 1)
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].axis("off")
    fig.suptitle(
        f"LAPLEX DIV2K-1024 (complex, k=1000, single-tau)  test_LL={test_ll:.1f}",
        fontsize=9,
    )
    plt.tight_layout()
    samples_path = os.path.join(_HERE, "demo_samples.png")
    plt.savefig(samples_path, dpi=120, bbox_inches="tight")
    print(f"Saved samples -> {samples_path}", flush=True)

    # --- Save results ---
    results_path = os.path.join(_HERE, "demo_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint:  {args.ckpt}\n")
        f.write(f"div2k_root:  {args.div2k_root}\n")
        f.write(f"val_images:  {len(val_ds)}\n")
        f.write(f"test_LL:     {test_ll:.4f}\n")
        f.write(f"eval_time:   {elapsed:.1f}s\n")
        f.write(f"device:      {device}\n")
        if device == "cuda":
            f.write(f"gpu:         {torch.cuda.get_device_name()}\n")
    print(f"Saved results -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
