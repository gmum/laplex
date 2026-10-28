#!/usr/bin/env python
# coding: utf-8
"""App. J: LAPLEX as a drop-in attention operator (table in App. J).

A pre-norm Transformer over pixel tokens in which only the attention sub-layer is
swapped: Softmax, Performer RFF, RFF with a learnable temperature (RFF_tau) and
LAPLEX (Structured, A_ij = exp(-|a_i - b_j| / tau), computed densely). One run per
call; run_appJ_table15.sh runs the sweep of the paper and appJ_attention_table.py builds
the table. torchvision datasets are read from --data_path (MNIST in it, CIFAR-10 in
cifar10/, SVHN in SVHN/); --download fetches them.

    python appJ_attention.py --dataset cifar10 --model Structured --epochs 200 \
        --lr 0.01 --att_dim 128 --att_depth 4 --data_path data/torchvision --download
"""
import json
import math
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# =========================================
# Softmax Attention
# =========================================
class SoftmaxAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)

    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        attn = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(Q.size(-1)), dim=-1)
        return attn @ V


# =========================================
# RFF Attention
# =========================================
class RFFAttention(nn.Module):
    def __init__(self, d, m=64, tau=None):
        super().__init__()
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.m = m
        self.register_buffer("omega", torch.randn(m, d))
        self.tau_param = tau if tau is None else nn.Parameter(torch.tensor(tau))

    def phi(self, x):
        if self.tau_param is not None:
            tau = nn.functional.softplus(self.tau_param) + 1e-6
            x = x / tau
        x_proj = nn.functional.linear(x, self.omega)
        return torch.exp(-torch.square(x_proj) / 2)

    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        Qf, Kf = self.phi(Q), self.phi(K)
        KV = Kf.transpose(-1, -2) @ V
        return Qf @ KV


# =========================================
# Structured Attention
# =========================================
class StructuredAttention(nn.Module):
    def __init__(self, d, tau=0.5):
        super().__init__()
        self.Wq = nn.Linear(d, 1)
        self.Wk = nn.Linear(d, 1)
        self.Wv = nn.Linear(d, d)
        self.tau_param = nn.Parameter(torch.tensor(tau))

    def forward(self, x):
        alpha, beta, V = self.Wq(x).squeeze(-1), self.Wk(x).squeeze(-1), self.Wv(x)
        diff = alpha.unsqueeze(-1) - beta.unsqueeze(-2)
        tau = nn.functional.softplus(self.tau_param) + 1e-6
        A = torch.exp(-diff.abs() / tau)
        return A @ V


# =========================================
# TRANSFORMER BLOCK (Pre-Norm Architecture)
# =========================================
class AttentionBlock(nn.Module):
    def __init__(self, d, attention_class):
        super().__init__()
        self.attn = attention_class(d)
        self.norm1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =========================================
# MODEL
# =========================================
class AttentionModel(nn.Module):
    def __init__(
        self, attention_class, d=64, depth=4, num_classes=10, input_channels=1
    ):
        super().__init__()
        self.embed = nn.Linear(input_channels, d)
        self.layers = nn.ModuleList(
            [AttentionBlock(d, attention_class) for _ in range(depth)]
        )
        self.ln_final = nn.LayerNorm(d)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H*W, C)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)

        x = self.embed(x)

        # Proste kodowanie pozycyjne (0 do 1)
        pos = torch.linspace(0, 1, x.size(1), device=x.device).view(1, -1, 1)
        x = x + pos

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x).mean(dim=1)
        return self.fc(x)


# =========================================
# Training & Evaluation
# =========================================
def train_evaluate(model, train_loader, test_loader, opts):
    model.to(opts.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=opts.scheduler_factor,
        patience=opts.scheduler_patience,
        min_lr=1e-5,
    )

    # --- Early Stopping Setup ---
    best_loss = float("inf")
    early_stop_counter = 0
    patience_early_stopping = getattr(opts, "early_stop_patience", 10)

    history = []
    for epoch in (pbar := tqdm(range(opts.epochs), desc="Epoch")):
        # --- TRENING ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for x, y in tqdm(train_loader, desc="Train", leave=False):
            x, y = x.to(opts.device), y.to(opts.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = nn.functional.cross_entropy(out, y, reduction="sum")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_total += y.size(0)
            train_correct += (out.argmax(1) == y).sum().item()

        loss_trn = train_loss / train_total
        acc_trn = train_correct / train_total * 100

        # --- EVALUATION ---
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(opts.device), y.to(opts.device)
                out = model(x)
                loss = nn.functional.cross_entropy(out, y, reduction="sum")

                test_loss += loss.item()
                test_total += y.size(0)
                test_correct += (out.argmax(1) == y).sum().item()

        loss_tst = test_loss / test_total
        acc_tst = test_correct / test_total * 100

        scheduler.step(loss_tst)
        current_lr = optimizer.param_groups[0]["lr"]

        pbar.write(
            f"Epoch {epoch+1:03d} | "
            f"Loss [trn|tst]: {loss_trn:.4g} | {loss_tst:.4g} | "
            f"Acc [trn|tst]: {acc_trn:.2f}% | {acc_tst:.2f}% | LR: {current_lr:.4g}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "loss_train": loss_trn,
                "loss_test": loss_tst,
                "acc_train": acc_trn,
                "acc_test": acc_tst,
                "lr": current_lr,
            }
        )

        # --- EARLY STOPPING LOGIC ---
        if loss_tst < best_loss:
            best_loss = loss_tst
            early_stop_counter = 0

            # torch.save(model.state_dict(), opts.outputs / "model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience_early_stopping:
                pbar.write(f" Early stopping triggered at epoch {epoch + 1}")
                break

    return pd.DataFrame(history)


def json_serial(obj):
    if isinstance(obj, (Path,)):
        return str(obj)
    elif isinstance(obj, (torch.device,)):
        return obj.type
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Mechanism Benchmark")
    parser.add_argument(
        "--dataset", default="mnist", choices=["mnist", "cifar10", "svhn"]
    )
    parser.add_argument(
        "--model", required=True, choices=["Softmax", "RFF", "RFF_tau", "Structured"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--outputs", type=str, default="results/attention")
    parser.add_argument("--init_tau", type=float, default=0.54)
    parser.add_argument("--att_dim", type=int, default=64)
    parser.add_argument("--att_depth", type=int, default=4)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--download", action="store_true",
                        help="download the dataset into --data_path if missing")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.outputs = (
        Path(args.outputs) / args.dataset / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    )
    args.outputs.mkdir(parents=True, exist_ok=True)
    print(vars(args))
    with open(args.outputs / "params.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, default=json_serial)

    args.data_path = Path(args.data_path)
    num_classes = 10
    download = args.download
    if args.dataset == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_ds = datasets.MNIST(
            args.data_path, train=True, download=download, transform=transform
        )
        test_ds = datasets.MNIST(
            args.data_path, train=False, download=download, transform=transform
        )
        in_channels = 1
    elif args.dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_ds = datasets.CIFAR10(
            args.data_path / "cifar10",
            train=True,
            download=download,
            transform=transform,
        )
        test_ds = datasets.CIFAR10(
            args.data_path / "cifar10",
            train=False,
            download=download,
            transform=transform,
        )
        in_channels = 3
    elif args.dataset == "svhn":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_ds = datasets.SVHN(
            args.data_path / "SVHN",
            split="train",
            download=download,
            transform=transform,
        )
        test_ds = datasets.SVHN(
            args.data_path / "SVHN",
            split="test",
            download=download,
            transform=transform,
        )
        in_channels = 3

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    if args.model == "Softmax":
        attn_fun = SoftmaxAttention
    elif args.model == "RFF":
        attn_fun = partial(RFFAttention, m=args.att_dim)
    elif args.model == "RFF_tau":
        attn_fun = partial(RFFAttention, m=args.att_dim, tau=args.init_tau)
    else:
        attn_fun = partial(StructuredAttention, tau=args.init_tau)

    print(f"\n--- Running Experiment: {args.model} on {args.dataset.upper()} ---")
    model = AttentionModel(
        attn_fun,
        d=args.att_dim,
        depth=args.att_depth,
        num_classes=num_classes,
        input_channels=in_channels,
    )
    df = train_evaluate(model, train_loader, test_loader, args)
    df.to_csv(args.outputs / "results.csv", index=False)
