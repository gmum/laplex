import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from imagenet10_dataset import get_imagenet10_loaders
from utils import TeedLog, EarlyStopping





class ExpKernelLayer(nn.Module):
    def __init__(self, d, m, init_tau=0.1):
        super().__init__()
        self.d = d
        self.m = m

        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(m))
        self.w = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))

    def forward(self, x):
        
        tau = F.softplus(self.log_tau) + 1e-6

        
        diff = torch.abs(self.alpha.unsqueeze(1) - self.beta.unsqueeze(0))
        K = torch.exp(-diff / tau)

        xw = x * self.w
        z = xw @ K
        return z


class Net(nn.Module):
    def __init__(self, d, m, num_classes=10):
        super().__init__()
        self.proj = ExpKernelLayer(d, m)
        self.classifier = nn.Linear(m, num_classes)

    def forward(self, x):
        z = self.proj(x)
        return self.classifier(z)





def train_model(d, m, num_classes, trn_loader, tst_loader, opts):
    model = Net(d=d, m=m, num_classes=num_classes).to(opts.device)

    
    
    
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.8, patience=7, min_lr=1e-5
    )

    early_stopping = EarlyStopping(patience=opts.patience, verbose=False)

    best_acc = 0
    res = []

    for epoch in (pbar := tqdm(range(opts.epochs), desc=f"{m=} Epoch")):
        model.train()
        train_loss, total_samples = 0.0, 0

        for x, y in trn_loader:
            x = x.view(x.size(0), -1).to(opts.device, non_blocking=True)
            y = y.to(opts.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        
        avg_train_loss = train_loss / total_samples

        
        model.eval()
        test_loss, correct, total_samples_test = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tst_loader:
                x = x.view(x.size(0), -1).to(opts.device, non_blocking=True)
                y = y.to(opts.device, non_blocking=True)

                out = model(x)
                loss = criterion(out, y)
                test_loss += loss.item() * x.size(0)

                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_samples_test += x.size(0)

        acc = correct / total_samples_test
        avg_test_loss = test_loss / total_samples_test

        scheduler.step(avg_test_loss)

        res.append(
            {
                "m": m,
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "acc": acc,
                "tau": F.softplus(model.proj.log_tau).item() + 1e-6,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if acc > best_acc:
            best_acc = acc

        if opts.save_model > 0 and m < 100 and (epoch + 1) % opts.save_model == 0:
            torch.save(model.state_dict(), opts.outputs / f"m{m}_ep{epoch}.pth")

        
        early_stopping(avg_test_loss)
        pbar.write(
            f"Epoch {epoch + 1}: Loss(trn/tst): {avg_train_loss:.4f}/{avg_test_loss:.4f}, lr: {res[-1]['lr']:.4g} | Acc: {acc * 100:.2f}%"
            f" [NoImp: {early_stopping.counter}/{early_stopping.patience}]"
        )
        if early_stopping.early_stop:
            pbar.write("Early stopping triggered. Training halted.")
            break

    return best_acc, pd.DataFrame(res)


def get_params():
    parser = argparse.ArgumentParser("Kernel Projection Ours")
    parser.add_argument("--root_dataset", type=str, default="./imagenet10")
    parser.add_argument("--dims", type=int, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--outputs", default="./outputs")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs to wait for improvement before stopping",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_params()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.outputs = Path(args.outputs) / datetime.now().strftime("%Y%m%d_%H%M%S")
    args.outputs.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeedLog(args.outputs)
    print(vars(args))

    
    if args.seed is None:
        args.seed = int(np.random.randint(1000, 10000))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_imagenet10_loaders(
        root=args.root_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    n_classes = len(train_loader.dataset.classes)
    sample_data, _ = train_loader.dataset[0]
    d_input = sample_data.numel()

    final_results = {}
    all_dfs = []

    for m_dim in args.dims:
        print(f"\n{'=' * 40}\nRunning experiment: m = {m_dim}\n{'=' * 40}")
        best_accuracy, history_df = train_model(
            d_input, m_dim, n_classes, train_loader, test_loader, args
        )
        final_results[m_dim] = best_accuracy
        all_dfs.append(history_df)

    full_df = pd.concat(all_dfs)
    full_df.to_csv(args.outputs / "results.csv", index=False)

    print("\nFinal Performance Summary:")
    print("-" * 25)
    for m, val in final_results.items():
        print(f"m={m:4d} | Best Acc: {val * 100:.2f}%")
