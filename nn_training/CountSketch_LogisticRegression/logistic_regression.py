import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import GaussianRandomProjection

from imagenet10_dataset import get_imagenet10_loaders
from utils import TeedLog


class TorchPCA:
    def __init__(self, n_components, device):
        self.n_components = n_components
        self.device = device
        self.mean = None
        self.components = None
        self.singular_values = None

    def fit(self, X):
        """
        X: Tensor (n_samples, n_features)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device).float()

        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean

        U, S, V = torch.pca_lowrank(X_centered, q=self.n_components, center=False)

        self.components = V
        self.singular_values = S

        return self

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device).float()

        X_centered = X - self.mean
        return torch.mm(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)





class CountSketch:
    def __init__(self, d, m, gaussian_weights=False):
        self.d = d
        self.m = m
        self.h = np.random.randint(0, m, size=d)
        if gaussian_weights:
            self.s = np.random.randn(d)
        else:
            self.s = np.random.choice([-1, 1], size=d)

    def transform(self, X):
        n = X.shape[0]
        Z = np.zeros((n, self.m))
        for i in range(self.d):
            Z[:, self.h[i]] += self.s[i] * X[:, i]
        return Z





class ExpKernelProjection:
    def __init__(
        self, d, m, tau=1.0, alpha_dist="normal", beta_dist="normal", weight_type="pm1"
    ):
        self.d = d
        self.m = m
        self.tau = tau

        self.alpha = np.random.randn(d) if alpha_dist == "normal" else np.random.rand(d)
        self.beta = np.random.randn(m) if beta_dist == "normal" else np.random.rand(m)
        self.w = (
            np.random.choice([-1, 1], size=d)
            if weight_type == "pm1"
            else np.random.randn(d)
        )

    def transform(self, X):
        
        
        diff = np.abs(self.alpha[:, np.newaxis] - self.beta[np.newaxis, :])
        K = np.exp(-diff / self.tau)  

        
        Xw = X * self.w  

        
        Z = Xw @ K
        return Z





def evaluate_projection(X_train, y_train, X_test, y_test, projector, n_iters):
    Z_train = projector.transform(X_train)
    Z_test = projector.transform(X_test)

    clf = LogisticRegression(max_iter=n_iters)
    clf.fit(Z_train, y_train)
    return clf.score(Z_test, y_test)


def get_params():
    parser = argparse.ArgumentParser("Base Methods")
    parser.add_argument("--root_dataset", type=str, default="./imagenet10")
    parser.add_argument("--dims", type=int, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--outputs", default="./outputs")
    return parser.parse_args()





def run_experiment():
    args = get_params()
    
    args.device = torch.device("cpu")

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

    X_train, y_train = [], []
    for data, lab in train_loader:
        data = data.flatten(start_dim=1).numpy()

        X_train.append(data)
        y_train.append(lab.numpy())

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_test, y_test = [], []
    for data, lab in test_loader:
        data = data.flatten(start_dim=1).numpy()

        X_test.append(data)
        y_test.append(lab.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    print(
        f"Train X: {X_train.shape}, Y: {y_train.shape}\nTest X: {X_test.shape}, Y: {y_test.shape}"
    )

    d = X_train.shape[1]

    dims = [25, 50, 100, 250, 500, 750, 1000]
    rows = []

    for m in dims:
        print(f"\n=== Target dimension m = {m} ===")

        
        print("Running PCA...")
        if d < 1000:
            pca = PCA(n_components=m)
            Z_tr_pca = pca.fit_transform(X_train)
            Z_te_pca = pca.transform(X_test)
        else:
            X_train_t = torch.from_numpy(X_train).to(args.device).float()
            X_test_t = torch.from_numpy(X_test).to(args.device).float()
            tpca = TorchPCA(n_components=m, device=args.device)
            Z_tr_pca_ = tpca.fit_transform(X_train_t)
            Z_te_pca_ = tpca.transform(X_test_t)

            Z_tr_pca = Z_tr_pca_.cpu().numpy()
            Z_te_pca = Z_te_pca_.cpu().numpy()

        print(f"\t - LogisticRegression...")
        clf = LogisticRegression(max_iter=args.iters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(Z_tr_pca, y_train)
        acc_pca = clf.score(Z_te_pca, y_test)
        print(f"\tPCA: {acc_pca:.4f}")
        rows.append({"m": m, "method": "PCA", "accuracy": acc_pca})

        
        print("Running GaussianRandomProjection...")
        rp = GaussianRandomProjection(n_components=m)
        Z_tr_rp = rp.fit_transform(X_train)
        Z_te_rp = rp.transform(X_test)
        print(f"\t - LogisticRegression...")
        clf = LogisticRegression(max_iter=args.iters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(Z_tr_rp, y_train)
        acc_rp = clf.score(Z_te_rp, y_test)
        print(f"\tRP: {acc_rp:.4f}")
        rows.append({"m": m, "method": "GaussianRP", "accuracy": acc_rp})

        
        print("Running CountSketch...")
        cs = CountSketch(d, m)
        acc_cs = evaluate_projection(X_train, y_train, X_test, y_test, cs, args.iters)
        print(f"\tCountSketch: {acc_cs:.4f}")
        rows.append({"m": m, "method": "CountSketch", "accuracy": acc_cs})

        
        print("Running Ours...")
        for tau in [0.01, 0.05, 0.1, 0.5, 1.0]:
            proj = ExpKernelProjection(d, m, tau=tau)
            acc_ours = evaluate_projection(
                X_train, y_train, X_test, y_test, proj, args.iters
            )
            print(f"\tOurs (tau={tau}): {acc_ours:.4f}")
            rows.append({"m": m, "method": f"Ours_tau={tau}", "accuracy": acc_ours})

        df = pd.DataFrame(rows)
        df.to_csv(Path(args.outputs) / "results.csv", index=False)


if __name__ == "__main__":
    run_experiment()
