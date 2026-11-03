"""Eksperyment MNIST dla GaussianKerLapWoodbury z szybkimi operatorami LAPLEX.

- Trenuje 25 epok, Adam lr=2e-3, batch=256.
- Loguje postęp co 10 epok do `progress.log` (bo conda run buforuje stdout).
- Ewaluuje LL na zbiorze testowym co 10 epok (1, 10, 20, 25) zamiast co epoka.
- Zapisuje próbki i checkpoint z sufiksem `_1tau` / `_2tau`.

Użycie:
    python _run_mnist.py                 # pojedyncze tau (domyślnie)
    python _run_mnist.py --two-tau       # dwie temperatury tau_L, tau_R
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Obejście konfliktu OpenMP (Intel libiomp5md.dll vs conda-forge) przy
# torch+matplotlib na Windows. Musi być ustawione PRZED importem torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Windows cp1250 nie łyka unicode'owego τ w printach — przestaw stdout na utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from laplex import laplex, laplex_gram, laplex_right


# ------------------------------------------------------------------
# Wrappery.  Konwencja LAPLEX:
#   K(a, b)[i, j] = exp( -|a_i - b_j| / tau_sign )
#   tau_sign      = tau_R   if a_i > b_j
#                   tau_L   otherwise
# Każdy wrapper akceptuje `tau_R=None`, co automatycznie daje single-tau.
# ------------------------------------------------------------------

def _as_tau_vec(tau):
    if not torch.is_tensor(tau):
        tau = torch.as_tensor(tau)
    return tau.reshape(1)


def X_mult_Ker(X, a, b, tau_L, tau_R=None):
    """X @ K(a, b).  X: (B, len(a)) -> (B, len(b))."""
    if tau_R is None:
        tau_R = tau_L
    squeeze_b = (X.ndim == 1)
    if squeeze_b:
        X = X.unsqueeze(0)
    Y = laplex(
        X.unsqueeze(1),
        a.unsqueeze(0),
        b.unsqueeze(0),
        _as_tau_vec(tau_L),
        _as_tau_vec(tau_R),
        normalize=False,
    ).squeeze(1)
    return Y.squeeze(0) if squeeze_b else Y


def Ker_mult_X(X, a, b, tau_L, tau_R=None):
    """K(a, b) @ X.  X: (len(b), B) -> (len(a), B)."""
    if tau_R is None:
        tau_R = tau_L
    squeeze_B = (X.ndim == 1)
    if squeeze_B:
        X = X.unsqueeze(1)
    X3 = X.T.unsqueeze(1)
    Y3 = laplex_right(
        X3,
        a.unsqueeze(0),
        b.unsqueeze(0),
        _as_tau_vec(tau_L),
        _as_tau_vec(tau_R),
        normalize=False,
    )
    Y = Y3.squeeze(1).T
    return Y.squeeze(1) if squeeze_B else Y


def Ker_gram(alpha_out, beta_sum, D_sum, tau_L, tau_R=None):
    """K(alpha_out, beta_sum) · diag(D_sum) · K(alpha_out, beta_sum)ᵀ.

    Zwraca (len(alpha_out), len(alpha_out)).
    Konwencja taus identyczna jak wyżej: tau_R gdy alpha_out > beta_sum.
    """
    if tau_R is None:
        tau_R = tau_L
    return laplex_gram(
        alpha_out.unsqueeze(0),
        beta_sum.unsqueeze(0),
        D_sum.unsqueeze(0),
        _as_tau_vec(tau_L),
        _as_tau_vec(tau_R),
        normalize=False,
    ).squeeze(0)


# ------------------------------------------------------------------
# Model.
# ------------------------------------------------------------------

class GaussianKerLap:
    """Funkcjonalny model z możliwie asymetrycznym jądrem Laplace.

    Jądro w modelu:
        K(α, β)[l, i] = exp( -|α_l - β_i| / τ_sign )
        τ_sign        = τ_R  gdy α_l > β_i
                        τ_L  w p.p.
    """

    def __init__(self, m, w, alpha, beta, A, s, tau_L, tau_R=None, eps=1e-12):
        self.m = m
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.s = torch.clamp(s, min=eps)
        self.tau_L = tau_L
        self.tau_R = tau_R if tau_R is not None else tau_L
        self.eps = eps

    def log_likelihood(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        n = self.m.numel()
        k = self.A.shape[0]

        D_inv = 1.0 / (self.s ** 2)

        # ===== G = Kᵀ · diag(w²/s²) · K   (k, k) ============================
        # G[i, i'] = Σ_l K(α,β)[l,i] (w²/s²)[l] K(α,β)[l,i'].
        # W Ker_gram(α_out, β_sum, D_sum, τ_L, τ_R):
        #   - α_out = oś wyjściowa (niesumowana),
        #   - β_sum = oś sumowana (z D po niej),
        #   - konwencja: τ_R gdy α_out > β_sum.
        # Dla naszego G oś wyjściowa to β (rozmiar k), sumujemy po α (n).
        # Ponieważ α_out↔β i β_sum↔α są zamienione względem definicji K(α,β),
        # warunek "α_out > β_sum" staje się "β > α", czyli w konwencji modelu
        # trafia to w τ_L (bo τ_R modelu = "α > β").  Dlatego TU zamieniamy
        # τ_L ↔ τ_R:
        w_tilde = self.w / self.s
        G = Ker_gram(
            self.beta, self.alpha, w_tilde ** 2,
            tau_L=self.tau_R,
            tau_R=self.tau_L,
        )

        # ===== M = I + A G Aᵀ ===============================================
        C = self.A @ G @ self.A.T
        C = 0.5 * (C + C.T)                       # symetryzacja vs roundoff
        M = torch.eye(k, device=x.device, dtype=x.dtype) + C
        L = torch.linalg.cholesky(M)

        # ===== quadratic =====================================================
        diff = x - self.m
        Dinv_diff = diff * D_inv
        v = Dinv_diff * self.w
        tmp = X_mult_Ker(v, self.alpha, self.beta, self.tau_L, self.tau_R)
        BtDinv = tmp @ self.A.T

        sol = torch.cholesky_solve(BtDinv.unsqueeze(-1), L).squeeze(-1)
        quad = (diff * Dinv_diff).sum(dim=1) - (sol * BtDinv).sum(dim=1)

        logdet = (
            torch.sum(torch.log(self.s ** 2))
            + 2 * torch.sum(torch.log(torch.diagonal(L)))
        )

        return -0.5 * (
            quad
            + logdet
            + n * torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
        )

    def sample(self, num_samples):
        n = self.m.numel()
        k = self.A.shape[0]

        z = torch.randn(num_samples, k, device=self.m.device)
        eps = torch.randn(num_samples, n, device=self.m.device)

        tmp = z @ self.A
        Ktmp = Ker_mult_X(tmp.T, self.alpha, self.beta, self.tau_L, self.tau_R)
        signal = Ktmp.T * self.w

        return self.m + signal + eps * self.s


class GaussianKerLapWoodbury(nn.Module):
    """Trenowalny wariant.  `single_tau=False` daje niezależne τ_L, τ_R.

    Jeśli `mean_init` jest podane, `m` jest zarejestrowany jako buffer
    (nietrenowalny, zamrożony) — pierwszy moment nie jest uczony SGD,
    bo i tak empiryczna średnia z N danych jest praktycznie MLE.
    """

    def __init__(self, n, k, eps=1e-4, single_tau=True, mean_init=None):
        super().__init__()
        self.n = n
        self.k = k
        self.eps = eps
        self.single_tau = single_tau

        if mean_init is not None:
            self.register_buffer("m", mean_init.detach().to(torch.float32).clone())
        else:
            self.m = nn.Parameter(torch.zeros(n))
        self.w = nn.Parameter(torch.randn(n) * 0.01)

        self.alpha = nn.Parameter(torch.linspace(-1, 1, n))
        self.beta = nn.Parameter(torch.linspace(-1, 1, k))

        # A jako dolnotrójkątna z dodatnią przekątną (Cholesky-shape).
        self.A_off = nn.Parameter(torch.randn(k, k) * 0.01)
        self.A_diag_raw = nn.Parameter(torch.full((k,), -4.6))

        self.s_raw = nn.Parameter(torch.zeros(n))

        if single_tau:
            self.tau_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.tau_L_raw = nn.Parameter(torch.tensor(0.0))
            self.tau_R_raw = nn.Parameter(torch.tensor(0.0))

    def tau_L(self):
        if self.single_tau:
            return F.softplus(self.tau_raw) + self.eps
        return F.softplus(self.tau_L_raw) + self.eps

    def tau_R(self):
        if self.single_tau:
            return F.softplus(self.tau_raw) + self.eps
        return F.softplus(self.tau_R_raw) + self.eps

    def s(self):
        return F.softplus(self.s_raw) + self.eps

    def A(self):
        L = torch.tril(self.A_off, diagonal=-1)
        diag = F.softplus(self.A_diag_raw) + self.eps
        return L + torch.diag(diag)

    def to_functional(self):
        return GaussianKerLap(
            self.m, self.w, self.alpha, self.beta, self.A(),
            self.s(), self.tau_L(), self.tau_R(),
        )

    def log_likelihood(self, x):
        return self.to_functional().log_likelihood(x)

    def sample(self, num_samples):
        return self.to_functional().sample(num_samples)


# ------------------------------------------------------------------
# Trening.
# ------------------------------------------------------------------

def empirical_mean(loader, n, device):
    """Jedno przebiegnięcie, zwraca empiryczną średnią (float32)."""
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
            ll = model.log_likelihood(x)
            total += ll.sum().item()
            count += x.shape[0]
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-tau", action="store_true",
                        help="Użyj osobnych τ_L i τ_R (asymetryczne jądro).")
    parser.add_argument("--complex", action="store_true",
                        help="Użyj zespolonego jądra A = Re[exp(-|α-β|/τ + i(γ-δ))].")
    parser.add_argument("--dense", action="store_true",
                        help="Zamiast kernela użyj dense low-rank W∈ℝ^{n×k_d} (PPCA-like).")
    parser.add_argument("--k", type=int, default=50,
                        help="Wymiar latent (β) dla kernel-modeli.")
    parser.add_argument("--k-dense", type=int, default=5,
                        help="Wymiar latent dla dense low-rank analog (--dense).")
    parser.add_argument("--learn-mean", action="store_true",
                        help="Ucz m od zera (domyślnie: zamrożone z Welforda).")
    args = parser.parse_args()

    if args.complex and args.dense:
        parser.error("--complex i --dense wykluczają się.")

    single_tau = not args.two_tau
    if args.dense:
        mode = f"dense_kd{args.k_dense}"
    else:
        kernel = "complex" if args.complex else "real"
        tau_str = "1tau" if single_tau else "2tau"
        mode = f"{kernel}_{tau_str}_k{args.k}"
    suffix = mode

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress_path = os.path.join(_HERE, "progress.log")
    samples_path = os.path.join(_HERE, f"samples_mnist_{suffix}.png")
    ckpt_path = os.path.join(_HERE, f"model_mnist_{suffix}.pt")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    open(progress_path, "w").close()
    log(f"device={device}  torch={torch.__version__}  mode={suffix}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    data_root = os.path.join(_HERE, "data")
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=0, pin_memory=(device == "cuda"))

    log(f"train={len(train_ds)}  test={len(test_ds)}")

    # Empiryczna średnia (Welford-style) — zamrożona w modelu, chyba że --learn-mean.
    if args.learn_mean:
        mean_init = None
        log("mean: learnable (init=0)")
    else:
        mean_init = empirical_mean(train_loader, 784, device=device)
        log(f"mean: Welford-frozen  ||m||={mean_init.norm().item():.3f}")

    if args.dense:
        from _model_dense import GaussianDenseLowRankWoodbury
        model = GaussianDenseLowRankWoodbury(784, args.k_dense, mean_init=mean_init).to(device)
    elif args.complex:
        from _model_complex import GaussianKerLapComplexWoodbury
        model = GaussianKerLapComplexWoodbury(784, args.k, single_tau=single_tau,
                                              mean_init=mean_init).to(device)
    else:
        model = GaussianKerLapWoodbury(784, args.k, single_tau=single_tau,
                                       mean_init=mean_init).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"model={mode}  n_params={n_params}  "
        f"(m {'learnable' if args.learn_mean else 'frozen'})")

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    num_epochs = 25
    eval_every = 10
    train_lls, test_lls = [], []

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            ll = model.log_likelihood(x)
            loss = -ll.mean()
            loss.backward()
            opt.step()
            total += ll.mean().item()

        train_ll = total / len(train_loader)
        train_lls.append((epoch, train_ll))

        do_eval = (epoch % eval_every == 0) or (epoch == num_epochs) or (epoch == 1)
        if do_eval:
            test_ll = evaluate(model, test_loader, device)
            test_lls.append((epoch, test_ll))

            elapsed = time.time() - t0
            eta = elapsed * (num_epochs - epoch) / max(epoch, 1)
            extra = ""
            if not single_tau:
                extra = f"  τ_L={model.tau_L().item():.3f}  τ_R={model.tau_R().item():.3f}"
            log(f"epoch {epoch:3d}/{num_epochs}  "
                f"train_LL={train_ll:.2f}  test_LL={test_ll:.2f}  "
                f"elapsed={elapsed:.1f}s  eta={eta:.1f}s{extra}")
        else:
            if epoch % 5 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (num_epochs - epoch) / max(epoch, 1)
                log(f"epoch {epoch:3d}/{num_epochs}  "
                    f"train_LL={train_ll:.2f}  "
                    f"elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    total_time = time.time() - t0
    log(f"done in {total_time:.1f}s")
    log(f"final train_LL = {train_lls[-1][1]:.2f}")
    log(f"final test_LL  = {test_lls[-1][1]:.2f}")
    log("eval points (epoch, test_LL): " +
        ", ".join(f"({e},{v:.2f})" for e, v in test_lls))
    if single_tau:
        log(f"τ = {model.tau_L().item():.4f}")
    else:
        log(f"τ_L = {model.tau_L().item():.4f}   τ_R = {model.tau_R().item():.4f}")

    torch.save(
        {"state_dict": model.state_dict(),
         "train_lls": train_lls,
         "test_lls": test_lls,
         "single_tau": single_tau},
        ckpt_path,
    )
    log(f"saved checkpoint -> {ckpt_path}")

    # Próbki.
    model.eval()
    with torch.no_grad():
        samples = model.sample(10).cpu()

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        img = samples[i].view(28, 28).clamp(0, 1)
        axes[i].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")
    if single_tau:
        title = (f"GaussianKerLapWoodbury 1-τ (LAPLEX) — "
                 f"test LL={test_lls[-1][1]:.1f}  τ={model.tau_L().item():.3f}")
    else:
        title = (f"GaussianKerLapWoodbury 2-τ (LAPLEX) — "
                 f"test LL={test_lls[-1][1]:.1f}  "
                 f"τ_L={model.tau_L().item():.3f}  τ_R={model.tau_R().item():.3f}")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(samples_path, dpi=120, bbox_inches="tight")
    log(f"saved samples -> {samples_path}")


if __name__ == "__main__":
    main()
