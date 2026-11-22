"""Operators compared in the benchmark figures.

LAPLEX implementations (``--impl``):

* ``fused``      -- default.  ``vendor/fused/``: ``kerlap_fused.py`` (method ``dl-fused``,
                    the code of the published Figs. 1/15 and Tab. 3) with one change
                    to the original in ``vendor/fused_orig/``, in ``prefix_suffix_dl.py``:
                    the scan grid becomes ``(tiles, B)`` when there are more than 65535 tiles,
                    which lifts the n < 2^26 cap of the 2026-05-03 file.  Outputs
                    match the original to its own run-to-run (atomicAdd) noise.
                    Reproduces the Fig. 1 H100 runtimes for n >= 2^21 to ~1 %.
* ``fused-orig`` -- the unmodified file, ``vendor/fused_orig/`` (fails for n >= 2^26).
* ``lib2026``    -- ``vendor/lib2026/cuda_kernel.py``, scan path forced
                    (``dense_budget=0, thin_dim_limit=0``); same algorithm as
                    ``public`` with launch grid ``(tiles, B)``.
* ``public-fix`` -- ``public-raw`` semantics (dense matmul when n*k <= 2^20, scan
                    otherwise) on the ``lib2026`` build, i.e. the released code
                    without the n < 2^26 cap.  Default of the figures that compare
                    LAPLEX with other methods (FFT, RFF), which the paper produced
                    with the released ``laplex.py``.
* ``public``     -- the released code (``comparison_experiments/``), always on
                    the scan path.  The released ``X_Laplex`` / ``Laplex_D_LaplexT``
                    silently switch to a dense matmul when ``n*k <= 2**20`` and
                    ignore the ``threshold`` argument.  Fails for n >= 2^26 (grid cap).
* ``public-raw`` -- the released functions called exactly as the original
                    scripts did (dense switch included).

Further variants, compared in impl_compare.py (Fig. 1 setting):

* ``lib2026-auto`` -- ``cuda_kernel.X_Laplex`` with its default dispatcher
                    (dense below 2^22 elements or for min(n, k) <= 512).
* ``merged``     -- ``vendor/merged_events.merged_K_at_x`` (two-pass
                    merged-events scan, meant for large B).  The original file
                    does not parse (unterminated CUDA source string); the copy
                    in ``vendor/`` adds the missing closing quotes only.
* ``dl``         -- ``vendor/kerlap/kerlap_ops.X_KerLap`` (2026-04-26, sorts a
                    internally).
* ``dl-opt``     -- ``vendor/kerlap_opt/kerlap_ops.X_KerLap`` (2026-07-01).
* ``mix-fused``  -- ``vendor/mix/laplex_mix_fused.laplex_mix`` with L = 1,
                    gamma = 1 (one CUDA launch, the x row lives in shared memory).
* ``nocuda``     -- ``laplex_noCUDA`` (torch.logcumsumexp) on the GPU.

``lib2026``/``public``/``public-raw`` import modules of the same name
(``prefix_suffix``) from different directories, so one process can use only
one of them.  Where each file in ``vendor/`` comes from: ``vendor/README.md``.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache

import torch

from common import HERE

VENDOR_DIR = HERE / "vendor"   # merged_events.py with a one-line syntax fix
PUBLIC_DIR = HERE / "comparison_experiments"
LIB_DIR = VENDOR_DIR / "lib2026"
FUSED_ORIG_DIR = VENDOR_DIR / "fused_orig"
FUSED_DIR = VENDOR_DIR / "fused"   # FUSED_ORIG_DIR + grid-layout fix
KERLAP_DIR = VENDOR_DIR / "kerlap"
KERLAP_OPT_DIR = VENDOR_DIR / "kerlap_opt"
MIX_DIR = VENDOR_DIR / "mix"
IMPLS = ("fused", "public-fix", "lib2026", "public", "public-raw", "fused-orig")
ALL_IMPLS = IMPLS + ("lib2026-auto", "merged", "dl", "dl-opt", "mix-fused", "nocuda")
DEFAULT_IMPL = "fused"
PAPER_COMPARISON_IMPL = "public-fix"   # figures comparing LAPLEX with FFT / RFF
_scan_dir = None


def _add_path(p):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def _claim_scan_dir(d):
    global _scan_dir
    if _scan_dir not in (None, d):
        raise RuntimeError(f"prefix_suffix already imported from {_scan_dir}; "
                           f"cannot also use the implementation in {d} in this process")
    _scan_dir = d
    _add_path(d)


@lru_cache(None)
def public():
    _claim_scan_dir(PUBLIC_DIR)
    import laplex
    return laplex


@lru_cache(None)
def lib2026():
    _claim_scan_dir(LIB_DIR)
    import cuda_kernel
    return cuda_kernel


@lru_cache(None)
def public_nocuda():
    _add_path(PUBLIC_DIR)
    import laplex_noCUDA
    return laplex_noCUDA


def _import_from(d, name):
    """Import module ``name`` from directory ``d`` (one implementation per process)."""
    _claim_scan_dir(d)
    import importlib
    return importlib.import_module(name)


@lru_cache(None)
def fused(orig: bool = False):
    _claim_scan_dir(FUSED_ORIG_DIR if orig else FUSED_DIR)
    import kerlap_fused
    return kerlap_fused


# ---------------------------------------------------------------------------
# X @ K(a, b)
# ---------------------------------------------------------------------------
def x_laplex(impl: str):
    """Return ``f(X, a, b)`` computing ``X @ K(a, b)``; ``a`` must be sorted."""
    if impl == "lib2026":
        ck = lib2026()

        def f(X, a, b):
            return ck.X_Laplex(X, a, b, a_sorted=True, dense_budget=0, thin_dim_limit=0)
    elif impl == "public-fix":
        ck = lib2026()

        def f(X, a, b):
            return ck.X_Laplex(X, a, b, a_sorted=True, dense_budget=2**20, thin_dim_limit=0)
    elif impl == "public":
        lap = public()

        def f(X, a, b):
            # Same branch choice as laplex.X_Laplex, without its dense shortcut.
            if a.numel() <= b.numel():
                return lap._xA(X, a, b, a_sorted=True)
            return lap._xAT(X, a, b, b_sorted=False)
    elif impl == "public-raw":
        lap = public()

        def f(X, a, b):
            return lap.X_Laplex(X, a, b, threshold=0, a_sorted=True)
    elif impl in ("fused", "fused-orig"):
        kf = fused(orig=impl == "fused-orig")

        def f(X, a, b):
            return kf.X_KerLap_fused(X, a, b, threshold=0, a_sorted=True, b_sorted=False)
    elif impl == "lib2026-auto":
        ck = lib2026()

        def f(X, a, b):
            return ck.X_Laplex(X, a, b, a_sorted=True)
    elif impl == "merged":
        me = _import_from(VENDOR_DIR, "merged_events")

        def f(X, a, b):
            # X @ K(a, b) = (K(b, a) @ X^T)^T
            return me.merged_K_at_x(b, a, X.T.contiguous()).T
    elif impl == "dl":
        ko = _import_from(KERLAP_DIR, "kerlap_ops")

        def f(X, a, b):
            return ko.X_KerLap(X, a, b, threshold=0)
    elif impl == "dl-opt":
        ko = _import_from(KERLAP_OPT_DIR, "kerlap_ops")

        def f(X, a, b):
            return ko.X_KerLap(X, a, b, threshold=0, a_sorted=True)
    elif impl == "mix-fused":
        mx = _import_from(MIX_DIR, "laplex_mix_fused")
        one = {}

        def f(X, a, b):
            # y[:, j] = sum_i X[:, i] K(b_j, a_i): queries b, anchors a.
            g = one.setdefault(X.device, torch.ones(1, device=X.device))
            return mx.laplex_mix(X, b.unsqueeze(0), a.unsqueeze(0), g)
    elif impl == "nocuda":
        lapnc = public_nocuda()

        def f(X, a, b):
            return lapnc.X_Laplex(X, a, b, threshold=0, a_sorted=True)
    else:
        raise ValueError(impl)
    return f


def x_laplex_cpu():
    """Pure-PyTorch reference (``laplex_noCUDA``), scan path (threshold=0 is honoured)."""
    lapnc = public_nocuda()

    def f(X, a, b):
        return lapnc.X_Laplex(X, a, b, threshold=0, a_sorted=True)
    return f


def dense_kernel(a, b):
    return torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())


def x_dense(X, a, b):
    """Explicit-dense baseline: materialise K and call one matmul."""
    return X @ dense_kernel(a, b)


# ---------------------------------------------------------------------------
# K(alpha, beta) diag(D) K(alpha, beta)^T
# ---------------------------------------------------------------------------
def _gram_public_scan(alpha_s, beta, D):
    """Body of laplex.Laplex_D_LaplexT (alpha sorted) without its dense shortcut."""
    from prefix_suffix import prefix_suffix_sep  # resolved via public() path
    n = alpha_s.numel()
    device, dtype = alpha_s.device, alpha_s.dtype
    idx = torch.bucketize(beta, alpha_s, right=True)
    alpha_ext = torch.cat([beta.min().unsqueeze(0), alpha_s, beta.max().unsqueeze(0)])
    alpha_ip1 = alpha_ext[idx + 1]
    u_vals = D * torch.exp(2.0 * (beta - alpha_ip1))
    u = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, u_vals)
    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_vals = D * torch.exp(2.0 * (alpha_i - beta))
    v = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, v_vals)
    w = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, D)
    U, V = prefix_suffix_sep(2.0 * alpha_s, u[:n].clone().unsqueeze(0),
                             v[1:n + 1].clone().unsqueeze(0))
    U, V = U.squeeze(0), V.squeeze(0)
    W = torch.cumsum(w[:n], dim=0)
    diff = torch.exp(-torch.relu(alpha_s.unsqueeze(1) - alpha_s.unsqueeze(0)))
    M_raw = diff * (U.unsqueeze(0) + (W.unsqueeze(1) - W.unsqueeze(0)) + V.unsqueeze(1))
    return torch.tril(M_raw) + torch.tril(M_raw, -1).T


def gram_laplex(impl: str):
    """Return ``f(alpha, beta, D)``; ``alpha`` must be sorted."""
    if impl == "lib2026":
        ck = lib2026()
        return lambda alpha, beta, D: ck.Laplex_D_LaplexT(alpha, beta, D, alpha_sorted=True,
                                                          dense_budget=0, thin_dim_limit=0)
    if impl == "public-fix":
        ck = lib2026()
        return lambda alpha, beta, D: ck.Laplex_D_LaplexT(alpha, beta, D, alpha_sorted=True,
                                                          dense_budget=2**20, thin_dim_limit=0)
    if impl == "public":
        public()
        return _gram_public_scan
    if impl == "public-raw":
        lap = public()
        return lambda alpha, beta, D: lap.Laplex_D_LaplexT(alpha, beta, D, threshold=0,
                                                          alpha_sorted=True)
    if impl in ("fused", "fused-orig"):
        kf = fused(orig=impl == "fused-orig")
        return lambda alpha, beta, D: kf.Ker_D_KerT_fused(alpha, beta, D, threshold=0,
                                                         alpha_sorted=True)
    raise ValueError(impl)


def gram_dense(alpha, beta, D):
    K = dense_kernel(alpha, beta)
    return (K * D.unsqueeze(0)) @ K.T


# ---------------------------------------------------------------------------
# FFT baselines (uniform grid a_i = b_i = i*dx, symmetric Toeplitz kernel)
# ---------------------------------------------------------------------------
def _cdt(dtype):
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def toeplitz_pre_fft(n, dx, dtype, device):
    """FFT of the 2n circulant embedding of K[i, j] = exp(-|i-j| dx)."""
    c = torch.exp(-torch.arange(n, device=device, dtype=dtype) * dx)
    emb = torch.zeros(2 * n, device=device, dtype=dtype)
    emb[:n] = c
    emb[n + 1:] = c[1:].flip(0)
    return torch.fft.fft(emb.to(_cdt(dtype)))


def toeplitz_matvec(X, C_fft):
    """(B, n) or (n,) -> same shape, via circulant embedding and three FFTs."""
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)
    B, n = X.shape
    X_pad = torch.zeros(B, 2 * n, device=X.device, dtype=_cdt(X.dtype))
    X_pad[:, :n] = X.to(X_pad.dtype)
    Y = torch.fft.ifft(torch.fft.fft(X_pad, dim=-1) * C_fft.unsqueeze(0), dim=-1)[:, :n].real
    return Y[0] if squeeze else Y


def toeplitz_problem(n, B, seed, device="cuda"):
    """Uniform grid a_i = i/n, X ~ N(0, 1) in fp32 and the fp64 dense reference."""
    torch.manual_seed(seed)
    dx = 1.0 / n
    a = torch.arange(n, device=device, dtype=torch.float32) * dx
    X = torch.randn(B, n, device=device, dtype=torch.float32)
    Y_ref = X.double() @ dense_kernel(a.double(), a.double())
    return dict(a=a, X=X, dx=dx, Y_ref=Y_ref)


def toeplitz_methods(lap):
    """fp32 routes compared on the Toeplitz problem (App. E.2)."""
    return {"laplex": lambda p: lap(p["X"], p["a"], p["a"]),
            "toeplitz": lambda p: toeplitz_matvec(
                p["X"], toeplitz_pre_fft(p["X"].shape[1], p["dx"], torch.float32, p["X"].device)),
            "dense": lambda p: x_dense(p["X"], p["a"], p["a"])}


# ---------------------------------------------------------------------------
# Random Fourier Features for exp(-|u - v|) (Cauchy spectral density)
# ---------------------------------------------------------------------------
def rff_matvec(x, a, b, D, seed):
    """RFF estimate of ``x @ K(a, b)``: (n,) -> (k,).  Differentiable in a, b."""
    device, dtype = x.device, x.dtype
    gen = torch.Generator(device=device).manual_seed(seed)
    U = torch.rand(D, generator=gen, device=device, dtype=dtype)
    w = torch.tan(math.pi * (U - 0.5))
    c = 2 * math.pi * torch.rand(D, generator=gen, device=device, dtype=dtype)
    scale = math.sqrt(2.0 / D)
    proj = x @ (scale * torch.cos(a.unsqueeze(1) * w + c))
    return (scale * torch.cos(b.unsqueeze(1) * w + c)) @ proj


def rel_errors(Y, Y_ref):
    """(relative l2, relative l_inf) of Y against an fp64 reference."""
    d = (Y.double() - Y_ref).abs()
    return d.norm().item() / Y_ref.norm().item(), d.max().item() / Y_ref.abs().max().item()
