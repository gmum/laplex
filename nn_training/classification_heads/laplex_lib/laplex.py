"""
laplex.py  --  auto-dispatch facade for the LAPLEX library
===========================================================

Top-level entry point.  At import time this module picks one of two
backends and re-exports its public API:

* ``cuda.py`` -- optimised CUDA backend (custom ``load_inline`` kernels,
  ~30 s first-time compile, requires nvcc + a working host compiler).
* ``pure.py`` -- pure-PyTorch fallback (runs anywhere PyTorch runs,
  including CPU-only boxes; algorithmically the same dispatcher).

Both backends expose the same public API:

    X_Laplex(X, a, b)                  = X @ K(a, b)
    Laplex_X(X, a, b)                  = K(a, b) @ X
    Laplex_D_LaplexT(alpha, beta, D)   = K diag(D) K^T

so ``from laplex import X_Laplex`` works regardless of which backend
ended up loaded.  The pure backend additionally exposes the phased
operations (``X_LaplexPhase`` etc.) -- the CUDA backend doesn't have
those yet, so to keep the facade's API consistent across backends the
phased names are sourced from ``pure.py`` unconditionally.

Selection rules
---------------
1.  ``LAPLEX_BACKEND=cuda`` (env var)  -> force CUDA;  raise on failure.
2.  ``LAPLEX_BACKEND=pure`` (env var)  -> force pure-PyTorch.
3.  Otherwise (default): try CUDA;  on any failure (no GPU, missing
    compiler, ``load_inline`` error, ...) emit one ImportWarning and
    transparently fall back to pure-PyTorch.

The chosen backend is exposed as ``laplex.BACKEND`` (``"cuda"`` or
``"pure"``).  Switching at runtime is possible via ``set_backend()``
but typically you set the env var once at process start.

The per-row (grouped) / L-kernel mixture API lives in a companion
module ``laplex_grouped`` (sibling repository).  When the CUDA backend
is loaded AND ``laplex_grouped`` is importable, its public names are
re-exported here too -- see the CUDA backend's bottom-of-file block.
On the pure-PyTorch backend grouped/mix are not available (there is no
pure-PyTorch implementation of them at present).
"""
from __future__ import annotations

import os
import sys
import warnings


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_forced = os.environ.get("LAPLEX_BACKEND", "").lower().strip()
_use_pure = False
_cuda_err = None


def _has_cuda_runtime() -> bool:
    """Cheap check: do we have torch + a visible CUDA device?  Avoids paying
    the ~30 s ``load_inline`` compile just to discover there's no GPU."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if _forced == "pure":
    _use_pure = True
elif _forced == "cuda":
    # Forced CUDA: import will raise on failure -- that is the documented
    # behaviour, the user explicitly asked for CUDA.
    import cuda_kernel as _backend  # noqa: F401
    _use_pure = False
else:
    # Auto: prefer CUDA when a GPU is present and the backend imports cleanly.
    if _has_cuda_runtime():
        try:
            import cuda_kernel as _backend  # noqa: F401
            _use_pure = False
        except Exception as _e:
            _cuda_err = _e
            _use_pure = True
    else:
        _use_pure = True


if _use_pure:
    if _cuda_err is not None:
        warnings.warn(
            f"laplex: CUDA backend unavailable "
            f"({type(_cuda_err).__name__}: {_cuda_err}); "
            f"falling back to pure-PyTorch.",
            ImportWarning,
        )
    from pure import *  # noqa: F401,F403
    from pure import __all__ as _pure_all
    BACKEND = "pure"
    __all__ = list(_pure_all) + ["BACKEND", "set_backend"]
else:
    from cuda_kernel import *  # noqa: F401,F403
    from cuda_kernel import __all__ as _cuda_all
    BACKEND = "cuda"
    __all__ = list(_cuda_all) + ["BACKEND", "set_backend"]


def set_backend(name: str) -> None:
    """Switch the active backend at runtime.

    ``name`` must be ``"cuda"`` or ``"pure"``.  This re-binds the module's
    public names to the chosen backend; existing references kept by callers
    (e.g.\\ ``X_Laplex = laplex.X_Laplex`` at the top of their script) are
    NOT updated.  Prefer setting ``LAPLEX_BACKEND`` before import for
    deterministic behaviour.
    """
    name = name.lower().strip()
    if name not in ("cuda", "pure"):
        raise ValueError(f"unknown backend {name!r}; expected 'cuda' or 'pure'")
    if name == BACKEND:
        return
    # Re-import via the same mechanism by setting env and reloading.
    os.environ["LAPLEX_BACKEND"] = name
    import importlib
    importlib.reload(sys.modules[__name__])


# ---------------------------------------------------------------------------
# Top-level smoke test (run: python laplex.py)
# ---------------------------------------------------------------------------
def _smoke_test():
    """Quick correctness check on the active backend against the dense
    reference, plus a report of which backend is in use."""
    import torch
    print("=" * 60)
    print(f"laplex.py: active backend = {BACKEND!r}")
    print(f"           module file    = {__file__}")
    if BACKEND == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA backend loaded but no GPU visible.")
    print("=" * 60)

    device = "cuda" if (BACKEND == "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(0)
    a = torch.randn(64, device=device, dtype=torch.float32)
    b = torch.randn(80, device=device, dtype=torch.float32)
    X = torch.randn(4, 64, device=device, dtype=torch.float32)
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())

    y     = X_Laplex(X, a, b)
    y_ref = X @ K
    err = (y - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-30)
    ok = err < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}  X_Laplex vs dense:  err={err:.2e}")
    return ok


if __name__ == "__main__":
    _smoke_test()
