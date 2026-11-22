"""Shared utilities for all paper benchmark scripts."""
from __future__ import annotations

import os, sys, time, gc

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import matplotlib.pyplot as plt
from functools import wraps


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


COL_OURS    = "C0"   
COL_DENSE   = "C1"   
COL_DENSE64 = "C2"   
COL_NOCUDA  = "C3"   

DEVICE_CUDA = "cuda" if torch.cuda.is_available() else "cpu"

def report_nan_when_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (torch.AcceleratorError, torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            return float("nan")

    return wrapper

@report_nan_when_error
def bench_fwd(call, warmup=10, iters=30):
    """Best-of-3 timing of the forward call (no_grad)."""
    def _one():
        with torch.no_grad():
            for _ in range(warmup):
                call()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            for _ in range(iters):
                call()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            return (time.perf_counter() - t0) / iters * 1e3
    return min(_one() for _ in range(3))

@report_nan_when_error
def bench_fb(make_step, warmup=2, iters=5):
    """Best-of-3 timing of one forward+backward step.

    `make_step()` should return a callable that does ONE
    forward+backward (typically: build leaf tensors, fwd, .sum().backward()).
    """
    def _one():
        for _ in range(warmup):
            make_step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(iters):
            make_step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - t0) / iters * 1e3
    return min(_one() for _ in range(3))


@report_nan_when_error
def peak_mem_mb(call, device="cuda"):
    """Peak GPU memory used by `call()` in MB.  Resets the peak counter first."""
    if device != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    call()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)



def save(fig, name):
    """Save figure as both PDF and PNG."""
    fig.tight_layout()
    pdf = f"{name}.pdf"
    png = f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"  saved: {pdf}\n         {png}")



def safe_call(call, label=""):
    """Run a call, return (result, ok, exception_msg).  Captures OOM."""
    try:
        out = call()
        return out, True, None
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "OutOfMemoryError" in str(type(e).__name__):
            torch.cuda.empty_cache(); gc.collect()
            return None, False, "OOM"
        raise















class DenseGate:
    def __init__(self, time_limit_ms=500.0, mem_limit_gb=1.5):
        self.alive        = True
        self.time_limit   = time_limit_ms
        self.mem_limit_gb = mem_limit_gb
        self.reason       = None

    def predicted_oom(self, *byte_sizes):
        """Pre-emptively skip if estimated tensor sizes exceed mem_limit_gb."""
        total_gb = sum(byte_sizes) / (1024**3)
        if total_gb > self.mem_limit_gb:
            self.alive  = False
            self.reason = f"predicted OOM ({total_gb:.1f} GB)"
            return True
        return False

    def try_time(self, bench_fn):
        """Run a benchmark, kill the gate if it exceeds time_limit or OOMs."""
        if not self.alive:
            return float("nan")
        try:
            t = bench_fn()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache(); gc.collect()
            self.alive  = False
            self.reason = "OOM"
            return float("nan")
        if t > self.time_limit:
            self.alive  = False
            self.reason = f"slow ({t:.0f} ms > {self.time_limit:.0f} ms)"
        return t



