# Copy of 2026-LAPLEX/merged_events.py (2026-05-18) with ONE fix: the closing """ of _CUDA_SRC
# (after the closing brace of merged_LR_launch) was missing, so the original does not parse.
"""
merged_events.py
================

Two-pass merged-events Laplace-kernel matvec  y = K(a, b) @ x  with full
autograd support.

Algorithm
---------
Sort  a  and  b  ascending, then iterate through the MERGED event list
(length n+k) twice in a single CUDA kernel: forward sweep builds L_part,
backward sweep builds R_part.  The two sweeps share a kernel launch but
are kept SEQUENTIAL in time (not interleaved) -- empirically this beats
fused-interleave by ~10% because (i) only one accumulator is live at a
time → higher SM occupancy, and (ii) the L2 cache reuses the same x
rows across the two passes (forward reads ascending b-events, backward
reads them descending) → better cache locality.

    K[i, j] = exp(-|a_i - b_j|)
    y[i, batch] = sum_j K[i, j] * x[j, batch]
                = L_part[i, batch] + R_part[i, batch]

with the "moving-observer" parametrisation
    L_part[i] = sum_{b_j <= a_i} exp(b_j - a_i) x_j
    R_part[i] = sum_{b_j >  a_i} exp(t - b_j) x_j

State evolution between events at t_prev and t_new (forward, t_new > t_prev):
    L *= exp(-(t_new - t_prev))     # exp arg < 0  ✓
At a b-event (b_k = t_new):  L += x[orig_b_idx, batch]    # no exp
At an a-event (a_i = t_new): L_part[orig_a_idx, batch] = L

The backward sweep is identical but iterates the event list in reverse.

All exp arguments are <= 0; L and R are bounded by Σ_j |x_j| -- no
overflow, plain FP32.

Optimisations vs the previous version
-------------------------------------
* Forward AND backward sweeps fused into a single kernel (one launch),
  but kept SEQUENTIAL in time -- benchmarks showed interleaving or
  vectorizing across batch lanes both REGRESSED 10-50% (register
  pressure → lower occupancy, cache thrashing).  Conclusion: the GPU
  scheduler already hides scalar dependency latency well; the original
  scalar form is the local optimum for this kernel shape.
* exp() removed from the hot loop: expm1_decay precomputed once on the
  host in the accumulator dtype (FP32 minimum, FP64 for FP64 I/O).
* expm1 form of the recurrence (numerical stability for small dt):
    L <- L + L * expm1_decay[e]    (single FMA, same cost)
* No materialised (E, B) s tensor -- the kernel reads x[idx, batch]
  directly when a b-event fires.
* No flip materialisation -- the backward sweep iterates the event
  array in reverse, in-place.
* Separate L_part / R_part outputs of shape (n, B) -- needed for the
  autograd backward, and avoids the read-modify-write into y.
* For autograd, the backward needs ONE additional kernel call (with
  swapped a, b on g_y) -- not two.

Autograd
--------
Wrapped in `MergedKAtX` (torch.autograd.Function).  Gradients:
    g_x[j, batch] = L'_part[j, batch] + R'_part[j, batch]
    g_a[i]        = sum_batch g_y[i, batch] * (R_part - L_part)[i, batch]
    g_b[j]        = sum_batch x[j, batch]    * (R'_part - L'_part)[j, batch]
where (L', R') = merged_LR(b, a, g_y).  See backward() below for the
short derivation.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline


_NVCC_FLAGS = ["-O3", "--use_fast_math", "--allow-unsupported-compiler",
               "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"]
if sys.platform == "win32":
    _NVCC_FLAGS += ["-DUSE_CUDA", "-Xcompiler", "/Zc:preprocessor"]


_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// One CUDA thread per batch lane.  The kernel walks the merged event
// list FORWARD to build L_part, then BACKWARD to build R_part, in a
// single launch.  x is read directly with the original-index lookup at
// each b-event (no intermediate (E, B) tensor needed).
//
// Layout:
//   expm1_decay (E,)        exp(-(t[e] - t[e-1])) - 1 for e>=1,
//                           expm1_decay[0] unused              T_acc
//   event_types (E,)        0 = a-event, 1 = b-event           int32
//   event_idxs  (E,)        original a- or b-index per event   int32
//   x           (K_in, B)   row-major                          T_io
//   L_out, R_out (N_out, B) row-major                          T_io
//
// Templated on TWO types:
//   T_io   storage type for global memory (FP16 / FP32 / FP64)
//   T_acc  internal accumulator type      (FP32 for FP16 I/O, else == T_io)
// This enables the mixed-precision inference pattern:  FP16 in global,
// FP32 accumulators in registers, FP16 out.
//
// Optimisation: the per-event decay factor is PRE-COMPUTED on the host
// once (a single E-wide elementwise expm1 call in T_acc precision) and
// passed to the kernel as `expm1_decay`.  This removes the high-latency
// SFU `exp` from the inner-loop critical path -- the kernel just loads
// the precomputed value and FMAs.  exp() latency is ~16-32 cycles on
// the SFU and was previously serialised by the L_e <- alpha_e * L_{e-1}
// dependency chain.  Removing it materially shortens the hot path.

// Conversion helpers (load: T_io -> T_acc;  store: T_acc -> T_io)
template<typename T_io, typename T_acc>
__device__ __forceinline__ T_acc load_as(const T_io* p) { return T_acc(*p); }
template<>
__device__ __forceinline__ float load_as<__half, float>(const __half* p) {
    return __half2float(*p);
}

template<typename T_io, typename T_acc>
__device__ __forceinline__ void store_as(T_io* p, T_acc v) { *p = T_io(v); }
template<>
__device__ __forceinline__ void store_as<__half, float>(__half* p, float v) {
    *p = __float2half(v);
}

// expm1 form of the recurrence (numerical stability for small dt):
//   L_new = L * exp_decay
//         = L * (1 + expm1_decay)            [where expm1_decay = exp_decay - 1]
//         = L + L * expm1_decay              [fused as a single FMA]
//
// Why bother?  When dt -> 0 we have  exp_decay -> 1  while  expm1_decay -> -dt.
// In FP16 storage,  1.0 - 1e-4  rounds to  1.0  exactly, so a naive
// exp_decay loses the meaningful -dt part for closely-spaced events.
// Storing expm1_decay (always in T_acc = FP32 or FP64) preserves that
// information.  The FMA "L + L * expm1" is one instruction on modern GPUs,
// same as the bare "L * exp_decay" multiply -- precision win is free.

// Sequential forward + backward sweeps in one kernel launch.  The two
// sweeps share neither register state nor output buffer, so they could
// in principle be interleaved -- but we tried that and it regressed
// 10%, and 2-lane vectorization on top of fusion regressed 50%.  The
// scalar sequential form turns out to be the local optimum on a
// memory-bound kernel of this shape: occupancy stays high (only one
// accumulator alive at a time → low register pressure → many warps
// resident per SM → strong latency hiding), and L2 reuses the same x
// rows across the two passes.
template<typename T_io, typename T_acc>
__global__ void merged_LR_kernel(
    const T_acc* __restrict__ expm1_decay,   // exp(-(t[e]-t[e-1])) - 1; [0] unused
    const int*   __restrict__ event_types,
    const int*   __restrict__ event_idxs,
    const T_io*  __restrict__ x,
    T_io*        __restrict__ L_out,
    T_io*        __restrict__ R_out,
    const int E,
    const int B)
{
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= B) return;

    // ── Forward sweep:  accumulate L, emit L_out at a-events ─────────
    T_acc L = T_acc(0);
    {
        const int t0   = event_types[0];
        const int idx0 = event_idxs [0];
        if (t0 == 1) {
            L = load_as<T_io, T_acc>(x + idx0 * B + batch);
        } else {
            store_as<T_io, T_acc>(L_out + idx0 * B + batch, T_acc(0));
        }
    }
    for (int e = 1; e < E; ++e) {
        // L *= exp_decay[e]   rewritten as   L = L + L * expm1_decay[e].
        L = L + L * expm1_decay[e];
        const int type = event_types[e];
        const int idx  = event_idxs [e];
        if (type == 1) {
            L += load_as<T_io, T_acc>(x + idx * B + batch);
        } else {
            store_as<T_io, T_acc>(L_out + idx * B + batch, L);
        }
    }

    // ── Backward sweep: accumulate R, emit R_out at a-events ─────────
    // The decay magnitudes are symmetric:  going from event e+1 back to
    // event e  uses  exp(-(t[e+1]-t[e])) - 1  =  expm1_decay[e+1].
    T_acc R = T_acc(0);
    {
        const int tL   = event_types[E - 1];
        const int idxL = event_idxs [E - 1];
        if (tL == 1) {
            R = load_as<T_io, T_acc>(x + idxL * B + batch);
        } else {
            store_as<T_io, T_acc>(R_out + idxL * B + batch, T_acc(0));
        }
    }
    for (int e = E - 2; e >= 0; --e) {
        R = R + R * expm1_decay[e + 1];
        const int type = event_types[e];
        const int idx  = event_idxs [e];
        if (type == 1) {
            R += load_as<T_io, T_acc>(x + idx * B + batch);
        } else {
            store_as<T_io, T_acc>(R_out + idx * B + batch, R);
        }
    }
}


// Host launcher: auto-dispatches between FP16, FP32, FP64 kernels by dtype.
// FP16  -- mixed precision (FP16 I/O, FP32 accumulator); halves bandwidth.
// FP32  -- straight FP32 throughout.
// FP64  -- straight FP64 throughout.
//
// `expm1_decay` is ALWAYS in the accumulator type (T_acc):
//   FP16 I/O -> FP32 expm1_decay  (precision preserved for small dt)
//   FP32 I/O -> FP32 expm1_decay
//   FP64 I/O -> FP64 expm1_decay
std::vector<torch::Tensor> merged_LR_launch(
    torch::Tensor expm1_decay,
    torch::Tensor event_types,
    torch::Tensor event_idxs,
    torch::Tensor x,
    int64_t n_out)
{
    TORCH_CHECK(expm1_decay.is_cuda() && x.is_cuda(), "CUDA required");
    TORCH_CHECK(x.dtype() == torch::kFloat16
             || x.dtype() == torch::kFloat32
             || x.dtype() == torch::kFloat64,
                "merged scan supports FP16, FP32, and FP64");
    auto acc_dtype = (x.dtype() == torch::kFloat64)
                     ? torch::kFloat64 : torch::kFloat32;
    TORCH_CHECK(expm1_decay.dtype() == acc_dtype,
                "expm1_decay must be in accumulator dtype "
                "(FP32 for FP16/FP32 I/O, FP64 for FP64 I/O)");
    TORCH_CHECK(event_types.dtype() == torch::kInt32, "types int32");
    TORCH_CHECK(event_idxs .dtype() == torch::kInt32, "idxs  int32");
    TORCH_CHECK(x.is_contiguous(),             "x contiguous");
    TORCH_CHECK(expm1_decay.is_contiguous(),   "expm1_decay  contig");
    TORCH_CHECK(event_types.is_contiguous(),   "event_types contig");
    TORCH_CHECK(event_idxs .is_contiguous(),   "event_idxs  contig");

    const int E = expm1_decay.size(0);
    const int B = x.size(1);

    auto L = torch::empty({n_out, (int64_t)B}, x.options());
    auto R = torch::empty({n_out, (int64_t)B}, x.options());

    const int NTHREADS = 128;
    const int n_blocks = (B + NTHREADS - 1) / NTHREADS;

    if (x.dtype() == torch::kFloat16) {
        // Mixed precision: __half I/O, float accumulator + FP32 expm1_decay.
        merged_LR_kernel<__half, float><<<n_blocks, NTHREADS>>>(
            expm1_decay.data_ptr<float>(),
            event_types.data_ptr<int32_t>(),
            event_idxs .data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(L.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(R.data_ptr<at::Half>()),
            E, B);
    } else if (x.dtype() == torch::kFloat32) {
        merged_LR_kernel<float, float><<<n_blocks, NTHREADS>>>(
            expm1_decay.data_ptr<float>(),
            event_types.data_ptr<int32_t>(),
            event_idxs .data_ptr<int32_t>(),
            x.data_ptr<float>(),
            L.data_ptr<float>(),
            R.data_ptr<float>(),
            E, B);
    } else {  // FP64
        merged_LR_kernel<double, double><<<n_blocks, NTHREADS>>>(
            expm1_decay.data_ptr<double>(),
            event_types.data_ptr<int32_t>(),
            event_idxs .data_ptr<int32_t>(),
            x.data_ptr<double>(),
            L.data_ptr<double>(),
            R.data_ptr<double>(),
            E, B);
    }

    return {L, R};
}
"""

_CPP_SRC = r"""
std::vector<torch::Tensor> merged_LR_launch(
    torch::Tensor expm1_decay,
    torch::Tensor event_types,
    torch::Tensor event_idxs,
    torch::Tensor x,
    int64_t n_out);
"""


_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = load_inline(
            name="laplex_merged_events_cuda",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["merged_LR_launch"],
            extra_cuda_cflags=_NVCC_FLAGS,
            verbose=False,
        )
    return _BACKEND


# ─────────────────────────── host-side helpers ──────────────────────────────


def _build_events(a: torch.Tensor, b: torch.Tensor):
    """Sort the merged event list  {a_i} ∪ {b_j}  ascending and pre-compute
    the per-step expm1 decay factors so the kernel's hot loop avoids exp().

    Tie-breaking:  b before a  (b's concatenated first + stable sort).
    Returns (expm1_decay, event_types, event_idxs) on a.device, all
    contiguous.  Types are int32; `expm1_decay` is FP32 for FP16/FP32
    inputs and FP64 for FP64 inputs (always the accumulator dtype).

    `expm1_decay[e]` is  exp(-(t[e] - t[e-1])) - 1  for e >= 1; [0] is
    set to 0 (unused; the very first event has no predecessor).

    Why expm1 and not exp?
    ----------------------
    The kernel recurrence  L <- L * exp(-dt)  is mathematically the same
    as  L <- L + L * (exp(-dt) - 1) = L + L * expm1(-dt).  For small dt
    (closely-spaced events),  exp(-dt) ≈ 1 - dt  rounds to 1 in low
    precision and loses the meaningful -dt part;  expm1(-dt) ≈ -dt is
    representable directly.  We always compute expm1 in T_acc precision
    (FP32 minimum), so even FP16 I/O paths get full FP32 decay accuracy.
    """
    n = a.numel()
    k = b.numel()
    device = a.device
    dtype  = a.dtype

    # T_acc: FP32 for FP16/FP32 I/O, FP64 for FP64 I/O.
    acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32

    all_vals  = torch.cat([b, a])
    all_types = torch.cat([
        torch.ones (k, dtype=torch.int32, device=device),
        torch.zeros(n, dtype=torch.int32, device=device),
    ])
    all_idxs = torch.cat([
        torch.arange(k, dtype=torch.int32, device=device),
        torch.arange(n, dtype=torch.int32, device=device),
    ])
    order = torch.argsort(all_vals, stable=True)
    ev_vals  = all_vals [order].to(acc_dtype).contiguous()
    ev_types = all_types[order].contiguous()
    ev_idxs  = all_idxs [order].contiguous()

    # Precompute expm1_decay[e] = exp(-(t[e]-t[e-1])) - 1 for e>=1; [0] unused.
    # Done once on the host in T_acc precision (FP32 / FP64) -- O(E),
    # cheap -- so the kernel hot loop only LOADS a precise decay factor.
    # The expm1 form preserves precision for small dt (close events).
    dt = ev_vals[1:] - ev_vals[:-1]                       # (E-1,) ; >= 0
    expm1_decay = torch.empty_like(ev_vals)
    expm1_decay[0]  = 0                                   # sentinel; unused
    expm1_decay[1:] = torch.expm1(-dt)
    return expm1_decay.contiguous(), ev_types, ev_idxs


def _merged_LR(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain CUDA call -- returns (L_part, R_part).  No autograd.

    Dispatches to FP16 / FP32 / FP64 kernel based on `x.dtype`.  All
    three inputs (a, b, x) must share dtype.
    """
    n = a.numel()
    expm1_decay, ev_types, ev_idxs = _build_events(a, b)
    out = _backend().merged_LR_launch(expm1_decay, ev_types, ev_idxs, x, n)
    return out[0], out[1]


# ─────────────────────────── autograd Function ──────────────────────────────


class MergedKAtX(Function):
    """``y = K(a, b) @ x``  with backward through a, b, x.

    Forward
    -------
    Computes  L_part, R_part = _merged_LR(a, b, x)  and returns
    y = L_part + R_part.  Saves a, b, x, L_part, R_part for backward.

    Backward
    --------
    Let y[i, batch] = sum_j K[i, j] x[j, batch], K[i, j] = exp(-|a_i - b_j|).

    *  dy/dx = K^T   =>   g_x = K(b, a) @ g_y
       which is one more merged-LR call with (b, a) swapped.
       Specifically  L', R' = _merged_LR(b, a, g_y)  and
       g_x[j, batch] = L'[j, batch] + R'[j, batch].

    *  dy[i]/da_i = R_part[i] - L_part[i]   (the only continuous a_i in
       y[i] sits inside the exp(±a_i) prefactors of L, R; the boundary
       "b_j <= a_i" is piecewise-constant in a_i a.s., so its
       contribution is zero).
       Hence  g_a[i] = sum_batch g_y[i, batch] * (R_part - L_part)[i, batch].

    *  dy[i]/db_j = sign(a_i - b_j) * K[i, j] * x[j, batch], which gives
       g_b[j] = sum_batch x[j, batch] * (R' - L')[j, batch]
       using the same (L', R') above.

    All three pieces share the single backward kernel invocation.
    """

    @staticmethod
    def forward(ctx, a, b, x):
        L, R = _merged_LR(a, b, x)
        ctx.save_for_backward(a, b, x, L, R)
        return L + R

    @staticmethod
    def backward(ctx, g_y):
        a, b, x, L, R = ctx.saved_tensors
        needs_a, needs_b, needs_x = ctx.needs_input_grad[:3]

        Lp = Rp = None
        if needs_b or needs_x:
            Lp, Rp = _merged_LR(b, a, g_y.contiguous())

        g_x = (Lp + Rp) if needs_x else None
        g_a = (g_y * (R - L)).sum(dim=1) if needs_a else None
        g_b = (x   * (Rp - Lp)).sum(dim=1) if needs_b else None
        return g_a, g_b, g_x


# ───────────────────────────── public API ───────────────────────────────────


def merged_K_at_x(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
                  normalize: bool = False,
                  norm_eps: Optional[float] = None,
                  B_chunk: Optional[int] = 4096) -> torch.Tensor:
    """y = K(a, b) @ x   via two-pass merged-events scan.

    Inputs
    ------
    a : (n,)   FP16 / FP32 / FP64 CUDA
    b : (k,)   same dtype as a
    x : (k, B) same dtype as a, contiguous

    Returns
    -------
    y : (n, B) same dtype as the inputs

    Normalisation
    -------------
    If `normalize=True`, returns y_i = (Σ_j K_ij · x_j) / (Σ_j K_ij).
    Implemented by appending a column of ones to x (denominator channel)
    and dividing post-scan inside a clamp.  `norm_eps` defaults to a
    dtype-appropriate floor (FP16 -> 1e-4, FP32 -> 1e-12, FP64 -> 1e-30).
    When ALL |a_i - b_j| are huge (every K entry underflows), the
    denominator hits the clamp floor and the output is silently 0;
    typical training keeps coordinates bounded so this is harmless.

    Numeric precision
    -----------------
    * FP32 / FP64 -- straight precision throughout (FP64 uses standard exp).
    * FP16 -- mixed precision: I/O in FP16 (half the bandwidth), but the
      L and R accumulators live in FP32 registers, so the output retains
      effectively FP32 accuracy at the cost of FP16 quantisation on
      load/store.  Suitable for inference; not recommended for training
      where FP16 storage of gradients amplifies rounding error.

    Autograd
    --------
    Differentiable through a, b, x via `MergedKAtX`.

    Chunking
    --------
    For very large B the (k, B) input + (n, B) outputs can exceed GPU
    memory.  `B_chunk` (default 4096) splits along B and runs the
    kernel on each slice;  autograd composes correctly across slices.
    Pass `None` to disable.
    """
    assert a.is_cuda and b.is_cuda and x.is_cuda
    assert a.dtype == b.dtype == x.dtype, \
        f"a, b, x must share dtype; got {a.dtype}, {b.dtype}, {x.dtype}"
    assert x.dtype in (torch.float16, torch.float32, torch.float64), \
        f"merged scan supports FP16 / FP32 / FP64; got {x.dtype}"
    assert x.dim() == 2 and x.size(0) == b.size(0), \
        f"x shape {tuple(x.shape)} mismatches b shape {tuple(b.shape)}"

    if normalize:
        if norm_eps is None:
            norm_eps = {torch.float16: 1e-4,
                        torch.float32: 1e-12,
                        torch.float64: 1e-30}[x.dtype]
        k = x.size(0)
        ones_col = torch.ones(k, 1, dtype=x.dtype, device=x.device)
        x_aug = torch.cat([x, ones_col], dim=1).contiguous()
        y_aug = _scalar_apply_chunked(a, b, x_aug, B_chunk)
        y     = y_aug[:, :-1]
        denom = y_aug[:, -1:].clamp(min=norm_eps)
        return y / denom

    return _scalar_apply_chunked(a, b, x, B_chunk)


def _scalar_apply_chunked(a, b, x, B_chunk):
    """Chunked dispatch helper for the shared-coords MergedKAtX Function."""
    B = x.size(1)
    if B_chunk is not None and B > B_chunk:
        parts = []
        for s in range(0, B, B_chunk):
            e = min(s + B_chunk, B)
            parts.append(MergedKAtX.apply(a, b, x[:, s:e].contiguous()))
        return torch.cat(parts, dim=1)
    return MergedKAtX.apply(a, b, x)
