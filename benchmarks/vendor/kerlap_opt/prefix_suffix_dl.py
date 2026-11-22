"""
prefix_suffix_dl.py
===================

Differentiable CUDA primitive:

    prefix_suffix(alpha, u, v)[b,i] = prefix(alpha, u)[b,i] + suffix(alpha, v)[b,i]

where:
    prefix(a, u)[b,i] = exp(-a[i]) * sum_{j<=i} exp(a[j]) * u[b,j]
    suffix(a, v)[b,i] = exp(+a[i]) * sum_{j>=i} exp(-a[j]) * v[b,j]

Requirements:
    alpha : [T]     float32 non-decreasing, CUDA
    u, v  : [B, T]  float32, CUDA

IMPLEMENTATION
-------------
Naively, exp(a) overflows for a > ~88 (float32). Instead, values are kept
in the representation (n, x) with value = x * 2^n, i.e. a mantissa x
and a separate integer exponent n -> no overflow. The scan uses the
Decoupled-Lookback algorithm (Merrill & Garland 2016) in a single pass.

FUSED KERNEL
------------
We scan the values multiplied by ef = exp2(frac(a*log2 e)); at the
end the result is divided by ef. Both steps (multiplication on input,
division on output) are done in registers inside the kernel, without
materializing a temporary (B, T) tensor -> ~2*B*T less memory traffic
per pair of scans compared with the non-fused version.

GRADIENT (for _PsFn / prefix_suffix)
    grad_u     = suffix(alpha, g)
    grad_v     = prefix(alpha, g)
    grad_alpha = (u*grad_u - v*grad_v + g*(suffix_fwd - prefix_fwd)).sum(dim=0)
"""

import os, sys, math, time
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

_LOG2E = math.log2(math.e)  # 1/ln(2)

_NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "--allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
]

# ============================================================
# CUDA source
# ============================================================
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Pair (n, x) representing the value x * 2^n.  n - int, x - float32.
struct Pair { int n; float x; };

// Associative combine: joins the prefix (a) with the next element (b).
// The result has exponent b.n; the mantissas are shifted to the common exponent.
__device__ __forceinline__ Pair combine(Pair a, Pair b) {
    return {b.n, b.x + ldexpf(a.x, a.n - b.n)};
}

// Warp scan over 32 threads with shfl_up.
__device__ __forceinline__ Pair warp_scan(Pair val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        Pair other;
        other.n = __shfl_up_sync(0xffffffff, val.n, offset);
        other.x = __shfl_up_sync(0xffffffff, val.x, offset);
        if ((threadIdx.x & 31) >= offset)
            val = combine(other, val);
    }
    return val;
}

// Decoupled-lookback state flags of every tile.
#define FLAG_UNAVAIL 0   // nothing computed yet
#define FLAG_LOCAL   1   // local_sum available, inclusive prefix not yet
#define FLAG_PREFIX  2   // inclusive prefix of everything so far available

// -------------------------------------------------------------------
// FORWARD scan (prefix): from left to right.
//
// Input:   n [T]      - exponent floor(alpha*log2e), int32
//           x [B,T]    - mantissa u
//           ef [T]     - exp2(frac(alpha*log2e)), float32
// Output:  out_y [B,T] - already decoded: prefix[b,i] = scan(x*ef)/ef
// DL state: state_*    - one slot per (batch, tile)
// -------------------------------------------------------------------
__global__ void dl_scan_fwd_fused(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    const float* __restrict__ ef,
    float* out_y,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles)
{
    int batch = blockIdx.x, tile = blockIdx.y;
    int sid   = batch * num_tiles + tile;
    const float* x_b  = x     + (size_t)batch * T;
    float*       oy_b = out_y + (size_t)batch * T;
    extern __shared__ Pair shared[];
    int tid  = tile * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    // Phase 0: vectorized load (4 elements per thread) + fusion: mantissa = x*ef.
    int4   ni  = ((const int4*)n )[tid];
    float4 xi  = ((const float4*)x_b)[tid];
    float4 efi = ((const float4*)ef)[tid];
    Pair a0={ni.x, xi.x*efi.x}, a1={ni.y, xi.y*efi.y},
         a2={ni.z, xi.z*efi.z}, a3={ni.w, xi.w*efi.w};

    // Phase 1: local inclusive scan within the thread, the warp and the block.
    a1=combine(a0,a1); a2=combine(a1,a2); a3=combine(a2,a3);
    Pair val = warp_scan(a3);
    Pair we; we.n=__shfl_up_sync(0xffffffff,val.n,1); we.x=__shfl_up_sync(0xffffffff,val.x,1);
    if (lane==0) we={0,0.0f};
    a0=combine(we,a0); a1=combine(we,a1); a2=combine(we,a2); a3=combine(we,a3);
    if (lane==31) shared[warp]=val;
    __syncthreads();
    if (warp==0) {
        int nw=blockDim.x>>5; Pair v=(lane<nw)?shared[lane]:Pair{0,0.0f};
        v=warp_scan(v); shared[lane]=v;
    }
    __syncthreads();
    if (warp>0) { Pair off=shared[warp-1]; a0=combine(off,a0); a1=combine(off,a1); a2=combine(off,a2); a3=combine(off,a3); }

    // Phase 2: decoupled lookback.  The last thread publishes local_sum,
    // looks back to the first tile with FLAG_PREFIX and aggregates.
    if (threadIdx.x==blockDim.x-1) {
        Pair local_sum=a3;
        state_ln[sid]=local_sum.n; state_lx[sid]=local_sum.x;
        __threadfence();
        atomicExch(&state_flag[sid], FLAG_LOCAL);

        Pair exc={0,0.0f};
        if (tile>0) {
            volatile int*   vflag=(volatile int*)  state_flag+batch*num_tiles;
            volatile int*   vln  =(volatile int*)  state_ln  +batch*num_tiles;
            volatile float* vlx  =(volatile float*)state_lx  +batch*num_tiles;
            volatile int*   vpn  =(volatile int*)  state_pn  +batch*num_tiles;
            volatile float* vpx  =(volatile float*)state_px  +batch*num_tiles;
            // Note: {0,0} is a LEFT identity of combine, but NOT a right one -
            // combine(p,{0,0}) overflows ldexpf when p.n is large.
            // Fix: the first hit initializes agg, the following ones are combined.
            Pair agg; bool first=true; int look=tile-1;
            while (look>=0) {
                int f; while ((f=vflag[look])==FLAG_UNAVAIL) {}
                Pair p;
                if (f==FLAG_PREFIX) p={vpn[look],vpx[look]};
                else                p={vln[look],vlx[look]};
                agg = first ? p : combine(p, agg);
                first = false;
                if (f==FLAG_PREFIX) break;
                look--;
            }
            exc=agg;
        }
        Pair inc=combine(exc,local_sum);
        state_pn[sid]=inc.n; state_px[sid]=inc.x;
        __threadfence();
        atomicExch(&state_flag[sid], FLAG_PREFIX);
        shared[32]=exc;
    }
    __syncthreads();

    // Phase 3: apply the prefix of the previous tiles and decode: /ef.
    Pair exc=shared[32];
    a0=combine(exc,a0); a1=combine(exc,a1); a2=combine(exc,a2); a3=combine(exc,a3);
    int base=tid*4;
    float4 yy;
    yy.x = a0.x / efi.x;
    yy.y = a1.x / efi.y;
    yy.z = a2.x / efi.z;
    yy.w = a3.x / efi.w;
    *((float4*)(oy_b+base))=yy;
}

// -------------------------------------------------------------------
// BACKWARD scan (suffix): from right to left.  tid=0 handles the
// rightmost quad (T-4..T-1), tile=0 = the rightmost tile.
// Writes to the original positions (no flip of the tensor).
// -------------------------------------------------------------------
__global__ void dl_scan_bwd_fused(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    const float* __restrict__ ef,
    float* out_y,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles)
{
    int batch = blockIdx.x, tile = blockIdx.y;
    int sid   = batch * num_tiles + tile;
    const float* x_b  = x     + (size_t)batch * T;
    float*       oy_b = out_y + (size_t)batch * T;
    extern __shared__ Pair shared[];

    int tid     = tile * blockDim.x + threadIdx.x;
    int tid_rev = T4 - 1 - tid;         // actual position in the tensor
    int lane    = threadIdx.x & 31, warp = threadIdx.x >> 5;

    // Load from the reversed position; inside the quad go .w -> .x.
    int4   ni  = ((const int4*)n)[tid_rev];
    float4 xi  = ((const float4*)x_b)[tid_rev];
    float4 efi = ((const float4*)ef)[tid_rev];
    Pair a0={ni.w, xi.w*efi.w}, a1={ni.z, xi.z*efi.z},
         a2={ni.y, xi.y*efi.y}, a3={ni.x, xi.x*efi.x};

    a1=combine(a0,a1); a2=combine(a1,a2); a3=combine(a2,a3);

    Pair val = warp_scan(a3);
    Pair we; we.n=__shfl_up_sync(0xffffffff,val.n,1); we.x=__shfl_up_sync(0xffffffff,val.x,1);
    if (lane==0) we={0,0.0f};
    a0=combine(we,a0); a1=combine(we,a1); a2=combine(we,a2); a3=combine(we,a3);
    if (lane==31) shared[warp]=val;
    __syncthreads();
    if (warp==0) {
        int nw=blockDim.x>>5; Pair v=(lane<nw)?shared[lane]:Pair{0,0.0f};
        v=warp_scan(v); shared[lane]=v;
    }
    __syncthreads();
    if (warp>0) { Pair off=shared[warp-1]; a0=combine(off,a0); a1=combine(off,a1); a2=combine(off,a2); a3=combine(off,a3); }

    if (threadIdx.x==blockDim.x-1) {
        Pair local_sum=a3;
        state_ln[sid]=local_sum.n; state_lx[sid]=local_sum.x;
        __threadfence();
        atomicExch(&state_flag[sid], FLAG_LOCAL);
        Pair exc={0,0.0f};
        if (tile>0) {
            volatile int*   vflag=(volatile int*)  state_flag+batch*num_tiles;
            volatile int*   vln  =(volatile int*)  state_ln  +batch*num_tiles;
            volatile float* vlx  =(volatile float*)state_lx  +batch*num_tiles;
            volatile int*   vpn  =(volatile int*)  state_pn  +batch*num_tiles;
            volatile float* vpx  =(volatile float*)state_px  +batch*num_tiles;
            Pair agg; bool first=true; int look=tile-1;
            while (look>=0) {
                int f; while ((f=vflag[look])==FLAG_UNAVAIL) {}
                Pair p;
                if (f==FLAG_PREFIX) p={vpn[look],vpx[look]};
                else                p={vln[look],vlx[look]};
                agg = first ? p : combine(p, agg);
                first = false;
                if (f==FLAG_PREFIX) break;
                look--;
            }
            exc=agg;
        }
        Pair inc=combine(exc,local_sum);
        state_pn[sid]=inc.n; state_px[sid]=inc.x;
        __threadfence();
        atomicExch(&state_flag[sid], FLAG_PREFIX);
        shared[32]=exc;
    }
    __syncthreads();

    Pair exc=shared[32];
    a0=combine(exc,a0); a1=combine(exc,a1); a2=combine(exc,a2); a3=combine(exc,a3);
    int base = tid_rev * 4;
    // a3 -> position base+0, a0 -> base+3.  efi.x holds ef of base+0.
    float4 yy;
    yy.x = a3.x / efi.x;
    yy.y = a2.x / efi.y;
    yy.z = a1.x / efi.z;
    yy.w = a0.x / efi.w;
    *((float4*)(oy_b+base))=yy;
}

// -------------------------------------------------------------------
// Launcher: runs both kernels and returns (prefix, suffix) [B,T].
// Requires T % 1024 == 0 (padding on the Python side).
// -------------------------------------------------------------------
std::vector<torch::Tensor> scan_ps_fused(
    torch::Tensor n_fwd, torch::Tensor ef_fwd, torch::Tensor x_fwd,
    torch::Tensor n_bwd, torch::Tensor ef_bwd, torch::Tensor x_bwd)
{
    int B=x_fwd.size(0), T=x_fwd.size(1), T4=T/4;
    const int threads=256;
    int tiles=(T4+threads-1)/threads;
    int64_t ns=(int64_t)B*tiles;

    auto out_fwd=torch::empty({B,T},x_fwd.options());
    auto out_bwd=torch::empty({B,T},x_bwd.options());

    auto oi=torch::TensorOptions().dtype(torch::kInt32).device(x_fwd.device());
    auto of=torch::TensorOptions().dtype(torch::kFloat32).device(x_fwd.device());
    int smem=33*(int)sizeof(Pair);
    dim3 grid(B, tiles);  // batch first -> better scheduling for large B

    auto sf0=torch::zeros({ns},oi); auto ln0=torch::empty({ns},oi);
    auto lx0=torch::empty({ns},of); auto pn0=torch::empty({ns},oi); auto px0=torch::empty({ns},of);
    auto sf1=torch::zeros({ns},oi); auto ln1=torch::empty({ns},oi);
    auto lx1=torch::empty({ns},of); auto pn1=torch::empty({ns},oi); auto px1=torch::empty({ns},of);

    dl_scan_fwd_fused<<<grid,threads,smem>>>(
        n_fwd.data_ptr<int>(), x_fwd.data_ptr<float>(), ef_fwd.data_ptr<float>(),
        out_fwd.data_ptr<float>(),
        sf0.data_ptr<int>(), ln0.data_ptr<int>(), lx0.data_ptr<float>(),
        pn0.data_ptr<int>(), px0.data_ptr<float>(), T, T4, tiles);

    dl_scan_bwd_fused<<<grid,threads,smem>>>(
        n_bwd.data_ptr<int>(), x_bwd.data_ptr<float>(), ef_bwd.data_ptr<float>(),
        out_bwd.data_ptr<float>(),
        sf1.data_ptr<int>(), ln1.data_ptr<int>(), lx1.data_ptr<float>(),
        pn1.data_ptr<int>(), px1.data_ptr<float>(), T, T4, tiles);

    return {out_fwd, out_bwd};
}

// -------------------------------------------------------------------
// Fused reduction of grad_alpha (used in the backward of _PrefSuffDiffFn).
//   out[t] = sum_b { u[b,t]*sgp[b,t] - v[b,t]*pgs[b,t]
//                  - gp[b,t]*P[b,t]  + gs[b,t]*S[b,t] }
// One pass over memory, no temporary (B, T) tensor.
// -------------------------------------------------------------------
__global__ void grad_alpha_reduce_kernel(
    const float* __restrict__ u,
    const float* __restrict__ sgp,
    const float* __restrict__ v,
    const float* __restrict__ pgs,
    const float* __restrict__ gp,
    const float* __restrict__ P,
    const float* __restrict__ gs,
    const float* __restrict__ S,
    float* __restrict__ out,
    int B, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        int idx = b * T + t;
        acc += u[idx] * sgp[idx] - v[idx] * pgs[idx]
             - gp[idx] * P[idx]  + gs[idx] * S[idx];
    }
    out[t] = acc;
}

torch::Tensor grad_alpha_reduce(
    torch::Tensor u, torch::Tensor sgp, torch::Tensor v, torch::Tensor pgs,
    torch::Tensor gp, torch::Tensor P, torch::Tensor gs, torch::Tensor S)
{
    int B = u.size(0), T = u.size(1);
    auto out = torch::empty({T}, u.options());
    int threads = 256;
    int blocks = (T + threads - 1) / threads;
    grad_alpha_reduce_kernel<<<blocks, threads>>>(
        u.data_ptr<float>(), sgp.data_ptr<float>(),
        v.data_ptr<float>(), pgs.data_ptr<float>(),
        gp.data_ptr<float>(), P.data_ptr<float>(),
        gs.data_ptr<float>(), S.data_ptr<float>(),
        out.data_ptr<float>(), B, T);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_ps_fused", &scan_ps_fused,
          "fused prefix+suffix DL scan (x*ef on input, /ef on output, 1 pass)");
    m.def("grad_alpha_reduce", &grad_alpha_reduce,
          "fused grad_alpha reduction (1-pass)");
}
"""

# ============================================================
# Building the extension
# ============================================================
print("Compiling prefix_suffix extension...", end=" ", flush=True)
_ext = load_inline(
    name="prefix_suffix_dl_fused",
    cpp_sources=[""],
    cuda_sources=[_CUDA_SRC],
    extra_cuda_cflags=_NVCC_FLAGS,
    verbose=False,
)
print("OK\n")

# ============================================================
# Python helpers
# ============================================================
_THREADS = 256
_PAD_TO = _THREADS * 4  # 1024; the kernel expects T % 1024 == 0


def _precompute(alpha: torch.Tensor):
    """
    Splits alpha into ef (mantissa < 2) and n (integer exponent).

    Returns:
        ef     = exp2(frac( alpha * log2e))  [T]   for the prefix
        n_fwd  = floor(alpha * log2e)        [T]   int32, non-decreasing
        ef_neg = exp2(frac(-alpha * log2e))  [T]   for the suffix
        n_bwd  = floor(-alpha * log2e)       [T]   int32, non-increasing
    """
    ap = alpha.float() * _LOG2E
    n_f = ap.floor().to(torch.int32)
    ef = torch.exp2(ap - n_f.float())

    an = -ap
    n_b = an.floor().to(torch.int32)
    ef_neg = torch.exp2(an - n_b.float())

    return ef, n_f, ef_neg, n_b


def _pad_pair(x: torch.Tensor, n1d: torch.Tensor):
    """Pad x [B,T] and n1d [T] to a multiple of 1024. Returns (x_p, n_p, T_orig)."""
    T = x.size(1)
    rem = T % _PAD_TO
    if rem == 0:
        return x, n1d, T
    pad = _PAD_TO - rem
    x_p = torch.nn.functional.pad(x, (0, pad))
    n_p = torch.cat([n1d, n1d[-1:].expand(pad)])
    return x_p, n_p, T


def _pad_ef(ef: torch.Tensor, T: int):
    """Pad ef [T] to a multiple of 1024 (repeating the last value)."""
    rem = T % _PAD_TO
    if rem == 0:
        return ef
    pad = _PAD_TO - rem
    return torch.cat([ef, ef[-1:].expand(pad)])


def _ps_raw(ef, n_fwd, ef_neg, n_bwd, u: torch.Tensor, v: torch.Tensor):
    """
    Raw call of the fused kernel.  The inputs (ef, n, u, v) must come from the
    same precompute; returns the final (prefix [B,T], suffix [B,T]) - the kernel
    decodes with /ef in its last phase.
    """
    T = u.size(1)
    u_pad, n_f_pad, T_orig = _pad_pair(u, n_fwd)
    v_pad, n_b_pad, _ = _pad_pair(v, n_bwd)
    ef_pad = _pad_ef(ef, T)
    ef_neg_pad = _pad_ef(ef_neg, T)

    out_f, out_b = _ext.scan_ps_fused(
        n_f_pad.contiguous(),
        ef_pad.contiguous(),
        u_pad.contiguous(),
        n_b_pad.contiguous(),
        ef_neg_pad.contiguous(),
        v_pad.contiguous(),
    )
    return out_f[:, :T_orig], out_b[:, :T_orig]


# ============================================================
# Autograd
# ============================================================
class _PsFn(Function):
    """
    Forward : returns prefix(alpha,u) + suffix(alpha,v), [B, T].
    Backward: grad_u = suffix(alpha, g),  grad_v = prefix(alpha, g),
              grad_alpha = (u*grad_u - v*grad_v
                            + g*(suffix_fwd - prefix_fwd)).sum(dim=0)
    """

    @staticmethod
    def forward(ctx, alpha, u, v):
        orig_dtype = alpha.dtype
        a32 = alpha.float()
        u32 = u.float()
        v32 = v.float()
        ef, n_fwd, ef_neg, n_bwd = _precompute(a32)
        prefix_r, suffix_r = _ps_raw(ef, n_fwd, ef_neg, n_bwd, u32, v32)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u32, v32, prefix_r, suffix_r)
        ctx.orig_dtype = orig_dtype
        return (prefix_r + suffix_r).to(orig_dtype)

    @staticmethod
    def backward(ctx, g):
        ef, n_fwd, ef_neg, n_bwd, u, v, prefix_r, suffix_r = ctx.saved_tensors
        od = ctx.orig_dtype
        g32 = g.float()
        # One _ps_raw computes prefix(a,g) and suffix(a,g) together.
        prefix_g, suffix_g = _ps_raw(ef, n_fwd, ef_neg, n_bwd, g32, g32)

        grad_u = suffix_g.to(od) if ctx.needs_input_grad[1] else None
        grad_v = prefix_g.to(od) if ctx.needs_input_grad[2] else None

        grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_alpha = (
                (u * suffix_g - v * prefix_g + g32 * (suffix_r - prefix_r))
                .sum(dim=0)
                .to(od)
            )

        return grad_alpha, grad_u, grad_v


# ============================================================
# Public API
# ============================================================
def prefix_suffix(
    alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    prefix_suffix(alpha, u, v) = prefix(alpha, u) + suffix(alpha, v).

    alpha : [T]    float32, non-decreasing, CUDA
    u, v  : [B, T] float32, CUDA
    returns: [B, T] differentiable tensor
    """
    return _PsFn.apply(alpha, u, v)


# ============================================================
# Autograd: prefix and suffix SEPARATELY (for different u, v)
#
# Differs from _PsFn in that it returns two tensors (not their sum) and uses
# the fused CUDA reduction (grad_alpha_reduce) in the backward.
# ============================================================
class _PrefSuffDiffFn(Function):
    """Returns (prefix(alpha, u), suffix(alpha, v)) as two separate tensors."""

    @staticmethod
    def forward(ctx, alpha, u, v):
        orig_dtype = alpha.dtype
        a32 = alpha.float()
        u32 = u.float()
        v32 = v.float()
        ef, n_fwd, ef_neg, n_bwd = _precompute(a32)
        P, S = _ps_raw(ef, n_fwd, ef_neg, n_bwd, u32, v32)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u32, v32, P, S)
        ctx.orig_dtype = orig_dtype
        return P.to(orig_dtype), S.to(orig_dtype)

    @staticmethod
    def backward(ctx, gP, gS):
        ef, n_fwd, ef_neg, n_bwd, u, v, P, S = ctx.saved_tensors
        od = ctx.orig_dtype
        gP32 = gP.float()
        gS32 = gS.float()

        # One call computes prefix(a, gS) and suffix(a, gP).
        prefix_gS, suffix_gP = _ps_raw(ef, n_fwd, ef_neg, n_bwd, gS32, gP32)

        grad_u = suffix_gP.to(od) if ctx.needs_input_grad[1] else None
        grad_v = prefix_gS.to(od) if ctx.needs_input_grad[2] else None

        grad_alpha = None
        if ctx.needs_input_grad[0]:
            # Fused reduction:
            #   grad_alpha[t] = sum_b (u*suffix_gP - v*prefix_gS - gP*P + gS*S)[b,t]
            grad_alpha = _ext.grad_alpha_reduce(
                u.contiguous(),
                suffix_gP.contiguous(),
                v.contiguous(),
                prefix_gS.contiguous(),
                gP32.contiguous(),
                P.contiguous(),
                gS32.contiguous(),
                S.contiguous(),
            ).to(od)

        return grad_alpha, grad_u, grad_v


def prefix_suffix_sep(alpha, u, v=None):
    """
    Returns (prefix(alpha, u), suffix(alpha, v)).  If v=None, v=u.

    alpha : [T]
    u, v  : [B, T] or [T]
    """
    if v is None:
        v = u
    squeeze = u.dim() == 1
    if squeeze:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0) if v is not u else u
    P, S = _PrefSuffDiffFn.apply(alpha, u, v)
    if squeeze:
        P = P.squeeze(0)
        S = S.squeeze(0)
    return P, S


# ============================================================
# GatherCombine: fusion of "X[:, idx_l] * w_l + X[:, idx_r] * w_r"
#
# Custom autograd so that:
#   - forward: one fused multiply-add without intermediate tensors
#   - backward: one scatter_add for grad_P and one for grad_S
#   - grad_w is skipped when it is not needed
# ============================================================
class _GatherCombineFn(Function):
    """
    Forward:
        y[b, j] = w_l[j] * P[b, c_l[j]] + w_r[j] * S[b, c_r[j]]

    Backward:
        grad_P[b, i] = sum_{j : c_l[j]=i} grad_y[b, j] * w_l[j]
        grad_S[b, i] = sum_{j : c_r[j]=i} grad_y[b, j] * w_r[j]
        grad_w_l[j]  = sum_b grad_y[b, j] * P[b, c_l[j]]
        grad_w_r[j]  = sum_b grad_y[b, j] * S[b, c_r[j]]
    """

    @staticmethod
    def forward(ctx, P, S, w_l, w_r, c_l, c_r):
        P_sel = P.index_select(1, c_l)  # [B, k]
        S_sel = S.index_select(1, c_r)  # [B, k]
        y = w_l.unsqueeze(0) * P_sel + w_r.unsqueeze(0) * S_sel
        ctx.save_for_backward(P_sel, S_sel, w_l, w_r, c_l, c_r)
        ctx.n = P.size(1)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        P_sel, S_sel, w_l, w_r, c_l, c_r = ctx.saved_tensors
        n = ctx.n
        B, k = grad_y.shape
        grad_P = grad_S = grad_w_l = grad_w_r = None

        if ctx.needs_input_grad[2]:
            grad_w_l = (grad_y * P_sel).sum(0)
        if ctx.needs_input_grad[3]:
            grad_w_r = (grad_y * S_sel).sum(0)

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            gy_wl = grad_y * w_l.unsqueeze(0) if ctx.needs_input_grad[0] else None
            gy_wr = grad_y * w_r.unsqueeze(0) if ctx.needs_input_grad[1] else None
            if ctx.needs_input_grad[0]:
                grad_P = torch.zeros(B, n, device=grad_y.device, dtype=grad_y.dtype)
                grad_P.scatter_add_(1, c_l.unsqueeze(0).expand(B, k), gy_wl)
            if ctx.needs_input_grad[1]:
                grad_S = torch.zeros(B, n, device=grad_y.device, dtype=grad_y.dtype)
                grad_S.scatter_add_(1, c_r.unsqueeze(0).expand(B, k), gy_wr)

        return grad_P, grad_S, grad_w_l, grad_w_r, None, None


# ============================================================
# Naive reference (for tests; overflows for alpha > ~88)
# ============================================================
def _naive_prefix(alpha, u):
    ea = torch.exp(alpha)
    ena = torch.exp(-alpha)
    return (u * ea.unsqueeze(0)).cumsum(dim=1) * ena.unsqueeze(0)


def _naive_suffix(alpha, v):
    ea = torch.exp(alpha)
    ena = torch.exp(-alpha)
    return ((v * ena.unsqueeze(0)).flip(-1).cumsum(dim=1).flip(-1)) * ea.unsqueeze(0)


def _naive_prefix_suffix(alpha, u, v):
    return _naive_prefix(alpha, u) + _naive_suffix(alpha, v)


# ============================================================
# Correctness tests
# ============================================================
def run_correctness_tests() -> bool:
    device = "cuda"
    torch.manual_seed(42)
    passed = failed = 0

    print("1. Correctness of prefix and suffix separately (vs naive):")
    for B, T, alo, ahi in [
        (1, 256, 0, 5),
        (4, 1024, 0, 10),
        (2, 512, -3, 0),
        (1, 4096, 0, 20),
    ]:
        alpha = torch.linspace(alo, ahi, T, device=device)
        u = torch.randn(B, T, device=device)
        v = torch.randn(B, T, device=device)

        ef, n_fwd, ef_neg, n_bwd = _precompute(alpha)
        p_got, s_got = _ps_raw(ef, n_fwd, ef_neg, n_bwd, u, v)
        p_ref = _naive_prefix(alpha, u)
        s_ref = _naive_suffix(alpha, v)

        ep = (p_got - p_ref).abs().max() / (p_ref.abs().max() + 1e-6)
        es = (s_got - s_ref).abs().max() / (s_ref.abs().max() + 1e-6)
        ok = ep < 2e-3 and es < 2e-3
        passed += ok
        failed += not ok
        print(
            f"   {'PASS' if ok else 'FAIL'}  B={B}, T={T:5d}, alpha={alo}..{ahi}  "
            f"err_prefix={ep:.1e}  err_suffix={es:.1e}"
        )

    print()
    print("2. Gradient (torch.autograd.gradcheck in fp64):")
    for B, T in [(1, 8), (2, 12), (1, 24)]:
        a_d = torch.linspace(
            0, 3, T, dtype=torch.float64, device=device
        ).requires_grad_(True)
        u_d = (
            0.3 * torch.randn(B, T, dtype=torch.float64, device=device)
        ).requires_grad_(True)
        v_d = (
            0.3 * torch.randn(B, T, dtype=torch.float64, device=device)
        ).requires_grad_(True)
        ok = torch.autograd.gradcheck(
            _PsFn.apply,
            (a_d, u_d, v_d),
            eps=1e-3,
            atol=5e-2,
            rtol=5e-2,
            raise_exception=False,
        )
        passed += ok
        failed += not ok
        print(f"   {'PASS' if ok else 'FAIL'}  B={B}, T={T}")

    print()
    print("3. Gradient vs autograd of the naive implementation (float32):")
    print(f"   {'Config':30s}  {'err_u':>7s}  {'err_v':>7s}  {'err_a':>7s}")
    print("   " + "-" * 58)
    for B, T, alo, ahi, desc in [
        (1, 256, 0, 5, "alpha 0..5"),
        (4, 1024, 0, 10, "alpha 0..10"),
        (8, 4096, 0, 20, "alpha 0..20"),
        (2, 512, -3, 0, "alpha -3..0"),
        (8, 16384, 0, 20, "alpha 0..20 duzy T"),
    ]:
        alpha = torch.linspace(alo, ahi, T, device=device)
        u_v = torch.randn(B, T, device=device)
        v_v = torch.randn(B, T, device=device)

        a1 = alpha.clone().requires_grad_(True)
        u1 = u_v.clone().requires_grad_(True)
        v1 = v_v.clone().requires_grad_(True)
        prefix_suffix(a1, u1, v1).sum().backward()

        a2 = alpha.clone().requires_grad_(True)
        u2 = u_v.clone().requires_grad_(True)
        v2 = v_v.clone().requires_grad_(True)
        _naive_prefix_suffix(a2, u2, v2).sum().backward()

        def re(a, b):
            return ((a - b).abs() / (b.abs().clamp(min=1e-6))).max().item()

        eu = re(u1.grad, u2.grad)
        ev = re(v1.grad, v2.grad)
        ea = re(a1.grad, a2.grad)
        tol_uv = 1e-2
        tol_a = max(2e-2, T * 1e-5)  # grad_a accumulates more error
        ok = eu < tol_uv and ev < tol_uv and ea < tol_a
        passed += ok
        failed += not ok
        print(
            f"   {'PASS' if ok else 'FAIL'}  B={B:2d},T={T:6d},{desc:16s}  {eu:.1e}  {ev:.1e}  {ea:.1e}"
        )

    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


if __name__ == "__main__":
    print("=== CORRECTNESS TESTS ===\n")
    run_correctness_tests()
