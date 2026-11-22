# Copy of KerLap implementation_cuda/faster/prefix_suffix_dl.py (2026-05-03).
# Only change (this copy): the scan grid is (B, tiles) as before when tiles <= 65535 and
# (tiles, B) otherwise; the kernels read the layout from gridDim.y.  The original fails
# with "invalid argument" for n >= 2^26 (gridDim.y <= 65535).
"""
prefix_suffix_dl.py

Implements and differentiates:

    prefix_suffix(alpha, u, v)[b,i] = prefix(alpha,u)[b,i] + suffix(alpha,v)[b,i]

where:
    prefix(a, u)[b,i] = exp(-a[i]) * sum_{j<=i} exp(a[j]) * u[b,j]
    suffix(a, v)[b,i] = exp(+a[i]) * sum_{j>=i} exp(-a[j]) * v[b,j]

Requirements:  alpha [T] float32 non-decreasing, CUDA
            u,v   [B,T] float32, CUDA

OPTIMIZATIONS over two separate calls of exp_weighted_cumsum_autograd:
  O1. ef  = exp2(frac( alpha*LOG2E))  [T]  computed ONCE, saved in ctx
      ef_neg= exp2(frac(-alpha*LOG2E)) [T]  likewise
      Backward reads ef, ef_neg from ctx instead of recomputing them.
  O2. No flip(): a dedicated CUDA kernel scans from right to left.
  O3. The kernel does not write out_n (always out_n[i]=n[i], a redundant write).

GRADIENT
--------
    grad_u  = suffix(alpha, g)     (scan from the right)
    grad_v  = prefix(alpha, g)     (scan from the left)
    grad_a  = (u*grad_u - v*grad_v + g*(suffix_fwd - prefix_fwd)).sum(0)

Backward cost = forward cost (the same 2 scans, the same preprocessing).
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
# CUDA source: two DL kernels (forward and backward)
# ============================================================
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct Pair { int n; float x; };

__device__ __forceinline__ Pair combine(Pair a, Pair b) {
    return {b.n, b.x + ldexpf(a.x, a.n - b.n)};
}

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

#define FLAG_UNAVAIL 0
#define FLAG_LOCAL   1
#define FLAG_PREFIX  2

// -------------------------------------------------------
// dl_scan_fwd: classic DL scan from left to right.
//   Writes only out_x (out_n is skipped, out_n[i]=n[i] always).
// -------------------------------------------------------
__global__ void dl_scan_fwd(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    float* out_x,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles)
{
    const bool bm_ = (gridDim.y == (unsigned)num_tiles);  /* repro: layout chosen by host */ int batch = bm_ ? blockIdx.x : blockIdx.y, tile = bm_ ? blockIdx.y : blockIdx.x;
    int sid   = batch * num_tiles + tile;
    const float* x_b  = x     + (size_t)batch * T;
    float*       ox_b = out_x + (size_t)batch * T;
    extern __shared__ Pair shared[];
    int tid  = tile * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    // Phase 1: local inclusive scan
    int4   ni = ((const int4*)n)[tid];
    float4 xi = ((const float4*)x_b)[tid];
    Pair a0={ni.x,xi.x}, a1={ni.y,xi.y}, a2={ni.z,xi.z}, a3={ni.w,xi.w};
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

    // Phase 2: decoupled lookback
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
            // BUGFIX: {0,0} is a LEFT identity for combine but NOT a right identity.
            // Original `combine(p, {0,0})` overflows when p.n is large (ldexpf(p.x, p.n)).
            // Fix: initialize agg from the first hit directly, combine on subsequent.
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

    // Phase 3: apply the prefix, write only out_x
    Pair exc=shared[32];
    a0=combine(exc,a0); a1=combine(exc,a1); a2=combine(exc,a2); a3=combine(exc,a3);
    int base=tid*4;
    float4 xx; xx.x=a0.x; xx.y=a1.x; xx.z=a2.x; xx.w=a3.x;
    *((float4*)(ox_b+base))=xx;
    // out_n is not written (out_n[i]=n[i] always)
}

// -------------------------------------------------------
// dl_scan_bwd: DL scan from right to left (no flip of the data!).
//   tid=0 handles the RIGHTMOST 4 elements (T-4..T-1),
//   tile=0 = the rightmost tile.
//   Writes the suffix scan to the original positions.
// -------------------------------------------------------
__global__ void dl_scan_bwd(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    float* out_x,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles)
{
    const bool bm_ = (gridDim.y == (unsigned)num_tiles);  /* repro: layout chosen by host */ int batch = bm_ ? blockIdx.x : blockIdx.y, tile = bm_ ? blockIdx.y : blockIdx.x;
    int sid   = batch * num_tiles + tile;
    const float* x_b  = x     + (size_t)batch * T;
    float*       ox_b = out_x + (size_t)batch * T;
    extern __shared__ Pair shared[];

    // tid_rev: the actual position (in int4/float4 units) read by this thread.
    // tid=0 -> tid_rev=T4-1 (rightmost quad), tid=T4-1 -> tid_rev=0 (leftmost).
    int tid     = tile * blockDim.x + threadIdx.x;
    int tid_rev = T4 - 1 - tid;
    int lane    = threadIdx.x & 31, warp = threadIdx.x >> 5;

    // Load from the reversed position; inside the quad go from the right: .w -> .z -> .y -> .x
    int4   ni = ((const int4*)n)[tid_rev];
    float4 xi = ((const float4*)x_b)[tid_rev];
    // a0 = rightmost (first in the scan), a3 = leftmost (last)
    Pair a0={ni.w,xi.w}, a1={ni.z,xi.z}, a2={ni.y,xi.y}, a3={ni.x,xi.x};

    // Phase 1: local inclusive scan (right to left within the thread)
    a1=combine(a0,a1); a2=combine(a1,a2); a3=combine(a2,a3);
    // a3.n = n[base+0] (leftmost position in this thread) - always the true n[i]

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

    // Phase 2: decoupled lookback (the same logic as in dl_scan_fwd)
    // tile=0 = the rightmost tile, it has no predecessors.
    // The predecessor of tile t is tile t-1 (to the left on the DL axis, i.e. to the right in the original coordinates).
    if (threadIdx.x==blockDim.x-1) {
        Pair local_sum=a3;   // sum of the whole tile (scan from right to left)
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
            // BUGFIX: {0,0} is a LEFT identity for combine but NOT a right identity.
            // Original `combine(p, {0,0})` overflows when p.n is large (ldexpf(p.x, p.n)).
            // Fix: initialize agg from the first hit directly, combine on subsequent.
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

    // Phase 3: apply the prefix from the right, write to the original positions
    Pair exc=shared[32];
    a0=combine(exc,a0); a1=combine(exc,a1); a2=combine(exc,a2); a3=combine(exc,a3);
    // a0 -> original position base+3, a3 -> base+0
    int base = tid_rev * 4;
    float4 xx; xx.x=a3.x; xx.y=a2.x; xx.z=a1.x; xx.w=a0.x;
    *((float4*)(ox_b+base))=xx;
}

// -------------------------------------------------------
// scan_ps: launches both kernels and returns (out_fwd, out_bwd)
//   n_fwd [T], x_fwd [B,T]  - input of the prefix scan
//   n_bwd [T], x_bwd [B,T]  - input of the suffix scan (n non-increasing)
//   T must be a multiple of 1024 (padding on the Python side)
// -------------------------------------------------------
std::vector<torch::Tensor> scan_ps(
    torch::Tensor n_fwd, torch::Tensor x_fwd,
    torch::Tensor n_bwd, torch::Tensor x_bwd)
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
    dim3 grid = (tiles <= 65535) ? dim3(B, tiles) : dim3(tiles, B);  /* repro: lift gridDim.y cap */  // batch first -> better scheduling for large B

    // Allocate the state of both kernels up front (memory safety)
    auto sf0=torch::zeros({ns},oi); auto ln0=torch::empty({ns},oi);
    auto lx0=torch::empty({ns},of); auto pn0=torch::empty({ns},oi); auto px0=torch::empty({ns},of);
    auto sf1=torch::zeros({ns},oi); auto ln1=torch::empty({ns},oi);
    auto lx1=torch::empty({ns},of); auto pn1=torch::empty({ns},oi); auto px1=torch::empty({ns},of);

    dl_scan_fwd<<<grid,threads,smem>>>(
        n_fwd.data_ptr<int>(), x_fwd.data_ptr<float>(), out_fwd.data_ptr<float>(),
        sf0.data_ptr<int>(), ln0.data_ptr<int>(), lx0.data_ptr<float>(),
        pn0.data_ptr<int>(), px0.data_ptr<float>(), T, T4, tiles);

    dl_scan_bwd<<<grid,threads,smem>>>(
        n_bwd.data_ptr<int>(), x_bwd.data_ptr<float>(), out_bwd.data_ptr<float>(),
        sf1.data_ptr<int>(), ln1.data_ptr<int>(), lx1.data_ptr<float>(),
        pn1.data_ptr<int>(), px1.data_ptr<float>(), T, T4, tiles);

    return {out_fwd, out_bwd};
}

// -------------------------------------------------------
// Segment-aware variants: grid is (B, tiles) instead of (tiles, B)
// so the CUDA scheduler processes tile 0 of ALL batches first,
// then tile 1 of all batches, etc.  This maximizes parallelism
// because tiles from different batches are independent.
// -------------------------------------------------------
__global__ void dl_scan_fwd_seg(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    float* out_x,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles)
{
    const bool bm_ = (gridDim.y == (unsigned)num_tiles);  /* repro: layout chosen by host */ int batch = bm_ ? blockIdx.x : blockIdx.y, tile = bm_ ? blockIdx.y : blockIdx.x;  // SWAPPED vs dl_scan_fwd
    int sid   = batch * num_tiles + tile;
    const float* x_b  = x     + (size_t)batch * T;
    float*       ox_b = out_x + (size_t)batch * T;
    extern __shared__ Pair shared[];
    int tid  = tile * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    int4   ni = ((const int4*)n)[tid];
    float4 xi = ((const float4*)x_b)[tid];
    Pair a0={ni.x,xi.x}, a1={ni.y,xi.y}, a2={ni.z,xi.z}, a3={ni.w,xi.w};
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
            // BUGFIX: {0,0} is a LEFT identity for combine but NOT a right identity.
            // Original `combine(p, {0,0})` overflows when p.n is large (ldexpf(p.x, p.n)).
            // Fix: initialize agg from the first hit directly, combine on subsequent.
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
    int base=tid*4;
    float4 xx; xx.x=a0.x; xx.y=a1.x; xx.z=a2.x; xx.w=a3.x;
    *((float4*)(ox_b+base))=xx;
}

__global__ void dl_scan_bwd_seg(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    float* out_x,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles)
{
    const bool bm_ = (gridDim.y == (unsigned)num_tiles);  /* repro: layout chosen by host */ int batch = bm_ ? blockIdx.x : blockIdx.y, tile = bm_ ? blockIdx.y : blockIdx.x;  // SWAPPED vs dl_scan_bwd
    int sid   = batch * num_tiles + tile;
    const float* x_b  = x     + (size_t)batch * T;
    float*       ox_b = out_x + (size_t)batch * T;
    extern __shared__ Pair shared[];

    int tid     = tile * blockDim.x + threadIdx.x;
    int tid_rev = T4 - 1 - tid;
    int lane    = threadIdx.x & 31, warp = threadIdx.x >> 5;

    int4   ni = ((const int4*)n)[tid_rev];
    float4 xi = ((const float4*)x_b)[tid_rev];
    Pair a0={ni.w,xi.w}, a1={ni.z,xi.z}, a2={ni.y,xi.y}, a3={ni.x,xi.x};

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
            // BUGFIX: {0,0} is a LEFT identity for combine but NOT a right identity.
            // Original `combine(p, {0,0})` overflows when p.n is large (ldexpf(p.x, p.n)).
            // Fix: initialize agg from the first hit directly, combine on subsequent.
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
    float4 xx; xx.x=a3.x; xx.y=a2.x; xx.z=a1.x; xx.w=a0.x;
    *((float4*)(ox_b+base))=xx;
}

std::vector<torch::Tensor> scan_ps_seg(
    torch::Tensor n_fwd, torch::Tensor x_fwd,
    torch::Tensor n_bwd, torch::Tensor x_bwd)
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
    dim3 grid = (tiles <= 65535) ? dim3(B, tiles) : dim3(tiles, B);  /* repro: lift gridDim.y cap */  // SWAPPED: batch first, tile second

    auto sf0=torch::zeros({ns},oi); auto ln0=torch::empty({ns},oi);
    auto lx0=torch::empty({ns},of); auto pn0=torch::empty({ns},oi); auto px0=torch::empty({ns},of);
    auto sf1=torch::zeros({ns},oi); auto ln1=torch::empty({ns},oi);
    auto lx1=torch::empty({ns},of); auto pn1=torch::empty({ns},oi); auto px1=torch::empty({ns},of);

    dl_scan_fwd_seg<<<grid,threads,smem>>>(
        n_fwd.data_ptr<int>(), x_fwd.data_ptr<float>(), out_fwd.data_ptr<float>(),
        sf0.data_ptr<int>(), ln0.data_ptr<int>(), lx0.data_ptr<float>(),
        pn0.data_ptr<int>(), px0.data_ptr<float>(), T, T4, tiles);

    dl_scan_bwd_seg<<<grid,threads,smem>>>(
        n_bwd.data_ptr<int>(), x_bwd.data_ptr<float>(), out_bwd.data_ptr<float>(),
        sf1.data_ptr<int>(), ln1.data_ptr<int>(), lx1.data_ptr<float>(),
        pn1.data_ptr<int>(), px1.data_ptr<float>(), T, T4, tiles);

    return {out_fwd, out_bwd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_ps", &scan_ps, "prefix+suffix decoupled-lookback scan");
    m.def("scan_ps_seg", &scan_ps_seg, "prefix+suffix DL scan (segment-aware scheduling)");
}
"""

# ============================================================
# Compilation
# ============================================================
print("Compiling prefix_suffix extension...", end=" ", flush=True)
_ext = load_inline(
    name="prefix_suffix_dl_v3",
    cpp_sources=[""],
    cuda_sources=[_CUDA_SRC],
    extra_cuda_cflags=_NVCC_FLAGS,
    verbose=False,
)
print("OK\n")

# ============================================================
# Python-level helpers
# ============================================================
_THREADS = 256
_PAD_TO = _THREADS * 4  # 1024


def _precompute(alpha: torch.Tensor):
    """
    Returns (ef, n_fwd, ef_neg, n_bwd) - shared preprocessing of alpha.

    ef     = exp2(frac(alpha * LOG2E))   [T]  for the prefix
    n_fwd  = floor(alpha * LOG2E)        [T]  int32, non-decreasing
    ef_neg = exp2(frac(-alpha * LOG2E))  [T]  for the suffix
    n_bwd  = floor(-alpha * LOG2E)       [T]  int32, non-increasing
    """
    ap = alpha.float() * _LOG2E
    n_f = ap.floor().to(torch.int32)
    ef = torch.exp2(ap - n_f.float())

    an = -ap
    n_b = an.floor().to(torch.int32)
    ef_neg = torch.exp2(an - n_b.float())

    return ef, n_f, ef_neg, n_b


def _pad_pair(x: torch.Tensor, n1d: torch.Tensor):
    """Pad x (B,T) and n1d (T) to a multiple of PAD_TO; returns (x_p, n_p, T_orig)."""
    T = x.size(1)
    rem = T % _PAD_TO
    if rem == 0:
        return x, n1d, T
    pad = _PAD_TO - rem
    x_p = torch.nn.functional.pad(x, (0, pad))
    n_p = torch.cat([n1d, n1d[-1:].expand(pad)])
    return x_p, n_p, T


def _ps_raw(ef, n_fwd, ef_neg, n_bwd, u: torch.Tensor, v: torch.Tensor):
    """
    Launches both DL kernels and decodes the results.
    Returns (prefix_result [B,T], suffix_result [B,T]).
    ef, n_fwd, ef_neg, n_bwd may be on the GPU.
    """
    # Prescaling: u' = u * ef,   v' = v * ef_neg
    u_p = u * ef.unsqueeze(0)  # [B, T]
    v_p = v * ef_neg.unsqueeze(0)

    # Pad to a multiple of 1024
    u_pad, n_f_pad, T_orig = _pad_pair(u_p, n_fwd)
    v_pad, n_b_pad, _ = _pad_pair(v_p, n_bwd)

    # Kernel
    out_f, out_b = _ext.scan_ps(
        n_f_pad.contiguous(),
        u_pad.contiguous(),
        n_b_pad.contiguous(),
        v_pad.contiguous(),
    )

    # Remove the padding and decode
    # prefix[i] = out_f[i] / ef[i]   (out_n_fwd[i] = n_fwd[i] always)
    # suffix[i] = out_b[i] / ef_neg[i]
    prefix_r = out_f[:, :T_orig] / ef.unsqueeze(0)
    suffix_r = out_b[:, :T_orig] / ef_neg.unsqueeze(0)
    return prefix_r, suffix_r


def _ps_raw_seg(ef, n_fwd, ef_neg, n_bwd, u: torch.Tensor, v: torch.Tensor):
    """
    Same as _ps_raw, but uses scan_ps_seg (segment-aware scheduling).
    The grid is (B, tiles) instead of (tiles, B), so the CUDA scheduler
    processes tile 0 of all batches at once (independent!),
    then tile 1, and so on.
    """
    u_p = u * ef.unsqueeze(0)
    v_p = v * ef_neg.unsqueeze(0)

    u_pad, n_f_pad, T_orig = _pad_pair(u_p, n_fwd)
    v_pad, n_b_pad, _ = _pad_pair(v_p, n_bwd)

    out_f, out_b = _ext.scan_ps_seg(
        n_f_pad.contiguous(),
        u_pad.contiguous(),
        n_b_pad.contiguous(),
        v_pad.contiguous(),
    )

    prefix_r = out_f[:, :T_orig] / ef.unsqueeze(0)
    suffix_r = out_b[:, :T_orig] / ef_neg.unsqueeze(0)
    return prefix_r, suffix_r


# ============================================================
# Autograd Function
# ============================================================
class _PsSegFn(Function):
    """Segment-aware version of _PsFn."""

    @staticmethod
    def forward(ctx, alpha, u, v):
        orig_dtype = alpha.dtype
        a32 = alpha.float()
        u32 = u.float()
        v32 = v.float()
        ef, n_fwd, ef_neg, n_bwd = _precompute(a32)
        prefix_r, suffix_r = _ps_raw_seg(ef, n_fwd, ef_neg, n_bwd, u32, v32)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u32, v32, prefix_r, suffix_r)
        ctx.orig_dtype = orig_dtype
        return (prefix_r + suffix_r).to(orig_dtype)

    @staticmethod
    def backward(ctx, g):
        ef, n_fwd, ef_neg, n_bwd, u, v, prefix_r, suffix_r = ctx.saved_tensors
        needs_a = ctx.needs_input_grad[0]
        needs_u = ctx.needs_input_grad[1]
        needs_v = ctx.needs_input_grad[2]
        orig_dtype = ctx.orig_dtype
        g32 = g.float()
        prefix_g, suffix_g = _ps_raw_seg(ef, n_fwd, ef_neg, n_bwd, g32, g32)
        grad_u = suffix_g.to(orig_dtype) if needs_u else None
        grad_v = prefix_g.to(orig_dtype) if needs_v else None
        grad_alpha = None
        if needs_a:
            grad_alpha = (
                (u * suffix_g - v * prefix_g + g32 * (suffix_r - prefix_r))
                .sum(dim=0)
                .to(orig_dtype)
            )
        return grad_alpha, grad_u, grad_v


class _PsFn(Function):

    @staticmethod
    def forward(ctx, alpha, u, v):
        """
        alpha : [T]    float32 or float64 (converted to float32 inside)
        u, v  : [B, T] float32 or float64
        returns: [B, T] in the original dtype
        """
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
        """
        g : [B, T]  upstream gradient
        grad_u  = suffix(alpha, g)
        grad_v  = prefix(alpha, g)
        grad_alpha = (u*grad_u - v*grad_v + g*(suffix_fwd - prefix_fwd)).sum(0)
        """
        ef, n_fwd, ef_neg, n_bwd, u, v, prefix_r, suffix_r = ctx.saved_tensors
        needs_a = ctx.needs_input_grad[0]
        needs_u = ctx.needs_input_grad[1]
        needs_v = ctx.needs_input_grad[2]
        orig_dtype = ctx.orig_dtype

        g32 = g.float()
        # Two scans of g32 (reusing the precomputed ef/n)
        prefix_g, suffix_g = _ps_raw(ef, n_fwd, ef_neg, n_bwd, g32, g32)

        grad_u = suffix_g.to(orig_dtype) if needs_u else None
        grad_v = prefix_g.to(orig_dtype) if needs_v else None

        grad_alpha = None
        if needs_a:
            grad_alpha = (
                (u * suffix_g - v * prefix_g + g32 * (suffix_r - prefix_r))
                .sum(dim=0)
                .to(orig_dtype)
            )

        return grad_alpha, grad_u, grad_v


# ============================================================
# Public API
# ============================================================
def prefix_suffix(
    alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    prefix_suffix(alpha, u, v) = prefix(alpha,u) + suffix(alpha,v)

    prefix(a,u)[b,i] = exp(-a[i]) * sum_{j<=i} exp(a[j])*u[b,j]
    suffix(a,v)[b,i] = exp(+a[i]) * sum_{j>=i} exp(-a[j])*v[b,j]

    alpha : [T]    float32, non-decreasing, CUDA
    u, v  : [B, T] float32, CUDA
    returns: [B, T] float32, differentiable
    """
    return _PsFn.apply(alpha, u, v)


def prefix_suffix_seg(
    alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Segment-aware prefix_suffix (better scheduling for large B)."""
    return _PsSegFn.apply(alpha, u, v)


def prefix_only(alpha: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Prefix part only, differentiable."""
    zeros = torch.zeros_like(u)
    return _PsFn.apply(alpha, u, zeros)


def suffix_only(alpha: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Suffix part only, differentiable."""
    zeros = torch.zeros_like(v)
    return _PsFn.apply(alpha, zeros, v)


# ============================================================
# Naive reference (via torch.cumsum, overflows for alpha > 88)
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

    print("1. Correctness of prefix and suffix separately:")
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
    print("2. Gradient (gradcheck float64, large eps/atol):")
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
    print("3. Gradients: DL vs naive autograd (float32):")
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

        # DL
        a1 = alpha.clone().requires_grad_(True)
        u1 = u_v.clone().requires_grad_(True)
        v1 = v_v.clone().requires_grad_(True)
        prefix_suffix(a1, u1, v1).sum().backward()

        # Naive
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
        tol_a = max(2e-2, T * 1e-5)  # grad_a accumulates more error than grad_u/v
        ok = eu < tol_uv and ev < tol_uv and ea < tol_a
        passed += ok
        failed += not ok
        print(
            f"   {'PASS' if ok else 'FAIL'}  B={B:2d},T={T:6d},{desc:16s}  {eu:.1e}  {ev:.1e}  {ea:.1e}"
        )

    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


# ============================================================
# Benchmarks
# ============================================================
def _bench(fn, warmup=20, iters=100):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _bench_fb(fn, alpha_v, u_v, v_v, warmup=20, iters=100):
    def step():
        a = alpha_v.detach().requires_grad_(True)
        u = u_v.detach().requires_grad_(True)
        v = v_v.detach().requires_grad_(True)
        fn(a, u, v).sum().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run_benchmarks():
    device = "cuda"
    hdr = (
        f"  {'Config':28s}"
        f"  {'naive fwd':>9s}  {'DL fwd':>7s}"
        f"  {'naive f+b':>9s}  {'DL f+b':>7s}"
        f"  {'speedup f+b':>11s}"
    )
    sep = "  " + "-" * 85

    print("--- Scaling T (B=8, alpha 0..20) ---")
    print(hdr)
    print(sep)
    B = 8
    for k in range(10, 21):
        T = 1 << k
        alpha = torch.linspace(0, 20, T, device=device)
        u = torch.randn(B, T, device=device)
        v = torch.randn(B, T, device=device)
        tf = _bench(lambda: _naive_prefix_suffix(alpha, u, v))
        tdf = _bench(lambda: prefix_suffix(alpha, u, v))
        tfb = _bench_fb(_naive_prefix_suffix, alpha, u, v)
        tdb = _bench_fb(prefix_suffix, alpha, u, v)
        desc = f"B={B},T=2^{k:<2d}={T:>8d}"
        print(
            f"  {desc:28s}  {tf:7.3f}ms  {tdf:5.3f}ms  {tfb:7.3f}ms  {tdb:5.3f}ms  {tdb/tfb:10.2f}x"
        )

    print()
    print("--- Scaling B (T=2^16, alpha 0..20) ---")
    print(hdr)
    print(sep)
    T = 1 << 16
    for B in [1, 2, 4, 8, 16, 32, 64, 128]:
        alpha = torch.linspace(0, 20, T, device=device)
        u = torch.randn(B, T, device=device)
        v = torch.randn(B, T, device=device)
        tf = _bench(lambda: _naive_prefix_suffix(alpha, u, v))
        tdf = _bench(lambda: prefix_suffix(alpha, u, v))
        tfb = _bench_fb(_naive_prefix_suffix, alpha, u, v)
        tdb = _bench_fb(prefix_suffix, alpha, u, v)
        desc = f"B={B:3d},T=2^16={T:>7d}"
        print(
            f"  {desc:28s}  {tf:7.3f}ms  {tdf:5.3f}ms  {tfb:7.3f}ms  {tdb:5.3f}ms  {tdb/tfb:10.2f}x"
        )
    print()


# ============================================================
if __name__ == "__main__":
    print("=== CORRECTNESS TESTS ===\n")
    run_correctness_tests()

    print("=== BENCHMARKI ===\n")
    run_benchmarks()
