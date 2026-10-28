"""
prefix_suffix.py
================

Self-contained, differentiable CUDA primitive for the *exponential-weighted*
prefix and suffix scans

    prefix(alpha, x)[b, i] = sum_{j <= i} exp(alpha[j] - alpha[i]) * x[b, j]
                           = exp(-alpha[i]) * cumsum_j (exp(alpha[j]) * x[b, j])
    suffix(alpha, x)[b, i] = sum_{j >= i} exp(alpha[i] - alpha[j]) * x[b, j].

Both float32 and float64 are supported in the same file: two `load_inline`
extensions are compiled at import time, and every call auto-dispatches on
the dtype of the first floating input — fp64 wins (any fp64 promotes the
whole call to fp64), fp16/bf16 are promoted to fp32.

Public API
----------
    prefix_suffix_sep(alpha, u, v=None)
        -> (prefix(alpha, u), suffix(alpha, v)).  v=None means v=u.
    prefix_suffix(alpha, u, v=None)
        -> prefix(alpha, u) + suffix(alpha, v).
    _GatherCombineFn.apply(P, S, w_l, w_r, c_l, c_r)
        -> w_l[j] * P[:, c_l[j]] + w_r[j] * S[:, c_r[j]]   (custom autograd)

Numerical trick
---------------
exp(alpha) overflows fp32 above ~88 (fp64 above ~709), so the sums are
done on a (n, x) representation that holds an integer power of 2 (n) and
a bounded mantissa (x), letting alpha range freely over thousands.  The
fused kernel multiplies by  ef = exp2(frac(alpha * log2 e))  on input and
divides by it on output, all in registers — no temporary (B, T) tensor.
The cross-tile aggregation uses Merrill & Garland's decoupled-lookback
algorithm, so the full scan is a single GPU pass.

Backward
--------
For   y = prefix(alpha, u) + suffix(alpha, v)   the gradient identities
    grad_u     = suffix(alpha, g),
    grad_v     = prefix(alpha, g),
    grad_alpha = sum_b ( u * grad_u - v * grad_v + g * (suffix - prefix) )
let us reuse the same CUDA scan in the backward pass.  A separate fused
kernel does the per-T reduction for `grad_alpha`.

Self-test
---------
    python prefix_suffix.py
"""
from __future__ import annotations

import math
import os
import sys

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline


_LOG2E = math.log2(math.e)   





_NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "--allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
]
if sys.platform == "win32":
    
    
    
    
    
    _NVCC_FLAGS += ["-DUSE_CUDA", "-Xcompiler", "/Zc:preprocessor"]





_CUDA_SRC_FP32 = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// (n, x): value = x * 2^n.  n integer exponent, x bounded mantissa in fp32.
struct Pair { int n; float x; };

// Associative combine: aggregate prefix `a` with next element `b`.
// Result keeps b's exponent; a's mantissa is rescaled into b's exponent.
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

// Forward (prefix) scan: left-to-right.
// Loads vector-wise (4 elements per thread via float4/int4),
// fuses input-side multiply by ef and output-side divide by ef in registers.
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

    int4   ni  = ((const int4*)n )[tid];
    float4 xi  = ((const float4*)x_b)[tid];
    float4 efi = ((const float4*)ef)[tid];
    Pair a0={ni.x, xi.x*efi.x}, a1={ni.y, xi.y*efi.y},
         a2={ni.z, xi.z*efi.z}, a3={ni.w, xi.w*efi.w};

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

    // Decoupled lookback: last thread publishes local sum, then walks back
    // through finished tiles aggregating their prefixes until it hits a
    // FLAG_PREFIX (definitive global prefix), then publishes our own.
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
            // Note: {0,0} is identity on the LEFT of combine; first hit
            // initialises agg, subsequent hits combine(p, agg).
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

    // Apply exclusive prefix from earlier tiles, then divide by ef.
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

// Backward (suffix) scan: right-to-left.  tid=0 owns the rightmost quad.
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
    int tid_rev = T4 - 1 - tid;
    int lane    = threadIdx.x & 31, warp = threadIdx.x >> 5;

    int4   ni  = ((const int4*)n)[tid_rev];
    float4 xi  = ((const float4*)x_b)[tid_rev];
    float4 efi = ((const float4*)ef)[tid_rev];
    // Reverse order within the quad.
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
    // a3 -> position base+0, a0 -> base+3 (reversed inside the quad).
    float4 yy;
    yy.x = a3.x / efi.x;
    yy.y = a2.x / efi.y;
    yy.z = a1.x / efi.z;
    yy.w = a0.x / efi.w;
    *((float4*)(oy_b+base))=yy;
}

// Launcher: runs both kernels and returns (prefix, suffix).  Requires T % 1024 == 0.
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
    dim3 grid(B, tiles);

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

// Fused per-T reduction for grad_alpha (saves one (B, T) temporary).
__global__ void grad_alpha_reduce_kernel(
    const float* __restrict__ u,   const float* __restrict__ sgp,
    const float* __restrict__ v,   const float* __restrict__ pgs,
    const float* __restrict__ gp,  const float* __restrict__ P,
    const float* __restrict__ gs,  const float* __restrict__ S,
    float* __restrict__ out, int B, int T)
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
    m.def("scan_ps_fused", &scan_ps_fused, "fused fp32 DL prefix+suffix scan");
    m.def("grad_alpha_reduce", &grad_alpha_reduce, "fused fp32 grad_alpha reduction");
}
"""







_CUDA_SRC_FP64 = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct Pair { int n; double x; };

__device__ __forceinline__ Pair combine(Pair a, Pair b) {
    return {b.n, b.x + ldexp(a.x, a.n - b.n)};
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

__global__ void dl_scan_fwd_fused(
    const int*    __restrict__ n,
    const double* __restrict__ x,
    const double* __restrict__ ef,
    double* out_y,
    int* state_flag, int* state_ln, double* state_lx,
    int* state_pn,   double* state_px,
    int T, int T4, int num_tiles)
{
    int batch = blockIdx.x, tile = blockIdx.y;
    int sid   = batch * num_tiles + tile;
    const double* x_b  = x     + (size_t)batch * T;
    double*       oy_b = out_y + (size_t)batch * T;
    extern __shared__ Pair shared[];
    int tid  = tile * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    // 4 scalar loads — nvcc coalesces into LD.128 issue pairs.
    int base = tid * 4;
    int n0 = n[base  ], n1 = n[base+1], n2 = n[base+2], n3 = n[base+3];
    double x0 = x_b[base  ], x1 = x_b[base+1], x2 = x_b[base+2], x3 = x_b[base+3];
    double e0 = ef[base  ], e1 = ef[base+1], e2 = ef[base+2], e3 = ef[base+3];
    Pair a0={n0, x0*e0}, a1={n1, x1*e1}, a2={n2, x2*e2}, a3={n3, x3*e3};

    a1=combine(a0,a1); a2=combine(a1,a2); a3=combine(a2,a3);
    Pair val = warp_scan(a3);
    Pair we; we.n=__shfl_up_sync(0xffffffff,val.n,1); we.x=__shfl_up_sync(0xffffffff,val.x,1);
    if (lane==0) we={0,0.0};
    a0=combine(we,a0); a1=combine(we,a1); a2=combine(we,a2); a3=combine(we,a3);
    if (lane==31) shared[warp]=val;
    __syncthreads();
    if (warp==0) {
        int nw=blockDim.x>>5; Pair v=(lane<nw)?shared[lane]:Pair{0,0.0};
        v=warp_scan(v); shared[lane]=v;
    }
    __syncthreads();
    if (warp>0) { Pair off=shared[warp-1]; a0=combine(off,a0); a1=combine(off,a1); a2=combine(off,a2); a3=combine(off,a3); }

    if (threadIdx.x==blockDim.x-1) {
        Pair local_sum=a3;
        state_ln[sid]=local_sum.n; state_lx[sid]=local_sum.x;
        __threadfence();
        atomicExch(&state_flag[sid], FLAG_LOCAL);

        Pair exc={0,0.0};
        if (tile>0) {
            volatile int*    vflag=(volatile int*)   state_flag+batch*num_tiles;
            volatile int*    vln  =(volatile int*)   state_ln  +batch*num_tiles;
            volatile double* vlx  =(volatile double*)state_lx  +batch*num_tiles;
            volatile int*    vpn  =(volatile int*)   state_pn  +batch*num_tiles;
            volatile double* vpx  =(volatile double*)state_px  +batch*num_tiles;
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
    oy_b[base+0] = a0.x / e0;
    oy_b[base+1] = a1.x / e1;
    oy_b[base+2] = a2.x / e2;
    oy_b[base+3] = a3.x / e3;
}

__global__ void dl_scan_bwd_fused(
    const int*    __restrict__ n,
    const double* __restrict__ x,
    const double* __restrict__ ef,
    double* out_y,
    int* state_flag, int* state_ln, double* state_lx,
    int* state_pn,   double* state_px,
    int T, int T4, int num_tiles)
{
    int batch = blockIdx.x, tile = blockIdx.y;
    int sid   = batch * num_tiles + tile;
    const double* x_b  = x     + (size_t)batch * T;
    double*       oy_b = out_y + (size_t)batch * T;
    extern __shared__ Pair shared[];

    int tid     = tile * blockDim.x + threadIdx.x;
    int tid_rev = T4 - 1 - tid;
    int lane    = threadIdx.x & 31, warp = threadIdx.x >> 5;

    int base_rev = tid_rev * 4;
    int n0 = n[base_rev+3], n1 = n[base_rev+2], n2 = n[base_rev+1], n3 = n[base_rev+0];
    double x0 = x_b[base_rev+3], x1 = x_b[base_rev+2], x2 = x_b[base_rev+1], x3 = x_b[base_rev+0];
    double e0 = ef[base_rev+3], e1 = ef[base_rev+2], e2 = ef[base_rev+1], e3 = ef[base_rev+0];
    Pair a0={n0, x0*e0}, a1={n1, x1*e1}, a2={n2, x2*e2}, a3={n3, x3*e3};

    a1=combine(a0,a1); a2=combine(a1,a2); a3=combine(a2,a3);

    Pair val = warp_scan(a3);
    Pair we; we.n=__shfl_up_sync(0xffffffff,val.n,1); we.x=__shfl_up_sync(0xffffffff,val.x,1);
    if (lane==0) we={0,0.0};
    a0=combine(we,a0); a1=combine(we,a1); a2=combine(we,a2); a3=combine(we,a3);
    if (lane==31) shared[warp]=val;
    __syncthreads();
    if (warp==0) {
        int nw=blockDim.x>>5; Pair v=(lane<nw)?shared[lane]:Pair{0,0.0};
        v=warp_scan(v); shared[lane]=v;
    }
    __syncthreads();
    if (warp>0) { Pair off=shared[warp-1]; a0=combine(off,a0); a1=combine(off,a1); a2=combine(off,a2); a3=combine(off,a3); }

    if (threadIdx.x==blockDim.x-1) {
        Pair local_sum=a3;
        state_ln[sid]=local_sum.n; state_lx[sid]=local_sum.x;
        __threadfence();
        atomicExch(&state_flag[sid], FLAG_LOCAL);
        Pair exc={0,0.0};
        if (tile>0) {
            volatile int*    vflag=(volatile int*)   state_flag+batch*num_tiles;
            volatile int*    vln  =(volatile int*)   state_ln  +batch*num_tiles;
            volatile double* vlx  =(volatile double*)state_lx  +batch*num_tiles;
            volatile int*    vpn  =(volatile int*)   state_pn  +batch*num_tiles;
            volatile double* vpx  =(volatile double*)state_px  +batch*num_tiles;
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
    oy_b[base_rev+0] = a3.x / e3;
    oy_b[base_rev+1] = a2.x / e2;
    oy_b[base_rev+2] = a1.x / e1;
    oy_b[base_rev+3] = a0.x / e0;
}

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
    auto of=torch::TensorOptions().dtype(torch::kFloat64).device(x_fwd.device());
    int smem=33*(int)sizeof(Pair);
    dim3 grid(B, tiles);

    auto sf0=torch::zeros({ns},oi); auto ln0=torch::empty({ns},oi);
    auto lx0=torch::empty({ns},of); auto pn0=torch::empty({ns},oi); auto px0=torch::empty({ns},of);
    auto sf1=torch::zeros({ns},oi); auto ln1=torch::empty({ns},oi);
    auto lx1=torch::empty({ns},of); auto pn1=torch::empty({ns},oi); auto px1=torch::empty({ns},of);

    dl_scan_fwd_fused<<<grid,threads,smem>>>(
        n_fwd.data_ptr<int>(), x_fwd.data_ptr<double>(), ef_fwd.data_ptr<double>(),
        out_fwd.data_ptr<double>(),
        sf0.data_ptr<int>(), ln0.data_ptr<int>(), lx0.data_ptr<double>(),
        pn0.data_ptr<int>(), px0.data_ptr<double>(), T, T4, tiles);

    dl_scan_bwd_fused<<<grid,threads,smem>>>(
        n_bwd.data_ptr<int>(), x_bwd.data_ptr<double>(), ef_bwd.data_ptr<double>(),
        out_bwd.data_ptr<double>(),
        sf1.data_ptr<int>(), ln1.data_ptr<int>(), lx1.data_ptr<double>(),
        pn1.data_ptr<int>(), px1.data_ptr<double>(), T, T4, tiles);

    return {out_fwd, out_bwd};
}

__global__ void grad_alpha_reduce_kernel(
    const double* __restrict__ u,   const double* __restrict__ sgp,
    const double* __restrict__ v,   const double* __restrict__ pgs,
    const double* __restrict__ gp,  const double* __restrict__ P,
    const double* __restrict__ gs,  const double* __restrict__ S,
    double* __restrict__ out, int B, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    double acc = 0.0;
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
        u.data_ptr<double>(), sgp.data_ptr<double>(),
        v.data_ptr<double>(), pgs.data_ptr<double>(),
        gp.data_ptr<double>(), P.data_ptr<double>(),
        gs.data_ptr<double>(), S.data_ptr<double>(),
        out.data_ptr<double>(), B, T);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_ps_fused", &scan_ps_fused, "fused fp64 DL prefix+suffix scan");
    m.def("grad_alpha_reduce", &grad_alpha_reduce, "fused fp64 grad_alpha reduction");
}
"""





print("Compiling prefix_suffix fp32 extension...", end=" ", flush=True)
_ext_fp32 = load_inline(
    name="prefix_suffix_dl_fused",
    cpp_sources=[""],
    cuda_sources=[_CUDA_SRC_FP32],
    extra_cuda_cflags=_NVCC_FLAGS,
    verbose=False,
)
print("OK")

print("Compiling prefix_suffix fp64 extension...", end=" ", flush=True)
_ext_fp64 = load_inline(
    name="prefix_suffix_dl_fused_fp64",
    cpp_sources=[""],
    cuda_sources=[_CUDA_SRC_FP64],
    extra_cuda_cflags=_NVCC_FLAGS,
    verbose=False,
)
print("OK")





_THREADS = 256
_PAD_TO  = _THREADS * 4   


def _backend_for(dtype):
    """Return (extension module, target float dtype) for the requested dtype."""
    return (_ext_fp64, torch.float64) if dtype == torch.float64 \
           else (_ext_fp32, torch.float32)


def _common_dtype(*tensors):
    """Promote: any fp64 -> fp64, otherwise fp32 (fp16/bf16 promoted)."""
    if any(t is not None and t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def _cast(dtype, *tensors):
    """Cast every floating tensor to `dtype` (None and same-dtype passthrough)."""
    out = []
    for t in tensors:
        if t is None:
            out.append(None)
        elif t.dtype.is_floating_point and t.dtype != dtype:
            out.append(t.to(dtype))
        else:
            out.append(t)
    return tuple(out)


def _precompute(alpha: torch.Tensor):
    """Decompose alpha into (ef, n_fwd, ef_neg, n_bwd) for both directions.

    ef     = exp2(frac( alpha * log2 e))                  (mantissa < 2)
    n_fwd  = floor(  alpha * log2 e)   int32 non-decreasing
    ef_neg = exp2(frac(-alpha * log2 e))
    n_bwd  = floor( -alpha * log2 e)   int32 non-increasing
    """
    dtype = alpha.dtype
    ap  =  alpha * _LOG2E
    n_f = ap.floor().to(torch.int32)
    ef  = torch.exp2(ap - n_f.to(dtype))
    an  = -ap
    n_b = an.floor().to(torch.int32)
    ef_neg = torch.exp2(an - n_b.to(dtype))
    return ef, n_f, ef_neg, n_b


def _pad_pair(x: torch.Tensor, n1d: torch.Tensor):
    """Pad x [B, T] and n1d [T] so that T is a multiple of _PAD_TO (1024)."""
    T = x.size(1)
    rem = T % _PAD_TO
    if rem == 0:
        return x, n1d, T
    pad = _PAD_TO - rem
    x_p = torch.nn.functional.pad(x, (0, pad))
    n_p = torch.cat([n1d, n1d[-1:].expand(pad)])
    return x_p, n_p, T


def _pad_ef(ef: torch.Tensor, T: int):
    """Pad ef [T] (replication of last value) to multiple of _PAD_TO."""
    rem = T % _PAD_TO
    if rem == 0:
        return ef
    pad = _PAD_TO - rem
    return torch.cat([ef, ef[-1:].expand(pad)])


def _ps_raw(ext, ef, n_fwd, ef_neg, n_bwd, u, v):
    """Raw fused-kernel call returning the already-decoded (P, S) pair."""
    T = u.size(1)
    u_pad, n_f_pad, T_orig = _pad_pair(u, n_fwd)
    v_pad, n_b_pad, _      = _pad_pair(v, n_bwd)
    ef_pad     = _pad_ef(ef,     T)
    ef_neg_pad = _pad_ef(ef_neg, T)
    out_f, out_b = ext.scan_ps_fused(
        n_f_pad.contiguous(), ef_pad.contiguous(),     u_pad.contiguous(),
        n_b_pad.contiguous(), ef_neg_pad.contiguous(), v_pad.contiguous(),
    )
    return out_f[:, :T_orig], out_b[:, :T_orig]





class _PrefSuffDiffFn(Function):
    """Returns (prefix(alpha, u), suffix(alpha, v)).  Auto-dispatches on dtype."""

    @staticmethod
    def forward(ctx, alpha, u, v):
        target = _common_dtype(alpha, u, v)
        ext, _ = _backend_for(target)
        a, u_, v_ = _cast(target, alpha, u, v)
        ef, n_fwd, ef_neg, n_bwd = _precompute(a)
        P, S = _ps_raw(ext, ef, n_fwd, ef_neg, n_bwd, u_, v_)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u_, v_, P, S)
        ctx.target = target
        ctx.orig_dtype = alpha.dtype
        return P, S

    @staticmethod
    def backward(ctx, gP, gS):
        ef, n_fwd, ef_neg, n_bwd, u, v, P, S = ctx.saved_tensors
        target = ctx.target
        ext, _ = _backend_for(target)
        gP_t, gS_t = _cast(target, gP, gS)

        
        prefix_gS, suffix_gP = _ps_raw(ext, ef, n_fwd, ef_neg, n_bwd, gS_t, gP_t)

        grad_u = suffix_gP if ctx.needs_input_grad[1] else None
        grad_v = prefix_gS if ctx.needs_input_grad[2] else None

        grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_alpha = ext.grad_alpha_reduce(
                u.contiguous(), suffix_gP.contiguous(),
                v.contiguous(), prefix_gS.contiguous(),
                gP_t.contiguous(), P.contiguous(),
                gS_t.contiguous(), S.contiguous(),
            )
        return grad_alpha, grad_u, grad_v


def prefix_suffix_sep(alpha, u, v=None):
    """Differentiable (prefix(alpha, u), suffix(alpha, v)).  v=None means v=u.

    alpha : (T,)            non-decreasing, CUDA, fp32 / fp64
    u, v  : (B, T) or (T,)  same dtype as alpha (or promotable)
    """
    if v is None:
        v = u
    squeeze = u.dim() == 1
    if squeeze:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0) if v is not u else u
    P, S = _PrefSuffDiffFn.apply(alpha, u, v)
    if squeeze:
        P, S = P.squeeze(0), S.squeeze(0)
    return P, S


def prefix_suffix(alpha, u, v=None):
    """prefix(alpha, u) + suffix(alpha, v)  as a single tensor."""
    if v is None:
        v = u
    P, S = prefix_suffix_sep(alpha, u, v)
    return P + S







class _GatherCombineFn(Function):
    """y[b, j] = w_l[j] * P[b, c_l[j]] + w_r[j] * S[b, c_r[j]]  (autograd-friendly)."""

    @staticmethod
    def forward(ctx, P, S, w_l, w_r, c_l, c_r):
        P_sel = P.index_select(1, c_l)              
        S_sel = S.index_select(1, c_r)              
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

        if ctx.needs_input_grad[0]:
            grad_P = torch.zeros(B, n, device=grad_y.device, dtype=grad_y.dtype)
            grad_P.scatter_add_(1, c_l.unsqueeze(0).expand(B, k),
                                grad_y * w_l.unsqueeze(0))
        if ctx.needs_input_grad[1]:
            grad_S = torch.zeros(B, n, device=grad_y.device, dtype=grad_y.dtype)
            grad_S.scatter_add_(1, c_r.unsqueeze(0).expand(B, k),
                                grad_y * w_r.unsqueeze(0))

        return grad_P, grad_S, grad_w_l, grad_w_r, None, None





def _smoke_test():
    assert torch.cuda.is_available(), "CUDA required (this is the CUDA backend)."
    device = "cuda"
    torch.manual_seed(0)

    print("=" * 60)
    print("prefix_suffix.py: smoke test (fp32 + fp64 dispatch)")
    print("=" * 60)

    passed = failed = 0

    def chk(name, got, ref, tol):
        nonlocal passed, failed
        ref64 = ref.detach().double()
        got64 = got.detach().double()
        err = (got64 - ref64).abs().max().item() / (ref64.abs().max().item() + 1e-30)
        ok = err < tol
        passed += ok; failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:48s} dtype={got.dtype} err={err:.2e}")

    def naive_prefix(alpha, u):
        ea = torch.exp(alpha); ena = torch.exp(-alpha)
        return (u * ea.unsqueeze(0)).cumsum(dim=1) * ena.unsqueeze(0)

    def naive_suffix(alpha, v):
        ea = torch.exp(alpha); ena = torch.exp(-alpha)
        return ((v * ena.unsqueeze(0)).flip(-1).cumsum(dim=1).flip(-1)) * ea.unsqueeze(0)

    
    print("\n--- prefix / suffix  vs naive (fp32) ---")
    for B, T in [(1, 256), (4, 1024), (8, 4096)]:
        alpha = torch.linspace(0, 5, T, device=device, dtype=torch.float32)
        u = torch.randn(B, T, device=device, dtype=torch.float32)
        v = torch.randn(B, T, device=device, dtype=torch.float32)
        P, S = prefix_suffix_sep(alpha, u, v)
        chk(f"prefix B={B}, T={T}", P, naive_prefix(alpha, u), 5e-4)
        chk(f"suffix B={B}, T={T}", S, naive_suffix(alpha, v), 5e-4)

    
    print("\n--- prefix / suffix  vs naive (fp64) ---")
    for B, T in [(2, 1024), (4, 4096)]:
        alpha = torch.linspace(0, 8, T, device=device, dtype=torch.float64)
        u = torch.randn(B, T, device=device, dtype=torch.float64)
        v = torch.randn(B, T, device=device, dtype=torch.float64)
        P, S = prefix_suffix_sep(alpha, u, v)
        chk(f"prefix fp64 B={B}, T={T}", P, naive_prefix(alpha, u), 1e-12)
        chk(f"suffix fp64 B={B}, T={T}", S, naive_suffix(alpha, v), 1e-12)

    
    print("\n--- prefix_suffix (combined) ---")
    alpha = torch.linspace(0, 5, 1024, device=device, dtype=torch.float32)
    u = torch.randn(4, 1024, device=device, dtype=torch.float32)
    v = torch.randn(4, 1024, device=device, dtype=torch.float32)
    chk("prefix_suffix fp32",
        prefix_suffix(alpha, u, v),
        naive_prefix(alpha, u) + naive_suffix(alpha, v), 5e-4)

    
    print("\n--- mixed dtypes (fp32+fp64 -> fp64) ---")
    a64 = torch.linspace(0, 5, 1024, device=device, dtype=torch.float64)
    u32 = torch.randn(2, 1024, device=device, dtype=torch.float32)
    P, _ = prefix_suffix_sep(a64, u32)
    assert P.dtype == torch.float64
    chk("prefix (a64, u32) -> fp64", P,
        naive_prefix(a64, u32.double()), 1e-12)

    
    print("\n--- _GatherCombineFn ---")
    P = torch.randn(3, 256, device=device, dtype=torch.float32)
    S = torch.randn(3, 256, device=device, dtype=torch.float32)
    w_l = torch.randn(64, device=device, dtype=torch.float32)
    w_r = torch.randn(64, device=device, dtype=torch.float32)
    c_l = torch.randint(0, 256, (64,), device=device, dtype=torch.int64)
    c_r = torch.randint(0, 256, (64,), device=device, dtype=torch.int64)
    Y = _GatherCombineFn.apply(P, S, w_l, w_r, c_l, c_r)
    Y_ref = w_l.unsqueeze(0) * P[:, c_l] + w_r.unsqueeze(0) * S[:, c_r]
    chk("_GatherCombineFn fp32", Y, Y_ref, 1e-6)

    
    print("\n--- gradient (fp32 vs naive) ---")
    for B, T in [(1, 256), (4, 1024)]:
        alpha = torch.linspace(0, 5, T, device=device, dtype=torch.float32)
        u_v = torch.randn(B, T, device=device, dtype=torch.float32)
        v_v = torch.randn(B, T, device=device, dtype=torch.float32)

        a1 = alpha.clone().requires_grad_(True)
        u1 = u_v.clone().requires_grad_(True)
        v1 = v_v.clone().requires_grad_(True)
        prefix_suffix(a1, u1, v1).sum().backward()

        a2 = alpha.clone().requires_grad_(True)
        u2 = u_v.clone().requires_grad_(True)
        v2 = v_v.clone().requires_grad_(True)
        (naive_prefix(a2, u2) + naive_suffix(a2, v2)).sum().backward()

        def re(a, b):
            return ((a - b).abs() / b.abs().clamp(min=1e-6)).max().item()
        eu = re(u1.grad, u2.grad); ev = re(v1.grad, v2.grad); ea = re(a1.grad, a2.grad)
        ok = eu < 1e-2 and ev < 1e-2 and ea < max(2e-2, T * 1e-5)
        passed += ok; failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  grad B={B}, T={T}  eu={eu:.1e} ev={ev:.1e} ea={ea:.1e}")

    print(f"\n{'='*60}\nResult: {passed} PASS, {failed} FAIL\n{'='*60}")
    return failed == 0


if __name__ == "__main__":
    _smoke_test()
