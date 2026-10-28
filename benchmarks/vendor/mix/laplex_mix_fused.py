"""
laplex_mix_fused.py
===================

Fully-fused CUDA kernel for the **shared-x multi-kernel Laplace mixture**

    y[b, p] = sum_{l=1..L}  gamma_l * sum_j exp(-|a_l[p] - b_l[j]|) * x[b, j]

i.e. ONE shared activation ``x`` is hit by ``L`` independent Laplace
kernels ``K(a_l, b_l)`` and the results are blended by signed weights
``gamma_l``.  This is the "function 2" semantics of the sibling
``laplex_multi.py`` (NOT the per-row Sortformer variant) -- here every
kernel sees the SAME ``x``.

Why a new kernel?
-----------------
The reference ``Laplex_X_grouped_mix`` scan path materialises three big
intermediates of size ``(L, B, k)`` / ``(L, B, n)`` (the per-kernel
prefix/suffix scans and the per-kernel outputs ``y_per_l``).  For the
target workload -- replacing every ``nn.Linear`` inside a ViT-S MLP,
where the effective batch is ``B = batch * tokens ~ 12k`` and ``L`` is
4-32 -- those intermediates are multiple GB and dominate both memory
and bandwidth.

This module fuses the entire pipeline into single CUDA launches:

* **forward**  : one block per ``x``-row.  ``x[b, :]`` is loaded ONCE into
  shared memory and reused across all ``L`` kernels.  For each ``l`` the
  block builds the weighted prefix/suffix arrays, runs an in-block scan
  (length ``k``), evaluates all ``n`` queries by a gather at the
  precomputed search-sorted boundaries, and accumulates the
  ``gamma_l``-weighted contribution into an in-smem ``y`` accumulator.
  No ``(L, B, *)`` tensor is ever written to DRAM.

* **backward** : ONE merged launch ``mix_bwd`` (one block per
  ``(l, b-strip)``).  Per ``(l,b)`` it recomputes the forward scans P,S AND
  the adjoint scans tildeP,tildeS (4 scans, no stash), emits ``grad_x`` via a
  low-contention global ``atomicAdd`` and accumulates the parameter grads in
  shared memory, flushed once per block.  Recomputing the scans (rather than
  stashing them from forward) keeps peak memory at ``O(B(k+n))`` -- no
  ``(L, B, k)`` intermediate.

Numerical model
---------------
All exponentials are precomputed on the PyTorch side with a **per-kernel
mid-range shift** ``r_l`` folded in: the kernel only ever sees
``exp(b_l[j] - r_l)`` and ``exp(-(a_l[p] - r_l))`` etc.  Because
``exp(-|a - b|)`` depends only on the difference, ``r_l`` cancels
exactly in the output, while keeping every stored exponential inside
FP32 range as long as the per-kernel coordinate spread is ``< ~80``
(always true for LayerNorm-bounded activations).  This makes the
in-block scan a plain FP32 accumulation -- no integer-exponent ``Pair``
bookkeeping needed (cf. ``prefix_suffix_multi.py``), which is both
simpler and faster.  The result is validated bit-close (FP32 noise) to
the dense reference and to the existing ``Laplex_X_grouped_mix``.

Public API
----------
    laplex_mix(x, a, b, gamma)        functional form, differentiable
                                      through (x, a, b, gamma)
    LaplexMixLinear(k, n, L=...)      nn.Module drop-in for nn.Linear(k, n)

Shapes
------
    x     : (..., k)        leading dims flattened to B = prod(...)
    a     : (L, n)          per-kernel query coords
    b     : (L, k)          per-kernel anchor coords
    gamma : (L,)            signed mixing weights
    y     : (..., n)
"""
from __future__ import annotations

# NOTE: torch MUST be imported before numpy on Windows (MKL/OpenMP DLL
# load-order, see project memory) -- and this module never imports numpy.
import os
import sys

import torch
from torch import Tensor
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline


# ---------------------------------------------------------------------------
# CUDA toolchain.  On this machine there is NO standalone CUDA Toolkit --
# everything CUDA-side lives in the conda ``ml`` env and CUDA_HOME is set by
# the launching .bat (run_verify.bat) to %CONDA_PREFIX%\Library.  A legacy
# standalone path is honoured only if it happens to exist.
# ---------------------------------------------------------------------------
_CUDA_LEGACY = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
if os.path.isdir(_CUDA_LEGACY):
    os.environ.setdefault("CUDA_HOME", _CUDA_LEGACY)
    os.environ.setdefault("CUDA_PATH", _CUDA_LEGACY)
    os.environ["PATH"] = os.path.join(_CUDA_LEGACY, "bin") + ";" + os.environ.get("PATH", "")
    import torch.utils.cpp_extension as _cext
    _cext.CUDA_HOME = _CUDA_LEGACY

# Verified working flag set for Windows + PyTorch 2.11 + cu128 (see project
# memory "Python + CUDA setup"): the 14.39 host compiler + the -DUSE_CUDA /
# /Zc:preprocessor pair bypass the compiled_autograd.h C2872 parse bug.
_NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "--allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
]
if sys.platform == "win32":
    _MSVC_1439 = (r"C:\Program Files (x86)\Microsoft Visual Studio\2022"
                  r"\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64")
    if os.path.isdir(_MSVC_1439):
        _NVCC_FLAGS += ["-ccbin", _MSVC_1439]
    _NVCC_FLAGS += ["-DUSE_CUDA", "-Xcompiler", "/Zc:preprocessor"]
else:
    # Linux (e.g. GMUM cu132 / sm_120): the MSVC-only flags above are skipped;
    # -DUSE_CUDA is the verified-working extra (TORCH_CUDA_ARCH_LIST injects the
    # -gencode, so don't add arch flags here).  See project memory gmum-singularity.
    _NVCC_FLAGS += ["-DUSE_CUDA"]


# ===========================================================================
# CUDA source
# ===========================================================================
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

// ----------------------------------------------------------------------
// In-block inclusive prefix scan of buf[0..len) (FP32, in place).
//   wsum : shared scratch of >= blockDim.x/32 floats (warp partials).
// Each thread owns a contiguous chunk; warp-shuffle scan of the 32 chunk
// totals, then warp 0 scans the per-warp totals.  Only TWO __syncthreads
// (vs Hillis-Steele's 2*log2(blockDim)).  Assumes blockDim a multiple of 32.
// ----------------------------------------------------------------------
__device__ __forceinline__ void block_prefix_inc(float* buf, int len, float* wsum) {
    int tid = threadIdx.x, nb = blockDim.x, lane = tid & 31, warp = tid >> 5;
    int nw = (nb + 31) >> 5;
    int per = (len + nb - 1) / nb;
    int lo = tid * per;
    int hi = lo + per; if (hi > len) hi = len;
    float acc = 0.f;
    for (int i = lo; i < hi; ++i) { acc += buf[i]; buf[i] = acc; }
    float v = acc;                                   // warp-inclusive scan
    for (int off = 1; off < 32; off <<= 1) {
        float t = __shfl_up_sync(0xffffffff, v, off);
        if (lane >= off) v += t;
    }
    if (lane == 31) wsum[warp] = v;                  // per-warp total
    __syncthreads();
    if (warp == 0) {
        float wv = (lane < nw) ? wsum[lane] : 0.f;
        for (int off = 1; off < 32; off <<= 1) {
            float t = __shfl_up_sync(0xffffffff, wv, off);
            if (lane >= off) wv += t;
        }
        if (lane < nw) wsum[lane] = wv;              // inclusive prefix of warp totals
    }
    __syncthreads();
    float warp_excl = (warp > 0) ? wsum[warp - 1] : 0.f;   // sum of all earlier warps
    // (v - acc) = this thread's EXCLUSIVE prefix within its warp: v is the
    // warp-inclusive scan of chunk totals, acc is this thread's own chunk total.
    float excl = warp_excl + (v - acc);              // total exclusive offset for my chunk
    for (int i = lo; i < hi; ++i) buf[i] += excl;
    __syncthreads();
}

// In-block inclusive SUFFIX scan: buf[i] <- sum_{j>=i} buf[j].
__device__ __forceinline__ void block_suffix_inc(float* buf, int len, float* wsum) {
    int tid = threadIdx.x, nb = blockDim.x, lane = tid & 31, warp = tid >> 5;
    int nw = (nb + 31) >> 5;
    int per = (len + nb - 1) / nb;
    int lo = tid * per;
    int hi = lo + per; if (hi > len) hi = len;
    float acc = 0.f;
    for (int i = hi - 1; i >= lo; --i) { acc += buf[i]; buf[i] = acc; }
    float v = acc;                                   // warp-inclusive suffix scan
    for (int off = 1; off < 32; off <<= 1) {
        float t = __shfl_down_sync(0xffffffff, v, off);
        if (lane + off < 32) v += t;
    }
    if (lane == 0) wsum[warp] = v;                   // per-warp total
    __syncthreads();
    if (warp == 0) {
        float wv = (lane < nw) ? wsum[lane] : 0.f;
        for (int off = 1; off < 32; off <<= 1) {
            float t = __shfl_down_sync(0xffffffff, wv, off);
            if (lane + off < 32) wv += t;
        }
        if (lane < nw) wsum[lane] = wv;              // inclusive suffix of warp totals
    }
    __syncthreads();
    float warp_excl = (warp + 1 < nw) ? wsum[warp + 1] : 0.f;   // sum of all later warps
    // (v - acc) = this thread's exclusive SUFFIX within its warp (mirror of the prefix case).
    float excl = warp_excl + (v - acc);              // total exclusive offset for my chunk
    for (int i = lo; i < hi; ++i) buf[i] += excl;
    __syncthreads();
}

// ======================================================================
// FORWARD : grid = (B), block = BLK  (one BLOCK per x-row, block-scan).
//   Block-per-row distributes the length-k scan across all BLK threads,
//   which beats warp-per-row when k is large (e.g. fc2 k=1536); the warp
//   variant is kept (warp_*_inc) for potential small-k dispatch.
//   x     : (B, k)
//   ebp   : (L, k)  = exp(b_sorted - r_l)
//   ebn   : (L, k)  = exp(-(b_sorted - r_l))
//   perm  : (L, k)  int32  sort order of b_l (gather index into x[b,:])
//   cl,cr : (L, n)  int32  clamped left/right boundary indices in [0,k)
//   ean   : (L, n)  = exp(-(a - r_l)) * gate_left
//   eap   : (L, n)  = exp( (a - r_l)) * gate_right
//   gamma : (L,)
//   y     : (B, n)
// smem  : (3k + n + BLK) floats
// ======================================================================
__global__ void mix_fwd(
    const float* __restrict__ x,
    const float* __restrict__ ebp, const float* __restrict__ ebn,
    const int*   __restrict__ perm,
    const int*   __restrict__ cl,  const int*   __restrict__ cr,
    const float* __restrict__ ean, const float* __restrict__ eap,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    int B, int L, int k, int n)
{
    int b = blockIdx.x, tid = threadIdx.x, nb = blockDim.x;
    extern __shared__ float sm[];
    float* xrow = sm;            // k
    float* bufP = xrow + k;      // k
    float* bufS = bufP + k;      // k
    float* yacc = bufS + k;      // n
    float* sp   = yacc + n;      // BLK  warp-partial scratch for the block scans

    const float* xb = x + (size_t)b * k;
    for (int i = tid; i < k; i += nb) xrow[i] = xb[i];
    for (int p = tid; p < n; p += nb) yacc[p] = 0.f;
    __syncthreads();

    for (int l = 0; l < L; ++l) {
        const float* ebp_l = ebp + (size_t)l * k;
        const float* ebn_l = ebn + (size_t)l * k;
        const int*   pm_l  = perm + (size_t)l * k;
        for (int j = tid; j < k; j += nb) {
            float xv = xrow[pm_l[j]];
            bufP[j] = ebp_l[j] * xv;
            bufS[j] = ebn_l[j] * xv;
        }
        __syncthreads();
        block_prefix_inc(bufP, k, sp);   // bufP[i] = sum_{j<=i} ebp*xs
        block_suffix_inc(bufS, k, sp);   // bufS[i] = sum_{j>=i} ebn*xs

        const int*   cl_l  = cl  + (size_t)l * n;
        const int*   cr_l  = cr  + (size_t)l * n;
        const float* ean_l = ean + (size_t)l * n;
        const float* eap_l = eap + (size_t)l * n;
        float gl = gamma[l];
        // The sort + searchsorted partition resolves the |a-b| absolute value:
        //   sum_j exp(-|a_p - b_j|) x_j
        //     = exp(-(a_p-r)) * sum_{b_j <= a_p} exp(b_j-r)  x_j   <- prefix P, read at cl
        //     + exp( (a_p-r)) * sum_{b_j >  a_p} exp(-(b_j-r)) x_j  <- suffix S, read at cr
        // ean = exp(-(a-r))*gate_l and eap = exp((a-r))*gate_r fold in the exps+gates;
        // cl = c (last anchor <= a) and cr = c+1 are the two sides of the boundary.
        for (int p = tid; p < n; p += nb)
            yacc[p] += gl * (ean_l[p] * bufP[cl_l[p]] + eap_l[p] * bufS[cr_l[p]]);
        __syncthreads();             // before reusing bufP/bufS next l
    }

    float* yb = y + (size_t)b * n;
    for (int p = tid; p < n; p += nb) yb[p] = yacc[p];
}

// ======================================================================
// BACKWARD (single merged kernel) : grid = (L, nstrips), block = BLK.
//   Each block owns one l and a grid-stride strip of b-rows.  Per (l,b) it
//   recomputes the forward scans P,S AND the adjoint scans tildeP,tildeS
//   (4 scans total -- no redundant work, vs the earlier 2-kernel split's
//   6).  It emits:
//     * grad_x[b, perm_l[j]]  via GLOBAL atomicAdd (only L collisions per
//       output element, so contention is low), and
//     * grad_ebp/ebn/ean/eap/gamma accumulated in smem over the strip,
//       flushed once per block via atomicAdd.
//   P,S are recomputed rather than stashed from forward to keep memory at
//   O(B(k+n)) -- no (L,B,k) intermediate.
// smem : (7k + 3n + BLK) floats
// ======================================================================
// Backward is REGISTER-bound on Ampere/Ada/Hopper (A100/4090/H100): the
// natural register use caps it at 2 blocks/SM (25% occ).  Forcing >=4
// blocks via __launch_bounds__ lifts it to 50% and ~halves backward
// latency on the ViT-S shapes (fc1 10.1->5.7 ms, profiled on A100).  On
// sm_120 (5090/5060) the backward is SMEM-bound, so the register cap would
// only risk spills with no occupancy gain -> skip it there.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 1000)
#define LAPLEX_BWD_LB __launch_bounds__(256, 4)
#else
#define LAPLEX_BWD_LB
#endif
__global__ void LAPLEX_BWD_LB mix_bwd(
    const float* __restrict__ x,                       // (B, k)
    const float* __restrict__ g,                       // (B, n)
    const float* __restrict__ ebp, const float* __restrict__ ebn,
    const int*   __restrict__ perm,
    const int*   __restrict__ cl,  const int*   __restrict__ cr,
    const float* __restrict__ ean, const float* __restrict__ eap,
    const float* __restrict__ gamma,
    float* __restrict__ grad_x,
    float* __restrict__ grad_ebp, float* __restrict__ grad_ebn,
    float* __restrict__ grad_ean, float* __restrict__ grad_eap,
    float* __restrict__ grad_gamma,
    int B, int L, int k, int n, int prof)
{
    // prof bitmask (0 = full/correct; nonzero gives WRONG grads, timing only):
    //   1 skip adjoint scatter, 2 skip grad_x atomics,
    //   4 skip the 2 forward (P,S) scans, 8 skip the 2 tilde scans.
    int l = blockIdx.x, strip = blockIdx.y, nstrips = gridDim.y;
    int tid = threadIdx.x, nb = blockDim.x;

    // Single-pass.  (A 2-pass "alias Pp/Ss into bP/bS" smem-reduction --
    // 7k+3n -> 5k+3n -- was tried to lift occupancy; profiling showed it
    // crossed NO blocks/SM threshold on any target shape, e.g. fc2 47->35KB
    // but 3 blocks/SM needs <=33KB, so it only added a 2nd n-pass and
    // regressed.  Occupancy is quantized; partial smem cuts don't pay.)
    extern __shared__ float sm[];
    float* xs   = sm;            // k
    float* Pp   = xs   + k;      // k   prefix P[i]
    float* Ss   = Pp   + k;      // k   suffix S[i]
    float* bP   = Ss   + k;      // k   barP (ean*grad scattered at cl) -> tildeP after scan
    float* bS   = bP   + k;      // k   barS (eap*grad scattered at cr) -> tildeS after scan
    float* grow = bS   + k;      // n   this row's upstream grad g[b,:] (grad-of-output)
    float* gebp = grow + n;      // k   accumulators (over the strip)
    float* gebn = gebp + k;      // k
    float* gean = gebn + k;      // n
    float* geap = gean + n;      // n
    float* sp   = geap + n;      // BLK  warp-partial scratch for the block scans
    __shared__ float ggamma;

    if (tid == 0) ggamma = 0.f;
    for (int j = tid; j < k; j += nb) { gebp[j] = 0.f; gebn[j] = 0.f; }
    for (int p = tid; p < n; p += nb) { gean[p] = 0.f; geap[p] = 0.f; }
    __syncthreads();

    const float* ebp_l = ebp + (size_t)l * k;
    const float* ebn_l = ebn + (size_t)l * k;
    const int*   pm_l  = perm + (size_t)l * k;
    const int*   cl_l  = cl  + (size_t)l * n;
    const int*   cr_l  = cr  + (size_t)l * n;
    const float* ean_l = ean + (size_t)l * n;
    const float* eap_l = eap + (size_t)l * n;
    float gl = gamma[l];

    for (int b = strip; b < B; b += nstrips) {
        const float* xb = x + (size_t)b * k;
        const float* gb = g + (size_t)b * n;
        for (int p = tid; p < n; p += nb) grow[p] = gb[p];
        for (int j = tid; j < k; j += nb) {
            float xv = xb[pm_l[j]];
            xs[j] = xv;
            Pp[j] = ebp_l[j] * xv;
            Ss[j] = ebn_l[j] * xv;
            bP[j] = 0.f; bS[j] = 0.f;
        }
        __syncthreads();
        if (!(prof & 4)) {
            block_prefix_inc(Pp, k, sp);     // P[i] = sum_{j<=i} ebp*xs
            block_suffix_inc(Ss, k, sp);     // S[i] = sum_{j>=i} ebn*xs
        }

        float gpart = 0.f;
        for (int p = tid; p < n; p += nb) {
            float Pc = Pp[cl_l[p]];
            float Sc = Ss[cr_l[p]];
            float gv = grow[p];
            gean[p] += gl * gv * Pc;
            geap[p] += gl * gv * Sc;
            gpart   += gv * (ean_l[p] * Pc + eap_l[p] * Sc);
            if (!(prof & 1)) {
                float w = gl * gv;
                atomicAdd(&bP[cl_l[p]], ean_l[p] * w);
                atomicAdd(&bS[cr_l[p]], eap_l[p] * w);
            }
        }
        __syncthreads();
        atomicAdd(&ggamma, gpart);
        // Adjoint of the forward gather.  Forward READ bufP[cl] (a prefix sum)
        // and bufS[cr] (a suffix sum); transposing that linear map turns the
        // cotangent of a prefix-read into a SUFFIX scan of the scattered grads
        // (and a suffix-read into a prefix scan).  So bP (grads scattered at cl)
        // gets a suffix scan, bS (scattered at cr) a prefix scan -> tildeP[j],
        // tildeS[j], the per-anchor cotangents feeding grad_ebp/ebn/grad_x below.
        if (!(prof & 8)) {
            block_suffix_inc(bP, k, sp);     // tildeP (adjoint of the prefix gather)
            block_prefix_inc(bS, k, sp);     // tildeS (adjoint of the suffix gather)
        }
        float* gxb = grad_x + (size_t)b * k;
        for (int j = tid; j < k; j += nb) {
            float tP = bP[j], tS = bS[j];
            gebp[j] += xs[j] * tP;
            gebn[j] += xs[j] * tS;
            // grad_x[b, perm_l[j]] -- summed over l across blocks => atomic
            if (!(prof & 2))
                atomicAdd(&gxb[pm_l[j]], ebp_l[j] * tP + ebn_l[j] * tS);
        }
        __syncthreads();
    }

    float* gebp_l = grad_ebp + (size_t)l * k;
    float* gebn_l = grad_ebn + (size_t)l * k;
    float* gean_l = grad_ean + (size_t)l * n;
    float* geap_l = grad_eap + (size_t)l * n;
    for (int j = tid; j < k; j += nb) {
        atomicAdd(&gebp_l[j], gebp[j]);
        atomicAdd(&gebn_l[j], gebn[j]);
    }
    for (int p = tid; p < n; p += nb) {
        atomicAdd(&gean_l[p], gean[p]);
        atomicAdd(&geap_l[p], geap[p]);
    }
    if (tid == 0) atomicAdd(&grad_gamma[l], ggamma);
}

// ----------------------------------------------------------------------
// Launchers
// ----------------------------------------------------------------------
static int BLK = 256;                 // threads per block (block-cooperative)

// Opt a kernel into >48KB dynamic smem, but only when the request GROWS and
// only with the EXACT bytes needed (setting the full sharedMemPerBlockOptin
// overshoots the driver's ~1KB reserve and yields cudaErrorInvalidValue).
// cudaFuncSetAttribute is NOT stream-capturable, so calling it per launch
// breaks CUDA-graph capture; the per-kernel highwater guard means that for a
// fixed (k,n) the call happens once during warmup and is a no-op at capture
// time.  For kernels needing <=48KB it is never called at all.
static int g_fwd_smem = 0, g_bwd_smem = 0;
static void optin_once(const void* fn, int* cur, int smem_bytes) {
    if (smem_bytes > 48 * 1024 && smem_bytes > *cur) {
        cudaFuncSetAttribute((void*)fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        *cur = smem_bytes;
    }
}

torch::Tensor mix_forward(
    torch::Tensor x, torch::Tensor ebp, torch::Tensor ebn, torch::Tensor perm,
    torch::Tensor cl, torch::Tensor cr, torch::Tensor ean, torch::Tensor eap,
    torch::Tensor gamma)
{
    // Block-per-row only.  A warp-per-row variant was tried and a
    // single-run A/B (probe_perf.py) showed block wins at EVERY shape,
    // including small k (fc1 k=384: warp 2.79 vs block 2.51 ms) -- the
    // occupancy probe explains why (warp needs 3k+n smem PER ROW, capping
    // it at ~17% SM occupancy; block reuses one row's smem across 256
    // threads and reaches ~100%).  An earlier "warp wins at fc1" reading
    // was a cross-run laptop-clock artifact.
    int B = x.size(0), k = x.size(1);
    int L = ebp.size(0), n = ean.size(1);
    auto y = torch::empty({B, n}, x.options());
    int smem = (3 * k + n + BLK) * (int)sizeof(float);
    optin_once((const void*)mix_fwd, &g_fwd_smem, smem);
    mix_fwd<<<B, BLK, smem>>>(
        x.data_ptr<float>(), ebp.data_ptr<float>(), ebn.data_ptr<float>(),
        perm.data_ptr<int>(), cl.data_ptr<int>(), cr.data_ptr<int>(),
        ean.data_ptr<float>(), eap.data_ptr<float>(), gamma.data_ptr<float>(),
        y.data_ptr<float>(), B, L, k, n);
    return y;
}

std::vector<torch::Tensor> mix_backward(
    torch::Tensor x, torch::Tensor g, torch::Tensor ebp, torch::Tensor ebn,
    torch::Tensor perm, torch::Tensor cl, torch::Tensor cr,
    torch::Tensor ean, torch::Tensor eap, torch::Tensor gamma, int64_t prof)
{
    int B = x.size(0), k = x.size(1);
    int L = ebp.size(0), n = ean.size(1);

    auto grad_x     = torch::zeros({B, k}, x.options());   // atomic-accumulated
    auto grad_ebp   = torch::zeros({L, k}, x.options());
    auto grad_ebn   = torch::zeros({L, k}, x.options());
    auto grad_ean   = torch::zeros({L, n}, x.options());
    auto grad_eap   = torch::zeros({L, n}, x.options());
    auto grad_gamma = torch::zeros({L},    x.options());

    // Block-cooperative.  nstrips trades parallelism against per-block
    // flush atomics (~nstrips*L*(2k+2n)); ~B/16 keeps flush below the
    // grad_x atomic traffic while giving plenty of blocks.
    int nstrips = (B + 15) / 16;
    if (nstrips > 1024) nstrips = 1024;
    if (nstrips < 1) nstrips = 1;
    dim3 grid_w(L, nstrips);
    int smem_w = (7 * k + 3 * n + BLK) * (int)sizeof(float);
    optin_once((const void*)mix_bwd, &g_bwd_smem, smem_w);
    mix_bwd<<<grid_w, BLK, smem_w>>>(
        x.data_ptr<float>(), g.data_ptr<float>(),
        ebp.data_ptr<float>(), ebn.data_ptr<float>(), perm.data_ptr<int>(),
        cl.data_ptr<int>(), cr.data_ptr<int>(),
        ean.data_ptr<float>(), eap.data_ptr<float>(), gamma.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_ebp.data_ptr<float>(), grad_ebn.data_ptr<float>(),
        grad_ean.data_ptr<float>(), grad_eap.data_ptr<float>(),
        grad_gamma.data_ptr<float>(), B, L, k, n, (int)prof);

    return {grad_x, grad_ebp, grad_ebn, grad_ean, grad_eap, grad_gamma};
}

// ----------------------------------------------------------------------
// Occupancy probe: prints theoretical max active blocks/SM (and the
// resulting warps/SM vs the SM cap) for the forward + backward kernels at
// the exact launch configs the dispatchers use.  Diagnoses whether the
// backward is shared-memory occupancy-bound.
// ----------------------------------------------------------------------
void report_occupancy(int k, int n, int L, int B) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    int maxWarps = prop.maxThreadsPerMultiProcessor / 32;
    printf("  device %s | SMs=%d | maxWarps/SM=%d | smem/SM=%d KB | smem/block max=%d KB\n",
           prop.name, prop.multiProcessorCount, maxWarps,
           (int)(prop.sharedMemPerMultiprocessor / 1024),
           (int)(prop.sharedMemPerBlockOptin / 1024));

    int nb = 0;
    {
        int blk = BLK, smem = (3*k + n + BLK) * (int)sizeof(float);
        optin_once((const void*)mix_fwd, &g_fwd_smem, smem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)mix_fwd, blk, smem);
        printf("  FWD block blk=%d smem=%2dKB -> %d blk/SM, %2d warps/SM (%3.0f%% of %d)\n",
               blk, smem/1024, nb, nb*blk/32, 100.0*nb*blk/32/maxWarps, maxWarps);
    }
    {
        int blk = BLK, smem = (7*k + 3*n + BLK) * (int)sizeof(float);
        optin_once((const void*)mix_bwd, &g_bwd_smem, smem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*)mix_bwd, blk, smem);
        printf("  BWD block blk=%d smem=%2dKB -> %d blk/SM, %2d warps/SM (%3.0f%% of %d)\n",
               blk, smem/1024, nb, nb*blk/32, 100.0*nb*blk/32/maxWarps, maxWarps);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mix_forward",  &mix_forward,  "fused LAPLEX multi-kernel mixture forward");
    m.def("mix_backward", &mix_backward, "fused LAPLEX multi-kernel mixture backward",
          pybind11::arg("x"), pybind11::arg("g"), pybind11::arg("ebp"),
          pybind11::arg("ebn"), pybind11::arg("perm"), pybind11::arg("cl"),
          pybind11::arg("cr"), pybind11::arg("ean"), pybind11::arg("eap"),
          pybind11::arg("gamma"),
          pybind11::arg("prof") = 0);   // prof>0: WRONG grads, perf-diagnostic only (see prof_bwd.py)
    m.def("report_occupancy", &report_occupancy, "print kernel occupancy for given (k,n,L,B)");
}
"""


print("Compiling laplex_mix_fused extension...", end=" ", flush=True)
_ext = load_inline(
    name="laplex_mix_fused_ext",
    cpp_sources=[""],
    cuda_sources=[_CUDA_SRC],
    extra_cuda_cflags=_NVCC_FLAGS,
    verbose=False,
)
print("OK")


# ===========================================================================
# Autograd Function : differentiable through (x, ebp, ebn, ean, eap, gamma)
# perm / cl / cr are detached integer metadata.
# ===========================================================================
class _LaplexMixFusedFn(Function):
    @staticmethod
    def forward(ctx, x, ebp, ebn, ean, eap, gamma, perm, cl, cr):
        x = x.contiguous()
        ebp = ebp.contiguous(); ebn = ebn.contiguous()
        ean = ean.contiguous(); eap = eap.contiguous()
        gamma = gamma.contiguous()
        perm = perm.contiguous(); cl = cl.contiguous(); cr = cr.contiguous()
        y = _ext.mix_forward(x, ebp, ebn, perm, cl, cr, ean, eap, gamma)
        ctx.save_for_backward(x, ebp, ebn, ean, eap, gamma, perm, cl, cr)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, ebp, ebn, ean, eap, gamma, perm, cl, cr = ctx.saved_tensors
        gy = gy.contiguous()
        gx, g_ebp, g_ebn, g_ean, g_eap, g_gamma = _ext.mix_backward(
            x, gy, ebp, ebn, perm, cl, cr, ean, eap, gamma)
        # order matches forward signature
        return gx, g_ebp, g_ebn, g_ean, g_eap, g_gamma, None, None, None


# ===========================================================================
# Functional form : precompute (sort / searchsorted / mid-range exps) then
# call the fused Function.  Differentiable through (x, a, b, gamma).
# ===========================================================================
def _laplex_precompute(x: Tensor, a: Tensor, b: Tensor):
    """Host-side prep shared by forward: sort b, search-sort a, build the
    mid-range-shifted exponentials.  Returns the tensors the fused
    Function consumes plus (lead, n) for the reshape.  Split out so it can
    be timed (and later CUDA-graphed) separately from the kernel."""
    L, n = a.shape
    Lb, k = b.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    if x2.dtype != torch.float32:
        x2 = x2.float()
    x2 = x2.contiguous()

    # --- detached integer metadata: sort b, search-sort a ---
    b_s, perm = torch.sort(b, dim=1)                       # (L,k)
    with torch.no_grad():
        c = torch.searchsorted(b_s.detach(), a.detach(), right=True) - 1   # (L,n)
        cl = c.clamp(0, k - 1).to(torch.int32).contiguous()
        cr = (c + 1).clamp(0, k - 1).to(torch.int32).contiguous()
        gate_l = (c >= 0).to(a.dtype)
        gate_r = ((c + 1) < k).to(a.dtype)
    perm_i = perm.to(torch.int32).contiguous()

    # --- per-kernel mid-range shift r_l (detached); cancels in output ---
    with torch.no_grad():
        lo = torch.minimum(a.min(dim=1).values, b_s.min(dim=1).values)
        hi = torch.maximum(a.max(dim=1).values, b_s.max(dim=1).values)
        r = (0.5 * (lo + hi))   # (L,) coord midpoint; exp(±r) cancels in exp(-|a-b|), keeps exps FP32-safe
    r = r[:, None]

    # --- differentiable exponentials (chain to a, b) ---
    ebp = torch.exp(b_s - r).float()                       # (L,k)
    ebn = torch.exp(-(b_s - r)).float()
    ean = (torch.exp(-(a - r)) * gate_l).float()           # (L,n)
    eap = (torch.exp(a - r) * gate_r).float()
    return x2, ebp, ebn, ean, eap, perm_i, cl, cr, lead, n


def laplex_mix(x: Tensor, a: Tensor, b: Tensor, gamma: Tensor) -> Tensor:
    """y = sum_l gamma_l * K(a_l, b_l) @ x   (shared x, fused CUDA).

    x     : (..., k)      activation (leading dims are flattened)
    a     : (L, n)        per-kernel query coords
    b     : (L, k)        per-kernel anchor coords
    gamma : (L,)          signed mixing weights
    returns y : (..., n)
    """
    assert a.dim() == 2 and b.dim() == 2 and gamma.dim() == 1
    L, n = a.shape
    Lb, k = b.shape
    assert Lb == L and gamma.numel() == L
    assert x.shape[-1] == k, f"x last dim {x.shape[-1]} != k {k}"

    x2, ebp, ebn, ean, eap, perm_i, cl, cr, lead, n = _laplex_precompute(x, a, b)
    y = _LaplexMixFusedFn.apply(x2, ebp, ebn, ean, eap, gamma.float(), perm_i, cl, cr)
    return y.reshape(*lead, n)


# ---------------------------------------------------------------------------
# Dense reference (pure torch, differentiable) -- for tests / CPU / tiny B.
# ---------------------------------------------------------------------------
def laplex_mix_dense(x: Tensor, a: Tensor, b: Tensor, gamma: Tensor) -> Tensor:
    """Reference y = sum_l gamma_l * exp(-|a_l - b_l|) @ x.  O(L n k B)."""
    L, n = a.shape
    k = b.shape[1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)                                  # (B,k)
    diff = a.unsqueeze(-1) - b.unsqueeze(-2)               # (L,n,k)
    K = torch.exp(-diff.abs())                             # (L,n,k)
    y = torch.einsum('lnk,bk->lbn', K, x2)                 # (L,B,n)
    y = (gamma[:, None, None] * y).sum(0)                  # (B,n)
    return y.reshape(*lead, n)


# ===========================================================================
# nn.Module drop-in for nn.Linear(k, n)
# ===========================================================================
class LaplexMixLinear(torch.nn.Module):
    """Drop-in replacement for ``nn.Linear(in_features=k, out_features=n)``
    whose weight matrix is an ``L``-kernel Laplace mixture
    ``W = sum_l gamma_l K(a_l, b_l)`` applied via the fused CUDA kernel.

    Parameter count: ``L*(n + k) + L`` (vs ``n*k`` for a dense linear).

        layer = LaplexMixLinear(k=384, n=1536, L=8).cuda()
        y = layer(x)                       # x : (..., 384) -> (..., 1536)
    """
    def __init__(self, k: int, n: int, L: int = 8,
                 coord_scale: float = 1.0, gamma_init: float = None,
                 bias: bool = True):
        super().__init__()
        self.k, self.n, self.L = k, n, L
        # a: query coords on the OUTPUT axis (length n); b: anchor coords on the
        # INPUT axis (length k).  W[p,j] = sum_l gamma_l * exp(-|a_l[p] - b_l[j]|).
        # Spread coords over a comparable range so the kernels are not all
        # identical at init; linspace + jitter gives distinct orderings.
        base_a = torch.linspace(-coord_scale, coord_scale, n)
        base_b = torch.linspace(-coord_scale, coord_scale, k)
        self.a = torch.nn.Parameter(
            base_a.unsqueeze(0).repeat(L, 1) + 0.1 * torch.randn(L, n))
        self.b = torch.nn.Parameter(
            base_b.unsqueeze(0).repeat(L, 1) + 0.1 * torch.randn(L, k))
        if gamma_init is None:
            gamma_init = 1.0 / L
        self.gamma = torch.nn.Parameter(gamma_init * torch.ones(L))
        self.bias = torch.nn.Parameter(torch.zeros(n)) if bias else None

    def forward(self, x):
        y = laplex_mix(x, self.a, self.b, self.gamma)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return (f"k={self.k}, n={self.n}, L={self.L}, "
                f"params={self.L * (self.n + self.k) + self.L}"
                f" (dense linear: {self.n * self.k})")


# ===========================================================================
# NOTE: a CUDA-graph wrapper (torch.cuda.make_graphed_callables) was prototyped
# to erase the ~15-20 tiny precompute launches, but capture fails with an
# illegal memory access for this custom-Function module (a plain nn.Linear
# graphs fine on the same box) AND it poisons the CUDA context, while the
# fwd+bwd payoff was only ~2-5%.  Abandoned -- see README "Status" for details.
# ===========================================================================
