"""
kerlap_fused.py

X_KerLap with fused CUDA kernels:
  1) fused_precompute: 7 PyTorch ops -> 1 kernel
  2) fused_gather_interp: ~20 PyTorch ops (searchsorted+clamp+exp+gather+sum) -> 1 kernel
  3) fused_scatter: ~20 PyTorch ops (searchsorted+scatter_add) -> 1 kernel

The DL scan (prefix_suffix) is unchanged.
"""

import os, sys, time, gc
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefix_suffix_dl import _ps_raw, _precompute, _PAD_TO
from prefix_suffix_dl import prefix_suffix as _prefix_suffix_sum

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
#include <cmath>

// ============================================================
// 1. Fused precompute: alpha -> ef, n_fwd, ef_neg, n_bwd
// ============================================================
__global__ void precompute_k(
    const float* __restrict__ alpha,
    float* __restrict__ ef,
    int*   __restrict__ n_fwd,
    float* __restrict__ ef_neg,
    int*   __restrict__ n_bwd,
    int T)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;

    float LOG2E = 1.4426950408889634f;

    float ap = alpha[i] * LOG2E;
    float fl_p = floorf(ap);
    float fr_p = ap - fl_p;
    ef[i]    = exp2f(fr_p);
    n_fwd[i] = __float2int_rz(fl_p);

    float an = -ap;
    float fl_n = floorf(an);
    float fr_n = an - fl_n;
    ef_neg[i] = exp2f(fr_n);
    n_bwd[i]  = __float2int_rz(fl_n);
}

std::vector<torch::Tensor> fused_precompute(torch::Tensor alpha) {
    int T = alpha.numel();
    auto ef     = torch::empty({T}, alpha.options());
    auto n_fwd  = torch::empty({T}, alpha.options().dtype(torch::kInt32));
    auto ef_neg = torch::empty({T}, alpha.options());
    auto n_bwd  = torch::empty({T}, alpha.options().dtype(torch::kInt32));

    int threads = 256;
    int blocks = (T + threads - 1) / threads;
    precompute_k<<<blocks, threads>>>(
        alpha.data_ptr<float>(),
        ef.data_ptr<float>(), n_fwd.data_ptr<int>(),
        ef_neg.data_ptr<float>(), n_bwd.data_ptr<int>(), T);
    return {ef, n_fwd, ef_neg, n_bwd};
}

// ============================================================
// 2. Fused gather+interpolate forward (_xA path)
// ============================================================
__global__ void gather_interp_fwd_k(
    const float* __restrict__ a_s,
    const float* __restrict__ b,
    const float* __restrict__ P,
    const float* __restrict__ S,
    float* __restrict__ y,
    int B, int n, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;

    float bj = b[j];

    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (a_s[mid] <= bj) lo = mid + 1;
        else hi = mid;
    }
    int c = lo - 1;

    int cl = c >= 0     ? c     : 0;
    int cr = c+1 < n    ? c+1   : n-1;
    float wl = c >= 0   ? expf(a_s[cl] - bj) : 0.0f;
    float wr = c+1 < n  ? expf(bj - a_s[cr]) : 0.0f;

    for (int batch = 0; batch < B; batch++) {
        y[batch*k + j] = wl * P[batch*n + cl] + wr * S[batch*n + cr];
    }
}

torch::Tensor gather_interp_fwd(
    torch::Tensor a_s, torch::Tensor b,
    torch::Tensor P, torch::Tensor S)
{
    a_s = a_s.contiguous(); b = b.contiguous();
    P = P.contiguous(); S = S.contiguous();
    int B = P.size(0), n = P.size(1), k = b.numel();
    auto y = torch::empty({B, k}, P.options());
    int threads = 256;
    int blocks = (k + threads - 1) / threads;
    gather_interp_fwd_k<<<blocks, threads>>>(
        a_s.data_ptr<float>(), b.data_ptr<float>(),
        P.data_ptr<float>(), S.data_ptr<float>(),
        y.data_ptr<float>(), B, n, k);
    return y;
}

// ============================================================
// 3. Gather+interpolate backward
// ============================================================
__global__ void gather_interp_bwd_k(
    const float* __restrict__ a_s,
    const float* __restrict__ b,
    const float* __restrict__ P,
    const float* __restrict__ S,
    const float* __restrict__ gy,
    float* __restrict__ gP,
    float* __restrict__ gS,
    float* __restrict__ ga_s,
    float* __restrict__ gb,
    int B, int n, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;

    float bj = b[j];
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (a_s[mid] <= bj) lo = mid + 1;
        else hi = mid;
    }
    int c = lo - 1;

    int cl = c >= 0    ? c    : 0;
    int cr = c+1 < n   ? c+1  : n-1;
    float wl = c >= 0  ? expf(a_s[cl] - bj) : 0.0f;
    float wr = c+1 < n ? expf(bj - a_s[cr]) : 0.0f;

    float gb_j = 0.0f;
    for (int batch = 0; batch < B; batch++) {
        float g = gy[batch*k + j];
        if (c >= 0) {
            atomicAdd(&gP[batch*n + cl], wl * g);
            atomicAdd(&ga_s[cl], wl * P[batch*n + cl] * g);
        }
        if (c+1 < n) {
            atomicAdd(&gS[batch*n + cr], wr * g);
            atomicAdd(&ga_s[cr], -wr * S[batch*n + cr] * g);
        }
        float d = 0.0f;
        if (c >= 0)  d -= wl * P[batch*n + cl];
        if (c+1 < n) d += wr * S[batch*n + cr];
        gb_j += d * g;
    }
    gb[j] = gb_j;
}

std::vector<torch::Tensor> gather_interp_bwd(
    torch::Tensor a_s, torch::Tensor b,
    torch::Tensor P, torch::Tensor S,
    torch::Tensor gy)
{
    a_s = a_s.contiguous(); b = b.contiguous();
    P = P.contiguous(); S = S.contiguous();
    gy = gy.contiguous();
    int B = P.size(0), n = P.size(1), k = b.numel();
    auto gP   = torch::zeros({B, n}, P.options());
    auto gS   = torch::zeros({B, n}, S.options());
    auto ga_s = torch::zeros({n}, a_s.options());
    auto gb   = torch::empty({k}, b.options());
    int threads = 256;
    int blocks = (k + threads - 1) / threads;
    gather_interp_bwd_k<<<blocks, threads>>>(
        a_s.data_ptr<float>(), b.data_ptr<float>(),
        P.data_ptr<float>(), S.data_ptr<float>(),
        gy.data_ptr<float>(),
        gP.data_ptr<float>(), gS.data_ptr<float>(),
        ga_s.data_ptr<float>(), gb.data_ptr<float>(),
        B, n, k);
    return {gP, gS, ga_s, gb};
}

// ============================================================
// 4. Fused scatter forward (_xAT path)
// ============================================================
__global__ void scatter_fwd_k(
    const float* __restrict__ b_s,
    const float* __restrict__ a,
    const float* __restrict__ X,
    float* __restrict__ u,
    float* __restrict__ v,
    int B, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ai = a[i];
    int lo = 0, hi = k;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (b_s[mid] <= ai) lo = mid + 1;
        else hi = mid;
    }
    int idx = lo - 1;

    int i_u = idx >= 0     ? idx   : 0;
    float vu = idx >= 0    ? expf(b_s[i_u] - ai) : 0.0f;
    int i_v = idx+1 < k    ? idx+1 : k-1;
    float vv = idx+1 < k   ? expf(ai - b_s[i_v]) : 0.0f;

    for (int batch = 0; batch < B; batch++) {
        float xi = X[batch*n + i];
        if (vu != 0.0f) atomicAdd(&u[batch*k + i_u], xi * vu);
        if (vv != 0.0f) atomicAdd(&v[batch*k + i_v], xi * vv);
    }
}

std::vector<torch::Tensor> fused_scatter_fwd(
    torch::Tensor b_s, torch::Tensor a, torch::Tensor X)
{
    b_s = b_s.contiguous(); a = a.contiguous(); X = X.contiguous();
    int B = X.size(0), n = a.numel(), k = b_s.numel();
    auto u = torch::zeros({B, k}, X.options());
    auto v = torch::zeros({B, k}, X.options());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scatter_fwd_k<<<blocks, threads>>>(
        b_s.data_ptr<float>(), a.data_ptr<float>(), X.data_ptr<float>(),
        u.data_ptr<float>(), v.data_ptr<float>(), B, n, k);
    return {u, v};
}

// ============================================================
// 5. Fused scatter backward
// ============================================================
__global__ void scatter_bwd_k(
    const float* __restrict__ b_s,
    const float* __restrict__ a,
    const float* __restrict__ X,
    const float* __restrict__ gu,
    const float* __restrict__ gv,
    float* __restrict__ gX,
    float* __restrict__ ga,
    float* __restrict__ gb_s,
    int B, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ai = a[i];
    int lo = 0, hi = k;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (b_s[mid] <= ai) lo = mid + 1;
        else hi = mid;
    }
    int idx = lo - 1;

    int i_u = idx >= 0   ? idx   : 0;
    float vu = idx >= 0  ? expf(b_s[i_u] - ai) : 0.0f;
    int i_v = idx+1 < k  ? idx+1 : k-1;
    float vv = idx+1 < k ? expf(ai - b_s[i_v]) : 0.0f;

    float ga_i = 0.0f;
    for (int batch = 0; batch < B; batch++) {
        float xi = X[batch*n + i];
        float gui = idx >= 0   ? gu[batch*k + i_u] : 0.0f;
        float gvi = idx+1 < k  ? gv[batch*k + i_v] : 0.0f;

        gX[batch*n + i] = gui * vu + gvi * vv;
        ga_i += xi * (-vu * gui + vv * gvi);

        if (idx >= 0)  atomicAdd(&gb_s[i_u],  gui * xi * vu);
        if (idx+1 < k) atomicAdd(&gb_s[i_v], -gvi * xi * vv);
    }
    ga[i] = ga_i;
}

std::vector<torch::Tensor> fused_scatter_bwd(
    torch::Tensor b_s, torch::Tensor a, torch::Tensor X,
    torch::Tensor gu, torch::Tensor gv)
{
    b_s = b_s.contiguous(); a = a.contiguous(); X = X.contiguous();
    gu = gu.contiguous(); gv = gv.contiguous();
    int B = X.size(0), n = a.numel(), k = b_s.numel();
    auto gX   = torch::empty({B, n}, X.options());
    auto ga   = torch::empty({n}, a.options());
    auto gb_s = torch::zeros({k}, b_s.options());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scatter_bwd_k<<<blocks, threads>>>(
        b_s.data_ptr<float>(), a.data_ptr<float>(), X.data_ptr<float>(),
        gu.data_ptr<float>(), gv.data_ptr<float>(),
        gX.data_ptr<float>(), ga.data_ptr<float>(), gb_s.data_ptr<float>(),
        B, n, k);
    return {gX, ga, gb_s};
}

// ============================================================
// 6. Fused build_M for Ker_D_KerT (forward)
//    M[i,j] = exp(a[j]-a[i]) * (U[j] + W[i] - W[j] + V[i])  for i>=j
//    M[i,j] = M[j,i]  (symmetric)
//    Replaces ~6 PyTorch ops (relu, exp, mul, tril, T, add).
// ============================================================
__global__ void build_M_fwd_k(
    const float* __restrict__ alpha_s,  // [n]
    const float* __restrict__ U,        // [n]
    const float* __restrict__ V,        // [n]
    const float* __restrict__ W,        // [n]
    float* __restrict__ M,              // [n, n]
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ai = alpha_s[i];
    float Vi = V[i];
    float Wi = W[i];

    for (int j = 0; j <= i; j++) {
        float e = expf(alpha_s[j] - ai);
        float inner = U[j] + Wi - W[j] + Vi;
        float m = e * inner;
        M[i * n + j] = m;
        if (j < i) M[j * n + i] = m;
    }
}

torch::Tensor build_M_fwd(
    torch::Tensor alpha_s, torch::Tensor U,
    torch::Tensor V, torch::Tensor W)
{
    alpha_s = alpha_s.contiguous(); U = U.contiguous();
    V = V.contiguous(); W = W.contiguous();
    int n = alpha_s.numel();
    auto M = torch::empty({n, n}, alpha_s.options());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    build_M_fwd_k<<<blocks, threads>>>(
        alpha_s.data_ptr<float>(), U.data_ptr<float>(),
        V.data_ptr<float>(), W.data_ptr<float>(),
        M.data_ptr<float>(), n);
    return M;
}

// ============================================================
// 7. Fused build_M backward
//    Given gM[n,n], compute g_alpha_s, g_U, g_V, g_W.
//    Each thread handles row i, loops over j=0..i.
// ============================================================
__global__ void build_M_bwd_k(
    const float* __restrict__ gM,       // [n, n]
    const float* __restrict__ alpha_s,  // [n]
    const float* __restrict__ U,        // [n]
    const float* __restrict__ V,        // [n]
    const float* __restrict__ W,        // [n]
    float* __restrict__ g_alpha_s,      // [n]
    float* __restrict__ g_U,            // [n]
    float* __restrict__ g_V,            // [n]
    float* __restrict__ g_W,            // [n]
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ai = alpha_s[i];
    float Vi = V[i];
    float Wi = W[i];

    float g_Vi = 0.0f;
    float g_Wi = 0.0f;
    float g_ai = 0.0f;

    for (int j = 0; j <= i; j++) {
        float aj = alpha_s[j];
        float e = expf(aj - ai);
        float inner = U[j] + Wi - W[j] + Vi;
        float m = e * inner;

        // Effective gradient: lower tri + transpose of strict lower tri
        float gm = gM[i * n + j];
        if (j < i) gm += gM[j * n + i];

        float gm_e = gm * e;

        // g_U[j] += gm * e  (column accumulation -> atomicAdd)
        atomicAdd(&g_U[j], gm_e);
        // g_V[i] += gm * e  (row accumulation -> local)
        g_Vi += gm_e;
        // g_W[i] += gm * e  (row, from +W[i] term -> local, added to g_W later)
        g_Wi += gm_e;
        // g_W[j] -= gm * e  (column, from -W[j] term -> atomicAdd)
        atomicAdd(&g_W[j], -gm_e);

        // g_alpha_s: dM/da[j] = +m, dM/da[i] = -m  (only off-diagonal)
        if (j < i) {
            atomicAdd(&g_alpha_s[j], gm * m);
            g_ai -= gm * m;
        }
    }

    // Write row-local accumulations (only thread i writes to index i)
    g_V[i] = g_Vi;
    atomicAdd(&g_W[i], g_Wi);
    atomicAdd(&g_alpha_s[i], g_ai);
}

std::vector<torch::Tensor> build_M_bwd(
    torch::Tensor gM, torch::Tensor alpha_s,
    torch::Tensor U, torch::Tensor V, torch::Tensor W)
{
    gM = gM.contiguous(); alpha_s = alpha_s.contiguous();
    U = U.contiguous(); V = V.contiguous(); W = W.contiguous();
    int n = alpha_s.numel();
    auto g_alpha_s = torch::zeros({n}, alpha_s.options());
    auto g_U       = torch::zeros({n}, U.options());
    auto g_V       = torch::empty({n}, V.options());
    auto g_W       = torch::zeros({n}, W.options());
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    build_M_bwd_k<<<blocks, threads>>>(
        gM.data_ptr<float>(), alpha_s.data_ptr<float>(),
        U.data_ptr<float>(), V.data_ptr<float>(), W.data_ptr<float>(),
        g_alpha_s.data_ptr<float>(), g_U.data_ptr<float>(),
        g_V.data_ptr<float>(), g_W.data_ptr<float>(), n);
    return {g_alpha_s, g_U, g_V, g_W};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_precompute",    &fused_precompute);
    m.def("gather_interp_fwd",   &gather_interp_fwd);
    m.def("gather_interp_bwd",   &gather_interp_bwd);
    m.def("fused_scatter_fwd",   &fused_scatter_fwd);
    m.def("fused_scatter_bwd",   &fused_scatter_bwd);
    m.def("build_M_fwd",         &build_M_fwd);
    m.def("build_M_bwd",         &build_M_bwd);
}
"""

print("Compiling fused kernels...", end=" ", flush=True)
_fused = load_inline(
    name="kerlap_fused_v2",
    cpp_sources=[""],
    cuda_sources=[_CUDA_SRC],
    extra_cuda_cflags=_NVCC_FLAGS,
    verbose=False,
)
print("OK")


# ============================================================
# Autograd wrappers
# ============================================================
class _PrefSuffFusedFn(Function):
    """prefix_suffix_sep with fused_precompute (1 kernel instead of 7)."""

    @staticmethod
    def forward(ctx, alpha, u, v):
        a32 = alpha.float()
        u32 = u.float()
        v32 = v.float()
        ef, n_fwd, ef_neg, n_bwd = _fused.fused_precompute(a32)
        P, S = _ps_raw(ef, n_fwd, ef_neg, n_bwd, u32, v32)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u32, v32, P, S)
        ctx.orig_dtype = alpha.dtype
        return P.to(alpha.dtype), S.to(alpha.dtype)

    @staticmethod
    def backward(ctx, gP, gS):
        ef, n_fwd, ef_neg, n_bwd, u, v, P, S = ctx.saved_tensors
        od = ctx.orig_dtype
        gP32 = gP.float()
        gS32 = gS.float()
        prefix_gS, suffix_gP = _ps_raw(ef, n_fwd, ef_neg, n_bwd, gS32, gP32)
        grad_u = suffix_gP.to(od) if ctx.needs_input_grad[1] else None
        grad_v = prefix_gS.to(od) if ctx.needs_input_grad[2] else None
        grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_alpha = (
                (u * suffix_gP - v * prefix_gS - gP32 * P + gS32 * S).sum(0).to(od)
            )
        return grad_alpha, grad_u, grad_v


class _GatherInterpFn(Function):
    """Fused searchsorted + gather + exp-weighted sum."""

    @staticmethod
    def forward(ctx, a_s, b, P, S):
        a32 = a_s.float().contiguous()
        b32 = b.float().contiguous()
        P32 = P.float().contiguous()
        S32 = S.float().contiguous()
        y = _fused.gather_interp_fwd(a32, b32, P32, S32)
        ctx.save_for_backward(a32, b32, P32, S32)
        ctx.orig_dtype = a_s.dtype
        return y.to(a_s.dtype)

    @staticmethod
    def backward(ctx, gy):
        a_s, b, P, S = ctx.saved_tensors
        gy32 = gy.float().contiguous()
        gP, gS, ga_s, gb = _fused.gather_interp_bwd(a_s, b, P, S, gy32)
        od = ctx.orig_dtype
        return (
            ga_s.to(od) if ctx.needs_input_grad[0] else None,
            gb.to(od) if ctx.needs_input_grad[1] else None,
            gP.to(od) if ctx.needs_input_grad[2] else None,
            gS.to(od) if ctx.needs_input_grad[3] else None,
        )


class _FusedBuildMFn(Function):
    """Fused matrix M construction for Ker_D_KerT.
    M[i,j] = exp(a[j]-a[i]) * (U[j]+W[i]-W[j]+V[i]) for i>=j, symmetric."""

    @staticmethod
    def forward(ctx, alpha_s, U, V, W):
        a32 = alpha_s.float().contiguous()
        U32 = U.float().contiguous()
        V32 = V.float().contiguous()
        W32 = W.float().contiguous()
        M = _fused.build_M_fwd(a32, U32, V32, W32)
        ctx.save_for_backward(a32, U32, V32, W32)
        ctx.orig_dtype = alpha_s.dtype
        return M.to(alpha_s.dtype)

    @staticmethod
    def backward(ctx, gM):
        a_s, U, V, W = ctx.saved_tensors
        gM32 = gM.float().contiguous()
        ga_s, gU, gV, gW = _fused.build_M_bwd(gM32, a_s, U, V, W)
        od = ctx.orig_dtype
        return (
            ga_s.to(od) if ctx.needs_input_grad[0] else None,
            gU.to(od) if ctx.needs_input_grad[1] else None,
            gV.to(od) if ctx.needs_input_grad[2] else None,
            gW.to(od) if ctx.needs_input_grad[3] else None,
        )


class _FusedScatterFn(Function):
    """Fused searchsorted + exp weights + scatter_add."""

    @staticmethod
    def forward(ctx, b_s, a, X):
        b32 = b_s.float()
        a32 = a.float()
        X32 = X.float()
        u, v = _fused.fused_scatter_fwd(b32, a32, X32)
        ctx.save_for_backward(b32, a32, X32)
        ctx.orig_dtype = b_s.dtype
        return u.to(b_s.dtype), v.to(b_s.dtype)

    @staticmethod
    def backward(ctx, gu, gv):
        b_s, a, X = ctx.saved_tensors
        gu32 = gu.float()
        gv32 = gv.float()
        gX, ga, gb_s = _fused.fused_scatter_bwd(b_s, a, X, gu32, gv32)
        od = ctx.orig_dtype
        return (
            gb_s.to(od) if ctx.needs_input_grad[0] else None,
            ga.to(od) if ctx.needs_input_grad[1] else None,
            gX.to(od) if ctx.needs_input_grad[2] else None,
        )


# ============================================================
# Fused X_KerLap
# ============================================================
def _xA_fused(X, a_s, b):
    """X @ K(a,b) — fused version, a_s already sorted."""
    P, S = _PrefSuffFusedFn.apply(a_s, X, X)
    return _GatherInterpFn.apply(a_s, b, P, S)


def _xAT_fused(X, a, b_s):
    """X @ K(a,b) — fused version (n>k path), b_s already sorted."""
    u, v = _FusedScatterFn.apply(b_s, a, X)
    return _prefix_suffix_sum(b_s, v, u)


def X_KerLap_fused(X, a, b, threshold=1024, a_sorted=False, b_sorted=False):
    """
    X @ K(a,b) with fused CUDA kernels.

    Same interface as X_KerLap, but ~20 kernel launches
    are replaced by 1 in the gather/scatter phase.
    """
    squeeze = X.dim() == 1
    if squeeze:
        X = X.unsqueeze(0)

    n = a.numel()
    k = b.numel()

    if min(n, k) <= threshold:
        K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        y = X @ K
        return y.squeeze(0) if squeeze else y

    if n <= k:
        if a_sorted:
            y = _xA_fused(X, a, b)
        else:
            a_s, perm = torch.sort(a)
            y = _xA_fused(X[:, perm], a_s, b)
    else:
        if b_sorted:
            y = _xAT_fused(X, a, b)
        else:
            b_s, perm_b = torch.sort(b)
            inv_perm_b = torch.argsort(perm_b)
            y = _xAT_fused(X, a, b_s)[:, inv_perm_b]

    return y.squeeze(0) if squeeze else y


# ============================================================
# Fused KerLap_X  =  K(a,b) @ X
# ============================================================
def KerLap_X_fused(X, a, b, threshold=1024, a_sorted=False, b_sorted=False):
    """
    K(a,b) @ X with fused CUDA kernels.
    X : (k, C) or (k,)
    a : (n,)
    b : (k,)
    returns : (n, C) or (n,)
    """
    result = X_KerLap_fused(
        X.T.contiguous(), b, a, threshold, a_sorted=b_sorted, b_sorted=a_sorted
    ).T
    return result


# ============================================================
# Fused Ker_D_KerT  =  K @ diag(D) @ K^T
# ============================================================
def _prefix_suffix_sep_fused(alpha, u, v=None):
    """prefix_suffix_sep z fused precompute."""
    if v is None:
        v = u
    squeeze = u.dim() == 1
    if squeeze:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0) if v is not u else u
    P, S = _PrefSuffFusedFn.apply(alpha, u, v)
    if squeeze:
        P = P.squeeze(0)
        S = S.squeeze(0)
    return P, S


def Ker_D_KerT_fused(alpha, beta, D, threshold=1024, alpha_sorted=False):
    """
    K(alpha,beta) @ diag(D) @ K(alpha,beta)^T with fused CUDA kernels.
    alpha : (n,)
    beta  : (k,)
    D     : (k,)
    returns : (n, n)
    """
    n = alpha.numel()
    k = beta.numel()

    if n <= threshold:
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        return (K * D.unsqueeze(0)) @ K.T

    return _Ker_D_KerT_scan_fused(alpha, beta, D, alpha_sorted=alpha_sorted)


def _Ker_D_KerT_scan_fused(alpha, beta, D, alpha_sorted=False):
    """Scan-based Ker_D_KerT z fused precompute."""
    n = alpha.numel()
    k = beta.numel()
    device = alpha.device
    dtype = alpha.dtype

    if alpha_sorted:
        alpha_s = alpha
        perm = None
        inv_perm = None
    else:
        alpha_s, perm = torch.sort(alpha)
        inv_perm = torch.argsort(perm)

    idx = torch.bucketize(beta, alpha_s, right=True)

    alpha_ext = torch.cat(
        [
            beta.min().unsqueeze(0),
            alpha_s,
            beta.max().unsqueeze(0),
        ]
    )

    alpha_ip1 = alpha_ext[idx + 1]
    u_vals = D * torch.exp(2.0 * (beta - alpha_ip1))
    u = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, u_vals)

    idx_left = (idx - 1).clamp(min=0)
    alpha_i = torch.where(idx > 0, alpha_s[idx_left], torch.zeros_like(beta))
    v_vals = D * torch.exp(2.0 * (alpha_i - beta))
    v = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, v_vals)

    w = torch.zeros(n + 1, device=device, dtype=dtype).scatter_add(0, idx, D)

    alpha2 = 2.0 * alpha_s
    # Fused prefix_suffix_sep (fused precompute instead of 7 PyTorch ops)
    U, V = _prefix_suffix_sep_fused(
        alpha2, u[:n].unsqueeze(0), v[1 : n + 1].unsqueeze(0)
    )
    U = U.squeeze(0)
    V = V.squeeze(0)

    W = torch.cumsum(w[:n], dim=0)

    # Fused matrix build (1 kernel instead of ~6 PyTorch ops)
    M = _FusedBuildMFn.apply(alpha_s, U, V, W)

    if inv_perm is not None:
        M = M[inv_perm][:, inv_perm]

    return M


# ============================================================
# Correctness + Gradient tests
# ============================================================
def _naive_X_KerLap(X, a, b):
    K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
    return X @ K


def run_tests():
    device = "cuda"
    torch.manual_seed(42)
    passed = failed = 0

    def check(name, got, ref, tol=2e-3):
        nonlocal passed, failed
        err = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
        ok = err < tol
        passed += ok
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:50s} err={err:.2e}")

    # --- Value tests ---
    print("=== Value tests: fused vs naive ===")
    for B, n, k in [(1, 50, 30), (4, 100, 200), (8, 200, 100), (2, 30, 50)]:
        a = torch.sort(torch.randn(n, device=device))[0]
        b = torch.randn(k, device=device)
        X = torch.randn(B, n, device=device)
        got = X_KerLap_fused(X, a, b, threshold=0, a_sorted=True)
        ref = _naive_X_KerLap(X, a, b)
        check(f"xA B={B} n={n} k={k}", got, ref)

    # xAT path
    for B, n, k in [(4, 200, 50), (8, 500, 100)]:
        a = torch.randn(n, device=device)
        b = torch.sort(torch.randn(k, device=device))[0]
        X = torch.randn(B, n, device=device)
        got = X_KerLap_fused(X, a, b, threshold=0, b_sorted=True)
        ref = _naive_X_KerLap(X, a, b)
        check(f"xAT B={B} n={n} k={k}", got, ref)

    # unsorted
    a = torch.randn(100, device=device)
    b = torch.randn(200, device=device)
    X = torch.randn(4, 100, device=device)
    got = X_KerLap_fused(X, a, b, threshold=0)
    ref = _naive_X_KerLap(X, a, b)
    check("unsorted a,b", got, ref)

    # --- KerLap_X value tests ---
    print("\n=== Value tests: KerLap_X_fused vs naive ===")
    for n, k, C in [(50, 30, 4), (100, 200, 8), (200, 50, 1)]:
        a = torch.sort(torch.randn(n, device=device))[0]
        b = torch.randn(k, device=device)
        X = torch.randn(k, C, device=device)
        K = torch.exp(-(a.unsqueeze(1) - b.unsqueeze(0)).abs())
        ref = K @ X
        got = KerLap_X_fused(X, a, b, threshold=0, a_sorted=True)
        check(f"KerLap_X n={n} k={k} C={C}", got, ref)

    # --- Ker_D_KerT value tests ---
    print("\n=== Value tests: Ker_D_KerT_fused vs naive ===")
    for n, k in [(50, 100), (30, 200), (80, 60)]:
        alpha = torch.sort(torch.randn(n, device=device))[0]
        beta = torch.randn(k, device=device)
        D = torch.randn(k, device=device)
        K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
        ref = (K * D.unsqueeze(0)) @ K.T
        got = Ker_D_KerT_fused(alpha, beta, D, threshold=0, alpha_sorted=True)
        check(f"Ker_D_KerT n={n} k={k}", got, ref, tol=5e-3)

    # unsorted
    alpha = torch.randn(50, device=device)
    beta = torch.randn(100, device=device)
    D = torch.randn(100, device=device)
    K = torch.exp(-(alpha.unsqueeze(1) - beta.unsqueeze(0)).abs())
    ref = (K * D.unsqueeze(0)) @ K.T
    got = Ker_D_KerT_fused(alpha, beta, D, threshold=0, alpha_sorted=False)
    check("Ker_D_KerT unsorted", got, ref, tol=5e-3)

    # --- Gradient tests ---
    print("\n=== Gradient tests: X_KerLap_fused vs naive ===")

    for n, k, desc in [(20, 30, "xA"), (40, 20, "xAT")]:
        a = torch.sort(torch.randn(n, device=device))[0]
        b = torch.sort(torch.randn(k, device=device))[0]
        X = torch.randn(4, n, device=device)

        a1 = a.clone().requires_grad_(True)
        b1 = b.clone().requires_grad_(True)
        x1 = X.clone().requires_grad_(True)
        X_KerLap_fused(
            x1, a1, b1, threshold=0, a_sorted=True, b_sorted=True
        ).sum().backward()

        a2 = a.clone().requires_grad_(True)
        b2 = b.clone().requires_grad_(True)
        x2 = X.clone().requires_grad_(True)
        _naive_X_KerLap(x2, a2, b2).sum().backward()

        ea = (a1.grad - a2.grad).abs().max().item() / (
            a2.grad.abs().max().item() + 1e-8
        )
        eb = (b1.grad - b2.grad).abs().max().item() / (
            b2.grad.abs().max().item() + 1e-8
        )
        ex = (x1.grad - x2.grad).abs().max().item() / (
            x2.grad.abs().max().item() + 1e-8
        )
        ok = max(ea, eb, ex) < 0.02
        passed += ok
        failed += not ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  X_KerLap {desc:5s} ea={ea:.2e} eb={eb:.2e} ex={ex:.2e}"
        )

    # KerLap_X gradient test
    print("\n=== Gradient tests: KerLap_X_fused vs naive ===")
    n, k, C = 30, 40, 4
    a = torch.sort(torch.randn(n, device=device))[0]
    b = torch.randn(k, device=device)
    X = torch.randn(k, C, device=device)

    a1 = a.clone().requires_grad_(True)
    b1 = b.clone().requires_grad_(True)
    x1 = X.clone().requires_grad_(True)
    KerLap_X_fused(x1, a1, b1, threshold=0, a_sorted=True).sum().backward()

    a2 = a.clone().requires_grad_(True)
    b2 = b.clone().requires_grad_(True)
    x2 = X.clone().requires_grad_(True)
    K = torch.exp(-(a2.unsqueeze(1) - b2.unsqueeze(0)).abs())
    (K @ x2).sum().backward()

    ea = (a1.grad - a2.grad).abs().max().item() / (a2.grad.abs().max().item() + 1e-8)
    eb = (b1.grad - b2.grad).abs().max().item() / (b2.grad.abs().max().item() + 1e-8)
    ex = (x1.grad - x2.grad).abs().max().item() / (x2.grad.abs().max().item() + 1e-8)
    ok = max(ea, eb, ex) < 0.02
    passed += ok
    failed += not ok
    print(
        f"  {'PASS' if ok else 'FAIL'}  KerLap_X   ea={ea:.2e} eb={eb:.2e} ex={ex:.2e}"
    )

    # Ker_D_KerT gradient test
    print("\n=== Gradient tests: Ker_D_KerT_fused vs naive ===")
    n, k = 30, 50
    alpha = torch.sort(torch.randn(n, device=device))[0]
    beta = torch.randn(k, device=device)
    D = torch.randn(k, device=device)

    al1 = alpha.clone().requires_grad_(True)
    be1 = beta.clone().requires_grad_(True)
    d1 = D.clone().requires_grad_(True)
    Ker_D_KerT_fused(al1, be1, d1, threshold=0, alpha_sorted=True).sum().backward()

    al2 = alpha.clone().requires_grad_(True)
    be2 = beta.clone().requires_grad_(True)
    d2 = D.clone().requires_grad_(True)
    K = torch.exp(-(al2.unsqueeze(1) - be2.unsqueeze(0)).abs())
    ((K * d2.unsqueeze(0)) @ K.T).sum().backward()

    ea = (al1.grad - al2.grad).abs().max().item() / (al2.grad.abs().max().item() + 1e-8)
    eb = (be1.grad - be2.grad).abs().max().item() / (be2.grad.abs().max().item() + 1e-8)
    ed = (d1.grad - d2.grad).abs().max().item() / (d2.grad.abs().max().item() + 1e-8)
    ok = max(ea, eb, ed) < 0.02
    passed += ok
    failed += not ok
    print(
        f"  {'PASS' if ok else 'FAIL'}  Ker_D_KerT ea={ea:.2e} eb={eb:.2e} ed={ed:.2e}"
    )

    print(f"\nTotal: {passed} PASS, {failed} FAIL\n")
    return failed == 0


# ============================================================
# Benchmarks: fused vs non-fused
# ============================================================
def _clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def bench_fwd(fn, warmup=10, iters=50):
    _clear()
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_fb(fn_create, warmup=10, iters=50):
    _clear()

    def step():
        args = fn_create()
        args[0](*args[1:]).sum().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _plot_comparison(
    axes_or_ax, title, logTs, orig_vals, fused_vals, ylabel="Time (ms)"
):
    """Helper: plots original vs fused on one axis."""
    import numpy as np

    ax = axes_or_ax
    o = np.array(orig_vals, dtype=float)
    f = np.array(fused_vals, dtype=float)
    ax.plot(logTs, o, "o-", color="#e74c3c", label="Original (sorted=True)", lw=2, ms=7)
    ax.plot(logTs, f, "^-", color="#2ecc71", label="Fused (sorted=True)", lw=2, ms=7)
    sp = o / f
    for i, (l, s) in enumerate(zip(logTs, sp)):
        if not np.isnan(s) and s > 1.02:
            ax.annotate(
                f"{s:.2f}x",
                xy=(l, f[i]),
                xytext=(0, -18),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                color="#27ae60",
                fontweight="bold",
            )
    ax.set_yscale("log")
    ax.set_xlabel("dimension (log2)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(logTs)
    ax.set_xticklabels([f"$2^{{{l}}}$" for l in logTs], fontsize=9)


def run_benchmarks():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    device = "cuda"
    torch.manual_seed(0)
    B = 8

    from kerlap_ops import X_KerLap as X_KerLap_orig
    from kerlap_ops import KerLap_X as KerLap_X_orig
    from kerlap_ops import Ker_D_KerT as Ker_D_KerT_orig

    outdir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(outdir, exist_ok=True)

    # ========================================
    # 1. X_KerLap benchmarks (3 configs)
    # ========================================
    configs = [
        ("n=k", lambda l: (1 << l, 1 << l)),
        ("k=1024, n var.", lambda l: (1 << l, 1024)),
        ("n=1024, k var.", lambda l: (1024, 1 << l)),
    ]

    all_xk_results = []

    for title, nk_fn in configs:
        print(f"--- X_KerLap: {title} ---")
        logTs = list(range(10, 21))
        orig_fwd = []
        fused_fwd = []
        orig_fb = []
        fused_fb = []

        for logT in logTs:
            n, k = nk_fn(logT)
            print(f"  n={n}, k={k}")

            a = torch.sort(torch.randn(n, device=device))[0]
            b = torch.sort(torch.randn(k, device=device))[0]
            X = torch.randn(B, n, device=device)

            t_orig = bench_fwd(
                lambda: X_KerLap_orig(X, a, b, threshold=0)
            )  # , a_sorted=True, b_sorted=True))
            t_fused = bench_fwd(
                lambda: X_KerLap_fused(
                    X, a, b, threshold=0, a_sorted=True, b_sorted=True
                )
            )
            orig_fwd.append(t_orig)
            fused_fwd.append(t_fused)

            def mk_orig():
                a_ = a.detach().requires_grad_(True)
                b_ = b.detach().requires_grad_(True)
                x_ = X.detach().requires_grad_(True)
                # return (lambda x,a,b: X_KerLap_orig(x,a,b, threshold=0, a_sorted=True, b_sorted=True)), x_, a_, b_
                return (lambda x, a, b: X_KerLap_orig(x, a, b, threshold=0)), x_, a_, b_

            def mk_fused():
                a_ = a.detach().requires_grad_(True)
                b_ = b.detach().requires_grad_(True)
                x_ = X.detach().requires_grad_(True)
                return (
                    (
                        lambda x, a, b: X_KerLap_fused(
                            x, a, b, threshold=0, a_sorted=True, b_sorted=True
                        )
                    ),
                    x_,
                    a_,
                    b_,
                )

            t_orig_fb = bench_fb(mk_orig)
            t_fused_fb = bench_fb(mk_fused)
            orig_fb.append(t_orig_fb)
            fused_fb.append(t_fused_fb)
            _clear()

        all_xk_results.append((title, logTs, orig_fwd, fused_fwd, orig_fb, fused_fb))

    # X_KerLap plots
    for mode, mode_label in [("fwd", "Forward"), ("fb", "Forward + Backward")]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (title, logTs, of, ff, ofb, ffb) in zip(axes, all_xk_results):
            vals_o = of if mode == "fwd" else ofb
            vals_f = ff if mode == "fwd" else ffb
            _plot_comparison(ax, title, logTs, vals_o, vals_f)
        fig.suptitle(
            f"X_KerLap: Original vs Fused -- {mode_label}  (B={B}, sorted=True)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        path = os.path.join(outdir, f"fused_xkerlap_{mode}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)

    # ========================================
    # 2. KerLap_X benchmarks (3 configs)
    # ========================================
    kx_configs = [
        ("n=k", lambda l: (1 << l, 1 << l)),
        ("k=1024, n var.", lambda l: (1 << l, 1024)),
        ("n=1024, k var.", lambda l: (1024, 1 << l)),
    ]

    all_kx_results = []

    for title, nk_fn in kx_configs:
        print(f"--- KerLap_X: {title} ---")
        logTs = list(range(10, 21))
        orig_fwd = []
        fused_fwd = []
        orig_fb = []
        fused_fb = []

        for logT in logTs:
            n, k = nk_fn(logT)
            C = B  # columns of X
            print(f"  n={n}, k={k}")

            a = torch.sort(torch.randn(n, device=device))[0]
            b = torch.sort(torch.randn(k, device=device))[0]
            X = torch.randn(k, C, device=device)

            t_orig = bench_fwd(
                lambda: KerLap_X_orig(X, a, b, threshold=0)
            )  # , a_sorted=True, b_sorted=True))
            t_fused = bench_fwd(
                lambda: KerLap_X_fused(
                    X, a, b, threshold=0, a_sorted=True, b_sorted=True
                )
            )
            orig_fwd.append(t_orig)
            fused_fwd.append(t_fused)

            def mk_orig():
                a_ = a.detach().requires_grad_(True)
                b_ = b.detach().requires_grad_(True)
                x_ = X.detach().requires_grad_(True)
                # return (lambda x,a,b: KerLap_X_orig(x,a,b, threshold=0, a_sorted=True, b_sorted=True)), x_, a_, b_
                return (lambda x, a, b: KerLap_X_orig(x, a, b, threshold=0)), x_, a_, b_

            def mk_fused():
                a_ = a.detach().requires_grad_(True)
                b_ = b.detach().requires_grad_(True)
                x_ = X.detach().requires_grad_(True)
                return (
                    (
                        lambda x, a, b: KerLap_X_fused(
                            x, a, b, threshold=0, a_sorted=True, b_sorted=True
                        )
                    ),
                    x_,
                    a_,
                    b_,
                )

            t_orig_fb = bench_fb(mk_orig)
            t_fused_fb = bench_fb(mk_fused)
            orig_fb.append(t_orig_fb)
            fused_fb.append(t_fused_fb)
            _clear()

        all_kx_results.append((title, logTs, orig_fwd, fused_fwd, orig_fb, fused_fb))

    # KerLap_X plots
    for mode, mode_label in [("fwd", "Forward"), ("fb", "Forward + Backward")]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (title, logTs, of, ff, ofb, ffb) in zip(axes, all_kx_results):
            vals_o = of if mode == "fwd" else ofb
            vals_f = ff if mode == "fwd" else ffb
            _plot_comparison(ax, title, logTs, vals_o, vals_f)
        fig.suptitle(
            f"KerLap_X: Original vs Fused -- {mode_label}  (C={B}, sorted=True)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        path = os.path.join(outdir, f"fused_kerlapx_{mode}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)

    # ========================================
    # 3. Ker_D_KerT benchmarks (n=1024, k varies)
    # ========================================
    print("--- Ker_D_KerT: n=1024, k varies ---")
    n_fixed = 1024
    logKs = list(range(10, 21))
    kdk_orig_fwd = []
    kdk_fused_fwd = []
    kdk_orig_fb = []
    kdk_fused_fb = []

    for logK in logKs:
        k = 1 << logK
        print(f"  n={n_fixed}, k={k}")

        alpha = torch.sort(torch.randn(n_fixed, device=device))[0]
        beta = torch.randn(k, device=device)
        D = torch.randn(k, device=device)

        t_orig = bench_fwd(
            lambda: Ker_D_KerT_orig(alpha, beta, D, threshold=0)
        )  # , alpha_sorted=True))
        t_fused = bench_fwd(
            lambda: Ker_D_KerT_fused(alpha, beta, D, threshold=0, alpha_sorted=True)
        )
        kdk_orig_fwd.append(t_orig)
        kdk_fused_fwd.append(t_fused)

        def mk_orig():
            a_ = alpha.detach().requires_grad_(True)
            b_ = beta.detach().requires_grad_(True)
            d_ = D.detach().requires_grad_(True)
            # return (lambda a,b,d: Ker_D_KerT_orig(a,b,d, threshold=0, alpha_sorted=True)), a_, b_, d_
            return (lambda a, b, d: Ker_D_KerT_orig(a, b, d, threshold=0)), a_, b_, d_

        def mk_fused():
            a_ = alpha.detach().requires_grad_(True)
            b_ = beta.detach().requires_grad_(True)
            d_ = D.detach().requires_grad_(True)
            return (
                (
                    lambda a, b, d: Ker_D_KerT_fused(
                        a, b, d, threshold=0, alpha_sorted=True
                    )
                ),
                a_,
                b_,
                d_,
            )

        t_orig_fb = bench_fb(mk_orig)
        t_fused_fb = bench_fb(mk_fused)
        kdk_orig_fb.append(t_orig_fb)
        kdk_fused_fb.append(t_fused_fb)
        _clear()

    # Ker_D_KerT plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mode, mode_label, o_vals, f_vals in [
        (axes[0], "fwd", "Forward", kdk_orig_fwd, kdk_fused_fwd),
        (axes[1], "fb", "Forward + Backward", kdk_orig_fb, kdk_fused_fb),
    ]:
        _plot_comparison(ax, mode_label, logKs, o_vals, f_vals, ylabel="Time (ms)")
        ax.set_xlabel("k (inner dimension, log2)", fontsize=11)
    fig.suptitle(
        "Ker_D_KerT: Original vs Fused  (n=1024, sorted=True)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(outdir, "fused_kerdkert.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ============================================================
if __name__ == "__main__":
    run_tests()
    print("=" * 60)
    print("BENCHMARKS: Original vs Fused")
    print("=" * 60)
    run_benchmarks()
    print("\nDone!")
