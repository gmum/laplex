"""
prefix_suffix.py
================

Rozszerzenie prefix_suffix_dl.py o WYMIAR L (multi-alpha).

Dla kazdego l = 0..L-1 obliczamy niezaleznie:
    prefix(alpha[l], u[l])[b, i] = exp(-alpha[l,i]) * sum_{j<=i} exp(alpha[l,j]) * u[l,b,j]
    suffix(alpha[l], v[l])[b, i] = exp(+alpha[l,i]) * sum_{j>=i} exp(-alpha[l,j]) * v[l,b,j]

Wszystko w JEDNYM wywolaniu kernela. Grid: (L*B, tiles) — decoupled
lookback dziala per-batch (batch = l_idx*B + b), wiec roznic l nie
mieszamy. Stany state_* sa indeksowane przez sid = batch*num_tiles+tile,
per-batch offset = batch*num_tiles; mechanism identyczny jak w oryginale.

API jest analogiczne do prefix_suffix_dl.py:
    prefix_suffix_multi(alpha, u, v)    -> (L, B, T)
    prefix_suffix_sep_multi(alpha, u, v) -> (P: (L,B,T), S: (L,B,T))

gdzie:
    alpha : (L, T)     niemalejace po osi T
    u, v  : (L, B, T)

UWAGA: ten plik KOMPILUJE extension CUDA (prefix_suffix_ext).
"""

import os, math
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

_LOG2E = math.log2(math.e)

# CUDA 12.1 (Windows): zgadza sie z prefix_suffix_dl.py
_CUDA_121 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
if os.path.isdir(_CUDA_121):
    os.environ["CUDA_HOME"] = _CUDA_121
    os.environ["CUDA_PATH"] = _CUDA_121
    os.environ["PATH"] = os.path.join(_CUDA_121, "bin") + ";" + os.environ.get("PATH", "")
    import torch.utils.cpp_extension as _cext
    _cext.CUDA_HOME = _CUDA_121

_NVCC_FLAGS = [
    "-O3", "--use_fast_math",
    "--allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
]

# ============================================================
# CUDA source — kopia oryginalu z dodatkowym wymiarem L
# ============================================================
_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
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

// Multi-alpha FORWARD scan.
//   batch index blockIdx.x in [0, L*B),  l_idx = batch / B_per_L
//   n  : [L, T]   ef : [L, T]   x  : [L*B, T]   out_y : [L*B, T]
__global__ void dl_scan_fwd_fused_multi(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    const float* __restrict__ ef,
    float* out_y,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles, int B_per_L)
{
    int batch = blockIdx.x, tile = blockIdx.y;
    int sid   = batch * num_tiles + tile;
    int l_idx = batch / B_per_L;

    const float* x_b  = x     + (size_t)batch * T;
    float*       oy_b = out_y + (size_t)batch * T;
    const int*   n_l  = n     + (size_t)l_idx * T;
    const float* ef_l = ef    + (size_t)l_idx * T;

    extern __shared__ Pair shared[];
    int tid  = tile * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    int4   ni  = ((const int4*)n_l )[tid];
    float4 xi  = ((const float4*)x_b)[tid];
    float4 efi = ((const float4*)ef_l)[tid];
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
    int base=tid*4;
    float4 yy;
    yy.x = a0.x / efi.x;
    yy.y = a1.x / efi.y;
    yy.z = a2.x / efi.z;
    yy.w = a3.x / efi.w;
    *((float4*)(oy_b+base))=yy;
}

// Multi-alpha BACKWARD scan (prawa->lewa).
__global__ void dl_scan_bwd_fused_multi(
    const int*   __restrict__ n,
    const float* __restrict__ x,
    const float* __restrict__ ef,
    float* out_y,
    int* state_flag, int* state_ln, float* state_lx,
    int* state_pn,   float* state_px,
    int T, int T4, int num_tiles, int B_per_L)
{
    int batch = blockIdx.x, tile = blockIdx.y;
    int sid   = batch * num_tiles + tile;
    int l_idx = batch / B_per_L;

    const float* x_b  = x     + (size_t)batch * T;
    float*       oy_b = out_y + (size_t)batch * T;
    const int*   n_l  = n     + (size_t)l_idx * T;
    const float* ef_l = ef    + (size_t)l_idx * T;

    extern __shared__ Pair shared[];
    int tid     = tile * blockDim.x + threadIdx.x;
    int tid_rev = T4 - 1 - tid;
    int lane    = threadIdx.x & 31, warp = threadIdx.x >> 5;

    int4   ni  = ((const int4*)n_l)[tid_rev];
    float4 xi  = ((const float4*)x_b)[tid_rev];
    float4 efi = ((const float4*)ef_l)[tid_rev];
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
    float4 yy;
    yy.x = a3.x / efi.x;
    yy.y = a2.x / efi.y;
    yy.z = a1.x / efi.z;
    yy.w = a0.x / efi.w;
    *((float4*)(oy_b+base))=yy;
}

// Launcher.
//   n_fwd  : [L, T]   ef_fwd : [L, T]   x_fwd : [L, B, T]
//   n_bwd, ef_bwd, x_bwd analogicznie.
std::vector<at::Tensor> scan_ps_fused_multi(
    at::Tensor n_fwd, at::Tensor ef_fwd, at::Tensor x_fwd,
    at::Tensor n_bwd, at::Tensor ef_bwd, at::Tensor x_bwd)
{
    int L = x_fwd.size(0);
    int B = x_fwd.size(1);
    int T = x_fwd.size(2);
    int LB = L * B;
    int T4 = T / 4;
    const int threads = 256;
    int tiles = (T4 + threads - 1) / threads;
    int64_t ns = (int64_t)LB * tiles;

    auto out_fwd = at::empty({L, B, T}, x_fwd.options());
    auto out_bwd = at::empty({L, B, T}, x_bwd.options());

    auto oi = at::TensorOptions().dtype(at::kInt).device(x_fwd.device());
    auto of = at::TensorOptions().dtype(at::kFloat).device(x_fwd.device());
    int smem = 33 * (int)sizeof(Pair);
    dim3 grid(LB, tiles);

    auto sf0=at::zeros({ns},oi); auto ln0=at::empty({ns},oi);
    auto lx0=at::empty({ns},of); auto pn0=at::empty({ns},oi); auto px0=at::empty({ns},of);
    auto sf1=at::zeros({ns},oi); auto ln1=at::empty({ns},oi);
    auto lx1=at::empty({ns},of); auto pn1=at::empty({ns},oi); auto px1=at::empty({ns},of);

    dl_scan_fwd_fused_multi<<<grid,threads,smem>>>(
        n_fwd.data_ptr<int>(), x_fwd.data_ptr<float>(), ef_fwd.data_ptr<float>(),
        out_fwd.data_ptr<float>(),
        sf0.data_ptr<int>(), ln0.data_ptr<int>(), lx0.data_ptr<float>(),
        pn0.data_ptr<int>(), px0.data_ptr<float>(), T, T4, tiles, B);

    dl_scan_bwd_fused_multi<<<grid,threads,smem>>>(
        n_bwd.data_ptr<int>(), x_bwd.data_ptr<float>(), ef_bwd.data_ptr<float>(),
        out_bwd.data_ptr<float>(),
        sf1.data_ptr<int>(), ln1.data_ptr<int>(), lx1.data_ptr<float>(),
        pn1.data_ptr<int>(), px1.data_ptr<float>(), T, T4, tiles, B);

    return {out_fwd, out_bwd};
}

// Fused redukcja grad_alpha po osi B dla multi.
//   in:  u, sgp, v, pgs, gp, P, gs, S :  [L, B, T]
//   out: [L, T]
//   out[l, t] = sum_b { u*sgp - v*pgs - gp*P + gs*S }[l, b, t]
__global__ void grad_alpha_reduce_kernel_multi(
    const float* __restrict__ u,
    const float* __restrict__ sgp,
    const float* __restrict__ v,
    const float* __restrict__ pgs,
    const float* __restrict__ gp,
    const float* __restrict__ P,
    const float* __restrict__ gs,
    const float* __restrict__ S,
    float* __restrict__ out,
    int L, int B, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (t >= T) return;
    size_t base_l = (size_t)l * B * T;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
        size_t idx = base_l + (size_t)b * T + t;
        acc += u[idx] * sgp[idx] - v[idx] * pgs[idx]
             - gp[idx] * P[idx]  + gs[idx] * S[idx];
    }
    out[(size_t)l * T + t] = acc;
}

at::Tensor grad_alpha_reduce_multi(
    at::Tensor u, at::Tensor sgp, at::Tensor v, at::Tensor pgs,
    at::Tensor gp, at::Tensor P, at::Tensor gs, at::Tensor S)
{
    int L = u.size(0), B = u.size(1), T = u.size(2);
    auto out = at::empty({L, T}, u.options());
    int threads = 256;
    dim3 blocks((T + threads - 1) / threads, L);
    grad_alpha_reduce_kernel_multi<<<blocks, threads>>>(
        u.data_ptr<float>(), sgp.data_ptr<float>(),
        v.data_ptr<float>(), pgs.data_ptr<float>(),
        gp.data_ptr<float>(), P.data_ptr<float>(),
        gs.data_ptr<float>(), S.data_ptr<float>(),
        out.data_ptr<float>(), L, B, T);
    return out;
}

// Two-tau fused grad_alpha redukcja.  Wykonuje OBIE redukcje (grad_aL i
// grad_aR) w jednym kernelu, czytajac kazdy z 8 tensorow wejsciowych
// tylko raz.
//   in:  u, sgp, gp, P, v, pgs, gs, S : [L, B, T]
//   out: outL, outR : [L, T]
//   outL[l,t] = sum_b { u*sgp - gp*P     }[l, b, t]
//   outR[l,t] = sum_b { gs*S  - v*pgs    }[l, b, t]
__global__ void grad_alpha_two_tau_kernel_multi(
    const float* __restrict__ u,
    const float* __restrict__ sgp,
    const float* __restrict__ gp,
    const float* __restrict__ P,
    const float* __restrict__ v,
    const float* __restrict__ pgs,
    const float* __restrict__ gs,
    const float* __restrict__ S,
    float* __restrict__ outL,
    float* __restrict__ outR,
    int L, int B, int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y;
    if (t >= T) return;
    size_t base_l = (size_t)l * B * T;
    float accL = 0.0f, accR = 0.0f;
    for (int b = 0; b < B; ++b) {
        size_t idx = base_l + (size_t)b * T + t;
        accL += u [idx] * sgp[idx] - gp[idx] * P  [idx];
        accR += gs[idx] * S  [idx] - v [idx] * pgs[idx];
    }
    outL[(size_t)l * T + t] = accL;
    outR[(size_t)l * T + t] = accR;
}

std::vector<at::Tensor> grad_alpha_two_tau_multi(
    at::Tensor u,   at::Tensor sgp,
    at::Tensor gp,  at::Tensor P,
    at::Tensor v,   at::Tensor pgs,
    at::Tensor gs,  at::Tensor S)
{
    int L = u.size(0), B = u.size(1), T = u.size(2);
    auto outL = at::empty({L, T}, u.options());
    auto outR = at::empty({L, T}, u.options());
    int threads = 256;
    dim3 blocks((T + threads - 1) / threads, L);
    grad_alpha_two_tau_kernel_multi<<<blocks, threads>>>(
        u.data_ptr<float>(),  sgp.data_ptr<float>(),
        gp.data_ptr<float>(), P.data_ptr<float>(),
        v.data_ptr<float>(),  pgs.data_ptr<float>(),
        gs.data_ptr<float>(), S.data_ptr<float>(),
        outL.data_ptr<float>(), outR.data_ptr<float>(),
        L, B, T);
    return {outL, outR};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan_ps_fused_multi", &scan_ps_fused_multi,
          "multi-alpha fused prefix+suffix DL scan");
    m.def("grad_alpha_reduce_multi", &grad_alpha_reduce_multi,
          "multi-alpha fused grad_alpha reduction");
    m.def("grad_alpha_two_tau_multi", &grad_alpha_two_tau_multi,
          "multi-alpha fused grad_aL+grad_aR reduction (two-tau)");
}
"""

print("Compiling prefix_suffix extension...", end=" ", flush=True)
_ext_multi = load_inline(
    name="prefix_suffix_ext",
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
_PAD_TO  = _THREADS * 4  # 1024


def _precompute_multi(alpha: torch.Tensor):
    """
    alpha : (L, T) niemalejace po dim=1
    Zwraca: ef (L,T), n_fwd (L,T), ef_neg (L,T), n_bwd (L,T).
    """
    ap  =  alpha.float() * _LOG2E
    n_f = ap.floor().to(torch.int32)
    ef  = torch.exp2(ap - n_f.float())

    an  = -ap
    n_b = an.floor().to(torch.int32)
    ef_neg = torch.exp2(an - n_b.float())
    return ef, n_f, ef_neg, n_b


def _pad_last_2d(n1d_LT: torch.Tensor, T: int):
    """Pad (L,T) -> (L, T_pad) gdzie T_pad = ceil(T/1024)*1024. Replikuje ostatnia kolumne."""
    rem = T % _PAD_TO
    if rem == 0:
        return n1d_LT, T
    pad = _PAD_TO - rem
    last = n1d_LT[:, -1:].expand(-1, pad)
    return torch.cat([n1d_LT, last], dim=1), T


def _pad_last_3d(x_LBT: torch.Tensor, T: int):
    """Pad (L,B,T) -> (L,B,T_pad). Zeros padding."""
    rem = T % _PAD_TO
    if rem == 0:
        return x_LBT, T
    pad = _PAD_TO - rem
    return torch.nn.functional.pad(x_LBT, (0, pad)), T


def _ps_raw_multi(ef, n_fwd, ef_neg, n_bwd, u: torch.Tensor, v: torch.Tensor):
    """
    ef, n_fwd, ef_neg, n_bwd : (L, T)
    u, v                     : (L, B, T)
    Zwraca: prefix (L,B,T), suffix (L,B,T) — nieobciete do T_orig, wlasciwe wartosci na [..., :T].
    """
    L, B, T = u.shape
    u_pad, T_orig   = _pad_last_3d(u,     T)
    v_pad, _        = _pad_last_3d(v,     T)
    ef_pad, _       = _pad_last_2d(ef,    T)
    ef_neg_pad, _   = _pad_last_2d(ef_neg,T)
    n_f_pad, _      = _pad_last_2d(n_fwd, T)
    n_b_pad, _      = _pad_last_2d(n_bwd, T)

    out_f, out_b = _ext_multi.scan_ps_fused_multi(
        n_f_pad.contiguous(), ef_pad.contiguous(),     u_pad.contiguous(),
        n_b_pad.contiguous(), ef_neg_pad.contiguous(), v_pad.contiguous(),
    )
    return out_f[..., :T_orig], out_b[..., :T_orig]


# ============================================================
# Autograd
# ============================================================
class _PrefSuffDiffMultiFn(Function):
    """
    (prefix(alpha_l, u_l), suffix(alpha_l, v_l)) dla kazdego l rownolegle.

    Forward:  alpha (L,T); u,v (L,B,T)  ->  P,S (L,B,T)
    Backward: analogicznie do _PrefSuffDiffFn, suma po B przez fused kernel.
    """

    @staticmethod
    def forward(ctx, alpha, u, v):
        orig_dtype = alpha.dtype
        a32 = alpha.float(); u32 = u.float(); v32 = v.float()
        ef, n_fwd, ef_neg, n_bwd = _precompute_multi(a32)
        P, S = _ps_raw_multi(ef, n_fwd, ef_neg, n_bwd, u32, v32)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u32, v32, P, S)
        ctx.orig_dtype = orig_dtype
        return P.to(orig_dtype), S.to(orig_dtype)

    @staticmethod
    def backward(ctx, gP, gS):
        ef, n_fwd, ef_neg, n_bwd, u, v, P, S = ctx.saved_tensors
        od = ctx.orig_dtype
        gP32 = gP.float().contiguous()
        gS32 = gS.float().contiguous()

        prefix_gS, suffix_gP = _ps_raw_multi(ef, n_fwd, ef_neg, n_bwd, gS32, gP32)

        grad_u = suffix_gP.to(od) if ctx.needs_input_grad[1] else None
        grad_v = prefix_gS.to(od) if ctx.needs_input_grad[2] else None

        grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_alpha = _ext_multi.grad_alpha_reduce_multi(
                u.contiguous(), suffix_gP.contiguous(),
                v.contiguous(), prefix_gS.contiguous(),
                gP32, P.contiguous(),
                gS32, S.contiguous(),
            ).to(od)

        return grad_alpha, grad_u, grad_v


def prefix_suffix_sep_multi(alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None):
    """
    Zwraca (prefix, suffix) dla kazdego l rownolegle.

    alpha : (L, T)   niemalejace po T
    u, v  : (L, B, T)   (jesli v is None, v = u)
    zwraca: (P, S) oba (L, B, T)
    """
    if v is None:
        v = u
    return _PrefSuffDiffMultiFn.apply(alpha, u, v)


# ============================================================
# Naiwna referencja
# ============================================================
def _naive_prefix_multi(alpha, u):
    # alpha (L,T), u (L,B,T)
    ea  = torch.exp(alpha).unsqueeze(1)   # (L,1,T)
    ena = torch.exp(-alpha).unsqueeze(1)
    return (u * ea).cumsum(dim=-1) * ena


def _naive_suffix_multi(alpha, v):
    ea  = torch.exp(alpha).unsqueeze(1)
    ena = torch.exp(-alpha).unsqueeze(1)
    return (v * ena).flip(-1).cumsum(dim=-1).flip(-1) * ea


# ============================================================
# Testy
# ============================================================
def run_correctness_tests() -> bool:
    device = "cuda"
    torch.manual_seed(0)
    passed = failed = 0

    def chk(name, got, ref, tol=3e-3):
        nonlocal passed, failed
        err = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
        ok = err < tol
        passed += ok; failed += not ok
        print(f"   {'PASS' if ok else 'FAIL'}  {name:40s}  err={err:.2e}")

    print("1. prefix/suffix multi vs naive:")
    for L, B, T in [(2, 1, 1024), (4, 2, 1024), (8, 4, 2048), (3, 2, 4096)]:
        alpha = torch.stack([
            torch.linspace(-2.0 + 0.3*i, 3.0 + 0.3*i, T, device=device)
            for i in range(L)
        ])
        u = torch.randn(L, B, T, device=device)
        v = torch.randn(L, B, T, device=device)

        ef, n_f, ef_neg, n_b = _precompute_multi(alpha)
        P, S = _ps_raw_multi(ef, n_f, ef_neg, n_b, u, v)
        P_ref = _naive_prefix_multi(alpha, u)
        S_ref = _naive_suffix_multi(alpha, v)
        chk(f"L={L} B={B} T={T} prefix", P, P_ref)
        chk(f"L={L} B={B} T={T} suffix", S, S_ref)

    print("\n2. Gradient multi vs naive (float32):")
    for L, B, T in [(4, 2, 1024), (8, 4, 2048)]:
        alpha0 = torch.stack([
            torch.linspace(0.0, 5.0 + 0.2*i, T, device=device) for i in range(L)
        ])
        u0 = torch.randn(L, B, T, device=device)
        v0 = torch.randn(L, B, T, device=device)

        a1 = alpha0.clone().requires_grad_(True)
        u1 = u0.clone().requires_grad_(True)
        v1 = v0.clone().requires_grad_(True)
        P1, S1 = prefix_suffix_sep_multi(a1, u1, v1)
        (P1 + S1).sum().backward()

        a2 = alpha0.clone().requires_grad_(True)
        u2 = u0.clone().requires_grad_(True)
        v2 = v0.clone().requires_grad_(True)
        P2 = _naive_prefix_multi(a2, u2)
        S2 = _naive_suffix_multi(a2, v2)
        (P2 + S2).sum().backward()

        def re(x, y):
            return ((x - y).abs() / y.abs().clamp(min=1e-6)).max().item()
        eu = re(u1.grad, u2.grad)
        ev = re(v1.grad, v2.grad)
        ea = re(a1.grad, a2.grad)
        ok = eu < 1e-2 and ev < 1e-2 and ea < max(2e-2, T * 1e-5)
        passed += ok; failed += not ok
        print(f"   {'PASS' if ok else 'FAIL'}  L={L} B={B} T={T}  "
              f"eu={eu:.1e}  ev={ev:.1e}  ea={ea:.1e}")

    print(f"\nWyniki: {passed} PASS, {failed} FAIL\n")
    return failed == 0


if __name__ == "__main__":
    run_correctness_tests()
