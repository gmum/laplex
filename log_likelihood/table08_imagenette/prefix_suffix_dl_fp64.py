"""prefix_suffix_dl_fp64.py — fp64 wariant DL-scan CUDA extension.

Kopia `prefix_suffix_dl.py` z float → double.  Używamy gdy fp32 precision
nie wystarcza (np. complex-kernel model z signed-D cancellation).

Zmiany vs fp32:
  - struct Pair: float x → double x
  - ldexpf/expf → ldexp/exp
  - float4 vec load → scalar x4 (nvcc coalesce-uje; LD.128 == 2×LD.64)
  - state buffers: float → double
  - torch::kFloat32 → torch::kFloat64

Warp shuffle w CUDA natywnie obsługuje `double` (__shfl_up_sync z double
parametrem, sm_30+).  `volatile double*` loads są atomowe dla wyrównanych
64-bitowych dostępów (Pascal+).

Performance: ~2× wolniej od fp32 (wymiar danych 2× większy), ale stabilne
dla single-tau scan-Gramu z signed-D weights.
"""

import os, sys, math, time
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

_LOG2E = math.log2(math.e)

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

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Pair: int32 exponent + fp64 mantissa (fp64 port of prefix_suffix_dl).
struct Pair { int n; double x; };

__device__ __forceinline__ Pair combine(Pair a, Pair b) {
    return {b.n, b.x + ldexp(a.x, a.n - b.n)};
}

__device__ __forceinline__ Pair warp_scan(Pair val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        Pair other;
        other.n = __shfl_up_sync(0xffffffff, val.n, offset);
        other.x = __shfl_up_sync(0xffffffff, val.x, offset);  // native double shuffle
        if ((threadIdx.x & 31) >= offset)
            val = combine(other, val);
    }
    return val;
}

#define FLAG_UNAVAIL 0
#define FLAG_LOCAL   1
#define FLAG_PREFIX  2

// -------------------------------------------------------------------
// FORWARD scan (prefix)
// -------------------------------------------------------------------
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

    // 4 scalar loads -- nvcc coalesces to efficient access.
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

// -------------------------------------------------------------------
// BACKWARD scan (suffix) -- reverse iteration
// -------------------------------------------------------------------
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

    // Load from reversed position (reverse order within the quad: .w to .x)
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
    int base = tid_rev * 4;
    // a3 -> position base+0, a0 -> base+3 (reversed order inside the quad)
    oy_b[base+0] = a3.x / e3;  // note: e3 corresponds to base+0 (see load above)
    oy_b[base+1] = a2.x / e2;
    oy_b[base+2] = a1.x / e1;
    oy_b[base+3] = a0.x / e0;
}

// Launcher
std::vector<at::Tensor> scan_ps_fused(
    at::Tensor n_fwd, at::Tensor ef_fwd, at::Tensor x_fwd,
    at::Tensor n_bwd, at::Tensor ef_bwd, at::Tensor x_bwd)
{
    int B=x_fwd.size(0), T=x_fwd.size(1), T4=T/4;
    const int threads=256;
    int tiles=(T4+threads-1)/threads;
    int64_t ns=(int64_t)B*tiles;

    auto out_fwd=at::empty({B,T},x_fwd.options());
    auto out_bwd=at::empty({B,T},x_bwd.options());

    auto oi=at::TensorOptions().dtype(at::kInt).device(x_fwd.device());
    auto of=at::TensorOptions().dtype(at::kDouble).device(x_fwd.device());
    int smem=33*(int)sizeof(Pair);
    dim3 grid(B, tiles);

    auto sf0=at::zeros({ns},oi); auto ln0=at::empty({ns},oi);
    auto lx0=at::empty({ns},of); auto pn0=at::empty({ns},oi); auto px0=at::empty({ns},of);
    auto sf1=at::zeros({ns},oi); auto ln1=at::empty({ns},oi);
    auto lx1=at::empty({ns},of); auto pn1=at::empty({ns},oi); auto px1=at::empty({ns},of);

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

// grad_alpha_reduce -- fp64
__global__ void grad_alpha_reduce_kernel(
    const double* __restrict__ u,
    const double* __restrict__ sgp,
    const double* __restrict__ v,
    const double* __restrict__ pgs,
    const double* __restrict__ gp,
    const double* __restrict__ P,
    const double* __restrict__ gs,
    const double* __restrict__ S,
    double* __restrict__ out,
    int B, int T)
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

at::Tensor grad_alpha_reduce(
    at::Tensor u, at::Tensor sgp, at::Tensor v, at::Tensor pgs,
    at::Tensor gp, at::Tensor P, at::Tensor gs, at::Tensor S)
{
    int B = u.size(0), T = u.size(1);
    auto out = at::empty({T}, u.options());
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
    m.def("scan_ps_fused", &scan_ps_fused, "fused fp64 DL scan");
    m.def("grad_alpha_reduce", &grad_alpha_reduce, "fused fp64 grad_alpha");
}
"""

print("Compiling prefix_suffix_fp64 extension...", end=" ", flush=True)
_ext = load_inline(
    name="prefix_suffix_dl_fused_fp64",
    cpp_sources=[""],
    cuda_sources=[_CUDA_SRC],
    extra_cuda_cflags=_NVCC_FLAGS,
    verbose=False,
)
print("OK")


_THREADS = 256
_PAD_TO  = _THREADS * 4   # 1024


def _precompute(alpha: torch.Tensor):
    """α → (ef, n_fwd, ef_neg, n_bwd) z rozkładem na mantysę fp64 i int32 exponent."""
    ap  =  alpha.double() * _LOG2E
    n_f = ap.floor().to(torch.int32)
    ef  = torch.exp2(ap - n_f.double())

    an  = -ap
    n_b = an.floor().to(torch.int32)
    ef_neg = torch.exp2(an - n_b.double())

    return ef, n_f, ef_neg, n_b


def _pad_pair(x: torch.Tensor, n1d: torch.Tensor):
    T   = x.size(1)
    rem = T % _PAD_TO
    if rem == 0:
        return x, n1d, T
    pad = _PAD_TO - rem
    x_p = torch.nn.functional.pad(x, (0, pad))
    n_p = torch.cat([n1d, n1d[-1:].expand(pad)])
    return x_p, n_p, T


def _pad_ef(ef: torch.Tensor, T: int):
    rem = T % _PAD_TO
    if rem == 0:
        return ef
    pad = _PAD_TO - rem
    return torch.cat([ef, ef[-1:].expand(pad)])


def _ps_raw(ef, n_fwd, ef_neg, n_bwd, u: torch.Tensor, v: torch.Tensor):
    T = u.size(1)
    u_pad, n_f_pad, T_orig = _pad_pair(u, n_fwd)
    v_pad, n_b_pad, _      = _pad_pair(v, n_bwd)
    ef_pad     = _pad_ef(ef,     T)
    ef_neg_pad = _pad_ef(ef_neg, T)

    out_f, out_b = _ext.scan_ps_fused(
        n_f_pad.contiguous(), ef_pad.contiguous(),     u_pad.contiguous(),
        n_b_pad.contiguous(), ef_neg_pad.contiguous(), v_pad.contiguous(),
    )
    return out_f[:, :T_orig], out_b[:, :T_orig]


class _PrefSuffDiffFn(Function):
    @staticmethod
    def forward(ctx, alpha, u, v):
        orig_dtype = alpha.dtype
        a64 = alpha.double(); u64 = u.double(); v64 = v.double()
        ef, n_fwd, ef_neg, n_bwd = _precompute(a64)
        P, S = _ps_raw(ef, n_fwd, ef_neg, n_bwd, u64, v64)
        ctx.save_for_backward(ef, n_fwd, ef_neg, n_bwd, u64, v64, P, S)
        ctx.orig_dtype = orig_dtype
        return P.to(orig_dtype), S.to(orig_dtype)

    @staticmethod
    def backward(ctx, gP, gS):
        ef, n_fwd, ef_neg, n_bwd, u, v, P, S = ctx.saved_tensors
        od = ctx.orig_dtype
        gP64 = gP.double(); gS64 = gS.double()

        prefix_gS, suffix_gP = _ps_raw(ef, n_fwd, ef_neg, n_bwd, gS64, gP64)

        grad_u = suffix_gP.to(od) if ctx.needs_input_grad[1] else None
        grad_v = prefix_gS.to(od) if ctx.needs_input_grad[2] else None

        grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_alpha = _ext.grad_alpha_reduce(
                u.contiguous(), suffix_gP.contiguous(),
                v.contiguous(), prefix_gS.contiguous(),
                gP64.contiguous(), P.contiguous(),
                gS64.contiguous(), S.contiguous(),
            ).to(od)

        return grad_alpha, grad_u, grad_v


def prefix_suffix_sep(alpha, u, v=None):
    """Zwraca (prefix(α,u), suffix(α,v)), fp64 internal."""
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
