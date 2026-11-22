"""
_magnitude.py
=============

The core two-tau magnitude primitive shared by `kerlap_real.py` and
`kerlap_complex.py`.

    K[l, k, j] = exp( -|a[l,k] - b[l,j]| / tau_sign )
    tau_sign   = tau_right  if a[l,k] > b[l,j]  else  tau_left

Two paths:

* `_magnitude_apply_dense` -- reference O(L n^2) softmax+einsum. Used on
  CPU, on non-float32 dtypes, and as the correctness baseline.
* `_magnitude_apply_scan`  -- fused O(L n) via decoupled-lookback prefix
  scan (`prefix_suffix_sep_multi` from prefix_suffix.py).
  Requires CUDA and float32 input. Handles two temperatures by running
  one scan on a_sorted/tau_L (prefix -> left side) and another on
  a_sorted/tau_R (suffix -> right side), both in a SINGLE scan launch
  by stacking along the L axis (2L rows).

`magnitude_apply(X, a, b, tau_L, tau_R, normalize)` picks the scan when
it can, dense otherwise. Complex X is handled by splitting into real and
imaginary parts (the magnitude kernel is real, so K @ (Re+i Im) = K@Re +
i K@Im).
"""

from __future__ import annotations

import os
import sys

import torch

# Make prefix_suffix importable from this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from prefix_suffix import (
        prefix_suffix_sep_multi,
        _precompute_multi,
        _pad_last_2d,
        _pad_last_3d,
        _ext_multi,
    )
    _HAS_SCAN = True
except Exception as _e:
    prefix_suffix_sep_multi = None
    _HAS_SCAN = False
    _SCAN_IMPORT_ERR = _e


# ----------------------------------------------------------------------------
# Two-tau prefix/suffix Function (single scan launch, no L doubling)
# ----------------------------------------------------------------------------

def _scan_ps_run(efL, n_fwd_L, ef_neg_R, n_bwd_R, u, v):
    """Single launch of `scan_ps_fused_multi` with forward side driven by
    alpha_L and backward side by alpha_R. Handles padding to 1024.
    Inputs: efL, n_fwd_L, ef_neg_R, n_bwd_R (L,T); u, v (L,B,T).
    Returns (P, S) (L,B,T) sliced back to T.
    """
    L, B, T = u.shape
    u_pad, T_orig = _pad_last_3d(u, T)
    v_pad, _      = _pad_last_3d(v, T)
    efL_pad, _    = _pad_last_2d(efL, T)
    efR_neg_pad, _= _pad_last_2d(ef_neg_R, T)
    nfL_pad, _    = _pad_last_2d(n_fwd_L, T)
    nbR_pad, _    = _pad_last_2d(n_bwd_R, T)
    out_f, out_b = _ext_multi.scan_ps_fused_multi(
        nfL_pad.contiguous(), efL_pad.contiguous(),     u_pad.contiguous(),
        nbR_pad.contiguous(), efR_neg_pad.contiguous(), v_pad.contiguous(),
    )
    return out_f[..., :T_orig], out_b[..., :T_orig]


class _TwoTauPrefSuffFn(torch.autograd.Function):
    """prefix(alpha_L, u) and suffix(alpha_R, v) in a single kernel launch."""

    @staticmethod
    def forward(ctx, alpha_L, alpha_R, u, v):
        aL32 = alpha_L.float(); aR32 = alpha_R.float()
        u32  = u.float();       v32  = v.float()
        # Precompute both alphas in full (efL/nFL/efnL/nBL and efR/nFR/efnR/nBR)
        # and save everything for backward — avoids computing the precompute twice.
        efL, n_fwd_L, ef_neg_L, n_bwd_L = _precompute_multi(aL32)
        efR, n_fwd_R, ef_neg_R, n_bwd_R = _precompute_multi(aR32)
        P, S = _scan_ps_run(efL, n_fwd_L, ef_neg_R, n_bwd_R, u32, v32)
        ctx.save_for_backward(
            u32, v32, P, S,
            efL, n_fwd_L, ef_neg_L, n_bwd_L,
            efR, n_fwd_R, ef_neg_R, n_bwd_R,
        )
        return P, S

    @staticmethod
    def backward(ctx, gP, gS):
        (u32, v32, P, S,
         efL, n_fwd_L, ef_neg_L, n_bwd_L,
         efR, n_fwd_R, ef_neg_R, n_bwd_R) = ctx.saved_tensors
        gP32 = gP.float().contiguous()
        gS32 = gS.float().contiguous()
        # du needs suffix(alpha_L, gP); dv needs prefix(alpha_R, gS).
        # Run single launch with fwd side = alpha_R (for prefix of gS) and
        # bwd side = alpha_L (for suffix of gP). Inputs swap: u=gS, v=gP.
        prefix_gS, suffix_gP = _scan_ps_run(
            efR, n_fwd_R, ef_neg_L, n_bwd_L, gS32, gP32)
        grad_u = suffix_gP if ctx.needs_input_grad[2] else None
        grad_v = prefix_gS if ctx.needs_input_grad[3] else None
        grad_aL = grad_aR = None
        need_aL = ctx.needs_input_grad[0]
        need_aR = ctx.needs_input_grad[1]
        if need_aL or need_aR:
            # Fused CUDA: one pass over all 8 tensors, returns both
            # reductions grad_aL, grad_aR. Replaces 4 einsums. All
            # inputs must be row-major contiguous [L,B,T] — P/S after the scan
            # are slices of the padded result, hence the forced .contiguous().
            gaL, gaR = _ext_multi.grad_alpha_two_tau_multi(
                u32.contiguous(),        suffix_gP.contiguous(),
                gP32,                    P.contiguous(),
                v32.contiguous(),        prefix_gS.contiguous(),
                gS32,                    S.contiguous(),
            )
            if need_aL: grad_aL = gaL
            if need_aR: grad_aR = gaR
        return grad_aL, grad_aR, grad_u, grad_v


def _two_tau_pref_suff(alpha_L, alpha_R, u, v):
    return _TwoTauPrefSuffFn.apply(alpha_L, alpha_R, u, v)


# ----------------------------------------------------------------------------
# Fused gather+combine (multi-L)
#
#   y[l, b, j] = w_l[l, j] * P[l, b, c_l[l, j]] + w_r[l, j] * S[l, b, c_r[l, j]]
#
# Multi-L variant of `_GatherCombineFn` from kerlap_norm. Saves P_sel/S_sel in
# forward and does scatter_add directly in backward (bypasses autograd gather).
# Replaces 4x torch.gather + mul + add in _magnitude_apply_scan_real with one
# Function with a fused backward.
# ----------------------------------------------------------------------------

class _FusedGatherCombineMultiFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, S, w_l, w_r, c_l, c_r):
        # P, S: (L, B, n_a); w_l, w_r, c_l, c_r: (L, n_b). Returns y: (L, B, n_b).
        L, B, n_a = P.shape
        n_b = c_l.shape[-1]
        idx_l = c_l.unsqueeze(1).expand(L, B, n_b)
        idx_r = c_r.unsqueeze(1).expand(L, B, n_b)
        P_sel = torch.gather(P, -1, idx_l)          # (L, B, n_b)
        S_sel = torch.gather(S, -1, idx_r)
        y = w_l.unsqueeze(1) * P_sel + w_r.unsqueeze(1) * S_sel
        ctx.save_for_backward(P_sel, S_sel, w_l, w_r, c_l, c_r)
        ctx.n_a = n_a
        return y

    @staticmethod
    def backward(ctx, grad_y):
        P_sel, S_sel, w_l, w_r, c_l, c_r = ctx.saved_tensors
        n_a = ctx.n_a
        L, B, n_b = grad_y.shape
        grad_P = grad_S = grad_w_l = grad_w_r = None

        if ctx.needs_input_grad[2]:
            grad_w_l = (grad_y * P_sel).sum(dim=1)       # (L, n_b)
        if ctx.needs_input_grad[3]:
            grad_w_r = (grad_y * S_sel).sum(dim=1)

        if ctx.needs_input_grad[0]:
            gy_wl  = grad_y * w_l.unsqueeze(1)
            idx_l  = c_l.unsqueeze(1).expand(L, B, n_b)
            grad_P = torch.zeros(L, B, n_a, device=grad_y.device, dtype=grad_y.dtype)
            grad_P.scatter_add_(-1, idx_l, gy_wl)
        if ctx.needs_input_grad[1]:
            gy_wr  = grad_y * w_r.unsqueeze(1)
            idx_r  = c_r.unsqueeze(1).expand(L, B, n_b)
            grad_S = torch.zeros(L, B, n_a, device=grad_y.device, dtype=grad_y.dtype)
            grad_S.scatter_add_(-1, idx_r, gy_wr)

        return grad_P, grad_S, grad_w_l, grad_w_r, None, None


def _fused_gather_combine_ml(P, S, w_l, w_r, c_l, c_r):
    return _FusedGatherCombineMultiFn.apply(P, S, w_l, w_r, c_l, c_r)


# ----------------------------------------------------------------------------
# Dense reference (O(L n^2))
# ----------------------------------------------------------------------------

def _magnitude_apply_dense(
    X: torch.Tensor,          # (B, L, n) real or complex
    a: torch.Tensor,          # (L, n)
    b: torch.Tensor,          # (L, n)
    tau_left:  torch.Tensor,  # (L,)
    tau_right: torch.Tensor,  # (L,)
    normalize: bool = True,
) -> torch.Tensor:
    L, n = a.shape
    diff   = a.unsqueeze(2) - b.unsqueeze(1)                      # (L, n, n)
    tau_lr = torch.where(diff > 0,
                         tau_right.view(L, 1, 1),
                         tau_left .view(L, 1, 1))
    logits = -diff.abs() / tau_lr
    if normalize:
        K = torch.softmax(logits, dim=1)
    else:
        K = torch.exp(logits)
    K = K.to(X.dtype)
    return torch.einsum("lkj,blk->blj", K, X)                     # (B, L, n)


# ----------------------------------------------------------------------------
# Fast scan path (O(L n))
# ----------------------------------------------------------------------------

def _magnitude_apply_scan_real(
    X: torch.Tensor,          # (B, L, n) float32 on CUDA
    a: torch.Tensor,          # (L, n) float32
    b: torch.Tensor,          # (L, n) float32
    tau_left:  torch.Tensor,  # (L,)   float32
    tau_right: torch.Tensor,  # (L,)   float32
    normalize: bool = True,
) -> torch.Tensor:
    """Fused DL-scan evaluation of the normalised (or raw) soft kernel.

    All log-domain weights used are <= 0 (because a_sorted[c_l] <= b <=
    a_sorted[c_r] when both gates are active), so exp(m) for the
    normalize=False post-multiplier cannot overflow.
    """
    B, L, n_a = X.shape
    n_b = b.shape[-1]
    # Multi-scan wants (L, B, n_a).
    X_LBn = X.transpose(0, 1).contiguous()

    # Sort a per-L; permute X to match.
    a_sorted, perm = torch.sort(a, dim=-1)                        # (L, n_a)
    perm_LBn = perm.unsqueeze(1).expand(L, B, n_a)
    X_perm = torch.gather(X_LBn, dim=-1, index=perm_LBn)          # (L, B, n_a)

    # One scan launch with forward side driven by alpha_L = a/tau_L and
    # backward side by alpha_R = a/tau_R. u carries [X; ones] for the
    # prefix; v carries the same for the suffix. No L-doubling.
    ones_row = torch.ones(L, 1, n_a, device=X.device, dtype=X.dtype)
    stacked = torch.cat([X_perm, ones_row], dim=1)                # (L, B+1, n_a)

    tau_l = tau_left.view(L, 1)
    tau_r = tau_right.view(L, 1)
    alpha_L = a_sorted / tau_l
    alpha_R = a_sorted / tau_r

    P_L, S_R = _two_tau_pref_suff(alpha_L, alpha_R, stacked, stacked)

    # searchsorted on a_sorted (unscaled — positive scaling preserves order).
    c      = torch.searchsorted(a_sorted, b, right=True) - 1      # (L, n_b), [-1, n_a-1]
    c_l    = c.clamp(min=0)
    c_r    = (c + 1).clamp(max=n_a - 1)
    gate_l = (c >= 0)
    gate_r = ((c + 1) < n_a)

    a_at_cl = torch.gather(a_sorted, dim=-1, index=c_l)           # (L, n_b)
    a_at_cr = torch.gather(a_sorted, dim=-1, index=c_r)

    log_w_l_raw = (a_at_cl - b) / tau_l                           # <= 0 where gated
    log_w_r_raw = (b - a_at_cr) / tau_r                           # <= 0 where gated

    neg_inf = torch.full_like(log_w_l_raw, float('-inf'))
    log_w_l = torch.where(gate_l, log_w_l_raw, neg_inf)
    log_w_r = torch.where(gate_r, log_w_r_raw, neg_inf)

    m   = torch.maximum(log_w_l, log_w_r)                         # <= 0
    w_l = torch.where(gate_l, torch.exp(log_w_l - m), torch.zeros_like(m))
    w_r = torch.where(gate_r, torch.exp(log_w_r - m), torch.zeros_like(m))

    # Fused multi-L gather+combine: num and den computed together on
    # the stacked (B+1) rows in a single custom-Function call. Backward
    # uses scatter_add directly (avoids autograd of torch.gather).
    combined = _fused_gather_combine_ml(P_L, S_R, w_l, w_r, c_l, c_r)  # (L, B+1, n)
    num = combined[:, :B, :]

    if normalize:
        den = combined[:, B:B+1, :]
        out_LBn = num / den
    else:
        # Recover raw (un-normalised) numerator: num * exp(m).
        out_LBn = num * torch.exp(m).unsqueeze(1)

    return out_LBn.transpose(0, 1).contiguous()                   # (B, L, n)


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------

def _can_use_scan(X: torch.Tensor, a: torch.Tensor) -> bool:
    if not _HAS_SCAN:
        return False
    if not (X.is_cuda and a.is_cuda):
        return False
    if X.dtype == torch.float32:
        return True
    if X.dtype == torch.complex64 and a.dtype == torch.float32:
        return True
    return False


def magnitude_apply(
    X: torch.Tensor,          # (B, L, n) real or complex
    a: torch.Tensor,          # (L, n) real
    b: torch.Tensor,          # (L, n) real
    tau_left:  torch.Tensor,  # (L,) positive
    tau_right: torch.Tensor,  # (L,) positive
    normalize: bool = True,
) -> torch.Tensor:
    if _can_use_scan(X, a):
        if X.is_complex():
            re = _magnitude_apply_scan_real(
                X.real.contiguous(), a, b, tau_left, tau_right, normalize)
            im = _magnitude_apply_scan_real(
                X.imag.contiguous(), a, b, tau_left, tau_right, normalize)
            return torch.complex(re, im)
        return _magnitude_apply_scan_real(
            X, a, b, tau_left, tau_right, normalize)
    return _magnitude_apply_dense(X, a, b, tau_left, tau_right, normalize)


# ----------------------------------------------------------------------------
# Correctness self-test
# ----------------------------------------------------------------------------

def _self_test():
    if not torch.cuda.is_available():
        print("CUDA unavailable — only running dense self-consistency check.")
        torch.manual_seed(0)
        B, L, n = 2, 4, 16
        X = torch.randn(B, L, n)
        a = torch.randn(L, n); b = torch.randn(L, n)
        tl = torch.rand(L) + 0.05; tr = torch.rand(L) + 0.05
        Y = _magnitude_apply_dense(X, a, b, tl, tr, normalize=True)
        print(f"dense OK, shape={tuple(Y.shape)}")
        return

    torch.manual_seed(0)
    device = "cuda"
    state = {"passed": 0, "failed": 0}

    def chk(name, got, ref, tol=1e-3):
        err = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
        ok  = err < tol
        state["passed"] += ok
        state["failed"] += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:50s}  err={err:.2e}")

    print("=== scan vs dense (real, float32) ===")
    for B, L, n in [(4, 8, 256), (16, 16, 784), (2, 4, 1024), (3, 2, 2048)]:
        X = torch.randn(B, L, n, device=device)
        a = torch.randn(L, n, device=device)
        b = torch.randn(L, n, device=device)
        tl = torch.rand(L, device=device) * 0.2 + 0.05
        tr = torch.rand(L, device=device) * 0.2 + 0.05
        for norm in (True, False):
            ref = _magnitude_apply_dense(X, a, b, tl, tr, normalize=norm)
            got = _magnitude_apply_scan_real(X, a, b, tl, tr, normalize=norm)
            chk(f"B={B} L={L} n={n} norm={norm}", got, ref)

    print("\n=== scan vs dense (complex64) ===")
    for B, L, n in [(4, 4, 512), (8, 8, 1024)]:
        Xr = torch.randn(B, L, n, device=device)
        Xi = torch.randn(B, L, n, device=device)
        X  = torch.complex(Xr, Xi)
        a = torch.randn(L, n, device=device)
        b = torch.randn(L, n, device=device)
        tl = torch.rand(L, device=device) * 0.2 + 0.05
        tr = torch.rand(L, device=device) * 0.2 + 0.05
        ref = _magnitude_apply_dense(X, a, b, tl, tr, normalize=True)
        got = magnitude_apply(X, a, b, tl, tr, normalize=True)
        chk(f"B={B} L={L} n={n} complex", got, ref)

    print("\n=== gradients (scan) ===")
    B, L, n = 4, 4, 512
    X = torch.randn(B, L, n, device=device)
    a  = torch.randn(L, n, device=device, requires_grad=True)
    b  = torch.randn(L, n, device=device, requires_grad=True)
    tl = (torch.rand(L, device=device) * 0.2 + 0.05).requires_grad_(True)
    tr = (torch.rand(L, device=device) * 0.2 + 0.05).requires_grad_(True)
    Y = _magnitude_apply_scan_real(X, a, b, tl, tr, normalize=True)
    Y.pow(2).sum().backward()
    print(f"  grads OK: a.grad={a.grad is not None} "
          f"b.grad={b.grad is not None} tl.grad={tl.grad is not None} "
          f"tr.grad={tr.grad is not None}")

    print(f"\n{state['passed']} PASS, {state['failed']} FAIL")


if __name__ == "__main__":
    _self_test()
