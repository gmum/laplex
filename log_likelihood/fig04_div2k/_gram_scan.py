"""Scan-based Laplace Gram (single-tau) — CUDA-fused fp64.

Używa `prefix_suffix_dl_fp64` z cuda_bench (własny fp64 port DL-scan-a).
~2× szybsze niż pure-PyTorch fp64 cumsum, ~1.3× wolniejsze od fp32 CUDA,
ale stabilne numerycznie dla signed-D gram-ów (cos2γ, sin2γ).

Koszt O(n² + n·k) flops; dokładność ~1e-15 (fp64 eps).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from prefix_suffix_dl_fp64 import prefix_suffix_sep


def ker_D_kerT_scan(alpha, beta, D, tau):
    """
    G[i, i'] = Σ_c D[c] · K[i, c] · K[i', c],   K[i, c] = exp(-|α_i - β_c|/τ).

    Wszystko liczone w fp64 (CUDA fused scan), zwraca fp64.
    """
    n = alpha.numel()
    device = alpha.device

    if not torch.is_tensor(tau):
        tau = torch.as_tensor(tau, dtype=torch.float64, device=device)

    # Skalujemy α,β przez τ: K(α,β;τ) = K(α/τ, β/τ; 1).  Pracujemy w fp64.
    a = (alpha / tau).to(torch.float64)
    b = (beta / tau).to(torch.float64)
    D_f = D.to(torch.float64)

    # Sort α
    a_s, perm = torch.sort(a)
    inv_perm = torch.argsort(perm)

    # Kubełek dla β[c]
    idx = torch.bucketize(b, a_s, right=True)

    # α-extended z sentinelami
    alpha_ext = torch.cat([
        b.min().unsqueeze(0),
        a_s,
        b.max().unsqueeze(0),
    ])

    # u[i] = Σ_{c: idx[c]==i} D[c] · exp(2·(β[c] - α_{i+1}))
    alpha_ip1 = alpha_ext[idx + 1]
    u_vals = D_f * torch.exp(2.0 * (b - alpha_ip1))
    u = torch.zeros(n + 1, dtype=torch.float64, device=device).scatter_add(0, idx, u_vals)

    # v[i] = Σ_{c: idx[c]==i} D[c] · exp(2·(α_{i-1} - β[c]))
    idx_left = (idx - 1).clamp(min=0)
    alpha_i_val = torch.where(idx > 0, a_s[idx_left], torch.zeros_like(b))
    v_vals = D_f * torch.exp(2.0 * (alpha_i_val - b))
    v = torch.zeros(n + 1, dtype=torch.float64, device=device).scatter_add(0, idx, v_vals)

    # w[i] = Σ_{c: idx[c]==i} D[c]
    w = torch.zeros(n + 1, dtype=torch.float64, device=device).scatter_add(0, idx, D_f)

    # Prefix/Suffix przez fused CUDA DL-scan (fp64).
    # Scan kernel computes:  P[i] = exp(-α_s[i]) · Σ_{j≤i} exp(α_s[j]) · u[j]
    # Uwaga: scan oczekuje (B, T).  Używamy B=1.
    alpha2 = 2.0 * a_s                                         # (n,) fp64, nondecreasing
    u_1d = u[:n].unsqueeze(0)                                  # (1, n)
    v_1d = v[1:n+1].unsqueeze(0)                               # (1, n)
    U_b, V_b = prefix_suffix_sep(alpha2, u_1d, v_1d)
    U = U_b.squeeze(0)
    V = V_b.squeeze(0)

    # W[i] = cumsum(w[:n])
    W = torch.cumsum(w[:n], dim=0)

    # M_raw = diff · (U[j] + (W[i]-W[j]) + V[i])
    ai = a_s.unsqueeze(1)
    aj = a_s.unsqueeze(0)
    diff = torch.exp(-torch.relu(ai - aj))

    M_raw = diff * (U.unsqueeze(0) + (W.unsqueeze(1) - W.unsqueeze(0)) + V.unsqueeze(1))

    # Symetryzacja tril + tril(-1)^T
    M = torch.tril(M_raw) + torch.tril(M_raw, -1).T

    # Un-sort (fp64 zachowany)
    return M[inv_perm][:, inv_perm]
