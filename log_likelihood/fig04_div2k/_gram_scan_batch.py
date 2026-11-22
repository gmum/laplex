"""Batched scan-based Laplace Gram — CUDA-fused fp64.

Batched `ker_D_kerT_scan`: computes B Gram matrices at once
    G_b[i, i'] = Σ_c D_b[c] · K[i,c] · K[i',c]

for B different diagonals D_b with a SHARED kernel K (shared α, β, τ).

Key observation: `prefix_suffix_sep` already handles a (B, T) batch,
and α is shared, so it suffices to prepare B versions of u/v/w
and launch ONE fused CUDA scan with B batches instead of B separate calls.

Cost: O(B·n + n² + B·n·k) vs B × O(n² + n·k) in the naive approach.
The CUDA scan is O(n) per batch (the B batches run in parallel on the GPU),
and the O(n²) construction of the Gram matrix shares its structure.

Use in multi-component LAPLEX:
    B = I(I+1)/2  (pairs of components)
    D_b = w_i · w_j / s²  for the pair (i,j)
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from prefix_suffix_dl_fp64 import prefix_suffix_sep


def ker_D_kerT_scan_batch(alpha, beta, D_batch, tau):
    """Batched Gram: G_b[i,i'] = Σ_c D_b[c] · K[i,c] · K[i',c].

    Args
    ----
    alpha : (k,)  — positions of the kernel "columns" (β side in our convention)
    beta  : (n,)  — positions of the kernel "rows" (α side)
    D_batch : (B, n) — B diagonals (e.g. w_i·w_j/s² for pairs of components)
    tau   : scalar or (1,) — shared kernel scale

    Returns
    -------
    G : (B, k, k) fp64 — B symmetric Gram matrices
    """
    k = alpha.numel()
    n = beta.numel()
    B = D_batch.shape[0]
    device = alpha.device

    if not torch.is_tensor(tau):
        tau = torch.as_tensor(tau, dtype=torch.float64, device=device)

    # Scale α, β by τ
    a = (alpha / tau).to(torch.float64)   # (k,) — Gram "columns"
    b = (beta / tau).to(torch.float64)    # (n,) — positions of the data / diagonal
    D_f = D_batch.to(torch.float64)       # (B, n)

    # Sort α (Gram columns)
    a_s, perm = torch.sort(a)
    inv_perm = torch.argsort(perm)

    # Buckets: the interval [a_s[i], a_s[i+1]) that each b[c] falls into
    idx = torch.bucketize(b, a_s, right=True)  # (n,)

    # α extended with sentinels
    alpha_ext = torch.cat([
        b.min().unsqueeze(0),
        a_s,
        b.max().unsqueeze(0),
    ])

    # --- Prepare u, v, w (batched over B) ---
    # M2: aggressive `del` of the transient (B, n) fp64 tensors after use,
    # to free memory immediately (effective under gradient checkpointing).
    idx_exp = idx.unsqueeze(0).expand(B, -1)           # (B, n) — cheap (view)

    # u[b, i] = Σ_{c: idx[c]==i} D_b[c] · exp(2·(b[c] - α_{i+1}))
    alpha_ip1 = alpha_ext[idx + 1]                     # (n,)
    exp_u = torch.exp(2.0 * (b - alpha_ip1))           # (n,)
    del alpha_ip1
    u_vals = D_f * exp_u.unsqueeze(0)                  # (B, n) — 240 MB transient
    del exp_u
    u = torch.zeros(B, k + 1, dtype=torch.float64, device=device)
    u.scatter_add_(1, idx_exp, u_vals)
    del u_vals                                          # free 240 MB

    # v[b, i] = Σ_{c: idx[c]==i} D_b[c] · exp(2·(α_{i-1} - b[c]))
    idx_left = (idx - 1).clamp(min=0)
    alpha_i_val = torch.where(idx > 0, a_s[idx_left], torch.zeros_like(b))
    del idx_left
    exp_v = torch.exp(2.0 * (alpha_i_val - b))        # (n,)
    del alpha_i_val
    v_vals = D_f * exp_v.unsqueeze(0)                  # (B, n) — 240 MB transient
    del exp_v
    v = torch.zeros(B, k + 1, dtype=torch.float64, device=device)
    v.scatter_add_(1, idx_exp, v_vals)
    del v_vals                                          # free 240 MB

    # w[b, i] = Σ_{c: idx[c]==i} D_b[c]
    w = torch.zeros(B, k + 1, dtype=torch.float64, device=device)
    w.scatter_add_(1, idx_exp, D_f)
    del D_f, idx_exp                                    # free fp64 D copy

    # --- Prefix/suffix scan (CUDA fused, B batches at once) ---
    alpha2 = 2.0 * a_s                                # (k,) fp64, nondecreasing
    u_in = u[:, :k]                                   # (B, k)
    v_in = v[:, 1:k+1]                                # (B, k)
    U_b, V_b = prefix_suffix_sep(alpha2, u_in, v_in)  # (B, k), (B, k)
    del u, v, u_in, v_in, alpha2                       # free (B, k+1) tensors

    # W[b, i] = cumsum(w[b, :k]) along dim=1
    W = torch.cumsum(w[:, :k], dim=1)                 # (B, k)
    del w

    # --- Build the k×k Gram matrices ---
    # M_raw[b, i, j] = exp(-|a_s[i] - a_s[j]|) · (U[b,j] + W[b,i] - W[b,j] + V[b,i])
    ai = a_s.unsqueeze(1)                             # (k, 1)
    aj = a_s.unsqueeze(0)                             # (1, k)
    diff_exp = torch.exp(-torch.relu(ai - aj))        # (k, k) — lower-tri kernel

    # Broadcasting: (B, k, k)
    # U[b,j] → (B, 1, k), W[b,i] → (B, k, 1), W[b,j] → (B, 1, k), V[b,i] → (B, k, 1)
    inner = (U_b.unsqueeze(1) +
             W.unsqueeze(2) - W.unsqueeze(1) +
             V_b.unsqueeze(2))                        # (B, k, k)
    del U_b, V_b, W

    M_raw = diff_exp.unsqueeze(0) * inner             # (B, k, k)
    del inner, diff_exp

    # Symmetrization: tril + tril(-1)^T
    M = torch.tril(M_raw) + torch.tril(M_raw, diagonal=-1).transpose(-2, -1)
    del M_raw

    # Un-sort: permute columns and rows
    G = M[:, inv_perm][:, :, inv_perm]

    return G


def ker_D_kerT_scan_single(alpha, beta, D, tau):
    """Single-diagonal wrapper (compatible with the original `ker_D_kerT_scan`)."""
    G = ker_D_kerT_scan_batch(alpha, beta, D.unsqueeze(0), tau)
    return G.squeeze(0)
