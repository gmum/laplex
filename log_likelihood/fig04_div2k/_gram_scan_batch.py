"""Batched scan-based Laplace Gram — CUDA-fused fp64.

Rozszerzenie `ker_D_kerT_scan` o batch: liczone jednocześnie B gramów
    G_b[i, i'] = Σ_c D_b[c] · K[i,c] · K[i',c]

dla B różnych diagonali D_b, przy WSPÓLNYM jądrze K (wspólne α, β, τ).

Kluczowa obserwacja: `prefix_suffix_sep` już obsługuje (B, T) batch,
a α jest współdzielone — więc wystarczy przygotować B wersji u/v/w
i odpalić JEDEN fused CUDA scan z B batchami zamiast B osobnych wywołań.

Koszt: O(B·n + n² + B·n·k) → vs B × O(n² + n·k) w naiwnym podejściu.
Scan CUDA jest O(n) per batch (B batchy lecą równolegle na GPU),
a konstrukcja Gram-a O(n²) jest wspólna z perspektywy struktury.

Użycie w multi-component LAPLEX:
    B = I(I+1)/2  (parami komponentów)
    D_b = w_i · w_j / s²  dla pary (i,j)
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
    alpha : (k,)  — pozycje "kolumn" jądra (β-side w naszej konwencji)
    beta  : (n,)  — pozycje "wierszy" jądra (α-side)
    D_batch : (B, n) — B diagonali (np. w_i·w_j/s² dla par komponentów)
    tau   : scalar lub (1,) — wspólna skala jądra

    Returns
    -------
    G : (B, k, k) fp64 — B symetrycznych gramów
    """
    k = alpha.numel()
    n = beta.numel()
    B = D_batch.shape[0]
    device = alpha.device

    if not torch.is_tensor(tau):
        tau = torch.as_tensor(tau, dtype=torch.float64, device=device)

    # Skalujemy α,β przez τ
    a = (alpha / tau).to(torch.float64)   # (k,) — "kolumny" Gram-a
    b = (beta / tau).to(torch.float64)    # (n,) — pozycje danych/diagonali
    D_f = D_batch.to(torch.float64)       # (B, n)

    # Sort α (kolumny Gram-a)
    a_s, perm = torch.sort(a)
    inv_perm = torch.argsort(perm)

    # Kubełki: do którego przedziału [a_s[i], a_s[i+1]) wpada każde b[c]
    idx = torch.bucketize(b, a_s, right=True)  # (n,)

    # Extended α z sentinelami
    alpha_ext = torch.cat([
        b.min().unsqueeze(0),
        a_s,
        b.max().unsqueeze(0),
    ])

    # --- Przygotowanie u, v, w (batchowane po B) ---
    # M2: agresywne `del` po użyciu transient (B, n) fp64 tensorów,
    # by zwolnić pamięć od razu (efektywne pod gradient checkpointingiem).
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

    # --- Prefix/Suffix scan (CUDA fused, B batches jednocześnie) ---
    alpha2 = 2.0 * a_s                                # (k,) fp64, nondecreasing
    u_in = u[:, :k]                                   # (B, k)
    v_in = v[:, 1:k+1]                                # (B, k)
    U_b, V_b = prefix_suffix_sep(alpha2, u_in, v_in)  # (B, k), (B, k)
    del u, v, u_in, v_in, alpha2                       # free (B, k+1) tensors

    # W[b, i] = cumsum(w[b, :k]) along dim=1
    W = torch.cumsum(w[:, :k], dim=1)                 # (B, k)
    del w

    # --- Konstrukcja k×k Gram-ów ---
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

    # Symetryzacja: tril + tril(-1)^T
    M = torch.tril(M_raw) + torch.tril(M_raw, diagonal=-1).transpose(-2, -1)
    del M_raw

    # Un-sort: permutacja kolumn i wierszy
    G = M[:, inv_perm][:, :, inv_perm]

    return G


def ker_D_kerT_scan_single(alpha, beta, D, tau):
    """Wrapper jednodiagonalny (kompatybilny z oryginalnym `ker_D_kerT_scan`)."""
    G = ker_D_kerT_scan_batch(alpha, beta, D.unsqueeze(0), tau)
    return G.squeeze(0)
