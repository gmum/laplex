import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laplex_lib"))
from laplex import X_Laplex


class LaplexLinear(nn.Module):
    """Low-rank linear layer using the Laplace kernel.

    Weight matrix: W[i, j] = Σ_r c_r * exp(-|a_r[i] - b_r[j]| / τ)

    Training forward uses X_Laplex for O(n log n) computation instead of
    materialising the O(n·k) weight matrix. Eval mode caches the dense W
    once for fast repeated inference.
    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        rank: Number of Laplace-kernel components used in the low-rank sum.
        bias: If True, adds a learnable bias term to the output.
        ab_scale: Initialization scale for `a` and `b`. If `"auto"`, it is set
            from `sigma_init` and the layer dimensions.
        c_scale: Initialization scale for `c`. If `"auto"`, it is set to
            `sigma_init`.
        sigma_init: Base scale used by the `"auto"` initialization rules.
        tau_init: Initial value of the kernel temperature parameter `tau`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        ab_scale: float | str = "auto",
        c_scale: float | str = "auto",
        sigma_init: float = 0.5,
        tau_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        if ab_scale == "auto":
            ab_scale = sigma_init / math.sqrt(math.sqrt(in_features * out_features))
        if c_scale == "auto":
            c_scale = sigma_init

        self.ab_scale = ab_scale
        self.c_scale = c_scale

        self.a = nn.Parameter(torch.empty(rank, in_features))
        self.b = nn.Parameter(torch.empty(rank, out_features))
        self.c = nn.Parameter(torch.empty(rank))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._cached_W: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.a, std=self.ab_scale)
        nn.init.normal_(self.b, std=self.ab_scale)
        nn.init.normal_(self.c, std=self.c_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._cached_W = None

    def _forward_train(self, x: Tensor) -> Tensor:
        tau = torch.exp(self.log_tau)
        out = x.new_zeros(x.shape[0], self.out_features)
        for r in range(self.rank):
            a_r = (self.a[r] / tau).to(x.dtype)
            b_r = (self.b[r] / tau).to(x.dtype)
            out = out + self.c[r].to(x.dtype) * X_Laplex(x, a_r, b_r)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out

    def _build_cached_weight(self) -> Tensor:
        with torch.no_grad():
            tau = torch.exp(self.log_tau)
            a = self.a / tau   # (rank, in_features)
            b = self.b / tau   # (rank, out_features)
            # diff[r, j, i] = |a_r[i] - b_r[j]|
            diff = (a.unsqueeze(1) - b.unsqueeze(2)).abs()  # (rank, out, in)
            K = torch.exp(-diff)                             # (rank, out, in)
            W = (self.c.view(-1, 1, 1) * K).sum(dim=0)     # (out, in)
        return W.detach()

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self._forward_train(x)
        if self._cached_W is None:
            self._cached_W = self._build_cached_weight()
        return F.linear(x, self._cached_W, self.bias)

    def train(self, mode: bool = True) -> "LaplexLinear":
        self._cached_W = None
        return super().train(mode)
