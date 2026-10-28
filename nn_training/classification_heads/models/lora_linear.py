import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoRALinear(nn.Module):
    """Low-rank linear layer for comparison with `LaplexLinear`.

    This layer represents the dense weight matrix as a low-rank product:

        W = B @ A

    where `A` has shape `(rank, in_features)` and `B` has shape
    `(out_features, rank)`. During training, the weight is computed from the
    factors directly. During evaluation, the dense weight matrix is cached
    for faster repeated inference.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        rank: Rank of the low-rank factorization.
        bias: If True, adds a learnable bias term to the output.
        sigma_init: Standard deviation used to initialize `A` and `B`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.sigma_init = sigma_init

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self._cached_W: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.A, std=self.sigma_init)
        nn.init.normal_(self.B, std=self.sigma_init)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._cached_W = None

    def _compute_weight(self) -> Tensor:
        if not self.training and self._cached_W is not None:
            return self._cached_W
        W = self.B @ self.A
        if not self.training:
            self._cached_W = W.detach()
        return W

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self._compute_weight(), self.bias)

    def train(self, mode: bool = True) -> "LoRALinear":
        self._cached_W = None
        return super().train(mode)
