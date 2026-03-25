"""Quadratic activation network from Cullen et al. (2026).

f_θ(x) = V σ(W^T x), where σ(x) = x²

W: Linear(2p -> K, bias=False)  with Kaiming init
V: Linear(K -> p, bias=False)   with Kaiming init
"""

import torch
import torch.nn as nn


class QuadraticNet(nn.Module):
    def __init__(self, p: int, K: int):
        super().__init__()
        self.p = p
        self.K = K
        self.W = nn.Linear(2 * p, K, bias=False)
        self.V = nn.Linear(K, p, bias=False)
        # Default Kaiming init from nn.Linear is correct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: V(W(x)²).

        Args:
            x: (batch, 2p) one-hot encoded input
        Returns:
            (batch, p) predictions
        """
        return self.V(self.W(x) ** 2)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
