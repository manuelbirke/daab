from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_sizes: list[int], output_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers += [nn.Linear(prev, output_dim)]
    return nn.Sequential(*layers)


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shared_sizes: list[int],
        value_sizes: list[int],
        adv_sizes: list[int],
    ):
        super().__init__()
        self.shared = _mlp(state_dim, shared_sizes[:-1], shared_sizes[-1])
        self.value_stream = _mlp(shared_sizes[-1], value_sizes, 1)
        self.adv_stream = _mlp(shared_sizes[-1], adv_sizes, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.shared(x)
        value = self.value_stream(z)
        adv = self.adv_stream(z)
        return value + (adv - adv.mean(dim=1, keepdim=True))
