# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""SGCRL networks: tanh-Gaussian actor and an inner-product contrastive critic."""

from __future__ import annotations

import math

import torch
from torch import nn


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(in_dim, h), nn.ReLU()]
    in_dim = h
  layers.append(nn.Linear(in_dim, out_dim))
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  """Running mean / std (Chan's parallel update), frozen unless ``update``d."""

  def __init__(self, dim: int, eps: float = 1e-4) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(eps))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1])
    n = x.shape[0]
    mean, var = x.mean(0), x.var(0, unbiased=False)
    total = self.count + n
    delta = mean - self.mean
    self.mean += delta * n / total
    self.var = (
      self.var * self.count + var * n + delta**2 * self.count * n / total
    ) / total
    self.count = total

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10.0, 10.0)


class ContrastiveCritic(nn.Module):
  """``phi(s, a)^T psi(g)``: logits[i, j] scores goal j for state-action i."""

  def __init__(
    self, state_dim: int, action_dim: int, goal_dim: int, hidden, repr_dim: int
  ) -> None:
    super().__init__()
    self.sa_encoder = mlp(state_dim + action_dim, hidden, repr_dim)
    self.g_encoder = mlp(goal_dim, hidden, repr_dim)

  def forward(self, state, action, goal) -> torch.Tensor:
    phi = self.sa_encoder(torch.cat((state, action), dim=-1))
    psi = self.g_encoder(goal)
    return phi @ psi.T

  def paired(self, state, action, goal) -> torch.Tensor:
    """Diagonal of ``forward`` without the [B, B] matrix."""
    phi = self.sa_encoder(torch.cat((state, action), dim=-1))
    return (phi * self.g_encoder(goal)).sum(-1)


class TanhGaussianActor(nn.Module):
  """pi(a | s, g): Gaussian squashed by tanh into [-1, 1]."""

  def __init__(
    self, state_dim: int, goal_dim: int, action_dim: int, hidden, min_std: float
  ) -> None:
    super().__init__()
    self.net = mlp(state_dim + goal_dim, hidden, 2 * action_dim)
    self.min_std = min_std

  def _dist(self, state, goal):
    mean, raw_std = self.net(torch.cat((state, goal), dim=-1)).chunk(2, dim=-1)
    std = nn.functional.softplus(raw_std) + self.min_std
    return mean, std

  def sample(self, state, goal) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterised action and its log-probability."""
    mean, std = self._dist(state, goal)
    pre = mean + std * torch.randn_like(mean)
    action = torch.tanh(pre)
    log_prob = (
      -0.5 * ((pre - mean) / std) ** 2 - std.log() - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    # tanh change of variables, in the numerically stable softplus form.
    log_prob -= (2.0 * (math.log(2.0) - pre - nn.functional.softplus(-2.0 * pre))).sum(
      -1
    )
    return action, log_prob

  def mode(self, state, goal) -> torch.Tensor:
    return torch.tanh(self._dist(state, goal)[0])
