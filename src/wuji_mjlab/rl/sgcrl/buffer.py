# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""GPU trajectory buffer with discounted future-goal sampling.

All environments step in lockstep, so slot ``t`` holds one step of every env.
Each env's episodes are told apart by an episode id; when an episode ends its
slots get the global step of its last state, so a future goal is drawn only
from later states of the same episode.
"""

from __future__ import annotations

import math

import torch

_ONGOING = torch.iinfo(torch.int64).max


class TrajectoryBuffer:
  def __init__(
    self,
    capacity: int,
    num_envs: int,
    state_dim: int,
    action_dim: int,
    goal_dim: int,
    device: str,
  ) -> None:
    self.capacity, self.num_envs, self.device = capacity, num_envs, device
    shape = (capacity, num_envs)
    self.state = torch.zeros(*shape, state_dim, device=device)
    self.action = torch.zeros(*shape, action_dim, device=device)
    self.goal = torch.zeros(*shape, goal_dim, device=device)  # achieved goal
    self.episode = torch.full(shape, -1, dtype=torch.int64, device=device)
    self.end_step = torch.full(shape, _ONGOING, dtype=torch.int64, device=device)
    self.slot_step = torch.full((capacity,), -1, dtype=torch.int64, device=device)
    self._env_episode = torch.zeros(num_envs, dtype=torch.int64, device=device)
    self.step = 0  # global step of the next insert

  @property
  def size(self) -> int:
    return min(self.step, self.capacity)

  def add(self, state, action, goal, done: torch.Tensor) -> None:
    """Store step ``self.step``; ``done`` marks envs whose episode ended after it."""
    slot = self.step % self.capacity
    self.state[slot], self.action[slot], self.goal[slot] = state, action, goal
    self.episode[slot] = self._env_episode
    self.end_step[slot] = _ONGOING
    self.slot_step[slot] = self.step
    if done.any():
      ending = (self.episode == self._env_episode) & done
      self.end_step[ending] = self.step
      self._env_episode += done.long()
    self.step += 1

  def sample(self, batch_size: int, discount: float, absorb: bool = True):
    """(state, action, future achieved goal) with the goal ``k`` steps ahead in
    the same episode, ``k ~ geometric(1 - discount)`` on 1, 2, ...

    ``absorb``: an ended episode stays in its last state (e.g. the dropped
    object), so an offset past the end takes that state; offsets past the
    newest step of a running episode are redrawn. Otherwise ``k`` is truncated
    to the later states in the buffer and renormalised, which hides what an
    episode ended in."""
    out_s, out_a, out_g, count = [], [], [], 0
    log_discount = math.log(discount)
    while count < batch_size:
      n = 2 * (batch_size - count)
      t = torch.randint(0, self.size, (n,), device=self.device)
      e = torch.randint(0, self.num_envs, (n,), device=self.device)
      step = self.slot_step[t]
      end = self.end_step[t, e]
      last = end.clamp_max(self.step - 1)
      avail = last - step  # later states of this episode in the buffer
      u = torch.rand(n, device=self.device)
      if absorb:
        k = 1 + torch.floor(torch.log1p(-u) / log_discount).long()
        ok = (k <= avail) | (end != _ONGOING)
        k = torch.minimum(k, avail)
      else:
        ok = avail >= 1
        # Truncated geometric over 1..avail by inverse CDF.
        tail = 1.0 - u * (1.0 - discount ** avail.clamp_min(1).double())
        k = 1 + torch.floor(torch.log(tail) / log_discount).long()
        k = torch.minimum(k.clamp_min(1), avail)
      t, e, k = t[ok], e[ok], k[ok]
      future = (t + k) % self.capacity
      out_s.append(self.state[t, e])
      out_a.append(self.action[t, e])
      out_g.append(self.goal[future, e])
      count += t.shape[0]
    cat = lambda xs: torch.cat(xs)[:batch_size]  # noqa: E731
    return cat(out_s), cat(out_a), cat(out_g)
