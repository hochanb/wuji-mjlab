# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Contact-force balance (Dexonomy's grasp quality), batched.

For contacts at ``p_i`` with inward normals ``n_i`` and friction ``mu``, find
contact forces inside the (pyramid) friction cones, total normal force 1 and
at least ``min_share`` of it spread evenly over the contacts, that minimise
the net wrench on the object

  min_a || sum_ik a_ik [e_ik; (p_i - c) / L x e_ik] - g ||,  e_ik = n_i + mu t_ik

``a >= 0``; ``L`` (object size) makes torque comparable to force; ``g`` is the
wrench the contacts must supply (0, or the object's weight per unit of total
normal force). The minimum is the residual: 0 when the forces balance, about
1 when nothing does. Solved by accelerated projected gradient (FISTA).
"""

from __future__ import annotations

import math

import torch


def _tangents(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  ref = torch.zeros_like(n)
  use_y = n[..., 0].abs() > 0.9
  ref[..., 0] = (~use_y).float()
  ref[..., 1] = use_y.float()
  d = torch.nn.functional.normalize(torch.cross(n, ref, dim=-1), dim=-1)
  return d, torch.cross(n, d, dim=-1)


def _project_simplex(v: torch.Tensor, mask: torch.Tensor, r: float) -> torch.Tensor:
  """Euclidean projection of each row onto {x >= 0, sum x = r, x = 0 off mask}."""
  v = v.masked_fill(~mask, -1e9)
  u, _ = torch.sort(v, dim=1, descending=True)
  css = u.cumsum(1) - r
  k = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)
  rho = ((u - css / k) > 0).sum(1).clamp_min(1)
  theta = css.gather(1, (rho - 1)[:, None]) / rho[:, None]
  return (v - theta).clamp_min(0.0) * mask


def balance_contact_forces(
  points: torch.Tensor,
  normals: torch.Tensor,
  mask: torch.Tensor,
  center: torch.Tensor,
  length: float,
  mu: float,
  min_share: float = 0.5,
  num_edges: int = 8,
  iterations: int = 300,
  target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """``points`` / ``normals`` (B, C, 3) (inward), ``mask`` (B, C) active
  contacts, ``center`` (B, 3), ``target`` (B, 6) wrench (default 0). Returns the residual (B,) and the contact forces
  (B, C, 3) (total normal force 1; 0 on inactive contacts). Rows with fewer
  than 2 active contacts get residual 1."""
  b, c, _ = points.shape
  d, t = _tangents(normals)
  ang = torch.arange(num_edges, device=points.device) * (2 * math.pi / num_edges)
  edges = normals[:, :, None] + mu * (
    torch.cos(ang)[:, None] * d[:, :, None] + torch.sin(ang)[:, None] * t[:, :, None]
  )  # (B, C, K, 3)
  lever = ((points - center[:, None]) / length)[:, :, None].expand_as(edges)
  wrench = torch.cat((edges, torch.cross(lever, edges, dim=-1)), -1)
  w = wrench.reshape(b, c * num_edges, 6).transpose(1, 2)  # (B, 6, n)

  active = mask.float()
  count = active.sum(1).clamp_min(1.0)
  base = (min_share / count)[:, None] * active / num_edges  # per edge
  a0 = base.repeat_interleave(num_edges, dim=1)
  emask = mask.repeat_interleave(num_edges, dim=1)
  r = 1.0 - min_share
  goal = torch.zeros(b, 6, device=points.device) if target is None else target
  lip = 2.0 * (w * w).sum((1, 2)).clamp_min(1e-9)
  x = _project_simplex(torch.zeros_like(a0), emask, r)
  y, step = x, 1.0
  for _ in range(iterations):
    a = a0 + y
    error = torch.einsum("bkn,bn->bk", w, a) - goal
    grad = 2.0 * torch.einsum("bkn,bk->bn", w, error)
    x_new = _project_simplex(y - grad / lip[:, None], emask, r)
    step_new = (1.0 + math.sqrt(1.0 + 4.0 * step * step)) / 2.0
    y = x_new + ((step - 1.0) / step_new) * (x_new - x)
    x, step = x_new, step_new
  a = a0 + x
  residual = (torch.einsum("bkn,bn->bk", w, a) - goal).norm(dim=1)
  residual = torch.where(mask.sum(1) >= 2, residual, torch.ones_like(residual))
  forces = (a.reshape(b, c, num_edges, 1) * edges).sum(2)
  return residual, forces
