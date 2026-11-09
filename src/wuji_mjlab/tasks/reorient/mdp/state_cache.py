# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Per-physics-state cache for quantities read many times a step.

Observations, rewards, metrics and the command read the same derived values
(wrist-tag pose, fingertip poses, world goal) over and over in one env step:
~24 ``site_pose_w`` calls per step, each converting every site's rotation
matrix. ``install(env)`` counts the simulation's ``step`` / ``forward`` /
``reset`` calls (the only places the kinematics change), and the helpers here
recompute a value only when that count moved. Without ``install`` they fall
back to computing every time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import torch
from mjlab.utils.lab_api.math import quat_from_matrix

_EPOCH = "_wuji_state_epoch"
_bypass = [0]


@contextlib.contextmanager
def bypass():
  """Compute everything fresh, neither reading nor filling caches (for CUDA
  graph capture: the graph must hold the whole computation)."""
  _bypass[0] += 1
  try:
    yield
  finally:
    _bypass[0] -= 1


def bypassed() -> bool:
  return _bypass[0] > 0


def install(env) -> None:
  """Count physics state changes of ``env.sim`` (idempotent; a stand-in env
  without a simulation is left uncached)."""
  sim = getattr(env, "sim", None)
  bridge = getattr(sim, "data", None)
  if bridge is None or getattr(bridge, _EPOCH, None) is not None:
    return
  counter = [0]
  for name in ("step", "forward", "reset"):
    fn = getattr(sim, name, None)
    if fn is None:
      continue

    def counted(*args, _fn=fn, **kwargs):
      try:
        return _fn(*args, **kwargs)
      finally:
        counter[0] += 1

    setattr(sim, name, counted)
  # The data bridge refuses plain attribute assignment (CUDA graph safety).
  object.__setattr__(bridge, _EPOCH, counter)


def epoch(entity) -> int | None:
  """The physics state count seen by ``entity`` (None: not installed)."""
  counter = getattr(getattr(entity.data, "data", None), _EPOCH, None)
  return None if counter is None else counter[0]


def memo(owner, name: str, key, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
  """``compute()`` cached on ``owner`` under ``name`` while ``key`` is equal
  (a key containing None is never cached)."""
  if key is None or (isinstance(key, tuple) and None in key) or bypassed():
    return compute()
  cache = owner.__dict__.setdefault("_wuji_memo", {})
  hit = cache.get(name)
  if hit is not None and hit[0] == key:
    return hit[1]
  value = compute()
  cache[name] = (key, value)
  return value


def constant(owner, name: str, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
  """``compute()`` once per ``owner`` (values that never change, such as index
  tensors; kept while capturing, which cannot copy from the CPU)."""
  cache = owner.__dict__.setdefault("_wuji_const", {})
  if name not in cache:
    cache[name] = compute()
  return cache[name]


def site_pose_w(entity, site_ids) -> torch.Tensor:
  """``entity.data.site_pose_w[:, site_ids]`` (num_envs, len(site_ids), 7),
  converting only those sites, once per physics state."""
  ids = tuple(int(i) for i in site_ids)
  data = entity.data
  key = epoch(entity)
  if key is None:  # not installed: exactly the entity's own property
    return data.site_pose_w[:, list(ids)]
  # The global ids, made once: indexing with a Python list would copy it to
  # the GPU every time, and that copy waits for the queued physics.
  gids = constant(data, f"site_ids{ids}", lambda: data.indexing.site_ids[list(ids)])

  def compute() -> torch.Tensor:
    pos = data.data.site_xpos[:, gids]
    quat = quat_from_matrix(data.data.site_xmat[:, gids])
    return torch.cat([pos, quat], dim=-1)

  return memo(data, f"site_pose_w{ids}", key, compute)
