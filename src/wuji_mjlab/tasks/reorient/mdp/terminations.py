# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .cage import (
  CageEscapePenalty,
  _compute_cage_aabb_escape_in_palm,
  _compute_cage_escape_dist,
  cage_drop,
  compute_cage_escalation,
  read_cage_penalty_counter,
  update_cage_penalty_counter,
)

__all__ = [
  "_compute_cage_aabb_escape_in_palm",
  "_compute_cage_escape_dist",
  "update_cage_penalty_counter",
  "compute_cage_escalation",
  "CageEscapePenalty",
  "read_cage_penalty_counter",
  "cage_drop",
  "goal_success",
  "object_speed_limit",
]


def goal_success(env, command_name: str = "reorient_command") -> torch.Tensor:
  """End the episode once the goal was reached and held (``single_success``)."""
  return env.command_manager.get_term(command_name).episode_success


def object_speed_limit(
  env,
  max_lin_vel: float = 5.0,
  max_ang_vel: float = 100.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  """End the episode once the object moves implausibly fast [m/s, rad/s].

  A dropped, elongated object can spin up numerically under implicitfast
  (explicit gyroscopic term at a 10 ms step) until it is flung away at
  hundreds of m/s, which poisons the observation normaliser and the value
  targets before ``cage_drop`` fires. In-hand speeds stay far below the limits.
  """
  data = env.scene[asset_cfg.name].data
  return (data.root_link_lin_vel_w.norm(dim=-1) > max_lin_vel) | (
    data.root_link_ang_vel_w.norm(dim=-1) > max_ang_vel
  )
