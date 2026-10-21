# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch

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
]


def goal_success(env, command_name: str = "reorient_command") -> torch.Tensor:
  """End the episode once the goal was reached and held (``single_success``)."""
  return env.command_manager.get_term(command_name).episode_success
