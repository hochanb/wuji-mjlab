# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math

import torch

from wuji_mjlab.tasks.reorient.mdp._env_utils import resolve_env_ids
from wuji_mjlab.utils.curriculum import get_curriculum_value  # noqa: F401 -- re-export

__all__ = [
  "adaptive_episode_curriculum",
  "get_curriculum_value",
  "get_linear_progress",
  "goal_rotation_curriculum",
  "reorient_success_curriculum",
  "resolve_env_ids",
  "wrist_range_curriculum",
]


def get_linear_progress(
  env,
  warmup_frac: float = 0.05,
  rampup_frac: float = 0.60,
) -> float:
  if not hasattr(env, "max_common_steps"):
    raise ValueError(
      "Set env.max_common_steps from the training runner's "
      "max_iterations * num_steps_per_env before using the time ramp."
    )
  effective_total_steps = max(float(env.max_common_steps), 1.0)
  step = float(getattr(env, "common_step_counter", 0))
  warmup = float(warmup_frac) * effective_total_steps
  rampup = float(rampup_frac) * effective_total_steps
  if step <= warmup:
    return 0.0
  if step >= rampup:
    return 1.0
  return float((step - warmup) / max(rampup - warmup, 1.0))


def reorient_success_curriculum(
  env,
  env_ids,
  command_name: str = "reorient_command",
  count_threshold: int = 3,
  delta_per_loop: float = 0.01,
) -> dict[str, float]:
  if not hasattr(env, "_reorient_success_curriculum"):
    env._reorient_success_curriculum = 0.0

  env_ids = resolve_env_ids(env, env_ids)

  command = env.command_manager.get_term(command_name)
  goal_reach_count = command.metrics["goal_reach_count"][env_ids]
  if goal_reach_count.numel() == 0:
    return {"value": float(env._reorient_success_curriculum)}

  delta_per_env = float(delta_per_loop) / float(env.num_envs)
  delta = torch.where(
    goal_reach_count >= float(count_threshold),
    torch.full_like(goal_reach_count, delta_per_env, dtype=torch.float32),
    torch.full_like(goal_reach_count, -delta_per_env, dtype=torch.float32),
  )
  delta_sum = float(delta.sum().item())
  env._reorient_success_curriculum = float(
    max(min(env._reorient_success_curriculum + delta_sum, 1.0), 0.0)
  )
  return {
    "value": float(env._reorient_success_curriculum),
    "mean_goal_reach_count": float(goal_reach_count.float().mean().item()),
    "success_env_frac": float(
      (goal_reach_count >= float(count_threshold)).float().mean().item()
    ),
    "delta": delta_sum,
  }


def adaptive_episode_curriculum(
  env,
  env_ids,
  target_steps: int = 800,
  inc_rate: float = 0.1,
  dec_rate: float = 0.2,
  min_scale: float = 0.05,
  drop_gamma: float = 1.0,
) -> dict[str, float]:
  if not hasattr(env, "_adaptive_episode_curriculum"):
    env._adaptive_episode_curriculum = float(min_scale)

  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return {"value": float(env._adaptive_episode_curriculum)}

  target_steps = max(int(target_steps), 1)
  episode_steps = env.episode_length_buf[env_ids].float()
  progress = torch.clamp(episode_steps / float(target_steps), 0.0, 1.0)
  reached_target = episode_steps >= float(target_steps)

  terminated = getattr(env.termination_manager, "terminated", None)
  if terminated is None:
    terminated = torch.zeros_like(reached_target)
  else:
    terminated = terminated[env_ids]
  early_drop = terminated & ~reached_target

  inc = reached_target.float() * float(inc_rate)
  pen = (
    early_drop.float() * torch.pow(1.0 - progress, float(drop_gamma)) * float(dec_rate)
  )

  delta = float((inc - pen).sum().item() / float(env.num_envs))
  env._adaptive_episode_curriculum = float(
    max(min(env._adaptive_episode_curriculum + delta, 1.0), float(min_scale))
  )
  return {
    "value": float(env._adaptive_episode_curriculum),
    "mean_episode_steps": float(episode_steps.mean().item()),
    "mean_progress": float(progress.mean().item()),
    "reached_target_frac": float(reached_target.float().mean().item()),
    "early_drop_frac": float(early_drop.float().mean().item()),
    "target_steps": float(target_steps),
    "delta": delta,
  }


def _success_level(
  env,
  env_ids,
  key: str,
  command_name: str,
  count_threshold: int,
  promote_frac: float,
  demote_frac: float,
  step: float,
  floor_warmup_frac: float,
  floor_rampup_frac: float,
) -> dict[str, float]:
  """Shared 0..1 difficulty level driven by success, with a time floor.

  Among the episodes ending now, the fraction that reached the goal at least
  ``count_threshold`` times raises the level by ``step`` (>= ``promote_frac``)
  or lowers it (< ``demote_frac``). The level never drops below a linear ramp
  over training progress, so a task that never succeeds still reaches full
  difficulty by ``floor_rampup_frac`` of training.
  """
  attr = f"_curriculum_level_{key}"
  level = float(getattr(env, attr, 0.0))
  env_ids = resolve_env_ids(env, env_ids)
  reach = env.command_manager.get_term(command_name).metrics["goal_reach_count"]
  frac = 0.0
  if env_ids.numel() > 0:
    frac = float((reach[env_ids] >= float(count_threshold)).float().mean().item())
    if frac >= promote_frac:
      level += step
    elif frac < demote_frac:
      level -= step
  floor = (
    get_linear_progress(env, floor_warmup_frac, floor_rampup_frac)
    if hasattr(env, "max_common_steps")
    else 0.0
  )
  level = min(max(level, floor, 0.0), 1.0)
  setattr(env, attr, level)
  return {"value": level, "success_env_frac": frac, "floor": floor}


def wrist_range_curriculum(
  env,
  env_ids,
  action_name: str = "wrist_pose",
  command_name: str = "reorient_command",
  min_scale: float = 0.1,
  count_threshold: int = 3,
  promote_frac: float = 0.5,
  demote_frac: float = 0.0,
  step: float = 0.01,
  floor_warmup_frac: float = 0.1,
  floor_rampup_frac: float = 0.5,
) -> dict[str, float]:
  """Grow the wrist action range from ``min_scale`` to its full limits.

  The fingers learn to manipulate first; the wrist then joins as an aid.
  """
  info = _success_level(
    env,
    env_ids,
    f"wrist_range_{action_name}",
    command_name,
    count_threshold,
    promote_frac,
    demote_frac,
    step,
    floor_warmup_frac,
    floor_rampup_frac,
  )
  scale = min_scale + info["value"] * (1.0 - min_scale)
  env.action_manager.get_term(action_name).range_scale = scale
  return {**info, "range_scale": scale}


def goal_rotation_curriculum(
  env,
  env_ids,
  command_name: str = "reorient_command",
  min_angle: float = math.radians(30.0),
  count_threshold: int = 3,
  promote_frac: float = 0.5,
  demote_frac: float = 0.1,
  step: float = 0.01,
  floor_warmup_frac: float = 0.1,
  floor_rampup_frac: float = 0.7,
) -> dict[str, float]:
  """Grow the goal rotation (from the object's current pose) up to full SO(3)."""
  info = _success_level(
    env,
    env_ids,
    f"goal_rotation_{command_name}",
    command_name,
    count_threshold,
    promote_frac,
    demote_frac,
    step,
    floor_warmup_frac,
    floor_rampup_frac,
  )
  max_angle = min_angle + info["value"] * (math.pi - min_angle)
  env.command_manager.get_term(command_name).goal_max_angle = max_angle
  return {**info, "goal_max_angle_deg": math.degrees(max_angle)}
