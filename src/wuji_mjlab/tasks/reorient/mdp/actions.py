# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


def warmup_steps(warmup_time_s: float, step_dt: float) -> int:
  """Control steps to hold at the default pose, as an exact integer count.

  The bound must not be tested by comparing ``step_index * step_dt`` against
  ``warmup_time_s`` in floating point.
  Mirrored in ``deploy/reorient/wuji_reorient_deploy/obs_builder.py``; a test pins them equal.
  """
  if step_dt <= 0.0:
    raise ValueError(f"step_dt must be positive, got {step_dt}")
  return max(0, math.ceil(warmup_time_s / step_dt - 1e-9))


class JointPositionOffsetEMAAction(JointPositionAction):
  cfg: "JointPositionOffsetEMAActionCfg"

  def __init__(self, cfg: "JointPositionOffsetEMAActionCfg", env):
    super().__init__(cfg, env)

    self._action_scale = cfg.action_scale
    self._ema_alpha = cfg.ema_alpha
    self._warmup_time_s = cfg.warmup_time_s
    self._warmup_steps = warmup_steps(cfg.warmup_time_s, env.step_dt)

    self._default_joint_pos = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()

    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower_limits = soft_limits[..., 0]
    self._upper_limits = soft_limits[..., 1]

    self._prev_target = self._default_joint_pos.clone()
    self._processed_actions.copy_(self._default_joint_pos)

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    clamped = torch.clamp(actions, -1.0, 1.0)

    raw_target = self._default_joint_pos + clamped * self._action_scale
    raw_target = torch.clamp(raw_target, self._lower_limits, self._upper_limits)

    smoothed = (
      self._ema_alpha * raw_target + (1.0 - self._ema_alpha) * self._prev_target
    )

    in_warmup = (self._env.episode_length_buf < self._warmup_steps).unsqueeze(-1)
    self._processed_actions = torch.where(in_warmup, self._default_joint_pos, smoothed)
    self._prev_target = self._processed_actions.clone()

  @property
  def processed_action(self) -> torch.Tensor:
    return self._processed_actions

  def reset(self, env_ids: torch.Tensor) -> None:
    super().reset(env_ids)
    self._prev_target[env_ids] = self._default_joint_pos[env_ids]
    self._processed_actions[env_ids] = self._default_joint_pos[env_ids]


@dataclass(kw_only=True)
class JointPositionOffsetEMAActionCfg(JointPositionActionCfg):
  """Configuration for offset + EMA action with warmup."""

  action_scale: float = 0.5
  ema_alpha: float = 0.5
  warmup_time_s: float = 0.4

  def build(self, env) -> JointPositionOffsetEMAAction:
    return JointPositionOffsetEMAAction(self, env)


class WristPoseAction(ActionTerm):
  """6-DoF wrist motion through actuated wrist joints.

  Action layout ``[rx, ry, rz, tx, ty, tz]``, each in ``[-1, 1]``, scaled to
  ``max_angle`` / ``max_translation`` and sent as position targets to the
  joints named in ``joint_names`` (same order). Targets are EMA-smoothed,
  rate-limited (``max_rate_normalized``, the same fraction of each joint's
  limit per second for rotation and translation) and interpolated over the
  physics substeps; the joints' position servos then move the hand base with
  real velocities, so contacts follow the motion.
  """

  cfg: "WristPoseActionCfg"

  def __init__(self, cfg: "WristPoseActionCfg", env):
    super().__init__(cfg, env)
    ids, _ = self._entity.find_joints(cfg.joint_names, preserve_order=True)
    if len(ids) != 6:
      raise ValueError(f"WristPoseAction needs 6 joints, got {cfg.joint_names}")
    n, device = self.num_envs, self.device
    self._joint_ids = torch.tensor(ids, device=device)
    self._raw_actions = torch.zeros(n, 6, device=device)
    self._smoothed = torch.zeros(n, 6, device=device)  # EMA, before the rate limit
    self._target = torch.zeros(n, 6, device=device)  # smoothed, rate-limited
    self._applied = torch.zeros(n, 6, device=device)  # current servo target
    self._step_start = torch.zeros(n, 6, device=device)
    self._substep = 0
    self._limit = torch.tensor(
      [cfg.max_angle] * 3 + [cfg.max_translation] * 3, device=device
    )
    self._max_step = self._limit * cfg.max_rate_normalized * env.step_dt
    self._warmup_steps = warmup_steps(cfg.warmup_time_s, env.step_dt)
    # Fraction of the joint limits the action may command (set by a curriculum).
    # Observations and penalties stay normalised by the full limits.
    self.range_scale = 1.0

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def processed_action(self) -> torch.Tensor:
    """Per-step servo target, normalised by the joint limits."""
    return self._target / self._limit

  @property
  def smoothed_action(self) -> torch.Tensor:
    """EMA-smoothed target before the rate limit, normalised by the joint limits."""
    return self._smoothed / self._limit

  @property
  def wrist_offset(self) -> torch.Tensor:
    """Measured wrist joint positions ``[rx, ry, rz (rad), tx, ty, tz (m)]``."""
    return self._entity.data.joint_pos[:, self._joint_ids]

  @property
  def wrist_offset_normalized(self) -> torch.Tensor:
    return self.wrist_offset / self._limit

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    raw_target = torch.clamp(actions, -1.0, 1.0) * self._limit * self.range_scale
    smoothed = (
      self.cfg.ema_alpha * raw_target + (1.0 - self.cfg.ema_alpha) * self._target
    )
    step = torch.clamp(smoothed - self._target, -self._max_step, self._max_step)
    target = self._target + step
    in_warmup = (self._env.episode_length_buf < self._warmup_steps).unsqueeze(-1)
    self._smoothed = torch.where(in_warmup, torch.zeros_like(smoothed), smoothed)
    self._target = torch.where(in_warmup, torch.zeros_like(target), target)
    self._step_start = self._applied.clone()
    self._substep = 0

  def apply_actions(self) -> None:
    self._substep += 1
    frac = min(self._substep / self._env.cfg.decimation, 1.0)
    self._applied = self._step_start + frac * (self._target - self._step_start)
    self._entity.set_joint_position_target(self._applied, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._smoothed[env_ids] = 0.0
    self._target[env_ids] = 0.0
    self._applied[env_ids] = 0.0
    self._step_start[env_ids] = 0.0


@dataclass(kw_only=True)
class WristPoseActionCfg(ActionTermCfg):
  """Actuated wrist rotation + translation (see ``WristPoseAction``)."""

  joint_names: tuple[str, ...]  # [rx, ry, rz, tx, ty, tz]
  max_angle: float = math.radians(15.0)  # [rad] per axis
  max_translation: float = 0.05  # [m] per axis
  # Per-axis rate limit in units of the axis limit per second, shared by rotation
  # and translation (5.0: 0 -> limit in 0.2 s, i.e. 75 deg/s and 0.25 m/s).
  max_rate_normalized: float = 5.0
  ema_alpha: float = 0.3
  warmup_time_s: float = 0.4

  def build(self, env) -> WristPoseAction:
    return WristPoseAction(self, env)
