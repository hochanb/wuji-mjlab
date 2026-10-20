# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
)

from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
  object_ghost_parts,
)
from wuji_mjlab.utils.math import random_quat_uniform

if TYPE_CHECKING:
  import viser


def random_quat_within_angle(
  n: int, max_angle: float, device, newton_iters: int = 12
) -> torch.Tensor:
  """Rotations of at most ``max_angle``, distributed as uniform SO(3) truncated.

  The rotation angle of a uniform SO(3) sample has CDF ``(a - sin a) / pi``;
  sampling that CDF below ``max_angle`` (Newton inversion) with a uniform axis
  reproduces uniform SO(3) exactly at ``max_angle = pi``.
  """
  max_angle = min(float(max_angle), math.pi)
  cdf_max = (max_angle - math.sin(max_angle)) / math.pi
  target = math.pi * cdf_max * torch.rand(n, device=device)  # a - sin(a)
  angle = torch.clamp((6.0 * target).pow(1.0 / 3.0), max=max_angle)
  for _ in range(newton_iters):
    f = angle - torch.sin(angle) - target
    angle = torch.clamp(
      angle - f / (1.0 - torch.cos(angle)).clamp_min(1e-6), 0.0, max_angle
    )
  axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
  half = 0.5 * angle.unsqueeze(-1)
  return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


class InHandReorientCommand(CommandTerm):
  cfg: InHandReorientCommandCfg

  def __init__(self, cfg: InHandReorientCommandCfg, env):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.robot: Entity = env.scene[cfg.robot_entity_name]
    palm_ids, _ = self.robot.find_bodies(cfg.palm_body_pattern)
    if not palm_ids:
      raise ValueError(
        f"InHandReorientCommand: no palm body matched pattern "
        f"'{cfg.palm_body_pattern}' on entity '{cfg.robot_entity_name}'."
      )
    self.palm_body_id = int(palm_ids[0])

    tag_ids, _ = self.robot.find_sites(cfg.tag_site_pattern)
    if not tag_ids:
      raise ValueError(
        f"InHandReorientCommand: no tag site matched '{cfg.tag_site_pattern}' "
        f"on entity '{cfg.robot_entity_name}'."
      )
    self.tag_site_id = int(tag_ids[0])

    # Goal pose lives in the wrist-tag frame: the task is to bring the object to
    # a pose relative to the wrist, so success, rewards and observations all see
    # the wrist-local object transform. Wrist motion (when actuated) only helps.
    self.goal_quat_tag = torch.zeros(self.num_envs, 4, device=self.device)
    self.goal_quat_tag[:, 0] = 1.0
    # Max goal rotation from the object's current pose (set by a curriculum);
    # >= pi samples uniformly on SO(3).
    self.goal_max_angle = math.pi
    # Bounded goals sampled on the reset path see stale kinematics (the reset
    # writes qpos only); they are re-drawn on the next per-step update.
    self._goal_pending = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Optional position goal, stored in the wrist-tag frame (deploy semantics).
    self.goal_pos_tag = torch.zeros(self.num_envs, 3, device=self.device)
    if cfg.goal_pos_range is not None:
      lo, hi = torch.tensor(cfg.goal_pos_range, device=self.device).T
      self._goal_pos_lo, self._goal_pos_hi = lo, hi
      self.goal_pos_tag[:] = 0.5 * (lo + hi)

    self.hold_counter = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self.goal_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    self.goal_reach_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self.in_success_window = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.window_timer = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self._reward_window_timer = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self.reward_hold_counter = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._reward_hold_counter_snapshot = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    self._success_achieved = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._goal_switched = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._within_threshold = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["goal_reach_count"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["ori_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["hold_counter"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_timer"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["in_success_window"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["window_timer"] = torch.zeros(self.num_envs, device=self.device)
    if cfg.goal_pos_range is not None:
      self.metrics["pos_error"] = torch.zeros(self.num_envs, device=self.device)

    self._visualization = ReorientCommandVisualization(entity_name=self.cfg.entity_name)
    # Without a compiled model (lightweight test envs) the plain box ghost is used.
    sim = getattr(env, "sim", None)
    indexing = getattr(self.object, "indexing", None)
    if sim is not None and indexing is not None:
      self._visualization.ghost_parts = object_ghost_parts(
        sim.mj_model, indexing.geom_ids.tolist()
      )
    if cfg.goal_pos_range is not None:
      self._visualization.pos_error_for_env = lambda i: float(self.pos_error()[i])

  @property
  def command(self) -> torch.Tensor:
    return self.goal_quat_w

  @property
  def goal_quat(self) -> torch.Tensor:
    return self.goal_quat_w

  @property
  def goal_quat_w(self) -> torch.Tensor:
    """World goal orientation from the current wrist-tag pose."""
    tag_quat_w = self.robot.data.site_pose_w[:, self.tag_site_id, 3:7]
    return quat_mul(tag_quat_w, self.goal_quat_tag)

  @property
  def goal_pos_w(self) -> torch.Tensor:
    tag_pose = self.robot.data.site_pose_w[:, self.tag_site_id]
    return tag_pose[:, :3] + quat_apply(tag_pose[:, 3:7], self.goal_pos_tag)

  def pos_error(self) -> torch.Tensor:
    return torch.norm(self.object.data.root_link_pos_w - self.goal_pos_w, dim=-1)

  def _update_metrics(self) -> None:
    self.metrics["goal_reach_count"] = self.goal_reach_count.float()
    ori_err = quat_error_magnitude(self.object.data.root_link_quat_w, self.goal_quat_w)
    self.metrics["ori_error"] = ori_err
    self.metrics["hold_counter"] = self.hold_counter.float()
    self.metrics["goal_timer"] = self.goal_timer.float()
    self.metrics["in_success_window"] = self.in_success_window.float()
    self.metrics["window_timer"] = self.window_timer.float()
    if self.cfg.goal_pos_range is not None:
      self.metrics["pos_error"] = self.pos_error()
    self._visualization.update_status_gui(
      num_envs=self.num_envs,
      policy_status_for_env=self._policy_status_for_env,
    )

  def _sample_goal_in_world(self, env_ids: torch.Tensor) -> None:
    """Sample a new goal for the given envs, in the wrist-tag frame.

    Orientation is uniform on SO(3) in the tag frame, matching deploy semantics
    (the deploy side samples goal_quat directly in tag frame); it stays fixed
    relative to the wrist (see ``goal_quat_w``).
    """
    n = len(env_ids)
    if self.cfg.goal_quat_tag_fixed is not None:
      self.goal_quat_tag[env_ids] = torch.tensor(
        self.cfg.goal_quat_tag_fixed, device=self.device
      )
    elif self.goal_max_angle >= math.pi:
      self.goal_quat_tag[env_ids] = random_quat_uniform(n, self.device)
    else:
      tag_quat_w = self.robot.data.site_pose_w[env_ids, self.tag_site_id, 3:7]
      object_in_tag = quat_mul(
        quat_inv(tag_quat_w), self.object.data.root_link_quat_w[env_ids]
      )
      delta = random_quat_within_angle(n, self.goal_max_angle, self.device)
      self.goal_quat_tag[env_ids] = quat_mul(object_in_tag, delta)
      self._goal_pending[env_ids] = True
    if self.cfg.goal_pos_range is not None:
      u = torch.rand(n, 3, device=self.device)
      self.goal_pos_tag[env_ids] = self._goal_pos_lo + u * (
        self._goal_pos_hi - self._goal_pos_lo
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._sample_goal_in_world(env_ids)
    self.hold_counter[env_ids] = 0
    self.goal_timer[env_ids] = 0
    self.goal_reach_count[env_ids] = 0
    self.in_success_window[env_ids] = False
    self.window_timer[env_ids] = 0
    self._reward_window_timer[env_ids] = 0
    self.reward_hold_counter[env_ids] = 0
    self._reward_hold_counter_snapshot[env_ids] = 0
    self._success_achieved[env_ids] = False
    self._goal_switched[env_ids] = False
    self._within_threshold[env_ids] = False

  def _min_goal_steps(self) -> int:
    return max(int(self.cfg.min_goal_interval / self._env.step_dt), 0)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None and bool(self._goal_pending.any()):
      # Per-step path: kinematics are fresh, so re-draw goals bounded relative
      # to the object pose drawn on the reset path.
      pending = self._goal_pending.nonzero(as_tuple=False).squeeze(-1)
      self._goal_pending[pending] = False
      self._sample_goal_in_world(pending)
      self._goal_pending[pending] = False
    selected = slice(None) if env_ids is None else env_ids
    goal_timer = self.goal_timer[selected] + 1
    ori_err = quat_error_magnitude(
      self.object.data.root_link_quat_w[selected], self.goal_quat_w[selected]
    )
    within_threshold = ori_err < self.cfg.success_threshold
    if self.cfg.pos_success_threshold is not None:
      pos_err = self.pos_error()[selected]
      within_threshold = within_threshold & (pos_err < self.cfg.pos_success_threshold)

    approaching_mask = ~self.in_success_window[selected]
    hold_counter = self.hold_counter[selected]
    hold_counter = torch.where(
      approaching_mask & within_threshold,
      hold_counter + 1,
      torch.where(approaching_mask, torch.zeros_like(hold_counter), hold_counter),
    )

    just_succeeded = approaching_mask & (hold_counter >= self.cfg.success_hold_steps)

    in_success_window = self.in_success_window[selected] | just_succeeded
    window_timer = self.window_timer[selected]
    window_timer = torch.where(
      just_succeeded, torch.zeros_like(window_timer), window_timer
    )
    hold_counter = torch.where(
      just_succeeded, torch.zeros_like(hold_counter), hold_counter
    )
    self.goal_reach_count[selected] += just_succeeded.long()

    self._success_achieved[selected] = just_succeeded

    window_mask = in_success_window
    window_timer = torch.where(window_mask, window_timer + 1, window_timer)

    reward_hold_counter = self.reward_hold_counter[selected]
    reward_hold_counter = torch.where(
      window_mask & within_threshold,
      reward_hold_counter + 1,
      reward_hold_counter,
    )
    reward_hold_counter = torch.where(
      just_succeeded, torch.zeros_like(reward_hold_counter), reward_hold_counter
    )

    min_goal_steps = self._min_goal_steps()
    should_switch = (
      window_mask
      & (window_timer >= self.cfg.goal_switch_delay)
      & (goal_timer >= min_goal_steps)
    )

    advance_ids = should_switch.nonzero(as_tuple=False).squeeze(-1)
    if env_ids is not None:
      advance_ids = env_ids[advance_ids]
    if advance_ids.numel() > 0:
      self._sample_goal_in_world(advance_ids)

    self._goal_switched[selected] = should_switch

    self._reward_window_timer[selected] = window_timer
    self._reward_hold_counter_snapshot[selected] = reward_hold_counter

    in_success_window = torch.where(
      should_switch, torch.zeros_like(in_success_window), in_success_window
    )
    window_timer = torch.where(
      should_switch, torch.zeros_like(window_timer), window_timer
    )
    hold_counter = torch.where(
      should_switch, torch.zeros_like(hold_counter), hold_counter
    )
    goal_timer = torch.where(should_switch, torch.zeros_like(goal_timer), goal_timer)
    reward_hold_counter = torch.where(
      should_switch, torch.zeros_like(reward_hold_counter), reward_hold_counter
    )

    self._within_threshold[selected] = within_threshold

    self.hold_counter[selected] = hold_counter
    self.goal_timer[selected] = goal_timer
    self.in_success_window[selected] = in_success_window
    self.window_timer[selected] = window_timer
    self.reward_hold_counter[selected] = reward_hold_counter

  def _debug_vis_impl(self, visualizer) -> None:
    self._visualization.draw_debug_visuals(
      visualizer=visualizer,
      num_envs=self.num_envs,
      palm_pose_w=self.robot.data.body_link_pose_w[:, self.palm_body_id, :],
      tag_pose_w=self.robot.data.site_pose_w[:, self.tag_site_id, :],
      object_pos_w=self.object.data.root_link_pos_w,
      goal_quat_w=self.goal_quat_w,
      goal_pos_w=self.goal_pos_w if self.cfg.goal_pos_range is not None else None,
      object_quat_w=self.object.data.root_link_quat_w,
      policy_status_for_env=self._policy_status_for_env,
    )

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx,
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    # This panel only displays status; mjlab's editing callbacks are unused.
    self._visualization.create_gui(
      name=name,
      server=server,
      get_env_idx=get_env_idx,
    )
    self._visualization.update_status_gui(
      num_envs=self.num_envs,
      policy_status_for_env=self._policy_status_for_env,
    )

  def _policy_status_for_env(self, env_idx: int) -> tuple[float, bool]:
    ori_error_rad = float(
      quat_error_magnitude(
        self.object.data.root_link_quat_w[env_idx : env_idx + 1],
        self.goal_quat_w[env_idx : env_idx + 1],
      )[0].item()
    )
    is_success = ori_error_rad < self.cfg.success_threshold
    if self.cfg.pos_success_threshold is not None:
      pos_err = float(self.pos_error()[env_idx].item())
      is_success = is_success and pos_err < self.cfg.pos_success_threshold
    return ori_error_rad, is_success

  @property
  def within_threshold(self) -> torch.Tensor:
    return self._within_threshold

  @property
  def success_achieved(self) -> torch.Tensor:
    return self._success_achieved

  @property
  def goal_switched(self) -> torch.Tensor:
    return self._goal_switched

  @property
  def reward_window_timer(self) -> torch.Tensor:
    return self._reward_window_timer

  @property
  def reward_hold_counter_snapshot(self) -> torch.Tensor:
    return self._reward_hold_counter_snapshot


@dataclass(kw_only=True)
class InHandReorientCommandCfg(CommandTermCfg):
  entity_name: str
  success_threshold: float = 0.2
  success_hold_steps: int = 5
  goal_switch_delay: int = 20
  min_goal_interval: float = 0.0
  # Per-axis (lo, hi) goal-position box in the wrist-tag frame [m]; None keeps
  # the orientation-only task. lo == hi pins the goal to a single point.
  goal_pos_range: tuple[tuple[float, float], ...] | None = None
  # Position gate for success [m]; None leaves success orientation-only.
  pos_success_threshold: float | None = None
  # Fixed goal orientation in the wrist-tag frame (wxyz); None samples goals.
  goal_quat_tag_fixed: tuple[float, float, float, float] | None = None
  debug_vis: bool = False
  robot_entity_name: str = "robot"
  palm_body_pattern: str = ".*_palm_link"
  tag_site_pattern: str = ".*_wrist_tag"

  def build(self, env) -> InHandReorientCommand:
    return InHandReorientCommand(self, env)
