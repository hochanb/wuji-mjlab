# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Goal command whose goal is a grasp drawn per episode from a grasp set.

The object pose of a grasp is given in the hand base (palm) frame and stored,
like the base command, in the wrist-tag frame (a fixed child of the palm), so
the reorient observations, rewards and viewer work unchanged. Each env also
carries its goal grasp's fingertip targets, tip / joint weights and joint
goal. Success G(s, xi) needs the object position (w_pos > 0) and
orientation (w_rot > 0) within the pose thresholds, every finger with
w_tip > 0 on its target and every finger with w_joint > 0 at its joint goal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from wuji_mjlab.tasks.reorient.mdp.commands import (
  InHandReorientCommand,
  InHandReorientCommandCfg,
)


def palm_to_tag(mj_model, palm_body_id: int, tag_site_id: int):
  """Tag site pose in the palm body frame (both fixed, so any qpos); global ids."""
  d = mujoco.MjData(mj_model)
  mujoco.mj_kinematics(mj_model, d)
  rb = d.xmat[palm_body_id].reshape(3, 3)
  pos = rb.T @ (d.site_xpos[tag_site_id] - d.xpos[palm_body_id])
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, (rb.T @ d.site_xmat[tag_site_id].reshape(3, 3)).ravel())
  return pos, quat


class GraspGoalCommand(InHandReorientCommand):
  cfg: GraspGoalCommandCfg

  def __init__(self, cfg: GraspGoalCommandCfg, env):
    super().__init__(cfg, env)
    dev, n = self.device, self.num_envs
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    pos_palm, quat_palm = t(cfg.goal_object_pos), t(cfg.goal_object_quat)
    # Goal object pose in the tag frame: T_tag^obj = (T_palm^tag)^-1 T_palm^obj.
    tag_pos, tag_quat = palm_to_tag(
      env.sim.mj_model,
      int(self.robot.indexing.body_ids[self.palm_body_id]),
      int(self.robot.indexing.site_ids[self.tag_site_id]),
    )
    tag_pos, tag_quat = t(tag_pos), t(tag_quat)
    g = pos_palm.shape[0]
    inv = quat_inv(tag_quat).expand(g, 4)
    self._goal_pos_tag_set = quat_apply(inv, pos_palm - tag_pos)
    self._goal_quat_tag_set = quat_mul(inv, quat_palm)
    self._tip_pos_set = t(cfg.goal_tip_points)
    self._tip_dir_set = torch.nn.functional.normalize(t(cfg.goal_tip_normals), dim=-1)
    self._tip_w_set = t(cfg.goal_tip_weights)
    self._pose_w_set = t(cfg.goal_pose_weights)
    self._joint_q_set = t(cfg.goal_joint_q)
    self._joint_w_set = t(cfg.goal_joint_weights)
    self._finger_joint_ids, _ = self.robot.find_joints(cfg.finger_joint_pattern)
    k = len(self._tip_site_ids)
    if self._joint_q_set.shape[-1] != len(self._finger_joint_ids):
      raise ValueError(
        f"GraspGoalCommand: joint goals have {self._joint_q_set.shape[-1]} angles, "
        f"'{cfg.finger_joint_pattern}' matched {len(self._finger_joint_ids)} joints"
      )
    # Per-env goal grasp (the base class reads _tip_target_*_b with expand).
    self.goal_index = torch.zeros(n, dtype=torch.long, device=dev)
    self._tip_target_pos_b = self._tip_pos_set[0].expand(n, k, 3).clone()
    self._tip_target_dir_b = self._tip_dir_set[0].expand(n, k, 3).clone()
    self.tip_weights = self._tip_w_set[0].expand(n, k).clone()
    self.pose_weights = self._pose_w_set[0].expand(n, 2).clone()  # (w_pos, w_rot)
    self.joint_goal = self._joint_q_set[0].expand(n, -1).clone()
    self.joint_weights = self._joint_w_set[0].expand(n, k).clone()
    self.metrics["tip_error"] = torch.zeros(n, device=dev)
    self.metrics["joint_error"] = torch.zeros(n, device=dev)
    # Grasp errors are read by rewards, the success gate, metrics and both
    # observation groups; compute them once per sim state. A step sees two
    # states: the rewards' (after physics) and, after the resets and
    # sim.forward(), the command update's and observations'. The key is the
    # step counter plus an epoch bumped when the command update starts and
    # whenever goals change.
    self._cache_epoch = 0
    self._cache_key: tuple[int, int] | None = None
    self._cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

  def compute(self, dt, env_ids=None) -> None:
    self.invalidate_grasp_errors()  # runs after sim.forward(): a new state
    super().compute(dt, env_ids)

  def invalidate_grasp_errors(self) -> None:
    self._cache_epoch += 1

  # -- goal ------------------------------------------------------------------
  def _sample_goal_in_world(self, env_ids: torch.Tensor) -> None:
    idx = torch.randint(
      0, self._tip_pos_set.shape[0], (env_ids.numel(),), device=self.device
    )
    self.goal_index[env_ids] = idx
    self.goal_pos_tag[env_ids] = self._goal_pos_tag_set[idx]
    self.goal_quat_tag[env_ids] = self._goal_quat_tag_set[idx]
    self._tip_target_pos_b[env_ids] = self._tip_pos_set[idx]
    self._tip_target_dir_b[env_ids] = self._tip_dir_set[idx]
    self.tip_weights[env_ids] = self._tip_w_set[idx]
    self.pose_weights[env_ids] = self._pose_w_set[idx]
    self.joint_goal[env_ids] = self._joint_q_set[idx]
    self.joint_weights[env_ids] = self._joint_w_set[idx]
    self.invalidate_grasp_errors()

  # -- errors ----------------------------------------------------------------
  def _grasp_errors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (int(self._env.common_step_counter), self._cache_epoch)
    if self._cache_key != key or self._cache is None:
      tip_pos, target_pos = self.fingertip_pos_w, self.fingertip_target_pos_w
      target_dir = self.fingertip_target_dir_w
      offset = tip_pos - (target_pos + self.cfg.pad_depth * target_dir)
      quat = self.object.data.root_link_quat_w.unsqueeze(1).expand(
        -1, offset.shape[1], -1
      )
      rel = tip_pos - target_pos
      cos = (rel * target_dir).sum(-1) / rel.norm(dim=-1).clamp_min(1e-6)
      q = self.robot.data.joint_pos[:, self._finger_joint_ids]
      err = (
        (q - self.joint_goal).abs().view(self.num_envs, self.tip_weights.shape[1], -1)
      )
      self._cache = (quat_apply_inverse(quat, offset), cos, err.amax(-1))
      self._cache_key = key
    return self._cache

  def tip_offset_b(self) -> torch.Tensor:
    """[N, F, 3] tip reference point minus where it sits when the pad touches
    the target (target + pad_depth along the normal), object frame."""
    return self._grasp_errors()[0]

  def tip_cos(self) -> torch.Tensor:
    """[N, F] cosine between (tip - target point) and the target normal."""
    return self._grasp_errors()[1]

  def joint_error(self) -> torch.Tensor:
    """[N, F] largest |q - q_goal| over each finger's joints [rad]."""
    return self._grasp_errors()[2]

  def _pose_weights(self) -> torch.Tensor:
    return self.pose_weights

  def _success_gate(self) -> torch.Tensor:
    dist = self.tip_offset_b().norm(dim=-1)
    # Outer side: the reference point lies within the cone around the normal.
    tip_ok = (dist < self.cfg.tip_success_dist) & (
      self.tip_cos() > math.cos(self.cfg.tip_success_angle)
    )
    joint_ok = self.joint_error() < self.cfg.joint_success_tol
    tip_ok = tip_ok | (self.tip_weights <= 0.0)
    joint_ok = joint_ok | (self.joint_weights <= 0.0)
    return tip_ok.all(-1) & joint_ok.all(-1)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    w = self.tip_weights > 0
    dist = self.tip_offset_b().norm(dim=-1)
    self.metrics["tip_error"] = (dist * w).sum(-1) / w.sum(-1).clamp_min(1)
    wj = self.joint_weights > 0
    self.metrics["joint_error"] = (self.joint_error() * wj).sum(-1) / wj.sum(
      -1
    ).clamp_min(1)


@dataclass(kw_only=True)
class GraspGoalCommandCfg(InHandReorientCommandCfg):
  # Goal grasp set: object pose in the hand base (palm) frame, fingertip
  # targets (object frame) and weights, finger joint goals and weights.
  goal_object_pos: tuple[tuple[float, ...], ...] = ()
  goal_object_quat: tuple[tuple[float, ...], ...] = ()
  goal_pose_weights: tuple[tuple[float, float], ...] = ()  # (w_pos, w_rot)
  goal_tip_points: tuple = ()
  goal_tip_normals: tuple = ()
  goal_tip_weights: tuple = ()
  goal_joint_q: tuple = ()
  goal_joint_weights: tuple = ()
  finger_joint_pattern: str = ".*_finger.*_joint.*"
  pad_depth: float = 0.005
  # Success G: tips with w_tip > 0 within this distance [m] of their contact
  # point and this angle [rad] of the normal; fingers with w_joint > 0 within
  # this joint error [rad].
  tip_success_dist: float = 0.01
  tip_success_angle: float = math.radians(60.0)
  joint_success_tol: float = 0.2

  def build(self, env) -> GraspGoalCommand:
    return GraspGoalCommand(self, env)
