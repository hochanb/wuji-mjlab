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

import copy
import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from wuji_mjlab.tasks.reorient.grasp_stage.mdp import (
  gather_rows,
  host_tensor,
  sample_rows,
)
from wuji_mjlab.tasks.reorient.grasp_stage.schema import load_grasp_sets
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
    if not cfg.goal_grasp_sets:
      raise ValueError("GraspGoalCommand: no goal grasp sets")
    goals = load_grasp_sets(cfg.goal_grasp_sets, cfg.hand)
    # The base class sizes its tip targets from fingertip_targets: the first
    # goal stands in until the per-env goals below take over.
    first = tuple(
      zip(goals.tip_points[0].tolist(), goals.tip_normals[0].tolist(), strict=True)
    )
    super().__init__(replace(cfg, fingertip_targets=first), env)
    dev, n = self.device, self.num_envs
    # The goal set stays in host memory (CPU tensors over the loaded arrays);
    # sampling sends only the drawn rows to the device (gather_rows).
    self._goal_tables = tuple(
      host_tensor(getattr(goals, a))
      for a in (
        "object_pos",
        "object_quat",
        "pose_weights",
        "tip_points",
        "tip_normals",
        "tip_weights",
        "joint_q",
        "joint_weights",
      )
    )
    # Goal object pose in the tag frame: T_tag^obj = (T_palm^tag)^-1 T_palm^obj,
    # applied to the drawn rows.
    tag_pos, tag_quat = palm_to_tag(
      env.sim.mj_model,
      int(self.robot.indexing.body_ids[self.palm_body_id]),
      int(self.robot.indexing.site_ids[self.tag_site_id]),
    )
    self._tag_pos = torch.tensor(tag_pos, dtype=torch.float32, device=dev)
    self._tag_quat_inv = quat_inv(
      torch.tensor(tag_quat, dtype=torch.float32, device=dev)
    )
    self._finger_joint_ids, _ = self.robot.find_joints(cfg.finger_joint_pattern)
    k = len(self._tip_site_ids)
    if goals.joint_q.shape[-1] != len(self._finger_joint_ids):
      raise ValueError(
        f"GraspGoalCommand: joint goals have {goals.joint_q.shape[-1]} angles, "
        f"'{cfg.finger_joint_pattern}' matched {len(self._finger_joint_ids)} joints"
      )
    # Per-env goal grasp (the base class reads _tip_target_*_b with expand);
    # every env starts from the first goal until its first resample.
    self.goal_index = torch.zeros(n, dtype=torch.long, device=dev)
    self._tip_target_pos_b = torch.zeros(n, k, 3, device=dev)
    self._tip_target_dir_b = torch.zeros(n, k, 3, device=dev)
    self.tip_weights = torch.zeros(n, k, device=dev)
    self.pose_weights = torch.zeros(n, 2, device=dev)  # (w_pos, w_rot)
    self.joint_goal = torch.zeros(n, len(self._finger_joint_ids), device=dev)
    self.joint_weights = torch.zeros(n, k, device=dev)
    # Grasp errors are read by rewards, the success gate, metrics and both
    # observation groups; compute them once per sim state. A step sees two
    # states: the rewards' (after physics) and, after the resets and
    # sim.forward(), the command update's and observations'. The key is the
    # step counter plus an epoch bumped when the command update starts and
    # whenever goals change.
    self._cache_epoch = 0
    self._cache_key: tuple[int, int] | None = None
    self._cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    first_idx = torch.zeros(n, dtype=torch.long)
    self._set_goal_rows(
      torch.arange(n, device=dev),
      first_idx,
      gather_rows(self._goal_tables, first_idx, dev),
    )
    self.metrics["tip_error"] = torch.zeros(n, device=dev)
    self.metrics["joint_error"] = torch.zeros(n, device=dev)
    self._ghost_model: mujoco.MjModel | None = None  # built on first draw
    self.show_goal_hand = cfg.goal_hand_alpha > 0

  def compute(self, dt, env_ids=None) -> None:
    self.invalidate_grasp_errors()  # runs after sim.forward(): a new state
    super().compute(dt, env_ids)

  def invalidate_grasp_errors(self) -> None:
    self._cache_epoch += 1

  # -- goal ------------------------------------------------------------------
  def _set_goal_rows(self, env_ids, idx, rows) -> None:
    pos, quat, pose_w, tip_pos, tip_dir, tip_w, joint_q, joint_w = rows
    n, k = len(idx), self._tip_target_pos_b.shape[1]
    inv = self._tag_quat_inv.expand(n, 4)
    self.goal_index[env_ids] = idx.to(self.device)
    self.goal_pos_tag[env_ids] = quat_apply(inv, pos - self._tag_pos)
    self.goal_quat_tag[env_ids] = quat_mul(inv, quat)
    self.pose_weights[env_ids] = pose_w
    self._tip_target_pos_b[env_ids] = tip_pos.view(n, k, 3)
    self._tip_target_dir_b[env_ids] = torch.nn.functional.normalize(
      tip_dir.view(n, k, 3), dim=-1
    )
    self.tip_weights[env_ids] = tip_w
    self.joint_goal[env_ids] = joint_q
    self.joint_weights[env_ids] = joint_w
    self.invalidate_grasp_errors()

  # -- goal ------------------------------------------------------------------
  def _sample_goal_in_world(self, env_ids: torch.Tensor) -> None:
    idx, rows = sample_rows(self._goal_tables, env_ids.numel(), self.device)
    self._set_goal_rows(env_ids, idx, rows)

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

  # -- viewer ----------------------------------------------------------------
  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
    super().create_gui(name, server, get_env_idx, on_change, request_action)
    with server.gui.add_folder("Grasp goal"):
      hand = server.gui.add_checkbox(
        "Goal hand (ghost)",
        initial_value=self.show_goal_hand,
        hint="The goal's finger joint angles on the current hand base.",
      )
      tips = server.gui.add_checkbox(
        "Tip -> target vectors",
        initial_value=self.show_tip_targets,
        hint="Fingertip target points on the object, their normals and the "
        "arrows from each target point to its fingertip.",
      )

    def toggle(attr, box):
      def update(_) -> None:
        setattr(self, attr, box.value)
        if on_change is not None:
          on_change()

      return update

    hand.on_update(toggle("show_goal_hand", hand))
    tips.on_update(toggle("show_tip_targets", tips))

  def _debug_vis_impl(self, visualizer) -> None:
    super()._debug_vis_impl(visualizer)
    alpha = self.cfg.goal_hand_alpha
    show = self.show_goal_hand and alpha > 0
    env_ids = visualizer.get_env_indices(self.num_envs) if show else []
    if not env_ids:
      return
    if self._ghost_model is None:
      # The hand's visual geoms only, in one colour; everything else hidden.
      model = copy.deepcopy(self._env.sim.mj_model)
      hand = set(self.robot.indexing.geom_ids.tolist())
      for g in range(model.ngeom):
        visual = model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
        if g in hand and visual:
          model.geom_rgba[g] = self.cfg.goal_hand_color
        else:
          model.geom_rgba[g, 3] = 0.0
      self._ghost_model = model
    # The hand where it is, with the fingers at the goal's joint angles.
    q_adr = self.robot.indexing.joint_q_adr[self._finger_joint_ids]
    data = self._env.sim.data
    mocap = self.robot.indexing.mocap_id
    for i in env_ids:
      qpos = data.qpos[i].clone()
      qpos[q_adr] = self.joint_goal[i]
      visualizer.add_ghost_mesh(
        qpos.cpu().numpy(),
        model=self._ghost_model,
        mocap_pos=data.mocap_pos[i].cpu().numpy() if mocap is not None else None,
        mocap_quat=data.mocap_quat[i].cpu().numpy() if mocap is not None else None,
        alpha=alpha,
        label=f"goal_hand_{i}",
      )


@dataclass(kw_only=True)
class GraspGoalCommandCfg(InHandReorientCommandCfg):
  # Goal grasp-set folders (concatenated), read when the command is built.
  goal_grasp_sets: tuple[str, ...] = ()
  hand: str = "wuji_hand2_right"
  finger_joint_pattern: str = ".*_finger.*_joint.*"
  pad_depth: float = 0.005
  # Success G: tips with w_tip > 0 within this distance [m] of their contact
  # point and this angle [rad] of the normal; fingers with w_joint > 0 within
  # this joint error [rad].
  tip_success_dist: float = 0.01
  tip_success_angle: float = math.radians(60.0)
  joint_success_tol: float = 0.2
  # Viewer: the goal hand shape (joint goal on the current hand base) as a
  # translucent ghost; alpha 0 hides it (also a checkbox in the viser panel).
  goal_hand_alpha: float = 0.35
  goal_hand_color: tuple[float, float, float, float] = (0.2, 0.8, 0.4, 1.0)

  def build(self, env) -> GraspGoalCommand:
    return GraspGoalCommand(self, env)
