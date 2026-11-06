# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Grasp-stage MDP terms: reset to an initial grasp, grasp observations,
grasp rewards and the object speed limit."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude, quat_mul

from wuji_mjlab.tasks.reorient.mdp._env_utils import resolve_env_ids
from wuji_mjlab.tasks.reorient.mdp.cage import read_cage_penalty_counter
from wuji_mjlab.tasks.reorient.mdp.observations import (
  goal_pos_err_in_tag,
  goal_rot_err_6d,
)
from wuji_mjlab.tasks.reorient.mdp.rewards import (
  orientation_alignment,
  position_alignment,
  tolerance,
)

_ROBOT = SceneEntityCfg("robot")
_OBJECT = SceneEntityCfg("object")
_FINGERS = ".*_finger.*_joint.*"


def _mocap_to_palm(env, robot: Entity, palm_pattern: str):
  """Palm pose in the mocap (mount) body frame with the wrist joints at zero."""
  cache = getattr(env, "_grasp_stage_mocap_to_palm", None)
  if cache is not None:
    return cache
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  palm = int(robot.indexing.body_ids[robot.find_bodies(palm_pattern)[0][0]])
  root = robot.indexing.root_body_id
  rr = d.xmat[root].reshape(3, 3)
  pos = rr.T @ (d.xpos[palm] - d.xpos[root])
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, (rr.T @ d.xmat[palm].reshape(3, 3)).ravel())
  cache = (
    torch.tensor(pos, dtype=torch.float32, device=env.device),
    torch.tensor(quat, dtype=torch.float32, device=env.device),
  )
  env._grasp_stage_mocap_to_palm = cache
  return cache


def _cached(env, key, build):
  """Per-env memo for values derived from term parameters (built once)."""
  cache = env.__dict__.setdefault("_grasp_stage_cache", {})
  if key not in cache:
    cache[key] = build()
  return cache[key]


def _param_tensor(env, data) -> torch.Tensor:
  """``data`` (a term parameter, fixed for the env's life) as a tensor, built
  on first use; keyed by identity, keeping ``data`` alive with it."""
  entry = _cached(
    env,
    ("tensor", id(data)),
    lambda: (data, torch.tensor(data, dtype=torch.float32, device=env.device)),
  )
  return entry[1]


def reset_to_init_grasp(
  env,
  env_ids,
  object_pos: tuple,
  object_quat: tuple,
  joint_q: tuple,
  palm_pattern: str = ".*_palm_link",
  finger_joint_pattern: str = _FINGERS,
  robot_cfg: SceneEntityCfg = _ROBOT,
  object_cfg: SceneEntityCfg = _OBJECT,
) -> None:
  """Put the fingers and the object in an initial grasp drawn per env.

  The object pose is in the hand base (palm) frame. The palm pose comes from
  the mount (mocap) pose just written by the robot reset with the wrist at
  zero, as body kinematics are stale on the reset path; run after it.
  """
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  dev, n = env.device, env_ids.numel()
  pos_set = _param_tensor(env, object_pos)
  quat_set = _param_tensor(env, object_quat)
  q_set = _param_tensor(env, joint_q)
  idx = torch.randint(0, pos_set.shape[0], (n,), device=dev)

  joint_ids = _cached(
    env,
    ("joints", finger_joint_pattern),
    lambda: robot.find_joints(finger_joint_pattern)[0],
  )
  robot.write_joint_state_to_sim(
    q_set[idx], torch.zeros_like(q_set[idx]), joint_ids=joint_ids, env_ids=env_ids
  )

  mocap = robot.indexing.mocap_id
  mount_pos = env.sim.data.mocap_pos[env_ids, mocap]
  mount_quat = env.sim.data.mocap_quat[env_ids, mocap]
  off_pos, off_quat = _mocap_to_palm(env, robot, palm_pattern)
  palm_pos = mount_pos + quat_apply(mount_quat, off_pos.expand(n, 3))
  palm_quat = quat_mul(mount_quat, off_quat.expand(n, 4))
  pos_w = palm_pos + quat_apply(palm_quat, pos_set[idx])
  quat_w = quat_mul(palm_quat, quat_set[idx])
  obj.write_root_link_pose_to_sim(torch.cat((pos_w, quat_w), -1), env_ids=env_ids)
  obj.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


# -- observations ------------------------------------------------------------
def grasp_tip_error(
  env, command_name: str = "reorient_command", noise=None
) -> torch.Tensor:
  """[N, F*3] tip offsets from their contact points, object frame; exactly
  zero for fingers with w_tip = 0 (``noise`` is added before the mask)."""
  cmd = env.command_manager.get_term(command_name)
  offset = cmd.tip_offset_b()
  if noise is not None:
    offset = noise.apply(offset)
  return (offset * (cmd.tip_weights > 0).unsqueeze(-1)).flatten(1)


def grasp_weights(env, command_name: str = "reorient_command") -> torch.Tensor:
  """[N, 2 + 2F] the goal grasp's (w_pos, w_rot), w_tip and w_joint."""
  cmd = env.command_manager.get_term(command_name)
  return torch.cat((cmd.pose_weights, cmd.tip_weights, cmd.joint_weights), dim=-1)


def grasp_rot_err_6d(
  env, command_name: str = "reorient_command", noise=None, **kwargs
) -> torch.Tensor:
  """``goal_rot_err_6d`` (same parameters), exactly zero for goals with
  w_rot = 0: ``noise`` (the term's sensor noise) is added before the mask."""
  cmd = env.command_manager.get_term(command_name)
  obs = goal_rot_err_6d(env, command_name, **kwargs)
  if noise is not None:
    obs = noise.apply(obs)
  return obs * (cmd.pose_weights[:, 1:2] > 0).float()


def grasp_pos_err(
  env, command_name: str = "reorient_command", noise=None, **kwargs
) -> torch.Tensor:
  """``goal_pos_err_in_tag`` (same parameters), exactly zero for goals with
  w_pos = 0: ``noise`` is added before the mask."""
  cmd = env.command_manager.get_term(command_name)
  obs = goal_pos_err_in_tag(env, command_name, **kwargs)
  if noise is not None:
    obs = noise.apply(obs)
  return obs * (cmd.pose_weights[:, 0:1] > 0).float()


# Observation groups are ordered [env obs, condition]: terms measured from the
# environment (they carry the sensor noise) first, then the goal condition,
# never noised.
CONDITION_TERMS = ("grasp_weights", "grasp_joint_goal", "object_encoding")


def observation_layout(env) -> dict[str, list[dict]]:
  """Per observation group, each term's slice, kind and noise, in order."""
  om, layout = env.observation_manager, {}
  for group, names in om.active_terms.items():
    cfg, start, rows = om.cfg[group], 0, []
    for name, dims in zip(names, om.group_obs_term_dim[group], strict=True):
      size = int(np.prod(dims))
      term = cfg.terms[name]
      noise = term.noise or term.params.get("noise")
      rows.append(
        {
          "term": name,
          "slice": [start, start + size],
          "kind": "condition" if name in CONDITION_TERMS else "env",
          "history": term.history_length,
          "noise": type(noise).__name__ if noise is not None else None,
        }
      )
      start += size
    layout[group] = rows
  return layout


def grasp_joint_goal(
  env, command_name: str = "reorient_command", asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """[N, F*J] goal finger joint angles, normalised like the joint observation
  (soft limits to [-1, 1])."""
  cmd = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[asset_cfg.name]
  limits = robot.data.soft_joint_pos_limits[:, cmd._finger_joint_ids]
  center = 0.5 * (limits[..., 0] + limits[..., 1])
  half = 0.5 * (limits[..., 1] - limits[..., 0])
  return ((cmd.joint_goal - center) / (half + 1e-6)).clamp(-1.0, 1.0)


def object_encoding(env, encoding: tuple[float, ...]) -> torch.Tensor:
  """[N, D] the object's preprocessed shape encoding (one object per stage)."""
  return _param_tensor(env, encoding).expand(env.num_envs, -1)


# -- rewards -----------------------------------------------------------------
def _pose_gate(cmd, ori_bound, ori_margin, pos_bound, pos_margin) -> torch.Tensor:
  """Closeness to the goal pose; a part weighted 0 does not gate."""
  ori_err = quat_error_magnitude(cmd.object.data.root_link_quat_w, cmd.goal_quat_w)
  ori = tolerance(ori_err, bounds=(0.0, ori_bound), margin=ori_margin)
  pos = tolerance(cmd.pos_error(), bounds=(0.0, pos_bound), margin=pos_margin)
  w = cmd.pose_weights
  return torch.where(w[:, 1] > 0, ori, 1.0) * torch.where(w[:, 0] > 0, pos, 1.0)


def grasp_orientation_alignment(
  env, command_name: str = "reorient_command", **kwargs
) -> torch.Tensor:
  """``orientation_alignment`` scaled by the goal's w_rot."""
  cmd = env.command_manager.get_term(command_name)
  return cmd.pose_weights[:, 1] * orientation_alignment(env, command_name, **kwargs)


def grasp_position_alignment(
  env, command_name: str = "reorient_command", **kwargs
) -> torch.Tensor:
  """``position_alignment`` scaled by the goal's w_pos."""
  cmd = env.command_manager.get_term(command_name)
  return cmd.pose_weights[:, 0] * position_alignment(env, command_name, **kwargs)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  return (values * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)


def grasp_tip_proximity(
  env,
  command_name: str = "reorient_command",
  scale: float = 0.02,
  angle_power: float = 2.0,
  ori_bound: float = 0.2,
  ori_margin: float = 0.3,
  pos_bound: float = 0.01,
  pos_margin: float = 0.02,
  gate_floor: float = 0.0,
) -> torch.Tensor:
  """w_tip-weighted mean over fingers of exp(-d / scale) (1 - theta/pi)^p, d
  the tip's distance from its contact point and theta its angle off the
  normal, scaled by how close the object is to its goal pose (see
  ``fingertip_target_proximity``). Zero when every w_tip is 0."""
  cmd = env.command_manager.get_term(command_name)
  d = cmd.tip_offset_b().norm(dim=-1)
  facing = (1.0 - torch.acos(cmd.tip_cos().clamp(-1.0, 1.0)) / math.pi).pow(angle_power)
  per_finger = torch.exp(-d / scale) * facing
  gate = _pose_gate(cmd, ori_bound, ori_margin, pos_bound, pos_margin)
  gate = gate_floor + (1.0 - gate_floor) * gate
  return gate * _weighted_mean(per_finger, cmd.tip_weights)


def grasp_joint_proximity(
  env,
  command_name: str = "reorient_command",
  scale: float = 0.2,
  ori_bound: float = 0.2,
  ori_margin: float = 0.3,
  pos_bound: float = 0.01,
  pos_margin: float = 0.02,
  gate_floor: float = 0.0,
) -> torch.Tensor:
  """w_joint-weighted mean over fingers of exp(-e / scale), e the finger's
  largest joint error [rad], with the same goal-pose gate. Zero when every
  w_joint is 0."""
  cmd = env.command_manager.get_term(command_name)
  per_finger = torch.exp(-cmd.joint_error() / scale)
  gate = _pose_gate(cmd, ori_bound, ori_margin, pos_bound, pos_margin)
  gate = gate_floor + (1.0 - gate_floor) * gate
  return gate * _weighted_mean(per_finger, cmd.joint_weights)


# -- terminations ------------------------------------------------------------
def object_speed_limit(
  env,
  max_lin_vel: float = 3.0,
  max_ang_vel: float = 20.0,
  max_ang_vel_in_cage: float = 50.0,
  asset_cfg: SceneEntityCfg = _OBJECT,
) -> torch.Tensor:
  """End the episode once the object moves implausibly fast [m/s, rad/s].

  mujoco-warp's implicitfast integrates a free body's gyroscopic term
  explicitly, so an object leaving the hand spinning at a few tens of rad/s
  spins up by itself and is flung away. The angular limit is ``max_ang_vel``
  outside the cage (dropping) and ``max_ang_vel_in_cage`` in the hand.
  """
  data = env.scene[asset_cfg.name].data
  ang = data.root_link_ang_vel_w.norm(dim=-1)
  counter = read_cage_penalty_counter(env)
  if counter is None:
    limit = torch.full_like(ang, max_ang_vel)
  else:
    limit = torch.where(counter > 0, max_ang_vel, max_ang_vel_in_cage)
  return (data.root_link_lin_vel_w.norm(dim=-1) > max_lin_vel) | (ang > limit)
