# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import pytest
import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  REORIENT_HAND2_MOUNT_POS,
)

_TASK = "WujiHand2_ReorientWrist_50Hz"
_SETTLE_STEPS = 60  # past the warmup and the servo transient


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(_TASK, play=True)
  cfg.scene.num_envs = 2
  cfg.terminations.pop("cage_drop")
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  yield env
  env.close()


def _palm_and_pivot(env):
  robot = env.scene["robot"]
  palm = robot.find_bodies(".*_palm_link")[0][0]
  pose = robot.data.body_link_pose_w[:, palm]
  pivot = torch.tensor(REORIENT_HAND2_MOUNT_POS).expand(env.num_envs, 3)
  return pose[:, 3:7].clone(), pose[:, :3] + quat_apply(pose[:, 3:7], pivot)


def _run(env, action, steps=_SETTLE_STEPS):
  for _ in range(steps):
    env.step(action)


def test_wrist_has_six_actuated_joints(env):
  assert env.action_manager.total_action_dim == 26
  assert env.observation_manager.group_obs_dim["policy"] == (243,)
  robot = env.scene["robot"]
  assert robot.joint_names[:6] == (
    "right_wrist_tx",
    "right_wrist_ty",
    "right_wrist_tz",
    "right_wrist_rx",
    "right_wrist_ry",
    "right_wrist_rz",
  )
  finger_ids = env.action_manager.get_term("finger_joint_pos")._target_ids.tolist()
  assert finger_ids == list(range(6, 26))


def test_wrist_rotates_about_pivot_within_limit(env):
  env.reset()
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  _run(env, zero)
  quat0, pivot0 = _palm_and_pivot(env)
  action = zero.clone()
  action[0, 20] = 5.0  # out of range: clipped to +1 -> +15 deg
  action[1, 21] = -1.0
  _run(env, action)
  quat, pivot = _palm_and_pivot(env)
  rot = torch.rad2deg(quat_error_magnitude(quat, quat0))
  torch.testing.assert_close(rot, torch.full((2,), 15.0), atol=1.0, rtol=0)
  assert torch.all((pivot - pivot0).norm(dim=-1) < 3e-3)  # servo sag only


def test_wrist_translates_within_limit(env):
  env.reset()
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  _run(env, zero)
  quat0, pivot0 = _palm_and_pivot(env)
  action = zero.clone()
  action[0, 23] = 3.0  # clipped to +5 cm along the mount x axis
  action[1, 25] = -1.0  # -5 cm along the mount z axis
  _run(env, action)
  quat, pivot = _palm_and_pivot(env)
  torch.testing.assert_close(
    (pivot - pivot0).norm(dim=-1), torch.full((2,), 0.05), atol=3e-3, rtol=0
  )
  assert torch.all(torch.rad2deg(quat_error_magnitude(quat, quat0)) < 1.0)


def test_wrist_rate_limited(env):
  env.reset()
  wrist = env.action_manager.get_term("wrist_pose")
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  _run(env, zero)
  action = zero.clone()
  action[:, 20] = 1.0
  action[:, 23] = 1.0
  env.step(action)
  # Rotation and translation move the same fraction of their limit per step.
  first = wrist._applied[:, [0, 3]] / wrist._limit[[0, 3]]
  expected = wrist.cfg.max_rate_normalized * env.step_dt
  torch.testing.assert_close(first, torch.full_like(first, expected), atol=1e-6, rtol=0)


def test_wrist_hf_penalty_reads_ema_before_rate_limit(env):
  env.reset()
  wrist = env.action_manager.get_term("wrist_pose")
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  _run(env, zero)
  action = zero.clone()
  action[:, 20] = 1.0
  env.step(action)
  # A full-range step: the EMA moves alpha of the way, the rate limit far less.
  smoothed = wrist.smoothed_action[:, 0]
  limited = wrist.processed_action[:, 0]
  torch.testing.assert_close(
    smoothed, torch.full_like(smoothed, wrist.cfg.ema_alpha), atol=1e-6, rtol=0
  )
  assert torch.all(limited < smoothed)
  params = env.reward_manager.get_term_cfg("wrist_action_hf").params
  assert params["signal"] == "smoothed_action"


def test_wrist_returns_to_mount_pose_on_reset(env):
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  env.reset()
  _run(env, zero, 5)
  quat_mount, _ = _palm_and_pivot(env)
  action = zero.clone()
  action[:, 20:26] = 1.0
  _run(env, action, 30)
  env.reset()
  _run(env, zero, 5)
  quat, _ = _palm_and_pivot(env)
  assert torch.all(torch.rad2deg(quat_error_magnitude(quat, quat_mount)) < 0.5)


def test_goal_is_wrist_local(env):
  """Goal pose rides with the wrist; errors are the object's wrist-local ones."""
  env.reset()
  cmd = env.command_manager.get_term("reorient_command")
  robot, obj = env.scene["robot"], env.scene["object"]
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
  _run(env, zero, 25)
  goal_tag = cmd.goal_quat_tag.clone()
  action = zero.clone()
  action[:, 20:26] = torch.tensor([1.0, -1.0, 0.5, 1.0, -1.0, 1.0])
  _run(env, action, 20)

  tag = robot.data.site_pose_w[:, cmd.tag_site_id]
  torch.testing.assert_close(cmd.goal_quat_tag, goal_tag)  # no goal switch here
  torch.testing.assert_close(cmd.goal_quat_w, quat_mul(tag[:, 3:7], goal_tag))
  cube_in_tag = quat_mul(quat_inv(tag[:, 3:7]), obj.data.root_link_quat_w)
  torch.testing.assert_close(
    quat_error_magnitude(obj.data.root_link_quat_w, cmd.goal_quat_w),
    quat_error_magnitude(cube_in_tag, goal_tag),
    atol=1e-5,
    rtol=0,
  )
