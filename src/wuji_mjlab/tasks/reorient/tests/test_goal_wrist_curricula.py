# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude
from wuji_mjlab.tasks.reorient.mdp.commands import random_quat_within_angle
from wuji_mjlab.tasks.reorient.mdp.curriculums import (
  goal_rotation_curriculum,
  wrist_range_curriculum,
)

_N = 100_000


def _angles(quat):
  return quat_error_magnitude(
    quat, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat), 4)
  )


def test_bounded_goal_rotation_respects_limit():
  for deg in (10.0, 30.0, 90.0):
    assert _angles(random_quat_within_angle(_N, math.radians(deg), "cpu")).max() <= (
      math.radians(deg) + 1e-5
    )


def test_bounded_goal_rotation_is_uniform_so3_at_pi():
  # Uniform SO(3): E[angle] = pi/2 + 2/pi.
  mean = _angles(random_quat_within_angle(_N, math.pi, "cpu")).mean().item()
  assert abs(mean - (math.pi / 2 + 2 / math.pi)) < 0.01


class _Env(SimpleNamespace):
  pass


def _env(reach: float, progress: float | None, single_success: bool = False):
  command = SimpleNamespace(
    metrics={"goal_reach_count": torch.full((8,), reach)},
    cfg=SimpleNamespace(single_success=single_success),
  )
  wrist = SimpleNamespace(range_scale=1.0)
  env = _Env(
    num_envs=8,
    device="cpu",
    command_manager=SimpleNamespace(get_term=lambda _: command),
    action_manager=SimpleNamespace(get_term=lambda _: wrist),
  )
  if progress is not None:
    env.max_common_steps = 1000
    env.common_step_counter = int(progress * 1000)
  return env, command, wrist


def test_curricula_start_easy_and_follow_time_floor():
  env, command, wrist = _env(reach=0.0, progress=0.0)
  ids = torch.arange(8)
  wrist_range_curriculum(env, ids, min_scale=0.1)
  goal_rotation_curriculum(env, ids, min_angle=math.radians(30.0))
  assert wrist.range_scale == 0.1
  assert math.isclose(command.goal_max_angle, math.radians(30.0))
  env.common_step_counter = 1000  # end of training: floors reach full range
  wrist_range_curriculum(env, ids)
  goal_rotation_curriculum(env, ids)
  assert wrist.range_scale == 1.0
  assert math.isclose(command.goal_max_angle, math.pi)


def test_curricula_promote_on_success():
  env, command, wrist = _env(reach=5.0, progress=None)
  ids = torch.arange(8)
  for _ in range(10):
    info = goal_rotation_curriculum(env, ids, step=0.05)
  assert math.isclose(info["value"], 0.5)
  assert command.goal_max_angle > math.radians(30.0)


def test_single_success_promotes_on_one_reach():
  ids = torch.arange(8)
  env, _, _ = _env(reach=1.0, progress=None)
  assert goal_rotation_curriculum(env, ids, step=0.05)["value"] == 0.0
  env, _, _ = _env(reach=1.0, progress=None, single_success=True)
  assert math.isclose(goal_rotation_curriculum(env, ids, step=0.05)["value"], 0.05)
