# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
import torch
from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
  compound_spec,
  stapler_fingertip_targets,
)


def _surface_distance(model, data, point, direction):
  """Signed distance from just outside ``point`` back to the stapler surface."""
  geomid = np.zeros(1, dtype=np.int32)
  start = np.asarray(point) + 0.01 * np.asarray(direction)
  return mujoco.mj_ray(model, data, start, -np.asarray(direction), None, 1, -1, geomid)


def test_stapler_targets_lie_on_the_surface_facing_out():
  model = compound_spec("stapler").compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  targets = stapler_fingertip_targets()
  assert len(targets) == 5
  for point, direction in targets:
    # A ray cast back along -direction from 1 cm out hits the surface at the point.
    np.testing.assert_allclose(
      _surface_distance(model, data, point, direction), 0.01, atol=1e-4
    )


def test_stapler_targets_layout():
  (thumb, _), *fingers = stapler_fingertip_targets()
  xs = [p[0] for p, _ in fingers]
  assert xs == sorted(xs, reverse=True), "index to pinky must run front to back"
  assert thumb[2] > max(p[2] for p, _ in fingers), "thumb on top, fingers below"
  assert thumb[0] > 0.0, "thumb at the front (+x, away from the hinge)"


def _reward(tip, ori_err=0.0, pos_err=0.0, **kwargs):
  from types import SimpleNamespace

  import torch
  from mjlab.utils.lab_api.math import quat_from_euler_xyz
  from wuji_mjlab.tasks.reorient.mdp.rewards import fingertip_target_proximity

  point = torch.zeros(1, 1, 3)
  direction = torch.tensor([[[0.0, 0.0, 1.0]]])
  tip = torch.tensor([[tip]])
  zero = torch.zeros(1)
  command = SimpleNamespace(
    fingertip_pos_w=tip,
    fingertip_target_pos_w=point,
    fingertip_target_dot=((tip - point) * direction).sum(-1),
    object=SimpleNamespace(
      data=SimpleNamespace(root_link_quat_w=quat_from_euler_xyz(zero, zero, zero))
    ),
    goal_quat=quat_from_euler_xyz(zero, zero, torch.full((1,), ori_err)),
    cfg=SimpleNamespace(goal_pos_range=((0.0, 0.0),) * 3),
    pos_error=lambda: torch.full((1,), pos_err),
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
  return float(fingertip_target_proximity(env, scale=0.02, pad_offset=0.0, **kwargs))


def test_proximity_favours_the_outer_side():
  front = _reward([0.0, 0.0, 0.005])
  behind = _reward([0.0, 0.0, -0.005])  # as close, but right through the wall
  beside = _reward([0.005, 0.0, 0.001])  # as close, but off to the side
  oblique_behind = _reward([0.004, 0.0, -0.003])
  far = _reward([0.0, 0.0, 0.02])
  assert behind < 1e-9
  assert front > beside > oblique_behind > behind  # rises as the tip wraps round
  assert front > far > 0.0


def test_proximity_is_gated_by_the_goal_error():
  tip = [0.0, 0.0, 0.005]
  at_goal = _reward(tip)
  assert _reward(tip, ori_err=0.15, pos_err=0.008) == at_goal  # within bounds
  assert 0.0 < _reward(tip, ori_err=0.35) < at_goal  # inside the margin
  assert _reward(tip, ori_err=1.5) < 1e-3 * at_goal  # mid-rotation: off
  assert _reward(tip, pos_err=0.05) < 1e-3 * at_goal


def test_gate_floor_keeps_a_fraction_far_from_the_goal():
  tip = [0.0, 0.0, 0.005]
  at_goal = _reward(tip, gate_floor=0.2)
  assert at_goal == pytest.approx(_reward(tip))
  assert _reward(tip, ori_err=2.5, gate_floor=0.2) == pytest.approx(0.2 * at_goal, rel=1e-3)


def test_finger_weights_select_fingers():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.rewards import fingertip_target_proximity

  # Five tips at the target point offset along its outward normal: thumb and
  # index touch (d = pad), the rest are 3 cm away.
  d = torch.tensor([[0.005, 0.005, 0.035, 0.035, 0.035]])
  target = torch.zeros(1, 5, 3)
  tips = torch.zeros(1, 5, 3)
  tips[..., 2] = d
  command = SimpleNamespace(
    fingertip_pos_w=tips,
    fingertip_target_pos_w=target,
    fingertip_target_dot=d,
    object=SimpleNamespace(
      data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]))
    ),
    goal_quat=torch.tensor([[1.0, 0, 0, 0]]),
    cfg=SimpleNamespace(goal_pos_range=None),
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
  thumb_only = fingertip_target_proximity(env, finger_weights=(1, 0, 0, 0, 0))
  torch.testing.assert_close(thumb_only, torch.tensor([1.0]))
  far_only = fingertip_target_proximity(env, finger_weights=(0, 0, 1, 1, 1))
  torch.testing.assert_close(far_only, torch.tensor([math.exp(-0.03 / 0.02)]))
  with pytest.raises(ValueError):
    fingertip_target_proximity(env, finger_weights=(1, 0, 0))


def _gate(tips, weights):
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.commands import InHandReorientCommand

  # Targets at the origin facing +z; tip reference points 5 mm inside the pad.
  tips = torch.tensor([tips])
  k = tips.shape[1]
  command = SimpleNamespace(
    fingertip_pos_w=tips,
    fingertip_target_pos_w=torch.zeros(1, k, 3),
    fingertip_target_dir_w=torch.tensor([0.0, 0.0, 1.0]).expand(1, k, 3),
    cfg=SimpleNamespace(
      tip_pad_depth=0.005,
      success_fingertip_dist=0.01,
      success_fingertip_angle=math.radians(60.0),
    ),
    _success_fingers=torch.tensor([w > 0.0 for w in weights]),
  )
  command.fingertip_on_target = lambda: InHandReorientCommand.fingertip_on_target(
    command
  )
  on = InHandReorientCommand.fingertip_on_target(command)[0].tolist()
  return on, bool(InHandReorientCommand.fingertips_in_place(command)[0])


def test_fingertip_success_gate():
  touching = [0.0, 0.0, 0.005]  # pad on the target
  near = [0.0, 0.0, 0.014]  # pad 9 mm off
  far = [0.0, 0.0, 0.016]  # pad 11 mm off
  behind = [0.0, 0.0, -0.005]  # as close, but through the wall
  side = [0.01, 0.0, 0.002]  # pad within 1 cm but 79 deg off the normal
  on, ok = _gate([touching, near, far, behind, side], (1, 1, 1, 1, 1))
  assert on == [True, True, False, False, False]
  assert not ok
  # Only the fingers weighted > 0 are required.
  assert _gate([touching, near, far, behind, side], (1, 0.3, 0, 0, 0))[1]
  assert not _gate([touching, near, far, behind, side], (1, 0, 0.5, 0, 0))[1]
  assert _gate([far, far, far, far, far], (0, 0, 0, 0, 0))[1]
