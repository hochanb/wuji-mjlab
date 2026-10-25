# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from wuji_mjlab.assets.objects.inhand_object.compound_objects import (
  compound_spec,
  smartphone_fingertip_targets,
  smartphone_key,
  smartphone_parts,
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


def test_smartphone_targets_lie_on_the_surface_facing_out():
  model = compound_spec("smartphone").compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for point, direction in smartphone_fingertip_targets():
    np.testing.assert_allclose(
      _surface_distance(model, data, point, direction), 0.01, atol=1e-4
    )


def test_smartphone_targets_layout():
  (thumb, thumb_dir), *fingers = smartphone_fingertip_targets(thumb_key=(2, 3))
  assert thumb_dir == (0.0, 0.0, 1.0), "thumb presses a front key"
  assert all(d == (0.0, 1.0, 0.0) for _, d in fingers), "fingers on the left side"
  xs = [p[0] for p, _ in fingers]
  assert xs == sorted(xs, reverse=True), "index to pinky run top to bottom"
  # Row 2 / column 3 from the top left: second-highest row, rightmost (-y) column.
  keys = smartphone_parts()[1:13]
  key = smartphone_key(smartphone_parts(), 2, 3)
  rows = sorted({k.pos[0] for k in keys}, reverse=True)
  assert key.pos[0] == rows[1]
  assert key.pos[1] == min(k.pos[1] for k in keys)


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


def test_finger_weights_scale_each_finger():
  tip = [0.0, 0.0, 0.005]
  assert _reward(tip, finger_weights=(0.2,)) == pytest.approx(0.2 * _reward(tip))
  assert _reward(tip, finger_weights=(0.0,)) == 0.0
