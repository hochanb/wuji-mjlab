# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.

from __future__ import annotations

import mujoco
import numpy as np
from wuji_mjlab.assets.objects.inhand_object.compound_objects import compound_spec


def _collision_geoms(model):
  return [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]


def _model():
  return compound_spec("smartphone").compile()


def test_smartphone_collides_as_boxes_only():
  model = _model()
  geoms = _collision_geoms(model)
  assert len(geoms) == 1 + 12 + 3  # body, 3 x 4 keys, 2 volume + 1 power
  assert all(model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX for i in geoms)
  assert model.nmesh == 0


def test_smartphone_keys_sit_on_the_front_in_a_3x4_grid():
  model = _model()
  body, *rest = _collision_geoms(model)
  keys = rest[:12]
  top = model.geom_pos[body][2] + model.geom_size[body][2]
  pos = np.array([model.geom_pos[i] for i in keys])
  half = np.array([model.geom_size[i] for i in keys])
  np.testing.assert_allclose(pos[:, 2] - half[:, 2], top, atol=1e-9)
  assert len(np.unique(pos[:, 0].round(6))) == 4  # rows along the long axis
  assert len(np.unique(pos[:, 1].round(6))) == 3
  # Keys stay inside the front face and do not overlap each other.
  assert np.all(np.abs(pos[:, :2]) + half[:, :2] <= model.geom_size[body][:2])
  gaps = np.diff(np.unique(pos[:, 0].round(6)))
  assert np.all(gaps > 2 * half[0, 0])


def test_smartphone_side_buttons():
  model = _model()
  body, *rest = _collision_geoms(model)
  side = rest[12:]
  center, size = model.geom_pos[body], model.geom_size[body]
  pos = np.array([model.geom_pos[i] for i in side])
  half = np.array([model.geom_size[i] for i in side])
  # Flush against a long side face, within the thickness.
  np.testing.assert_allclose(np.abs(pos[:, 1] - center[1]) - half[:, 1], size[1], atol=1e-9)
  assert np.all(np.abs(pos[:, 2] - center[2]) + half[:, 2] < size[2])
  up, down, power = pos - center
  assert up[1] > 0 and down[1] > 0 and power[1] < 0  # volume left (+y), power right
  assert up[0] > down[0] > size[0] / 3  # volume in the upper part
  assert 0.0 < power[0] < size[0] / 2  # power just above the middle
