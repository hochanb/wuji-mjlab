# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from functools import partial

import mujoco
import numpy as np
import pytest
import torch
from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import (
  _get_spec,
  wuji_hand2_xml_path,
)
from wuji_mjlab.grasp_synth import (
  GRASP_TYPES,
  GraspObject,
  GraspSynthCfg,
  GraspSynthesizer,
)
from wuji_mjlab.grasp_synth.grasp_types import resolve_contact
from wuji_mjlab.grasp_synth.wrench import balance_contact_forces
from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  get_hand2_reorient_spec,
)

# 54 mm cube in front of the palm (the reorient cube spawn, palm frame).
CUBE_POS = (0.0042, 0.0791, -0.0673)


def _balance(points, mask):
  p = torch.tensor([points], dtype=torch.float32)
  n = -torch.nn.functional.normalize(p, dim=-1)
  return balance_contact_forces(
    p, n, torch.tensor([mask]).bool(), torch.zeros(1, 3), 0.03, mu=0.5
  )


def test_balance_opposing_contacts_cancel():
  residual, forces = _balance([[0.03, 0, 0], [-0.03, 0, 0], [0, 0.03, 0]], [1, 1, 0])
  assert residual.item() < 1e-4
  assert forces[0, :, :].sum(0).norm() < 1e-4
  assert forces[0, 2].norm() == 0  # inactive contact carries nothing
  # Normal force: total 1, each active contact at least min_share / count.
  normal = (
    forces[0]
    * -torch.nn.functional.normalize(
      torch.tensor([[0.03, 0, 0], [-0.03, 0, 0], [0, 0.03, 0]]), dim=-1
    )
  ).sum(-1)
  assert normal.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_balance_one_sided_contacts_do_not():
  residual, _ = _balance([[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]], [1, 1, 1])
  assert residual.item() > 0.3
  residual, _ = _balance([[0.03, 0, 0], [-0.03, 0, 0]], [1, 0])
  assert residual.item() == 1.0  # fewer than two contacts


@pytest.mark.parametrize("name", sorted(GRASP_TYPES))
def test_grasp_type_contacts_resolve(name):
  spec = get_hand2_reorient_spec(partial(_get_spec, wuji_hand2_xml_path("right")))
  model = spec.compile()
  for c in GRASP_TYPES[name].contacts:
    r = resolve_contact(model, c)
    assert np.linalg.norm(r.normal) == pytest.approx(1.0)
    assert model.body(r.body_id).name.startswith(f"right_finger{c.finger}_link")


@pytest.mark.slow
def test_pinch_grasps_on_cube_are_valid():
  """Grasps found for a cube touch it with thumb and index, without
  penetration, inside the soft joint limits."""
  if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable")
  obj = GraspObject.box((0.027, 0.027, 0.027), mass=0.12)
  # No gravity: two-point grasps rarely hold the cube's weight (condim 3).
  cfg = GraspSynthCfg(
    num_worlds=1024, explore_worlds=256, attempts_per_sample=5, gravity=0.0
  )
  syn = GraspSynthesizer(cfg)
  pose = np.array([[*CUBE_POS, 1.0, 0.0, 0.0, 0.0]])
  r = syn.generate(obj, pose, "pinch", 2)
  assert r.attempts[0] <= 2 * cfg.attempts_per_sample
  (q,) = r.joint_pos
  assert q.shape[1] == 20 and len(q) >= 1
  assert r.joint_target[0].shape == q.shape

  m = syn.model
  lo, hi = m.jnt_range[:20].T
  mid, half = (lo + hi) / 2, (hi - lo) / 2 * cfg.soft_limit_factor
  data = mujoco.MjData(m)
  obj_adr = m.joint("object_free").qposadr[0]
  obj_body = m.body("object").id
  for row in q:
    assert ((row >= mid - half - 1e-5) & (row <= mid + half + 1e-5)).all()
    data.qpos[:20] = row
    data.qpos[obj_adr : obj_adr + 7] = pose[0]
    m.geom_margin[:] = cfg.touch_dist
    mujoco.mj_forward(m, data)
    touching = set()
    for c in data.contact[: data.ncon]:
      assert c.dist >= -cfg.max_penetration - 1e-4
      bodies = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
      if obj_body in bodies and c.dist < cfg.touch_dist:
        name = m.body((bodies - {obj_body}).pop()).name
        touching.add(int(name[len("right_finger")]) if "finger" in name else 0)
    assert {1, 2} <= touching
