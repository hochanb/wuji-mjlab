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
    body = model.body(r.body_id).name
    if c.finger == 0:
      assert body == "right_palm_link"
    else:
      assert body.startswith(f"right_finger{c.finger}_link")


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


def test_rotation_helpers():
  from wuji_mjlab.grasp_synth.dexonomy import (
    _axis_angle,
    _mat_to_quat,
    _rotation_between,
  )
  from wuji_mjlab.grasp_synth.synth import _quat_to_mat

  a = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
  b = torch.nn.functional.normalize(torch.randn(64, 3), dim=-1)
  b[0] = -a[0]  # opposite vectors
  r = _rotation_between(a, b)
  assert torch.allclose(torch.einsum("nij,nj->ni", r, a), b, atol=1e-5)
  r = _axis_angle(torch.randn(64, 3))
  assert torch.allclose(r @ r.transpose(1, 2), torch.eye(3).expand(64, 3, 3), atol=1e-5)
  assert torch.allclose(_quat_to_mat(_mat_to_quat(r)), r, atol=1e-5)


def test_template_library_loads():
  from wuji_mjlab.grasp_synth import load_templates

  lib = load_templates()
  for name, need in (("pinch", 2), ("tripod", 3), ("five_tip", 5), ("power", 5)):
    assert lib[name], name
    for t in lib[name]:
      assert len(t.joint_pos) == 20 and len(t.fingers) >= min(need, 2)
      for c in t.contacts:
        assert np.linalg.norm(c.normal) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_dexonomy_tripod_on_cube():
  """Object poses come out in the wrist frame: placing the cube there with
  the returned joints touches it with the tripod fingers, no penetration."""
  if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable")
  from wuji_mjlab.grasp_synth import DexonomyCfg, DexonomySynthesizer

  obj = GraspObject.box((0.027, 0.027, 0.027), mass=0.12)
  cfg = DexonomyCfg(align_samples=2048, grow_templates=True)
  dex = DexonomySynthesizer(cfg)
  before = len(dex.library["tripod"])
  r = dex.generate(obj, "tripod", 2)
  # Validated grasps may become templates (in memory only).
  assert len(dex.library["tripod"]) == before + r.stats["new_templates"]
  assert len(r.joint_pos) >= 1 and r.object_pose.shape == (len(r.joint_pos), 7)
  assert r.attempts <= 2 * cfg.synth.attempts_per_sample
  np.testing.assert_allclose(np.linalg.norm(r.object_pose[:, 3:], axis=1), 1, atol=1e-4)

  syn = GraspSynthesizer(cfg.synth)
  m = syn._build_model(obj)
  data = mujoco.MjData(m)
  adr = m.joint("object_free").qposadr[0]
  obj_body = m.body("object").id
  m.geom_margin[:] = cfg.synth.touch_dist
  for q, pose in zip(r.joint_pos, r.object_pose, strict=True):
    data.qpos[:20], data.qpos[adr : adr + 7] = q, pose
    mujoco.mj_forward(m, data)
    touching = set()
    for c in data.contact[: data.ncon]:
      assert c.dist >= -cfg.synth.max_penetration - 1e-4
      bodies = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
      if obj_body in bodies and c.dist < cfg.synth.touch_dist:
        name = m.body((bodies - {obj_body}).pop()).name
        if "finger" in name:
          touching.add(int(name[len("right_finger")]))
    assert {1, 2, 3} <= touching


def test_template_library_save_load_and_default_protected(tmp_path):
  from wuji_mjlab.grasp_synth import DexonomySynthesizer, load_templates
  from wuji_mjlab.grasp_synth.templates import TEMPLATES_PATH

  dex = DexonomySynthesizer()
  path = tmp_path / "lib.yaml"
  dex.save_templates(path)
  assert load_templates(path) == dex.library
  with pytest.raises(ValueError, match="default library"):
    dex.save_templates(TEMPLATES_PATH)
