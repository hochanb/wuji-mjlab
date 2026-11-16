# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Grasp templates: a hand shape and the hand surface points that touch.

As in Dexonomy, a template is the finger joint angles plus its contacts: per
hand link, a point on the link surface and the outward surface normal, in
the link frame. Templates are made from validated grasps
(``template_from_grasp``, Dexonomy's template update: the contacts that
actually touch the object, averaged per link). ``templates.yaml`` holds the
library, a list of templates per grasp type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

TEMPLATES_PATH = Path(__file__).with_name("templates.yaml")


@dataclass(frozen=True)
class HandContact:
  body: str
  point: tuple[float, float, float]  # on the link surface, link frame [m]
  normal: tuple[float, float, float]  # outward, link frame

  @property
  def finger(self) -> int:
    """1 (thumb) .. 5 (pinky); 0 for the palm."""
    name = self.body.removeprefix("right_finger")
    return int(name[0]) if name != self.body else 0


@dataclass(frozen=True)
class GraspTemplate:
  name: str
  joint_pos: tuple[float, ...]  # 20, finger1 joint1 .. finger5 joint4
  contacts: tuple[HandContact, ...]

  @property
  def fingers(self) -> tuple[int, ...]:
    """Fingers with a contact (the palm excluded)."""
    return tuple(sorted({c.finger for c in self.contacts} - {0}))


def template_from_grasp(
  model: mujoco.MjModel,
  name: str,
  joint_pos,
  object_pose,
  touch_dist: float = 0.002,
) -> GraspTemplate:
  """Contacts of a grasp: hand-object contacts within ``touch_dist`` with the
  hand at ``joint_pos`` and the object (body ``object``) at ``object_pose``
  in the wrist frame, averaged per hand link."""
  data = mujoco.MjData(model)
  margin = model.geom_margin.copy()
  gap = model.geom_gap.copy()
  model.geom_margin[:] = touch_dist
  model.geom_gap[:] = touch_dist
  try:
    adr = model.joint("object_free").qposadr[0]
    data.qpos[:20] = joint_pos
    data.qpos[adr : adr + 7] = object_pose
    mujoco.mj_forward(model, data)
  finally:
    model.geom_margin[:] = margin
    model.geom_gap[:] = gap
  obj = model.body("object").id
  per_body: dict[int, list] = {}
  for c in data.contact[: data.ncon]:
    b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
    if (b1 == obj) == (b2 == obj) or c.dist >= touch_dist:
      continue
    hand, n = (b2, -c.frame[:3]) if b1 == obj else (b1, c.frame[:3].copy())
    # contact.pos is midway between the surfaces; n points hand -> object.
    per_body.setdefault(hand, []).append((c.pos - 0.5 * c.dist * n, n))
  contacts = []
  for b, items in sorted(per_body.items()):
    p = np.mean([i[0] for i in items], 0)
    n = np.mean([i[1] for i in items], 0)
    n /= np.linalg.norm(n)
    rot = data.xmat[b].reshape(3, 3)
    contacts.append(
      HandContact(
        model.body(b).name,
        tuple(np.round(rot.T @ (p - data.xpos[b]), 5).tolist()),
        tuple(np.round(rot.T @ n, 4).tolist()),
      )
    )
  q = tuple(np.round(np.asarray(joint_pos, float), 4).tolist())
  return GraspTemplate(name, q, tuple(contacts))


def save_templates(
  templates: dict[str, list[GraspTemplate]], path=TEMPLATES_PATH
) -> None:
  doc = {
    name: [
      {
        "joint_pos": list(t.joint_pos),
        "contacts": [
          {"body": c.body, "point": list(c.point), "normal": list(c.normal)}
          for c in t.contacts
        ],
      }
      for t in items
    ]
    for name, items in templates.items()
  }
  header = (
    "# Grasp templates per grasp type (grasp_synth.templates): finger joint\n"
    "# angles and, per touching hand link, a surface point and outward normal\n"
    "# in the link frame. Made from validated grasps of the 54 mm cube.\n"
  )
  Path(path).write_text(
    header + yaml.safe_dump(doc, sort_keys=False, width=100, default_flow_style=None)
  )


def load_templates(path=TEMPLATES_PATH) -> dict[str, list[GraspTemplate]]:
  doc = yaml.safe_load(Path(path).read_text())
  return {
    name: [
      GraspTemplate(
        name,
        tuple(t["joint_pos"]),
        tuple(
          HandContact(c["body"], tuple(c["point"]), tuple(c["normal"]))
          for c in t["contacts"]
        ),
      )
      for t in items
    ]
    for name, items in doc.items()
  }
