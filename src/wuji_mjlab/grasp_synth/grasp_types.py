# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Grasp types for Wuji Hand 2: which hand surface points touch the object.

A grasp type is a set of hand contacts, the counterpart of a Dexonomy
template's annotated contacts. Each contact is a reference point fixed in a
hand link, ``depth`` inside the link surface, with the outward surface normal;
when the surface touches the object the reference point sits ``depth`` off the
object surface along its normal. Fingers with a contact are driven onto the
object; the others keep their initial joint angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  HAND2_TIP_PAD_POINTS,
)

# Depth of HAND2_TIP_PAD_POINTS inside the pad; the same depth is used for the
# phalanx points.
PAD_DEPTH = 0.005
# Palmar side of every finger link (and tip site) frame.
_PALMAR = np.array([0.0, 1.0, 0.0])

Link = Literal["tip", "middle", "proximal"]


@dataclass(frozen=True)
class ContactSpec:
  """A hand contact: ``finger`` 1 (thumb) .. 5 (pinky) and the link touching.

  ``tip`` is the fingertip pad (``HAND2_TIP_PAD_POINTS``); ``middle`` /
  ``proximal`` are the palmar side of the middle / proximal phalanx (link3 /
  link2), at the middle of its collision capsule.
  """

  finger: int
  link: Link


@dataclass(frozen=True)
class GraspType:
  name: str
  contacts: tuple[ContactSpec, ...]
  description: str = ""

  @property
  def fingers(self) -> tuple[int, ...]:
    return tuple(sorted({c.finger for c in self.contacts}))


def _tips(*fingers: int) -> tuple[ContactSpec, ...]:
  return tuple(ContactSpec(f, "tip") for f in fingers)


GRASP_TYPES: dict[str, GraspType] = {
  t.name: t
  for t in (
    GraspType("pinch", _tips(1, 2), "thumb and index fingertips"),
    GraspType("tripod", _tips(1, 2, 3), "thumb, index and middle fingertips"),
    GraspType("five_tip", _tips(1, 2, 3, 4, 5), "all five fingertips"),
    GraspType(
      "power",
      _tips(1, 2, 3, 4, 5)
      + tuple(ContactSpec(f, "middle") for f in range(1, 6))
      + tuple(ContactSpec(f, "proximal") for f in range(2, 6)),
      "fingertips plus the middle and proximal phalanges (wrap)",
    ),
  )
}


@dataclass(frozen=True)
class ResolvedContact:
  body_id: int
  point: np.ndarray  # (3,) in the body frame, PAD_DEPTH inside the surface
  normal: np.ndarray  # (3,) outward, body frame
  finger: int


def resolve_contact(model: mujoco.MjModel, spec: ContactSpec) -> ResolvedContact:
  """Body and body-frame point / normal of a contact in a Wuji Hand 2 model."""
  if spec.link == "tip":
    site = model.site(f"right_finger{spec.finger}_tip")
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, site.quat)
    rot = rot.reshape(3, 3)
    point = site.pos + rot @ np.asarray(HAND2_TIP_PAD_POINTS[spec.finger - 1])
    return ResolvedContact(int(site.bodyid[0]), point, rot @ _PALMAR, spec.finger)
  link = {"middle": 3, "proximal": 2}[spec.link]
  body = model.body(f"right_finger{spec.finger}_link{link}")
  capsules = [
    g
    for g in range(model.ngeom)
    if model.geom_bodyid[g] == body.id
    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
    and (model.geom_contype[g] or model.geom_conaffinity[g])
  ]
  if len(capsules) != 1:
    raise ValueError(f"{body.name}: no collision capsule for a {spec.link} contact")
  g = capsules[0]
  axis = np.empty(9)
  mujoco.mju_quat2Mat(axis, model.geom_quat[g])
  if abs(axis.reshape(3, 3)[2, 2]) < 0.99:  # capsule axis must be the link z
    raise ValueError(
      f"{body.name}: capsule is not along the link, no {spec.link} contact"
    )
  radius = float(model.geom_size[g, 0])
  point = model.geom_pos[g] + (radius - PAD_DEPTH) * _PALMAR
  return ResolvedContact(body.id, point, _PALMAR.copy(), spec.finger)
