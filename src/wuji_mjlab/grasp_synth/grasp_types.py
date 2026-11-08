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

Link = Literal["tip", "middle", "proximal", "palm"]
# Palm contact: where a power grasp presses the object, palm frame (x, z) [m];
# the point is found on the palm surface along the palm normal (+y).
PALM_CONTACT_XZ = (-0.005, -0.045)


@dataclass(frozen=True)
class ContactSpec:
  """A hand contact: ``finger`` 1 (thumb) .. 5 (pinky) and the link touching.

  ``tip`` is the fingertip pad (``HAND2_TIP_PAD_POINTS``); ``middle`` /
  ``proximal`` are the palmar side of the middle / proximal phalanx (link3 /
  link2), at the middle of its collision capsule; ``palm`` (finger 0) is the
  palm surface at ``PALM_CONTACT_XZ``. The palm does not move: a palm contact
  is a requirement (the object must touch it), not something refined.
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
    GraspType(
      "power_palm",
      _tips(1, 2, 3, 4, 5)
      + tuple(ContactSpec(f, "middle") for f in range(1, 6))
      + tuple(ContactSpec(f, "proximal") for f in range(2, 6))
      + (ContactSpec(0, "palm"),),
      "power wrap with the object against the palm",
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
  if spec.link == "palm":
    return _palm_contact(model)
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


def _palm_contact(model: mujoco.MjModel) -> ResolvedContact:
  """The palm surface at PALM_CONTACT_XZ: a ray along -y onto the palm's
  collision mesh, at the zero joint pose."""
  palm = model.body("right_palm_link").id
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  rot, pos = data.xmat[palm].reshape(3, 3), data.xpos[palm]
  x, z = PALM_CONTACT_XZ
  start = pos + rot @ np.array([x, 0.1, z])
  geom = model.geom("right_palm_collision").id
  dist = mujoco.mj_rayMesh(model, data, geom, start, rot @ -_PALMAR)
  if dist < 0:
    raise ValueError("palm surface not found at PALM_CONTACT_XZ")
  point = np.array([x, 0.1 - dist, z]) - PAD_DEPTH * _PALMAR
  return ResolvedContact(palm, point, _PALMAR.copy(), 0)
