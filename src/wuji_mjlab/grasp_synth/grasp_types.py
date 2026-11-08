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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import yaml

from wuji_mjlab.tasks.reorient.config.wuji_hand2.hand2_constants import (
  HAND2_TIP_PAD_POINTS,
)

# Depth of HAND2_TIP_PAD_POINTS inside the pad; the same depth is used for the
# phalanx points.
PAD_DEPTH = 0.005
# Palmar side of every finger link (and tip site) frame.
_PALMAR = np.array([0.0, 1.0, 0.0])

Link = Literal["tip", "middle", "proximal", "palm"]
Side = Literal["palmar", "radial", "ulnar", "dorsal"]
# Outward directions of the link faces in the finger link frames (+y palmar,
# +x towards the thumb, the finger along -z, as the palm frame at zero pose).
_SIDES = {
  "palmar": np.array([0.0, 1.0, 0.0]),
  "radial": np.array([1.0, 0.0, 0.0]),
  "ulnar": np.array([-1.0, 0.0, 0.0]),
  "dorsal": np.array([0.0, -1.0, 0.0]),
}
# Palm contact: where a power grasp presses the object, palm frame (x, z) [m];
# the point is found on the palm surface along the palm normal (+y).
PALM_CONTACT_XZ = (-0.005, -0.045)
# Thumb joints (joint1..4) of the adducted thumb: alongside the index's
# radial side in the palm plane, ~1 mm off the index / palm. Grasp types
# without thumb contacts keep the thumb where the initial joints put it.
ADDUCTED_THUMB_JOINT_POS = (-0.03, 0.41, 0.6, 0.0)


@dataclass(frozen=True)
class ContactSpec:
  """A hand contact: ``finger`` 1 (thumb) .. 5 (pinky) and the link touching.

  ``tip`` is the fingertip pad (``HAND2_TIP_PAD_POINTS``); ``middle`` /
  ``proximal`` are the palmar side of the middle / proximal phalanx (link3 /
  link2), at the middle of its collision capsule; ``palm`` (finger 0) is the
  palm surface at ``PALM_CONTACT_XZ``. The palm does not move: a palm contact
  is a requirement (the object must touch it), not something refined.
  ``side``: the link face touching, ``palmar`` (default), ``radial`` (thumb
  side), ``ulnar`` (pinky side) or ``dorsal``; the others are found by a ray
  onto the link's collision geometry at the same place along the link.
  """

  finger: int
  link: Link
  side: Side = "palmar"


@dataclass(frozen=True)
class GraspType:
  name: str
  contacts: tuple[ContactSpec, ...]
  description: str = ""
  joint_bounds: tuple[tuple[str, float, float], ...] = ()
  """(joint regex, lo, hi) [rad]: the type's posture (e.g. straight fingers
  for parallel extension, no hyperextension for adduction grip). Refinement
  keeps the joints inside, and grasps outside fail."""

  def bounds(self, joint_names, lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """``lo`` / ``hi`` (per joint) narrowed by ``joint_bounds``."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    for pattern, b_lo, b_hi in self.joint_bounds:
      for j, name in enumerate(joint_names):
        if re.fullmatch(pattern, name):
          lo[j], hi[j] = max(lo[j], b_lo), min(hi[j], b_hi)
    return lo, np.maximum(hi, lo)

  @property
  def fingers(self) -> tuple[int, ...]:
    return tuple(sorted({c.finger for c in self.contacts}))


def _adducted_thumb_bounds() -> tuple[tuple[str, float, float], ...]:
  return tuple(
    (f"right_finger1_joint{j + 1}", q - 0.15, q + 0.15)
    for j, q in enumerate(ADDUCTED_THUMB_JOINT_POS)
  )


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
    GraspType(
      "adducted_thumb",
      tuple(ContactSpec(f, "tip") for f in range(2, 6))
      + tuple(ContactSpec(f, "middle") for f in range(2, 6))
      + tuple(ContactSpec(f, "proximal") for f in range(2, 6))
      + (ContactSpec(0, "palm"),),
      "power wrap by the fingers against the palm, the thumb adducted beside the"
      " index (initial thumb joints ADDUCTED_THUMB_JOINT_POS)",
      _adducted_thumb_bounds(),
    ),
  )
}


def _load_taxonomy() -> dict[str, tuple[GraspType, tuple[float, ...]]]:
  path = Path(__file__).with_name("taxonomy.yaml")
  doc = yaml.safe_load(path.read_text())
  return {
    name: (
      GraspType(
        name,
        tuple(ContactSpec(f, link, side) for f, link, side in entry["contacts"]),
        "GRASP taxonomy (Dexonomy annotation)",
        tuple(
          (k, float(lo), float(hi))
          for k, (lo, hi) in entry.get("joint_bounds", {}).items()
        ),
      ),
      tuple(entry["joint_pos"]),
    )
    for name, entry in doc.items()
  }


# GRASP taxonomy types ("1_large_diameter" .. "33_inferior_pincer") with
# their starting hand shapes (taxonomy.yaml).
TAXONOMY = _load_taxonomy()
GRASP_TYPES.update({name: t for name, (t, _) in TAXONOMY.items()})
TAXONOMY_JOINT_POS = {name: q for name, (_, q) in TAXONOMY.items()}


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
  if spec.side != "palmar":
    return _side_contact(model, spec)
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


def _side_contact(model: mujoco.MjModel, spec: ContactSpec) -> ResolvedContact:
  """A non-palmar face: from the link's axis (the pad / capsule centre) a ray
  comes back from outside along -side onto the link's collision geoms, at
  the zero joint pose."""
  f = spec.finger
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  if spec.link == "tip":
    site = model.site(f"right_finger{f}_tip")
    body = int(site.bodyid[0])
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, site.quat)
    axis_b = site.pos + rot.reshape(3, 3) @ np.array(
      [0, 0, HAND2_TIP_PAD_POINTS[f - 1][2]]
    )
    geoms = [f"right_finger{f}_link4_col", f"right_finger{f}_tip_sensor_col"]
  else:
    link = {"middle": 3, "proximal": 2}[spec.link]
    body = model.body(f"right_finger{f}_link{link}").id
    geoms = [f"right_finger{f}_link{link}_col"]
    axis_b = model.geom(geoms[0]).pos.copy()
  rot_b, pos_b = data.xmat[body].reshape(3, 3), data.xpos[body]
  direction = rot_b @ _SIDES[spec.side]
  start = pos_b + rot_b @ axis_b + 0.04 * direction
  hits = []
  for name in geoms:
    g = model.geom(name).id
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      dist = mujoco.mj_rayMesh(model, data, g, start, -direction)
    else:
      dist = mujoco.mju_rayGeom(
        data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], start, -direction,
        int(model.geom_type[g]),
      )  # fmt: skip
    if dist >= 0:
      hits.append(dist)
  if not hits:
    raise ValueError(f"finger {f} {spec.link}: no {spec.side} surface")
  surface = start - min(hits) * direction
  point = rot_b.T @ (surface - PAD_DEPTH * direction - pos_b)
  return ResolvedContact(body, point, _SIDES[spec.side].copy(), f)
