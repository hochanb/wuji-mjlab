# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""In-hand objects assembled from primitive geoms.

Each object is one free body named ``cube`` whose origin (the pivot) is the total
centre of mass, so the reorient task terms that look up ``cube`` /
``cube_visual`` / ``cube_center`` work unchanged. The first part keeps the
``cube`` / ``cube_visual`` geom names; other parts are ``cube_<i>`` /
``cube_<i>_visual``. Friction and condim match ``xmls/cube.xml``; the contact
softness does not (see ``_SOLMIX``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg

_IDENTITY = (1.0, 0.0, 0.0, 0.0)
_CYL_ALONG_Y = (float(np.cos(np.pi / 4)), float(-np.sin(np.pi / 4)), 0.0, 0.0)

# Same density as the 54 mm / 120 g reorient cube.
CUBE_DENSITY = 0.120 / 0.054**3

# Hand-object contact softness. The hand geoms (priority 1) are soft pads with a
# few mm of give, which lets a squeeze push a finger through a thin part such as
# the 8 mm stapler base. At equal priority MuJoCo averages solref/solimp by
# solmix and takes the larger friction, so a dominant solmix makes these stiff
# values win while the hand's friction (and its randomisation) still applies.
_PRIORITY = 1
_SOLMIX = 100.0
_SOLREF = (0.02, 1.0)  # 0.02 s: the 2 * timestep floor at dt = 0.01
_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class Part:
  """A primitive in the object frame before COM centring.

  ``size`` follows MuJoCo: box half extents, cylinder (radius, half length)
  along the geom z axis.
  """

  kind: str  # "box" | "cylinder"
  size: tuple[float, ...]
  pos: tuple[float, float, float]
  rgba: tuple[float, float, float, float]
  quat: tuple[float, float, float, float] = _IDENTITY
  density: float = CUBE_DENSITY

  @property
  def volume(self) -> float:
    if self.kind == "box":
      return 8.0 * float(np.prod(self.size))
    return float(np.pi * self.size[0] ** 2 * 2.0 * self.size[1])


def _quat_mul(a, b) -> np.ndarray:
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
  )


def _rotate_about(parts: list[Part], pivot: np.ndarray, quat) -> list[Part]:
  """Rigidly rotate parts (position and orientation) about a pivot point."""
  conj = (quat[0], -quat[1], -quat[2], -quat[3])
  out = []
  for p in parts:
    rel = _quat_mul(_quat_mul(quat, (0.0, *(np.asarray(p.pos) - pivot))), conj)[1:]
    out.append(
      Part(
        kind=p.kind,
        size=p.size,
        pos=tuple((pivot + rel).tolist()),
        rgba=p.rgba,
        quat=tuple(_quat_mul(quat, p.quat).tolist()),
        density=p.density,
      )
    )
  return out


def stapler_parts(open_deg: float = 15.0) -> list[Part]:
  """Stapler: base plate, y-axis hinge cylinder, and an upper assembly (outer
  U frame around an inner staple magazine) opened by ``open_deg`` about the
  hinge. Long axis is x; the hinge is at the -x end."""
  black = (0.1, 0.1, 0.1, 1.0)
  gray = (0.5, 0.5, 0.52, 1.0)
  red = (0.8, 0.12, 0.12, 1.0)
  silver = (0.78, 0.8, 0.82, 1.0)

  base_hx, base_hy, base_hz = 0.065, 0.0175, 0.004  # 13.0 x 3.5 x 0.8 cm
  hinge_r, hinge_hl = 0.0085, 0.015
  hinge = np.array([-base_hx + hinge_r, 0.0, base_hz + hinge_r])
  lift = 0.005  # upper assembly raised so the protruding magazine clears the base
  x0, x1 = float(hinge[0]), 0.06  # upper assembly spans hinge -> front
  cx, hx = (x0 + x1) / 2, (x1 - x0) / 2

  # Closed-pose z levels (before lift): walls 0.004..0.020, top plate 0.020..0.024.
  wall_y, wall_hy = 0.01325, 0.00175
  inner_hy = wall_y - wall_hy  # frame inner half width
  gap = 0.001  # magazine side clearance to the frame inner faces
  mag_bottom, mag_top = 0.004 - 0.005, 0.020  # protrudes 5 mm below the frame
  upper = [
    Part("box", (hx, 0.015, 0.002), (cx, 0.0, 0.022 + lift), red),
    Part("box", (hx, wall_hy, 0.008), (cx, wall_y, 0.012 + lift), red),
    Part("box", (hx, wall_hy, 0.008), (cx, -wall_y, 0.012 + lift), red),
    Part(
      "box",
      (0.055, inner_hy - gap, (mag_top - mag_bottom) / 2),
      (0.002, 0.0, (mag_top + mag_bottom) / 2 + lift),
      silver,
    ),
  ]
  # The +x end lifts: rotate about the hinge's y axis by -open_deg.
  half = np.radians(-open_deg) / 2
  open_quat = (float(np.cos(half)), 0.0, float(np.sin(half)), 0.0)
  return [
    Part("box", (base_hx, base_hy, base_hz), (0.0, 0.0, 0.0), black),
    Part("cylinder", (hinge_r, hinge_hl), tuple(hinge.tolist()), gray, _CYL_ALONG_Y),
    *_rotate_about(upper, hinge, open_quat),
  ]


COMPOUND_OBJECTS = {"stapler": stapler_parts}


def compound_spec(name: str) -> mujoco.MjSpec:
  parts = COMPOUND_OBJECTS[name]()
  masses = np.array([p.density * p.volume for p in parts])
  com = (masses[:, None] * np.array([p.pos for p in parts])).sum(0) / masses.sum()

  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_freejoint")
  for i, (part, mass) in enumerate(zip(parts, masses, strict=True)):
    if part.kind == "box":
      geom_type, size = mujoco.mjtGeom.mjGEOM_BOX, list(part.size)
    else:
      geom_type, size = mujoco.mjtGeom.mjGEOM_CYLINDER, [*part.size, 0.0]
    common = dict(
      type=geom_type,
      size=size,
      pos=(np.asarray(part.pos) - com).tolist(),
      quat=list(part.quat),
    )
    suffix = "" if i == 0 else f"_{i}"
    body.add_geom(
      name=f"cube{suffix}_visual",
      contype=0,
      conaffinity=0,
      density=0,
      group=2,
      rgba=list(part.rgba),
      **common,
    )
    body.add_geom(
      name=f"cube{suffix}",
      mass=float(mass),
      group=3,
      friction=[0.3, 0.005, 0.0001],
      conaffinity=2,
      condim=3,
      priority=_PRIORITY,
      solmix=_SOLMIX,
      solref=list(_SOLREF),
      solimp=list(_SOLIMP),
      **common,
    )
  body.add_site(name="cube_center", pos=[0.0, 0.0, 0.0], group=4)
  return spec


def get_compound_object_cfg(name: str) -> EntityCfg:
  from functools import partial

  from mjlab.entity import EntityCfg

  return EntityCfg(spec_fn=partial(compound_spec, name))
