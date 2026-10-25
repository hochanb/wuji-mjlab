# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""In-hand objects assembled from primitive geoms.

Each object is one free body named ``cube`` whose origin (the pivot) is the total
centre of mass, so the reorient task terms that look up ``cube`` /
``cube_visual`` / ``cube_center`` work unchanged. The first part keeps the
``cube`` / ``cube_visual`` geom names; other parts are ``cube_<i>`` /
``cube_<i>_visual``. Contact parameters match ``xmls/cube.xml``.
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


def _rotate_vec(quat, v) -> np.ndarray:
  conj = (quat[0], -quat[1], -quat[2], -quat[3])
  return _quat_mul(_quat_mul(quat, (0.0, *v)), conj)[1:]


def _parts_com(parts: list[Part]) -> np.ndarray:
  masses = np.array([p.density * p.volume for p in parts])
  return (masses[:, None] * np.array([p.pos for p in parts])).sum(0) / masses.sum()


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


def stapler_fingertip_targets(
  open_deg: float = 15.0,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
  """Fingertip (point, outward direction) targets in the stapler body frame.

  Thumb first: on the top plate, 1 cm in from its front (+x, away from the
  hinge) edge. Index to pinky: along the underside of the base plate, front to
  back. See ``InHandReorientCommandCfg.fingertip_targets``.
  """
  parts = stapler_parts(open_deg)
  com = _parts_com(parts)
  base, top = parts[0], parts[2]

  def on(part: Part, local, normal):
    point = np.asarray(part.pos) + _rotate_vec(part.quat, local) - com
    direction = _rotate_vec(part.quat, normal)
    return tuple(point.round(5).tolist()), tuple(direction.round(5).tolist())

  thumb = on(top, (top.size[0] - 0.01, 0.0, top.size[2]), (0.0, 0.0, 1.0))
  fingers = [
    on(base, (x, 0.0, -base.size[2]), (0.0, 0.0, -1.0))
    for x in (0.045, 0.015, -0.015, -0.045)
  ]
  return (thumb, *fingers)


def smartphone_parts() -> list[Part]:
  """Smartphone: a slab (12.15 x 6.48 x 0.8 cm, long axis x, top end +x,
  screen +z) with a 3 x 4 grid of thin square keys (1.5 mm tall) on the front,
  90% of the width wide and as far from the bottom (-x) edge as from the long
  sides, two thin volume buttons on the upper left
  side and a power button just above the middle of the right side (left / right
  as seen facing the screen with the top up: +y / -y). Every part is a box
  primitive, so collisions use the exact boxes rather than a mesh's convex
  hull."""
  body_color = (0.15, 0.15, 0.17, 1.0)
  button_color = (0.75, 0.76, 0.78, 1.0)

  body_hx, body_hy, body_hz = 0.06075, 0.0324, 0.004
  # Key / pitch proportions of 8 mm keys at 10.5 mm (rows, along x) and 11 mm
  # (columns, along y) pitch, scaled so the grid spans 90% of the width.
  grid_w = 0.9 * 2 * body_hy
  unit = grid_w / (2 * 11 + 8)
  key_h, key_hz = 4 * unit, 0.00075
  pitch_x, pitch_y = 10.5 * unit, 11 * unit
  margin = body_hy - grid_w / 2
  grid_cx = -body_hx + margin + 1.5 * pitch_x + key_h
  z = body_hz + key_hz
  keys = [
    Part(
      "box",
      (key_h, key_h, key_hz),
      (grid_cx + (row - 1.5) * pitch_x, (col - 1) * pitch_y, z),
      button_color,
    )
    for row in range(4)
    for col in range(3)
  ]
  side_hy, side_hz = 0.0005, 0.0015  # 1 mm proud of the side, 3 mm tall
  y = body_hy + side_hy
  side = [
    Part("box", (0.006, side_hy, side_hz), (0.03645, y, 0.0), button_color),  # volume up
    Part("box", (0.006, side_hy, side_hz), (0.02349, y, 0.0), button_color),  # volume down
    Part("box", (0.008, side_hy, side_hz), (0.01215, -y, 0.0), button_color),  # power
  ]
  return [
    Part("box", (body_hx, body_hy, body_hz), (0.0, 0.0, 0.0), body_color),
    *keys,
    *side,
  ]


def smartphone_key(parts: list[Part], row: int, col: int) -> Part:
  """Front key at ``row`` / ``col`` (1-based) counted from the top left as seen
  facing the screen with the top (+x) up. ``smartphone_parts`` lists the keys
  row-major from the bottom (-x) row, each row from the right (-y)."""
  return parts[1 + (4 - row) * 3 + (3 - col)]


def smartphone_fingertip_targets(
  thumb_key: tuple[int, int] = (2, 3),
  finger_x: tuple[float, float, float, float] = (0.012, -0.007, -0.031, -0.052),
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
  """Fingertip (point, outward direction) targets in the smartphone body frame.

  Thumb first: the top face centre of the front key ``thumb_key`` (see
  ``smartphone_key``). Index to pinky: on the left (+y) side face, mid
  thickness, at ``finger_x`` along the long axis (pre-COM, top end +x).
  """
  parts = smartphone_parts()
  com = _parts_com(parts)
  body, key = parts[0], smartphone_key(parts, *thumb_key)

  def target(point, normal):
    return tuple((np.asarray(point) - com).round(5).tolist()), normal

  thumb = target(np.asarray(key.pos) + (0.0, 0.0, key.size[2]), (0.0, 0.0, 1.0))
  fingers = [target((x, body.size[1], 0.0), (0.0, 1.0, 0.0)) for x in finger_x]
  return (thumb, *fingers)


COMPOUND_OBJECTS = {"stapler": stapler_parts, "smartphone": smartphone_parts}


def compound_spec(name: str) -> mujoco.MjSpec:
  parts = COMPOUND_OBJECTS[name]()
  masses = np.array([p.density * p.volume for p in parts])
  com = _parts_com(parts)

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
      solref=[0.015, 1.0],
      solimp=[0.85, 0.85, 0.001, 0.5, 2.0],
      **common,
    )
  body.add_site(name="cube_center", pos=[0.0, 0.0, 0.0], group=4)
  return spec


def get_compound_object_cfg(name: str) -> EntityCfg:
  from functools import partial

  from mjlab.entity import EntityCfg

  return EntityCfg(spec_fn=partial(compound_spec, name))
