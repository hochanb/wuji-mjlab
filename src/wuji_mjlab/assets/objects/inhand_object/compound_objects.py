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


def cube_parts() -> list[Part]:
  """The 54 mm / 120 g reorient cube as a single box."""
  return [Part("box", (0.027, 0.027, 0.027), (0.0, 0.0, 0.0), (0.85, 0.55, 0.2, 1.0))]


def stapler_half_open_parts() -> list[Part]:
  """The stapler opened 7.5 deg, half the default."""
  return stapler_parts(open_deg=7.5)


def spray_parts(
  radius: float = 0.0203125,
  height: float = 0.1171875,
  trigger_radius: float = 0.01,
  trigger_height: float = 0.01875,
) -> list[Part]:
  """Spray can: a cylindrical body along z (40.6 mm across, 117.2 mm tall)
  and a cylindrical trigger (button) centred on its top (20 mm across,
  18.75 mm tall); the reorient cube's density (~121 g)."""
  body = Part("cylinder", (radius, height / 2), (0.0, 0.0, 0.0), (0.2, 0.45, 0.85, 1.0))
  trigger = Part(
    "cylinder",
    (trigger_radius, trigger_height / 2),
    (0.0, 0.0, height / 2 + trigger_height / 2),
    (0.85, 0.15, 0.12, 1.0),
  )
  return [body, trigger]


def remocon_parts(
  length: float = 0.16 * 2 / 3,
  width: float = 0.045,
  thickness: float = 0.018,
  key: float = 0.009,
  pitch: float = 0.012,
  key_height: float = 0.0015,
  grid_x: float = -0.015,
  power_inset: float = 0.02,
  power_key: float = 0.014,
  scale: float = 1.25,
) -> list[Part]:
  """Remote control: a slab (10.7 x 4.5 x 1.8 cm before ``scale``; 13.3 x
  5.6 x 2.25 cm at 1.25; long axis x, top end +x, buttons +z) with a 3 x 4
  grid of square keys (9 mm, 1.5 mm proud, 12 mm pitch before ``scale``; 3
  across, 4 along the length) centred across the width at ``grid_x`` and a
  power key (``power_key`` across, 14 mm before ``scale``) ``power_inset`` below
  the top end. ``scale`` multiplies every
  length. Box primitives only, so collisions use the exact boxes (no convex
  hull)."""
  body_color = (0.12, 0.12, 0.14, 1.0)
  key_color = (0.75, 0.76, 0.78, 1.0)
  length, width, thickness = length * scale, width * scale, thickness * scale
  key, pitch, key_height = key * scale, pitch * scale, key_height * scale
  grid_x, power_inset, power_key = (
    grid_x * scale,
    power_inset * scale,
    power_key * scale,
  )
  hx, hy, hz = length / 2, width / 2, thickness / 2
  kh, kz = key / 2, key_height / 2
  z = hz + kz
  keys = [
    Part(
      "box",
      (kh, kh, kz),
      (grid_x + (1.5 - row) * pitch, (1 - col) * pitch, z),
      key_color,
    )
    for row in range(4)  # top row first
    for col in range(3)  # left (+y, seen facing the buttons with the top up) first
  ]
  ph = power_key / 2
  power = Part("box", (ph, ph, kz), (hx - power_inset, 0.0, z), (0.8, 0.12, 0.1, 1.0))
  return [Part("box", (hx, hy, hz), (0.0, 0.0, 0.0), body_color), *keys, power]


_CYL_ALONG_X = (float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0)

# Flashlight (long axis x, head at +x): body tube, a slightly thicker head and a
# thin flat slide switch on the body's +z side, a little towards the head.
FLASHLIGHT_SCALE = 1.2 * 1.25  # on the 2.6 cm x 11 cm body below
FLASHLIGHT_BODY_RADIUS = 0.013 * FLASHLIGHT_SCALE  # 3.9 cm diameter
# The body's front end (where the head attaches) is fixed; its tail quarter is
# cut off: 16.5 cm -> 12.4 cm.
FLASHLIGHT_BODY_FRONT_X = 0.055 * FLASHLIGHT_SCALE
FLASHLIGHT_BODY_HALF_LENGTH = 0.055 * FLASHLIGHT_SCALE * 0.75  # 12.4 cm long
FLASHLIGHT_HEAD_RADIUS = 0.016 * FLASHLIGHT_SCALE  # 4.8 cm diameter
FLASHLIGHT_HEAD_HALF_LENGTH = (
  0.015 * FLASHLIGHT_SCALE * 0.75
)  # 3.4 cm long (3/4 length)
FLASHLIGHT_SWITCH_SCALE = 1.5  # on the switch at FLASHLIGHT_SCALE
FLASHLIGHT_SWITCH_HALF = tuple(
  h * FLASHLIGHT_SCALE * FLASHLIGHT_SWITCH_SCALE for h in (0.006, 0.004, 0.0015)
)  # 2.7 x 1.8 x 0.675 cm
FLASHLIGHT_SWITCH_X = (
  0.015 * FLASHLIGHT_SCALE
)  # switch centre (x = 0: uncut body middle)


def flashlight_parts() -> list[Part]:
  """Flashlight: body cylinder, thicker head cylinder at +x, slide switch on +z."""
  body_r, body_hl = FLASHLIGHT_BODY_RADIUS, FLASHLIGHT_BODY_HALF_LENGTH
  sx, sy, sz = FLASHLIGHT_SWITCH_HALF
  return [
    Part(
      "cylinder",
      (body_r, body_hl),
      (FLASHLIGHT_BODY_FRONT_X - body_hl, 0.0, 0.0),
      (0.15, 0.15, 0.17, 1.0),
      _CYL_ALONG_X,
    ),
    Part(
      "cylinder",
      (FLASHLIGHT_HEAD_RADIUS, FLASHLIGHT_HEAD_HALF_LENGTH),
      (FLASHLIGHT_BODY_FRONT_X + FLASHLIGHT_HEAD_HALF_LENGTH, 0.0, 0.0),
      (0.75, 0.76, 0.78, 1.0),
      _CYL_ALONG_X,
    ),
    Part(
      "box",
      (sx, sy, sz),
      (FLASHLIGHT_SWITCH_X, 0.0, body_r + sz),
      (0.9, 0.35, 0.05, 1.0),
    ),
  ]


def iphone18pro_parts(
  length: float = 0.150,
  width: float = 0.0719,
  thickness: float = 0.00875,
  mass: float = 0.211,
) -> list[Part]:
  """Phone sized as an iPhone 18 Pro (15.0 x 7.19 x 0.875 cm, 211 g;
  long axis x, top end +x, screen +z) with a 3 x 4 grid of thin square keys
  (1.5 mm tall) on the front, 90% of the width wide and as far from the
  bottom (-x) edge as from the long sides, two thin volume buttons on the
  upper left side and a power button just above the middle of the right side
  (left / right as seen facing the screen with the top up: +y / -y). One
  uniform density giving ``mass`` in total. Every part is a box primitive, so
  collisions use the exact boxes rather than a mesh's convex hull."""
  body_color = (0.15, 0.15, 0.17, 1.0)
  button_color = (0.75, 0.76, 0.78, 1.0)

  body_hx, body_hy, body_hz = length / 2, width / 2, thickness / 2
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
    (
      "box",
      (key_h, key_h, key_hz),
      (grid_cx + (row - 1.5) * pitch_x, (col - 1) * pitch_y, z),
    )
    for row in range(4)
    for col in range(3)
  ]
  side_hy, side_hz = 0.0005, 0.0015  # 1 mm proud of the side, 3 mm tall
  y = body_hy + side_hy
  side = [  # at fractions of the half length above the middle
    ("box", (0.006, side_hy, side_hz), (0.6 * body_hx, y, 0.0)),  # volume up
    ("box", (0.006, side_hy, side_hz), (0.3867 * body_hx, y, 0.0)),  # volume down
    ("box", (0.008, side_hy, side_hz), (0.2 * body_hx, -y, 0.0)),  # power
  ]
  body = ("box", (body_hx, body_hy, body_hz), (0.0, 0.0, 0.0))
  shapes = [body, *keys, *side]
  volume = sum(8.0 * float(np.prod(size)) for _, size, _ in shapes)
  density = mass / volume
  colors = [body_color] + [button_color] * (len(shapes) - 1)
  return [
    Part(kind, size, pos, rgba, density=density)
    for (kind, size, pos), rgba in zip(shapes, colors, strict=True)
  ]


def iphone18pro_key(parts: list[Part], row: int, col: int) -> Part:
  """Front key at ``row`` / ``col`` (1-based) counted from the top left as seen
  facing the screen with the top (+x) up. ``iphone18pro_parts`` lists the keys
  row-major from the bottom (-x) row, each row from the right (-y)."""
  return parts[1 + (4 - row) * 3 + (3 - col)]


COMPOUND_OBJECTS = {
  "stapler": stapler_parts,
  "stapler_half_open": stapler_half_open_parts,
  "cube": cube_parts,
  "spray": spray_parts,
  "remocon": remocon_parts,
  "flashlight": flashlight_parts,
  "iphone18pro": iphone18pro_parts,
}


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
