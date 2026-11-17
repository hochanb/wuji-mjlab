# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""The Wuji Hand 2 as it is used (task independent): the calibration rig it
is mounted on (mount, jig, wrist tag), its capsule collision model, contact
parameters, and the actuator gains and force limits measured on the
hardware, optionally with the glove covers (``glove_covers.yaml``).
``get_hand2_model_spec`` builds it; tasks add their simulation options (e.g.
the reorient task's ``get_hand2_reorient_spec``) and tools (grasp synthesis)
theirs."""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import mujoco
import yaml

from wuji_mjlab.assets.robots.wuji_hand2.wuji_hand2_cfg import wuji_hand2_mesh_dir

# Calibration rig on the palm: mount, jig (visual) and wrist tag poses.
HAND2_MOUNT_POS = (-0.0030000041, -0.000250110803, 0.0285)
HAND2_MOUNT_ROT = (1.0, 0.0, 0.0, -0.0000081995)
HAND2_TAG_POS = (0.0, 0.037408, 0.034737)
HAND2_TAG_ROT = (0.5, -0.5, -0.5, -0.5)
HAND2_JIG_POS = (0.0, -0.1062088, 0.0178309)
HAND2_JIG_ROT = (0.541668573, -0.454527399, 0.454527399, 0.541668573)

HAND2_GEOM_DEFAULTS = {
  "group": 3,
  "friction": (0.7, 0.005, 0.0001),
  "solref": (0.02, 1.5),
  "solimp": (0.9, 0.95, 0.001, 0.5, 2.0),
  "priority": 1,
}

HAND2_ACTUATOR_GAINS = {
  "right_finger1_joint1": (1.354985, 0.04769),
  "right_finger1_joint2": (2.198821, 0.062838),
  "right_finger1_joint3": (0.886467, 0.027982),
  "right_finger1_joint4": (0.654026, 0.020457),
  "right_finger2_joint1": (1.345207, 0.047402),
  "right_finger2_joint2": (0.88946, 0.027239),
  "right_finger2_joint3": (0.779361, 0.024489),
  "right_finger2_joint4": (0.656424, 0.020508),
  "right_finger3_joint1": (1.299313, 0.045699),
  "right_finger3_joint2": (0.942641, 0.027846),
  "right_finger3_joint3": (0.73725, 0.023408),
  "right_finger3_joint4": (0.710314, 0.022366),
  "right_finger4_joint1": (1.276892, 0.046209),
  "right_finger4_joint2": (1.013334, 0.028704),
  "right_finger4_joint3": (0.859159, 0.027436),
  "right_finger4_joint4": (0.648064, 0.020448),
  "right_finger5_joint1": (1.102915, 0.043816),
  "right_finger5_joint2": (0.910201, 0.026593),
  "right_finger5_joint3": (0.861968, 0.027476),
  "right_finger5_joint4": (0.656704, 0.020611),
}

HAND2_ACTUATOR_FORCE_LIMITS = {
  "right_finger1_joint1": 1.02,
  "right_finger1_joint2": 0.83,
  "right_finger1_joint3": 0.35,
  "right_finger1_joint4": 0.21,
  "right_finger2_joint1": 1.05,
  "right_finger2_joint2": 0.38,
  "right_finger2_joint3": 0.31,
  "right_finger2_joint4": 0.30,
  "right_finger3_joint1": 1.06,
  "right_finger3_joint2": 0.39,
  "right_finger3_joint3": 0.32,
  "right_finger3_joint4": 0.29,
  "right_finger4_joint1": 1.02,
  "right_finger4_joint2": 0.41,
  "right_finger4_joint3": 0.28,
  "right_finger4_joint4": 0.30,
  "right_finger5_joint1": 0.87,
  "right_finger5_joint2": 0.36,
  "right_finger5_joint3": 0.28,
  "right_finger5_joint4": 0.30,
}

HAND2_COLLISION_CAPSULES = {
  "right_finger1_link1": (
    (
      0.0001600994,
      -0.0025818024,
      -0.001025414,
      0.0010229472,
      0.00022010189,
      -0.014007877,
    ),
    0.008326602,
  ),
  "right_finger1_link2": (
    (
      1.549657e-06,
      -3.15346e-06,
      -0.03143319,
      1.556016e-06,
      -3.148208e-06,
      -0.004805587,
    ),
    0.00874906,
  ),
  "right_finger1_link3": (
    (
      -5.1675061e-05,
      -9.0148385e-07,
      -0.034401508,
      -5.1675061e-05,
      -9.0148385e-07,
      -0.00053389498,
    ),
    0.008768497,
  ),
  "right_finger2_link1": (
    (
      2.0003946e-20,
      -0.0012000001,
      -0.00087331082,
      -7.5588199e-19,
      -0.0012000001,
      -0.01354449,
    ),
    0.009388893,
  ),
  "right_finger2_link2": (
    (
      1.43177e-05,
      1.68479e-07,
      -0.003500009,
      -3.205759e-06,
      1.684791e-07,
      -0.03710302,
    ),
    0.007507256,
  ),
  "right_finger2_link3": (
    (
      -1.805026e-18,
      -1.1052597e-34,
      -0.029478312,
      2.1212593e-21,
      1.2988967e-37,
      3.4642794e-05,
    ),
    0.00801817,
  ),
  "right_finger3_link1": (
    (
      2.0003946e-20,
      -0.0012000001,
      -0.00087331082,
      -7.5588199e-19,
      -0.0012000001,
      -0.01354449,
    ),
    0.009388893,
  ),
  "right_finger3_link2": (
    (
      1.43177e-05,
      1.68479e-07,
      -0.003500009,
      -3.205759e-06,
      1.684791e-07,
      -0.03710302,
    ),
    0.007507256,
  ),
  "right_finger3_link3": (
    (
      -1.805026e-18,
      -1.1052597e-34,
      -0.029478312,
      2.1212593e-21,
      1.2988967e-37,
      3.4642794e-05,
    ),
    0.00801817,
  ),
  "right_finger4_link1": (
    (
      2.0003946e-20,
      -0.0012000001,
      -0.00087331082,
      -7.5588199e-19,
      -0.0012000001,
      -0.01354449,
    ),
    0.009388893,
  ),
  "right_finger4_link2": (
    (
      1.43177e-05,
      1.68479e-07,
      -0.003500009,
      -3.205759e-06,
      1.684791e-07,
      -0.03710302,
    ),
    0.007507256,
  ),
  "right_finger4_link3": (
    (
      -1.805026e-18,
      -1.1052597e-34,
      -0.029478312,
      2.1212593e-21,
      1.2988967e-37,
      3.4642794e-05,
    ),
    0.00801817,
  ),
  "right_finger5_link1": (
    (
      2.0003975e-20,
      -0.0012000005,
      -0.00087331082,
      -7.5588196e-19,
      -0.0012000005,
      -0.01354449,
    ),
    0.009407771,
  ),
  "right_finger5_link2": (
    (
      -2.479326e-08,
      9.787299e-07,
      -0.03333651,
      -2.479326e-08,
      9.787299e-07,
      -0.003263492,
    ),
    0.007500022,
  ),
  "right_finger5_link3": (
    (
      -1.6782627e-18,
      -1.0276395e-34,
      -0.02740811,
      -2.177578e-21,
      -1.333382e-37,
      -3.5562547e-05,
    ),
    0.008049908,
  ),
}

_HAND2_MESH_DIR = wuji_hand2_mesh_dir("right")
_HAND2_GLOVE_COVERS = Path(__file__).resolve().parent / "glove_covers.yaml"
HAND2_GLOVE_RGBA = (0.05, 0.05, 0.05, 1.0)


def apply_hand2_actuator_gains(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Apply the Wuji Hand 2 actuator gains measured on the hardware."""
  actuators = {actuator.name: actuator for actuator in spec.actuators}
  for joint_name, (kp, kv) in HAND2_ACTUATOR_GAINS.items():
    actuator_name = f"{joint_name}_actuator"
    try:
      actuator = actuators[actuator_name]
    except KeyError as exc:
      raise ValueError(f"missing Wuji Hand 2 actuator {actuator_name!r}") from exc
    actuator.set_to_position(kp=kp, kv=kv)
  return spec


def apply_hand2_force_limits(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Joint actuator force limits must match actuator limits or they can clip the requested torque before the actuator limit is reached."""
  actuators = {actuator.name: actuator for actuator in spec.actuators}
  for joint_name, limit in HAND2_ACTUATOR_FORCE_LIMITS.items():
    actuator_name = f"{joint_name}_actuator"
    try:
      actuator = actuators[actuator_name]
    except KeyError as exc:
      raise ValueError(f"missing Wuji Hand 2 actuator {actuator_name!r}") from exc
    actuator.forcerange = [-limit, limit]
    actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint = spec.joint(joint_name)
    joint.actfrcrange = [-limit, limit]
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
  return spec


def apply_hand2_collision_overlay(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  collision_meshes = {geom.name: geom for geom in spec.geoms}
  for body_name, (fromto, radius) in HAND2_COLLISION_CAPSULES.items():
    geom_name = f"{body_name}_col"
    try:
      collision_mesh = collision_meshes[geom_name]
    except KeyError as exc:
      raise ValueError(f"missing hand2 collision mesh {geom_name!r}") from exc
    collision_mesh.name = f"{geom_name}_mesh_disabled"
    collision_mesh.contype = 0
    collision_mesh.conaffinity = 0

    link_number = int(body_name[-1])
    enabled = link_number != 1
    spec.body(body_name).add_geom(
      name=geom_name,
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=fromto,
      size=(radius, 0.0, 0.0),
      contype=int(enabled),
      conaffinity=int(enabled),
      group=3,
      density=0,
    )

  while spec.excludes:
    spec.delete(spec.excludes[0])
  spec.add_exclude(
    bodyname1="right_finger1_link2",
    bodyname2="right_palm_link",
  )
  return spec


def add_frame_axes(
  body,
  *,
  half_length: float = 0.03,
  half_width: float = 0.0015,
  origin_radius: float = 0.0035,
) -> None:
  """Add visual-only XYZ axes to a body frame."""
  body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(origin_radius, 0.0, 0.0),
    rgba=(1.0, 1.0, 0.0, 1.0),
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
  )
  for axis, rgba in (
    (0, (1.0, 0.0, 0.0, 1.0)),
    (1, (0.0, 1.0, 0.0, 1.0)),
    (2, (0.0, 0.0, 1.0, 1.0)),
  ):
    size = [half_width, half_width, half_width]
    size[axis] = half_length
    pos = [0.0, 0.0, 0.0]
    pos[axis] = half_length
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=pos,
      rgba=rgba,
      contype=0,
      conaffinity=0,
      group=1,
      density=0,
    )


def inject_hand2_rig(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Attach the calibration mount, jig, and wrist tag to the hand2 palm."""
  assets = dict(spec.assets)
  for mesh_name in ("r_mount", "jig_table"):
    filename = f"{mesh_name}.STL"
    spec.add_mesh(name=mesh_name, content_type="model/stl", file=filename)
    asset_key = f"{spec.meshdir}/{filename}" if spec.meshdir else filename
    assets[asset_key] = (_HAND2_MESH_DIR / filename).read_bytes()
  spec.assets = assets

  palm = spec.body("right_palm_link")
  mount = palm.add_body(
    name="r_mount",
    pos=HAND2_MOUNT_POS,
    quat=HAND2_MOUNT_ROT,
    mass=0.069,
    ipos=(-0.00285955798829, -0.00108266456258, -0.0169824080918),
    iquat=(-0.51751125, -0.18292100, 0.83364234, 0.06133891),
    inertia=(2.60409226e-05, 2.50870849e-05, 1.69216352e-05),
    explicitinertial=1,
  )
  mount.add_geom(
    name="r_mount_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="r_mount",
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
    rgba=(0.75, 0.75, 0.75, 1.0),
  )
  mount.add_geom(
    name="jig_table_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="jig_table",
    pos=HAND2_JIG_POS,
    quat=HAND2_JIG_ROT,
    contype=0,
    conaffinity=0,
    group=1,
    density=0,
    rgba=(0.35, 0.55, 0.9, 0.55),
  )

  tag = mount.add_body(
    name="right_wrist_tag_frame",
    pos=HAND2_TAG_POS,
    quat=HAND2_TAG_ROT,
  )
  tag.add_site(
    name="right_wrist_tag",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.0252, 0.0252, 0.0006),
    rgba=(1.0, 0.9, 0.0, 0.5),
    group=1,
  )
  add_frame_axes(tag)
  return spec


def apply_hand2_contact(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Contact parameters of the hand geoms left at the defaults
  (HAND2_GEOM_DEFAULTS); the tip sites go to group 4."""
  defaults = spec.default.geom
  for geom in spec.geoms:
    if geom.group == defaults.group:
      geom.group = HAND2_GEOM_DEFAULTS["group"]
    if (geom.friction == defaults.friction).all():
      geom.friction = HAND2_GEOM_DEFAULTS["friction"]
    if (geom.solref == defaults.solref).all():
      geom.solref = HAND2_GEOM_DEFAULTS["solref"]
    if (geom.solimp == defaults.solimp).all():
      geom.solimp = HAND2_GEOM_DEFAULTS["solimp"]
    if geom.priority == defaults.priority:
      geom.priority = HAND2_GEOM_DEFAULTS["priority"]
  for finger in range(1, 6):
    spec.site(f"right_finger{finger}_tip").group = 4
  return spec


@cache
def hand2_glove_covers() -> dict:
  """``glove_covers.yaml``: {"covers": {name: parent, mesh, pose, inertia},
  "excludes": [[body, body], ...]}."""
  return yaml.safe_load(_HAND2_GLOVE_COVERS.read_text())


def apply_hand2_glove(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Add the glove's soft covers (palm, proximal and middle phalanges; the
  fingertips stay bare): per cover a body welded to its link with a black
  visual mesh and a collision mesh (``<cover>_col``, the hand's contact
  parameters via ``apply_hand2_contact``), plus the upstream cover contact
  exclusions. Call after ``apply_hand2_collision_overlay``, which clears the
  exclusions."""
  data = hand2_glove_covers()
  assets = dict(spec.assets)
  for name, cover in data["covers"].items():
    mesh, filename = cover["mesh"], f"{cover['mesh']}.STL"
    spec.add_mesh(name=mesh, content_type="model/stl", file=filename)
    asset_key = f"{spec.meshdir}/{filename}" if spec.meshdir else filename
    assets[asset_key] = (_HAND2_MESH_DIR / filename).read_bytes()
    body = spec.body(cover["parent"]).add_body(
      name=name,
      pos=cover["pos"],
      quat=cover["quat"],
      mass=cover["mass"],
      ipos=cover["ipos"],
      iquat=cover["iquat"],
      inertia=cover["inertia"],
      explicitinertial=1,
    )
    body.add_geom(
      name=f"{name}_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh,
      contype=0,
      conaffinity=0,
      group=1,
      density=0,
      rgba=HAND2_GLOVE_RGBA,
    )
    body.add_geom(
      name=f"{name}_col",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh,
      density=0,
    )
  spec.assets = assets
  for body1, body2 in data["excludes"]:
    spec.add_exclude(bodyname1=body1, bodyname2=body2)
  return spec


def get_hand2_model_spec(
  base_spec_fn: Callable[[], mujoco.MjSpec],
  glove: bool = False,
) -> mujoco.MjSpec:
  """The hand on its calibration rig: rig, capsule collisions, contact
  parameters, measured actuator gains and force limits (no simulation
  options); ``glove=True`` adds the glove covers (``apply_hand2_glove``)."""
  spec = inject_hand2_rig(base_spec_fn())
  spec = apply_hand2_collision_overlay(spec)
  if glove:
    spec = apply_hand2_glove(spec)
  spec = apply_hand2_contact(spec)
  spec = apply_hand2_actuator_gains(spec)
  return apply_hand2_force_limits(spec)
